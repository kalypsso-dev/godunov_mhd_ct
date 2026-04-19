// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV0.cpp
 *
 * Godunov time integration implementation detail version v0.
 */
#include <godunov_mhd_ct/scheme/GodunovImplemV0.h>

// Compute functors
#include <godunov_mhd_ct/scheme/ConvertToPrimitivesVariablesFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeUpdatedPrimitiveVariablesFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeLimitedSlopesFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeHydroFluxesAndStoreFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeViscousFluxesAndStoreFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeElectricFieldFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeSourceFaceMagFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeHydroFluxesAndUpdateFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeEmfAndStoreFunctor.h>
#include <godunov_mhd_ct/scheme/ComputeViscousFluxesAndStoreFunctor.h>
#include <godunov_mhd_ct/scheme/ReadFluxesAndConservativeUpdateFunctor.h>
#include <godunov_mhd_ct/scheme/CorrectEmfAtBlockBorderFunctor.h>
#include <godunov_mhd_ct/scheme/ReadEmfAndUpdateFunctor.h>
#include <kalypsso/core/FillBlockGhostFaces.h>
#include <kalypsso/core/CheckEdgeSiblingsConnectivity.h>

// profiling colors
#include <godunov_mhd_ct/profiling.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::resize_auxiliary_data()
{
  const auto num_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_outside = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants_outside();
  const auto num_ghosts = this->m_amr_mesh.local_num_ghosts();
  const auto num_mirrors = this->m_amr_mesh.local_num_mirrors();
  const auto num_outsides = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants_outside();
  const auto num_outsides_ghost =
    this->m_mesh_map.get_amr_mesh_info().local_num_quadrants_outside_ghost();

  m_Q_ghosted.resize_and_reset(num_owned + num_ghosts + num_outside);
  m_Q2_ghosted.resize_and_reset(num_owned + num_ghosts + num_outside);

  // note: m_Q_ghosted_mg is sized with the number of mirror + number of ghost quadrants,
  // because ghost quadrant will be populated with MeshGhostsExchanger
  m_Q_ghosted_mg.resize_and_reset(num_mirrors + num_ghosts);

  m_Bface_ghosted.resize_and_reset(num_owned + num_ghosts + num_outsides + num_outsides_ghost);

  m_Slopes_x.resize_and_reset(num_owned + num_ghosts + num_outside);
  m_Slopes_y.resize_and_reset(num_owned + num_ghosts + num_outside);
  if constexpr (dim == 3)
    m_Slopes_z.resize_and_reset(num_owned + num_ghosts + num_outside);

  // Flux is resized large enough to store fluxes for owned + ghost blocks
  // we don't need to compute in outside quad, since outside quad are at the same level of AMR than
  // the neighbor directly inside
  m_Fluxes.resize_and_reset(num_owned + num_ghosts + num_outside);

  m_elec_field.resize_and_reset(num_owned + num_ghosts + num_outside);
  m_sFaceMag.resize_and_reset(num_owned + num_ghosts + num_outside);

  m_emf.resize_and_reset(num_owned + num_ghosts + num_outside);
  m_emf2.resize_and_reset(num_owned + num_ghosts + num_outside);

} // GodunovImplemV0<dim, device_t>::resize_auxiliary_data

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
uint64_t
GodunovImplemV0<dim, device_t>::total_mem_size_in_bytes()
{
  uint64_t total = 0;
  total += m_Q_ghosted.allocated_size_in_bytes();
  total += m_Q2_ghosted.allocated_size_in_bytes();
  total += m_Q_ghosted_mg.allocated_size_in_bytes();
  total += m_Bface_ghosted.allocated_size_in_bytes();
  total += m_Slopes_x.allocated_size_in_bytes();
  total += m_Slopes_y.allocated_size_in_bytes();
  total += m_Slopes_z.allocated_size_in_bytes();
  total += m_Fluxes.allocated_size_in_bytes();
  total += m_elec_field.allocated_size_in_bytes();
  total += m_sFaceMag.allocated_size_in_bytes();
  total += m_emf.allocated_size_in_bytes();

  return total;
} // GodunovImplemV0<dim, device_t>::total_mem_size_in_bytes

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::do_time_step(DataArrayBlock_t     U,
                                             DataArrayBlock_t     U2,
                                             FaceDataArrayBlock_t Bface,
                                             FaceDataArrayBlock_t Bface2,
                                             real_t               dt)
{

  /*
   * Note about conservativity:
   *
   * To do a conservative update, we need to ensure that at the interface between two neighbors
   * quads living at different AMR levels, the coarse quad get updated using the fluxes computed by
   * the fine quad.
   *
   * Currently we compute fluxes in all quadrants (owned + ghosts).
   * Computing fluxes in ghost quadrants is only need in case a owned quadrant has a face neighbor
   * at finer AMR level that is ghost; in that case the flux in the ghost must be used to update
   * current owned quadrant.
   */

  // start main computation
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME);

  // from Bface (non-ghosted, owned+ghost) to m_Bface_ghosted (owned)
  // fill face-centered magnetic field in owned + ghost quads + outside (only inner)
  this->fill_Bface_ghosted(Bface);

  // convert conservative variable into primitives ones m_U ==> m_Q_ghosted_mg
  // also perform a half time step gravity predictor
  this->convert_to_primitives_in_mirror_quads(U, m_Bface_ghosted);

  // do mpi comm to populate ghosts quads in m_Q_ghosted_mg (see MeshGhostExchanger)
  this->mpi_exchange_mirrors_and_ghosts(m_Q_ghosted_mg);

  // update primitive variables in owned quadrants + copy ghost quads
  // fill m_Q_ghosted (2 ghost cells all around owned blocks)
  this->convert_to_primitives(U, m_Bface_ghosted);

  this->compute_limited_slopes_in_owned_and_ghosts();

  //
  // Compute primitive variables predictor at t_{n+1/2}
  //
  this->compute_primitives_predictor(dt);

  //
  // compute electric field
  //
  this->compute_elec_field_in_owned_and_ghosts();

  //
  // compute source term magnetic field face
  //
  this->compute_sFaceMag_in_owned_and_ghosts(dt);

  //
  // compute flux along X, store and update
  //
  this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IX);
  this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IX);

  //
  // compute flux along Y, store and update
  //
  this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IY);
  this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IY);

  //
  // compute flux along Z, store and update
  //
  if constexpr (dim == 3)
  {
    this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IZ);
    this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IZ);
  }

  if (this->m_viscosity.enabled)
  {
    this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IX);
    this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IX);

    this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IY);
    this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IY);

    if constexpr (dim == 3)
    {
      this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IZ);
      this->read_fluxes_and_update_in_owned(U2, Bface2, dt, IZ);
    }
  }

  //
  // Update magnetic field
  //
  this->compute_emf_and_store_in_owned_and_ghosts(dt);
  this->read_emf_and_update_in_owned(Bface2);

  //
  // Add gravity source term when enabled
  //
  const auto gravity_enabled = this->m_config_map.getBool("gravity", "enabled", false);
  if (gravity_enabled)
    this->add_gravity_source_term(U, U2, dt);

} // GodunovImplemV0<dim, device_t>::do_time_step

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::fill_Bface_ghosted(FaceDataArrayBlock_t Bface)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_BFACE_GHOSTED);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();
  const auto num_quadrants_outside =
    this->m_mesh_map.get_amr_mesh_info().local_num_quadrants_outside();
  const auto num_quadrants_outside_ghost =
    this->m_mesh_map.get_amr_mesh_info().local_num_quadrants_outside_ghost();
  const auto amr_mesh_info = this->m_mesh_map.get_amr_mesh_info();

  FillBlockGhostFacesFunctor<dim, device_t>::apply(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    num_quadrants_owned + num_quadrants_ghost + num_quadrants_outside + num_quadrants_outside_ghost,
    0,
    num_quadrants_owned + num_quadrants_ghost + num_quadrants_outside + num_quadrants_outside_ghost,
    Bface,
    this->m_Bface_ghosted,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    amr_mesh_info);

} // GodunovImplemV0<dim, device_t>::fill_Bface_ghosted

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::convert_to_primitives_in_mirror_quads(
  DataArrayBlock_t     U,
  FaceDataArrayBlock_t Bface_ghosted)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  // compute primitive variables in owned mirror blocks (U must have MPI ghost up to date)
  ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_in_mirrors(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.mirror_orchard_keys(),
    this->m_mesh_map.get_amr_mesh_info(),
    U,
    Bface_ghosted,
    m_Q_ghosted_mg,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_mhd_settings,
    this->m_par_env);

} // GodunovImplemV0<dim, device_t>::convert_to_primitives_in_mirror_quads

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::convert_to_primitives(DataArrayBlock_t     U,
                                                      FaceDataArrayBlock_t Bface_ghosted)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  //
  // step 1: convert to primitive variables in owned quadrants
  //

  const auto num_quads_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quads_mirrors = this->m_amr_mesh.local_num_mirrors();
  const auto num_quads_ghosts = this->m_amr_mesh.local_num_ghosts();

  // compute primitive variables in all owned blocks (U must have MPI ghost up to date)
  ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.get_amr_mesh_info(),
    0,
    num_quads_owned,
    U,
    Bface_ghosted,
    m_Q_ghosted,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_mhd_settings,
    this->m_par_env);

  //
  // step 2: copy primitives variables in ghost
  //
  const auto num_ghosted_cells = m_Q_ghosted.num_cells();
  const auto num_vars = m_Q_ghosted.num_vars();

  auto range_in = std::pair<std::size_t, std::size_t>(
    num_ghosted_cells * num_vars * num_quads_mirrors,
    num_ghosted_cells * num_vars * (num_quads_mirrors + num_quads_ghosts));

  auto data_in = Kokkos::subview(m_Q_ghosted_mg.flat_view(), range_in);

  auto range_out = std::pair<std::size_t, std::size_t>(
    num_ghosted_cells * num_vars * num_quads_owned,
    num_ghosted_cells * num_vars * (num_quads_owned + num_quads_ghosts));

  auto data_out = Kokkos::subview(m_Q_ghosted.flat_view(), range_out);

  Kokkos::deep_copy(data_out, data_in);

} // GodunovImplemV0<dim, device_t>::convert_to_primitives

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_limited_slopes_in_owned_and_ghosts()
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SLOPES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // compute limited slopes in all quadrants (owned + ghost)
  ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_group(m_Q_ghosted,
                                                             m_Slopes_x,
                                                             m_Slopes_y,
                                                             m_Slopes_z,
                                                             fm,
                                                             num_quadrants_owned +
                                                               num_quadrants_ghost,
                                                             this->m_mhd_settings);

} // GodunovImplemV0<dim, device_t>::compute_limited_slopes_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_primitives_predictor(real_t dt)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_PRIM_UPDATE_PREDICTOR);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // compute primitive variables updated to t_{n+1/2} (predictor) in all quadrants (owned + ghost)
  ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::apply_on_group(
    this->m_config_map,
    this->m_mesh_map.orchard_keys(),
    m_Q_ghosted,
    m_Q2_ghosted,
    m_Slopes_x,
    m_Slopes_y,
    m_Slopes_z,
    fm,
    num_quadrants_owned + num_quadrants_ghost,
    this->m_mhd_settings,
    this->m_viscosity,
    dt);

} // GodunovImplemV0<dim, device_t>::compute_primitives_predictor

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_elec_field_in_owned_and_ghosts()
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_ELEC_FIELD);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // compute electric field in all quadrants of q_group
  ComputeElectricFieldFunctor<dim, device_t>::apply_on_group(
    m_Q_ghosted, m_elec_field, fm, num_quadrants_owned + num_quadrants_ghost);

} // GodunovImplmemV0<dim, device_t>::compute_elec_field_in_owned_and_ghosts

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_sFaceMag_in_owned_and_ghosts(real_t dt)
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SFACEMAG);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // compute source term for magnetic field in all quadrants
  ComputeSourceFaceMagFunctor<dim, device_t>::apply_on_group(m_sFaceMag,
                                                             m_elec_field,
                                                             fm,
                                                             num_quadrants_owned +
                                                               num_quadrants_ghost,
                                                             dt,
                                                             this->m_config_map,
                                                             this->m_mesh_map.orchard_keys());

} // GodunovImplemV0<dim, device_t>::compute_sFaceMag_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_fluxes_and_store_in_owned_and_ghosts(real_t dt,
                                                                             int    direction)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_FLUXES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // reshape flux to be a flux array in given direction
  auto flux_block_sizes = m_Q_ghosted.block_size();
  flux_block_sizes[static_cast<size_t>(direction)]++;
  m_Fluxes.reshape(flux_block_sizes);

  // compute fluxes and update all quadrants in a group of quadrants
  ComputeHydroFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                          this->m_mesh_map.orchard_keys(),
                                                          this->m_mesh_map.get_amr_mesh_info(),
                                                          m_Fluxes,
                                                          m_Q_ghosted,
                                                          m_Q2_ghosted,
                                                          m_Slopes_x,
                                                          m_Slopes_y,
                                                          m_Slopes_z,
                                                          m_sFaceMag,
                                                          fm,
                                                          0,
                                                          num_quadrants_owned + num_quadrants_ghost,
                                                          direction,
                                                          this->m_mhd_settings,
                                                          dt);

} // GodunovImplemV0<dim, device_t>::compute_fluxes_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_viscous_fluxes_and_store_in_owned_and_ghosts(real_t dt,
                                                                                     int direction)
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_VISCOUS_FLUXES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // reshape flux to be a flux array in given direction
  auto flux_block_sizes = m_Q_ghosted.block_size();
  flux_block_sizes[static_cast<size_t>(direction)]++;
  m_Fluxes.reshape(flux_block_sizes);

  // compute fluxes and update all quadrants in a group of quadrants
  ComputeViscousFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                            this->m_mesh_map.orchard_keys(),
                                                            this->m_mesh_map.get_amr_mesh_info(),
                                                            m_Fluxes,
                                                            m_Q_ghosted,
                                                            fm,
                                                            0,
                                                            num_quadrants_owned +
                                                              num_quadrants_ghost,
                                                            direction,
                                                            this->m_viscosity,
                                                            dt);

} // GodunovImplemV0<dim, device_t>::compute_viscous_fluxes_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::read_fluxes_and_update_in_owned(DataArrayBlock_t     u_out,
                                                                FaceDataArrayBlock_t b_out,
                                                                real_t               dt,
                                                                int                  direction)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // check flux array sizes
  {
    [[maybe_unused]] auto flux_block_sizes = m_Q_ghosted.block_size();
    flux_block_sizes[static_cast<size_t>(direction)]++;
    assertm(flux_block_sizes == m_Fluxes.shape(), "Flux array has incompatible shape.");
  }

  // compute incoming fluxes in all ghost quadrants and perform conservative update only at
  // interface between ghost and owned quadrants (in case the owned quadrant is coarser than the
  // ghost quadrant)
  ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(this->m_config_map,
                                                               this->m_mesh_map.hashmap(),
                                                               this->m_mesh_map.orchard_keys(),
                                                               this->m_mesh_map.conformal_status(),
                                                               this->m_mesh_map.get_amr_mesh_info(),
                                                               u_out,
                                                               b_out,
                                                               m_Fluxes,
                                                               fm,
                                                               direction,
                                                               this->m_brick_sizes,
                                                               this->m_is_brick_periodic,
                                                               this->m_mhd_settings,
                                                               dt);

} // GodunovImplemV0<dim, device_t>::read_fluxes_and_update_in_owned

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_emf_and_store_in_owned_and_ghosts(real_t dt)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_EMF);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_model.get_fieldmap();

  // check emf array sizes
  {
    [[maybe_unused]] auto emf_block_sizes = m_Q_ghosted.block_size() + 1;
    assertm(emf_block_sizes == m_emf.block_size(), "EMF array has incompatible shape.");
  }

  // compute incoming fluxes in all ghost quadrants and perform conservative update only at
  // interface between ghost and owned quadrants (in case the owned quadrant is coarser than the
  // ghost quadrant)
  ComputeEmfAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                  this->m_mesh_map.orchard_keys(),
                                                  this->m_mesh_map.get_amr_mesh_info(),
                                                  m_emf,
                                                  m_Q_ghosted,
                                                  m_Q2_ghosted,
                                                  m_Slopes_x,
                                                  m_Slopes_y,
                                                  m_Slopes_z,
                                                  m_sFaceMag,
                                                  fm,
                                                  0,
                                                  num_quadrants_owned + num_quadrants_ghost,
                                                  this->m_mhd_settings,
                                                  dt);

} // GodunovImplemV0<dim, device_t>::compute_emf_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::read_emf_and_update_in_owned(FaceDataArrayBlock_t b_out)
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE_MAG);

  Kokkos::deep_copy(m_emf2.logical_view(), m_emf.logical_view());

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
  KALYPSSO_INFO("==============================================================================");
  KALYPSSO_INFO("Begin checking edge connectivity at block borders");
  KALYPSSO_INFO("==============================================================================");

  core::CheckEdgeSiblingsConnectivity<dim, device_t>::apply(this->m_block_sizes,
                                                            this->m_mesh_map.hashmap(),
                                                            this->m_mesh_map.orchard_keys(),
                                                            this->m_mesh_map.get_amr_mesh_info(),
                                                            this->m_config_map,
                                                            this->m_brick_sizes,
                                                            this->m_is_brick_periodic);

  KALYPSSO_INFO("==============================================================================");
  KALYPSSO_INFO("End   checking edge connectivity at block borders");
  KALYPSSO_INFO("==============================================================================");

#endif

  // step 1: apply correction at all edge location that is non-conforming
  CorrectEmfAtBlockBorderFunctor<dim, device_t>::apply(this->m_config_map,
                                                       this->m_mesh_map.hashmap(),
                                                       this->m_mesh_map.orchard_keys(),
                                                       this->m_mesh_map.conformal_status(),
                                                       this->m_mesh_map.get_amr_mesh_info(),
                                                       this->m_block_sizes,
                                                       m_emf,
                                                       m_emf2,
                                                       this->m_brick_sizes,
                                                       this->m_is_brick_periodic,
                                                       this->m_par_env);

  // step 2: actual update of magnetic field
  ReadEmfAndUpdateFunctor<dim, device_t>::apply(
    this->m_mesh_map.get_amr_mesh_info(), b_out, m_emf2);

} // GodunovImplemV0<dim, device_t>::read_emf_and_update_in_owned

// explicit template instantiation
template class GodunovImplemV0<2, kalypsso::DefaultDevice>;
template class GodunovImplemV0<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
