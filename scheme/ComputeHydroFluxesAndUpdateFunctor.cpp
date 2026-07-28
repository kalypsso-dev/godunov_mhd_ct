// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeHydroFluxesAndUpdateFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeHydroFluxesAndUpdateFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::ComputeHydroFluxesAndUpdateFunctor(
  ConfigMap const &                  config_map,
  amr_hashmap_t const &              amr_hashmap,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  DataArrayBlock_t const &           u_in,
  DataArrayBlock_t const &           u_out,
  DataArrayGhostedBlock_t const &    q,
  DataArrayGhostedBlock_t const &    q2,
  DataArrayGhostedBlock_t const &    slopes_x,
  DataArrayGhostedBlock_t const &    slopes_y,
  DataArrayGhostedBlock_t const &    slopes_z,
  DataArrayGhostedBlock_t const &    sFaceMag,
  FieldMap<models::MHD>              fm,
  int32_t                            iOct_begin,
  int32_t                            num_octants,
  brick_size_t<dim> const &          brick_sizes,
  Kokkos::Array<bool, dim> const &   is_brick_periodic,
  MHDSettings const &                mhd_settings,
  real_t                             dt)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_conformal_status(conformal_status)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Uin(u_in)
  , m_Uout(u_out)
  , m_q(q)
  , m_q2(q2)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_sFaceMag(sFaceMag)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_block_sizes(slopes_x.block_size())
  , m_block_sizes_fluxes(slopes_x.block_size() + 1)
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_nbFluxesPerLeaf(Kokkos::dim_prod(m_block_sizes_fluxes))
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
  , m_stencil_helper(amr_hashmap,
                     orchard_keys,
                     slopes_x.block_size(),
                     brick_sizes,
                     is_brick_periodic)
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                  config_map,
  amr_hashmap_t const &              amr_hashmap,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  DataArrayBlock_t const &           Uin,
  DataArrayBlock_t const &           Uout,
  DataArrayGhostedBlock_t const &    q,
  DataArrayGhostedBlock_t const &    q2,
  DataArrayGhostedBlock_t const &    slopes_x,
  DataArrayGhostedBlock_t const &    slopes_y,
  DataArrayGhostedBlock_t const &    slopes_z,
  DataArrayGhostedBlock_t const &    sFaceMag,
  FieldMap<models::MHD>              fm,
  int32_t                            iOct_begin,
  int32_t                            num_octants,
  brick_size_t<dim> const &          brick_sizes,
  Kokkos::Array<bool, dim> const &   is_brick_periodic,
  MHDSettings const &                mhd_settings,
  real_t                             dt)
{

  ComputeHydroFluxesAndUpdateFunctor<dim, device_t> functor(
    config_map,
    amr_hashmap,
    orchard_keys,
    conformal_status,
    amr_mesh_info,
    Uin,
    Uout,
    q,
    q2,
    slopes_x,
    slopes_y,
    slopes_z,
    sFaceMag,
    fm,
    iOct_begin,  // first index to process
    num_octants, // number of octants to process
    brick_sizes,
    is_brick_periodic,
    mhd_settings,
    dt);

  // we use atomic update, computations is flux-oriented
  const auto nbIterations = num_octants * functor.nb_fluxes_per_leaf();

  // launch computation
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::ComputeFluxesAndConservativeUpdateFunctor - group",
    Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbIterations),
    functor);

} // apply_on_group

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::apply_on_ghosts(
  ConfigMap const &                  config_map,
  amr_hashmap_t const &              amr_hashmap,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  DataArrayBlock_t const &           Uin,
  DataArrayBlock_t const &           Uout,
  DataArrayGhostedBlock_t const &    q,
  DataArrayGhostedBlock_t const &    q2,
  DataArrayGhostedBlock_t const &    slopes_x,
  DataArrayGhostedBlock_t const &    slopes_y,
  DataArrayGhostedBlock_t const &    slopes_z,
  DataArrayGhostedBlock_t const &    sFaceMag,
  FieldMap<models::MHD>              fm,
  brick_size_t<dim> const &          brick_sizes,
  Kokkos::Array<bool, dim> const &   is_brick_periodic,
  MHDSettings const &                mhd_settings,
  real_t                             dt)
{

  ComputeHydroFluxesAndUpdateFunctor<dim, device_t> functor(
    config_map,
    amr_hashmap,
    orchard_keys,
    conformal_status,
    amr_mesh_info,
    Uin,
    Uout,
    q,
    q2,
    slopes_x,
    slopes_y,
    slopes_z,
    sFaceMag,
    fm,
    amr_mesh_info.local_num_mirrors(), // first ghost after all mirror quads
    amr_mesh_info.local_num_ghosts(),  // number of octants to process
    brick_sizes,
    is_brick_periodic,
    mhd_settings,
    dt);

  // we use atomic update, computations is flux-oriented
  const auto nbIterations = amr_mesh_info.local_num_ghosts() * functor.nb_fluxes_per_leaf();

  // launch computation
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::ComputeFluxesAndConservativeUpdateFunctor - ghosts",
    Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbIterations),
    functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::reconstruct_state_2d(const MHDStateCell & q,
                                                                        int32_t              is,
                                                                        int32_t              js,
                                                                        int32_t     iOct_local,
                                                                        int         dir,
                                                                        face_type_t face) const
{
  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

  const auto offset = face == FACE_LEFT ? KALYPSSO_NUM(-0.5) : KALYPSSO_NUM(0.5);

  // get limited slopes along given direction "dir"
  const MHDStateCell dq = get_state(get_slopes(dir), is, js, iOct_local);

  // reconstruct state on face center
  MHDStateCell qr;

  qr[ID] = q[ID] + offset * dq[ID];
  qr[IP] = q[IP] + offset * dq[IP];
  qr[IU] = q[IU] + offset * dq[IU];
  qr[IV] = q[IV] + offset * dq[IV];
  qr[IW] = q[IW] + offset * dq[IW];
  qr[IA] = q[IA] + offset * dq[IA];
  qr[IB] = q[IB] + offset * dq[IB];
  qr[IC] = q[IC] + offset * dq[IC];

  qr[ID] = fmax(smallr, qr[ID]);
  qr[IP] = fmax(smallp * qr[ID], qr[IP]);

  return qr;

} // reconstruct_state_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::reconstruct_state_3d(const MHDStateCell & q,
                                                                        int32_t              is,
                                                                        int32_t              js,
                                                                        int32_t              ks,
                                                                        int32_t     iOct_local,
                                                                        int         dir,
                                                                        face_type_t face) const
{

  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

  const auto offset = face == FACE_LEFT ? KALYPSSO_NUM(-0.5) : KALYPSSO_NUM(0.5);

  // get limited slopes along given direction "dir"
  const MHDStateCell dq = get_state(get_slopes(dir), is, js, ks, iOct_local);

  // reconstruct state on interface
  MHDStateCell qr;

  qr[ID] = q[ID] + offset * dq[ID];
  qr[IP] = q[IP] + offset * dq[IP];
  qr[IU] = q[IU] + offset * dq[IU];
  qr[IV] = q[IV] + offset * dq[IV];
  qr[IW] = q[IW] + offset * dq[IW];
  qr[IA] = q[IA] + offset * dq[IA];
  qr[IB] = q[IB] + offset * dq[IB];
  qr[IC] = q[IC] + offset * dq[IC];

  qr[ID] = fmax(smallr, qr[ID]);
  qr[IP] = fmax(smallp * qr[ID], qr[IP]);

  return qr;

} // reconstruct_state_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d_group(
  index_t const & cell_index,
  index_t const & iOct_local) const
{

  auto const iOct_global = m_iOct_begin + iOct_local;

  // i and j enumerate block faces
  // i and j are aligned with inner block (non-ghosted block)
  // i is in range [0, bx] ===> bx+1 values
  // j is in range [0, by] ===> by+1 values
  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const iq2 = i + 1;
  auto const jq2 = j + 1;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 4;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 1
  auto qprim = get_state(m_q2, iq2, jq2, iOct_local);

  /*
   * compute flux from left face along X dir and update both sides
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_state(m_q2, iq2 - 1, jq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(qprim_n, is - 1, js, iOct_local, IX, FACE_RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_2d(qprim, is, js, iOct_local, IX, FACE_LEFT);

    // fix normal component of B
    {
      const auto AL =
        m_q(i + 2, j + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, IX, iOct_local);
      qR[MHD::IA] = AL;
      qL[MHD::IA] = AL;
    }

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    const auto face_xmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, iOct_global, flux_cur);
      }

      if (i == 0 and face_xmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ -1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }

    const auto face_xmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i > 0 and j < m_block_sizes[IY])
    {
      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i - 1, j, iOct_global, flux_cur);
      }

      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<2>       coords2{ i - 1, j }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }
  }

  /*
   * compute flux from left face along Y dir and update both sides
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_state(m_q2, iq2, jq2 - 1, iOct_local);

    // step 1 :
    // reconstruct state at right face in neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(qprim_n, is, js - 1, iOct_local, IY, FACE_RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_2d(qprim, is, js, iOct_local, IY, FACE_LEFT);

    // fix normal component of B
    {
      const auto BL =
        m_q(i + 2, j + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, IY, iOct_local);
      qR[MHD::IB] = BL;
      qL[MHD::IB] = BL;
    }

    // swap IU / IV
    my_swap(qL[MHD::IU], qL[MHD::IV]);
    my_swap(qR[MHD::IU], qR[MHD::IV]);

    // swap IA / IB
    my_swap(qL[MHD::IA], qL[MHD::IB]);
    my_swap(qR[MHD::IA], qR[MHD::IB]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IV]);
    my_swap(flux[MHD::IA], flux[MHD::IB]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_ymin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, iOct_global, flux_cur);
      }

      if (j == 0 and face_ymin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, -1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }

    const auto face_ymax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (j > 0 and i < m_block_sizes[IX])
    {
      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j - 1, iOct_global, flux_cur);
      }

      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<2>       coords2{ i, j - 1 }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }
  }

} // compute_fluxes_and_update_2d_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d_ghost(
  index_t const & cell_index,
  index_t const & first_ghost,
  index_t const & iGhost) const
{

  KOKKOS_ASSERT((first_ghost + iGhost) < m_q2.num_quadrants() && "Invalid access to m_q2 view.");

  // iOct local : to be used when accessing array sized upon num_mirrors + num_ghosts like m_q
  // (aka m_Qghosted_mg in SolverHydroMusclBlock)
  auto const iOct_local = first_ghost + iGhost;

  // iOct global : to be used when accessing global arrays like U, U2 (here u_in, u_out), orchard
  // keys, hashmap, ...
  auto const iOct_global = m_amr_mesh_info.local_num_quadrants() + iGhost;

  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const iq2 = i + 1;
  auto const jq2 = j + 1;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  // auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 4;

  // get current location primitive variables state
  // note: primitive variables Q2 is a ghosted array with ghost width of 1
  const auto qprim = get_state(m_q2, iq2, jq2, iGhost);

  /*
   * Face XMIN - we only need to update a coarser neighbor
   */
  const auto face_xmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmin_neighbor_is_coarser and i == 0 and j < m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ -1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_state(m_q2, iq2 - 1, jq2, iGhost);

      // step 1 :
      // reconstruct state at right face in neighbor (index relative to slopes array)
      auto qL = reconstruct_state_2d(qprim_n, is - 1, js, iGhost, IX, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_2d(qprim, is, js, iGhost, IX, FACE_LEFT);

      // fix normal component of B
      {
        const auto AL = m_q(i + 2, j + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, IX, iGhost);
        qR[MHD::IA] = AL;
        qL[MHD::IA] = AL;
      }

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face XMAX - we only need to update a coarser neighbor
   */
  const auto face_xmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmax_neighbor_is_coarser and i == m_block_sizes[IX] and j < m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<2>       coords2{ i - 1, j }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_state(m_q2, iq2 - 1, jq2, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_2d(qprim_n, is - 1, js, iGhost, IX, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_2d(qprim, is, js, iGhost, IX, FACE_LEFT);

      // fix normal component of B
      {
        const auto AL = m_q(i + 2, j + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, IX, iGhost);
        qR[MHD::IA] = AL;
        qL[MHD::IA] = AL;
      }

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face YMIN - we only need to update a coarser neighbor
   */
  const auto face_ymin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymin_neighbor_is_coarser and i < m_block_sizes[IX] and j == 0)
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, -1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2 - 1, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_2d(qprim_n, is, js - 1, iGhost, IY, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_2d(qprim, is, js, iGhost, IY, FACE_LEFT);

      // fix normal component of B
      {
        const auto BL = m_q(i + 2, j + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, IY, iGhost);
        qR[MHD::IB] = BL;
        qL[MHD::IB] = BL;
      }

      // swap IU / IV
      my_swap(qL[MHD::IU], qL[MHD::IV]);
      my_swap(qR[MHD::IU], qR[MHD::IV]);

      // swap IA / IB
      my_swap(qL[MHD::IA], qL[MHD::IB]);
      my_swap(qR[MHD::IA], qR[MHD::IB]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IV]);
      my_swap(flux[MHD::IA], flux[MHD::IB]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face YMAX - we only need to update a coarser neighbor
   */
  const auto face_ymax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymax_neighbor_is_coarser and i < m_block_sizes[IX] and j == m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<2>       coords2{ i, j - 1 }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2 - 1, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_2d(qprim_n, is, js - 1, iGhost, IY, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_2d(qprim, is, js, iGhost, IY, FACE_LEFT);

      // fix normal component of B
      {
        const auto BL = m_q(i + 2, j + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, IY, iGhost);
        qR[MHD::IB] = BL;
        qL[MHD::IB] = BL;
      }

      // swap IU / IV
      my_swap(qL[MHD::IU], qL[MHD::IV]);
      my_swap(qR[MHD::IU], qR[MHD::IV]);

      // swap IA / IB
      my_swap(qL[MHD::IA], qL[MHD::IB]);
      my_swap(qR[MHD::IA], qR[MHD::IB]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IV]);
      my_swap(flux[MHD::IA], flux[MHD::IB]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

} // compute_fluxes_and_update_2d_ghost

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d_group(
  const index_t & cell_index,
  const index_t & iOct_local) const
{
  auto const iOct_global = m_iOct_begin + iOct_local;

  // i,j,k enumerate block faces
  // i,j,k are aligned with inner block (non-ghosted block)
  // i is in range [0, bx] ===> bx+1 values
  // j is in range [0, by] ===> by+1 values
  // k is in range [0, bz] ===> bz+1 values
  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // coordinates to access m_q2 and slopes array (which have a ghost width of 1)
  auto const iq2 = i + 1;
  auto const jq2 = j + 1;
  auto const kq2 = k + 1;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;
  auto const & ks = kq2;

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 8 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 8;


  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_state(m_q2, iq2, jq2, kq2, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_state(m_q2, iq2 - 1, jq2, kq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in the left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is - 1, js, ks, iOct_local, IX, FACE_RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IX, FACE_LEFT);

    // fix normal component of B
    {
      const auto AL =
        m_q(i + 2, j + 2, k + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IX, iOct_local);
      qR[MHD::IA] = AL;
      qL[MHD::IA] = AL;
    }

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    const auto face_xmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (i == 0 and face_xmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ -1, 0, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }

    const auto face_xmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i > 0 and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i - 1, j, k, iOct_global, flux_cur);
      }

      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 1, 0, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i - 1, j, k }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along X

  /*
   * compute flux from left face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_state(m_q2, iq2, jq2 - 1, kq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in the left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is, js - 1, ks, iOct_local, IY, FACE_RIGHT);

    // step 2:
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IY, FACE_LEFT);

    // fix normal component of B
    {
      const auto BL =
        m_q(i + 2, j + 2, k + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IY, iOct_local);
      qR[MHD::IB] = BL;
      qL[MHD::IB] = BL;
    }

    // swap IU / IV
    my_swap(qL[MHD::IU], qL[MHD::IV]);
    my_swap(qR[MHD::IU], qR[MHD::IV]);

    // swap IA / IB
    my_swap(qL[MHD::IA], qL[MHD::IB]);
    my_swap(qR[MHD::IA], qR[MHD::IB]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IV]);
    my_swap(flux[MHD::IA], flux[MHD::IB]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_ymin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (j == 0 and face_ymin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, -1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }


    const auto face_ymax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j > 0 and k < m_block_sizes[IZ])
    {
      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j - 1, k, iOct_global, flux_cur);
      }

      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i, j - 1, k }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along Y

  /*
   * compute flux from left face along Z dir
   */
  {
    // get state in neighbor along Z
    auto qprim_n = get_state(m_q2, iq2, jq2, kq2 - 1, iOct_local);

    // step 1 :
    // reconstruct state at right face in the left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is, js, ks - 1, iOct_local, IZ, FACE_RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IZ, FACE_LEFT);

    // fix normal component of B
    {
      const auto CL =
        m_q(i + 2, j + 2, k + 2, MHD::ICL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IZ, iOct_local);
      qR[MHD::IC] = CL;
      qL[MHD::IC] = CL;
    }

    // swap IU / IW
    my_swap(qL[MHD::IU], qL[MHD::IW]);
    my_swap(qR[MHD::IU], qR[MHD::IW]);

    // swap IA / IC
    my_swap(qL[MHD::IA], qL[MHD::IC]);
    my_swap(qR[MHD::IA], qR[MHD::IC]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IW]);
    my_swap(flux[MHD::IA], flux[MHD::IC]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_zmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_zmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (k == 0 and face_zmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (k == 0 and face_zmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 0, -1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }


    const auto face_zmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_zmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k > 0)
    {
      if (k == m_block_sizes[IZ] and face_zmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j, k - 1, iOct_global, flux_cur);
      }

      if (k == m_block_sizes[IZ] and face_zmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 0, 1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i, j, k - 1 }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along Z

} // compute_fluxes_and_update_3d_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d_ghost(
  index_t const & cell_index,
  index_t const & first_ghost,
  index_t const & iGhost) const
{

  KOKKOS_ASSERT((first_ghost + iGhost) < m_q2.num_quadrants() && "Invalid access to m_q2 view.");

  // iOct local : to be used when accessing array sized upon num_mirrors + num_ghosts like m_q2
  // (aka m_Qghosted_mg in SolverHydroMusclBlock)
  auto const iOct_local = first_ghost + iGhost;

  // iOct global : to be used when accessing global arrays like U, U2 (here u_in, u_out), orchard
  // keys, hashmap, ...
  auto const iOct_global = m_amr_mesh_info.local_num_quadrants() + iGhost;

  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const iq2 = i + 1;
  auto const jq2 = j + 1;
  auto const kq2 = k + 1;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;
  auto const & ks = kq2;

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  // auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 8;

  // get current location primitive variables state
  // note: primitive variables Q2 is a ghosted array with ghost width of 1
  const auto qprim = get_state(m_q2, iq2, jq2, kq2, iGhost);

  /*
   * Face XMIN - we only need to update a coarser neighbor
   */
  const auto face_xmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmin_neighbor_is_coarser and i == 0 and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ -1, 0, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_state(m_q2, iq2 - 1, jq2, kq2, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is - 1, js, ks, iGhost, IX, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IX, FACE_LEFT);

      // fix normal component of B
      {
        const auto AL =
          m_q(i + 2, j + 2, k + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IX, iGhost);
        qR[MHD::IA] = AL;
        qL[MHD::IA] = AL;
      }

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face XMIN

  /*
   * Face XMAX - we only need to update a coarser neighbor
   */
  const auto face_xmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmax_neighbor_is_coarser and i == m_block_sizes[IX] and j < m_block_sizes[IY] and
      k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 1, 0, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i - 1, j, k }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_state(m_q2, iq2 - 1, jq2, kq2, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is - 1, js, ks, iGhost, IX, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IX, FACE_LEFT);

      // fix normal component of B
      {
        const auto AL =
          m_q(i + 2, j + 2, k + 2, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IX, iGhost);
        qR[MHD::IA] = AL;
        qL[MHD::IA] = AL;
      }

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face XMAX

  /*
   * Face YMIN - we only need to update a coarser neighbor
   */
  const auto face_ymin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymin_neighbor_is_coarser and i < m_block_sizes[IX] and j == 0 and k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, -1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2 - 1, kq2, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is, js - 1, ks, iGhost, IY, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IY, FACE_LEFT);

      // fix normal component of B
      {
        const auto BL =
          m_q(i + 2, j + 2, k + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IY, iGhost);
        qR[MHD::IB] = BL;
        qL[MHD::IB] = BL;
      }

      // swap IU / IV
      my_swap(qL[MHD::IU], qL[MHD::IV]);
      my_swap(qR[MHD::IU], qR[MHD::IV]);

      // swap IA / IB
      my_swap(qL[MHD::IA], qL[MHD::IB]);
      my_swap(qR[MHD::IA], qR[MHD::IB]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IV]);
      my_swap(flux[MHD::IA], flux[MHD::IB]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face YMIN

  /*
   * Face YMAX - we only need to update a coarser neighbor
   */
  const auto face_ymax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymax_neighbor_is_coarser and i < m_block_sizes[IX] and j == m_block_sizes[IY] and
      k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i, j - 1, k }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2 - 1, kq2, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is, js - 1, ks, iGhost, IY, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IY, FACE_LEFT);

      // fix normal component of B
      {
        const auto BL =
          m_q(i + 2, j + 2, k + 2, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IY, iGhost);
        qR[MHD::IB] = BL;
        qL[MHD::IB] = BL;
      }

      // swap IU / IV
      my_swap(qL[MHD::IU], qL[MHD::IV]);
      my_swap(qR[MHD::IU], qR[MHD::IV]);

      // swap IA / IB
      my_swap(qL[MHD::IA], qL[MHD::IB]);
      my_swap(qR[MHD::IA], qR[MHD::IB]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IV]);
      my_swap(flux[MHD::IA], flux[MHD::IB]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face YMAX

  /*
   * Face ZMIN - we only need to update a coarser neighbor
   */
  const auto face_zmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_zmin_neighbor_is_coarser and i < m_block_sizes[IX] and j < m_block_sizes[IY] and k == 0)
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 0, -1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2, kq2 - 1, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is, js, ks - 1, iGhost, IZ, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IZ, FACE_LEFT);

      // fix normal component of B
      {
        const auto CL =
          m_q(i + 2, j + 2, k + 2, MHD::ICL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IZ, iGhost);
        qR[MHD::IC] = CL;
        qL[MHD::IC] = CL;
      }

      // swap IU / IW
      my_swap(qL[MHD::IU], qL[MHD::IW]);
      my_swap(qR[MHD::IU], qR[MHD::IW]);

      // swap IA / IC
      my_swap(qL[MHD::IA], qL[MHD::IC]);
      my_swap(qR[MHD::IA], qR[MHD::IC]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IW]);
      my_swap(flux[MHD::IA], flux[MHD::IC]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face ZMIN

  /*
   * Face ZMAX - we only need to update a coarser neighbor
   */
  const auto face_zmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_zmax_neighbor_is_coarser and i < m_block_sizes[IX] and j < m_block_sizes[IY] and
      k == m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 0, 1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i, j, k - 1 }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_state(m_q2, iq2, jq2, kq2 - 1, iGhost);

      // step 1 :
      // reconstruct state at right face in the left neighbor
      auto qL = reconstruct_state_3d(qprim_n, is, js, ks - 1, iGhost, IZ, FACE_RIGHT);

      // step 2 :
      // reconstruct state at left face in current cell
      auto qR = reconstruct_state_3d(qprim, is, js, ks, iGhost, IZ, FACE_LEFT);

      // fix normal component of B
      {
        const auto CL =
          m_q(i + 2, j + 2, k + 2, MHD::ICL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IZ, iGhost);
        qR[MHD::IC] = CL;
        qL[MHD::IC] = CL;
      }

      // swap IU / IW
      my_swap(qL[MHD::IU], qL[MHD::IW]);
      my_swap(qR[MHD::IU], qR[MHD::IW]);

      // swap IA / IC
      my_swap(qL[MHD::IA], qL[MHD::IC]);
      my_swap(qR[MHD::IA], qR[MHD::IC]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_mhd(qL, qR, m_mhd_settings);

      my_swap(flux[MHD::IU], flux[MHD::IW]);
      my_swap(flux[MHD::IA], flux[MHD::IC]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face ZMAX

} // compute_fluxes_and_update_3d_ghost

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                              const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  auto const iOct_local = global_index / m_nbFluxesPerLeaf;
  auto const cell_index = global_index - iOct_local * m_nbFluxesPerLeaf;

  if constexpr (dim == 2)
    compute_fluxes_and_update_2d_group(cell_index, iOct_local);
  else if constexpr (dim == 3)
    compute_fluxes_and_update_3d_group(cell_index, iOct_local);

} // operator () - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndUpdateFunctor<dim, device_t>::operator()(TagComputeGhostQuad const &,
                                                              const index_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost quadrant
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_nbFluxesPerLeaf;
  const auto cell_index = global_index - iGhost * m_nbFluxesPerLeaf;

  if constexpr (dim == 2)
    compute_fluxes_and_update_2d_ghost(cell_index, first_ghost, iGhost);
  else if constexpr (dim == 3)
    compute_fluxes_and_update_3d_ghost(cell_index, first_ghost, iGhost);

} // operator () - TagComputeAllQuadInGroup

// explicit template instantiation
template class ComputeHydroFluxesAndUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ComputeHydroFluxesAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
