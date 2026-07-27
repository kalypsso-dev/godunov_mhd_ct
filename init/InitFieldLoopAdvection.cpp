// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitFieldLoopAdvection.cpp
 */

#include <godunov_mhd_ct/init/InitFieldLoopAdvection.h>
#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/problems/init_cond_utils.h>

#include "FixVectorPotentialAtNonConformalEdges.h"

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitFieldLoopAdvectionDataFunctor<dim, device_t>::InitFieldLoopAdvectionDataFunctor(
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  FieldMap<models::MHD>  fm,
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  ConfigMap const &            config_map)
  : m_Udata(Udata)
  , m_Bface(Bface)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_mhd_settings(config_map)
  , m_flaParams(config_map)
  , m_fla_vector_potential(m_flaParams)
  , m_vectorPotential("vector_potential", m_Udata.block_size(), m_Udata.num_quadrants())
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFieldLoopAdvectionDataFunctor<dim, device_t>::apply(DataArrayBlock_t            Udata,
                                                        FaceDataArrayBlock_t        Bface,
                                                        FieldMap<models::MHD> fm,
                                                        MeshMap<dim, device_t> &    mesh_map,
                                                        int32_t           local_num_octants,
                                                        ConfigMap const & config_map)
{
  // data init functor
  InitFieldLoopAdvectionDataFunctor functor(
    Udata, Bface, fm, mesh_map.orchard_keys(), local_num_octants, config_map);

  // for cell-centered variables init : compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // in 2D only Az (z component of vector potential) is useful
  // in 3D, all component are used
  const auto nbEdgesPerLeaf = functor.vectorPotential().num_elements_per_octant();
  const auto nbEdgesTotal = local_num_octants * nbEdgesPerLeaf;


  // initialize all hydro variables but not total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitFieldLoopAdvectionDataFunctor - all hydro "
                       "variables except total energy",
                       Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
                       functor);

  // initialize vector potential (edge-centered)
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::InitFieldLoopAdvectionDataFunctor - vector potential",
    Kokkos::RangePolicy<exec_space, TagInitVectorPotential>(0, nbEdgesTotal),
    functor);

  // fix vector potential at non-conformal edges
  const auto flaParams = FieldLoopAdvectionParams(config_map);

  FieldLoopAdvectionVectorPotential<dim> analytical_vec_pot(flaParams);
  FixVectorPotentialAtNonConformalEdges<dim, device_t, FieldLoopAdvectionVectorPotential<dim>>::
    apply(config_map,
          mesh_map.orchard_keys(),
          functor.vectorPotential(),
          analytical_vec_pot,
          mesh_map.update_conformal_full_status(),
          local_num_octants);

  // initialize magnetic field (i.e. compute curl of vector_potential)
  auto Bface2 =
    EdgeDataArrayBlock_t::compute_curl(functor.vectorPotential(), mesh_map.orchard_keys());
  Kokkos::deep_copy(Bface.logical_view(), Bface2.logical_view());

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitFieldLoopAdvectionDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);

} // InitFieldLoopAdvectionDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFieldLoopAdvectionDataFunctor<dim, device_t>::operator()(TagInitHydroVar,
                                                             const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  constexpr auto ID = models::MHD::ID;
  // constexpr auto IP = models::MHD::IP;
  constexpr auto IE = models::MHD::IE;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;

  // field loop advection problem parameters
  const auto                  xc = m_flaParams.xc;
  const auto                  yc = m_flaParams.yc;
  [[maybe_unused]] const auto zc = m_flaParams.zc;

  const real_t gamma0 = m_mhd_settings.hydro.gamma0;

  const real_t cos_theta = KALYPSSO_NUM(2.0) / SQRT_5_F;
  const real_t sin_theta = KALYPSSO_NUM(1.0) / SQRT_5_F;

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // compute distance to field loop center
  auto r = (xyz[IX] - xc) * (xyz[IX] - xc) + (xyz[IY] - yc) * (xyz[IY] - yc);

  if constexpr (dim == 3)
  {
    r += (xyz[IZ] - zc) * (xyz[IZ] - zc);
  }
  r = sqrt(r);

  // total energy is only partially initialized
  // we only init thermal energy here

  if (r <= m_flaParams.radius)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = m_flaParams.density;
  }
  else
  {
    m_Udata(cell_index, m_fm[ID], iOct) = 1.0;
  }

  m_Udata(cell_index, m_fm[IU], iOct) =
    m_Udata(cell_index, m_fm[ID], iOct) * m_flaParams.vflow * cos_theta;

  m_Udata(cell_index, m_fm[IV], iOct) =
    m_Udata(cell_index, m_fm[ID], iOct) * m_flaParams.vflow * sin_theta;

  m_Udata(cell_index, m_fm[IW], iOct) = 0.0;

  m_Udata(cell_index, m_fm[IE], iOct) = m_flaParams.pressure / (gamma0 - 1);


  // add kinetic energy
  m_Udata(cell_index, m_fm[IE], iOct) +=
    dim == 2 ? HALF_F *
                 (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                  m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct)) /
                 m_Udata(cell_index, m_fm[ID], iOct)
             : HALF_F *
                 (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                  m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct) +
                  m_Udata(cell_index, m_fm[IW], iOct) * m_Udata(cell_index, m_fm[IW], iOct)) /
                 m_Udata(cell_index, m_fm[ID], iOct);

} // end InitFieldLoopAdvectionDataFunctor::operator () - TagInitHydroVar

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFieldLoopAdvectionDataFunctor<dim, device_t>::operator()(TagInitVectorPotential,
                                                             const int32_t & global_index) const
{
  // convert global index into
  // - octant id
  // - edge_flat_index inside block
  //
  const auto & nbEdgesPerLeaf = m_vectorPotential.num_elements_per_octant();

  const auto iOct = global_index / nbEdgesPerLeaf;
  const auto edge_flat_index = static_cast<int32_t>(global_index - iOct * nbEdgesPerLeaf);

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto edge_indexes =
    edge_flat_index_unravel<dim>(edge_flat_index, block_sizes, m_vectorPotential.offsets());

  auto const & bSize = block_sizes[IX];

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that face center
  const auto xyz = orchard_key_to_edgecenter_real_space<dim>(
    key, edge_indexes, bSize, m_scaling_factor, m_xyz_min);

  const auto & ivar = edge_indexes[dim];

  m_vectorPotential(edge_indexes, iOct) = m_fla_vector_potential(xyz, ivar);

} // end InitFieldLoopAdvectionDataFunctor::operator () - TagInitVectorPotential

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFieldLoopAdvectionDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy,
                                                             const int32_t & global_index) const
{
  constexpr auto IE = models::MHD::IE;

  const auto & block_sizes = m_Udata.block_size();

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // mag energy energy
  real_t mag_energy = ZERO_F;

  // add magnetic energy
  if constexpr (dim == 2)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];

    const auto a = HALF_F * (m_Bface(i, j, IX, iOct) + m_Bface(i + 1, j, IX, iOct));
    const auto b = HALF_F * (m_Bface(i, j, IY, iOct) + m_Bface(i, j + 1, IY, iOct));
    const auto c = m_Bface(i, j, IZ, iOct);
    mag_energy += HALF_F * (a * a + b * b + c * c);
  }
  else
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    auto const & k = iCoord[IZ];

    const auto a = HALF_F * (m_Bface(i, j, k, IX, iOct) + m_Bface(i + 1, j, k, IX, iOct));
    const auto b = HALF_F * (m_Bface(i, j, k, IY, iOct) + m_Bface(i, j + 1, k, IY, iOct));
    const auto c = HALF_F * (m_Bface(i, j, k, IZ, iOct) + m_Bface(i, j, k + 1, IZ, iOct));
    mag_energy += HALF_F * (a * a + b * b + c * c);
  }

  m_Udata(cell_index, m_fm[IE], iOct) += mag_energy;

} // end InitFieldLoopAdvectionDataFunctor::operator () - TagInitTotalEnergy

// explicit template instantiation
template class InitFieldLoopAdvectionDataFunctor<2, kalypsso::DefaultDevice>;
template class InitFieldLoopAdvectionDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitFieldLoopAdvectionRefineFunctor<dim, device_t>::InitFieldLoopAdvectionRefineFunctor(
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  FieldMap<models::MHD>  fm,
  orchard_key_view_t<device_t> orchard_keys,
  amrflags_view_t              amrflags,
  int32_t                      local_num_octants,
  int                          level_refine,
  ConfigMap const &            config_map)
  : m_Udata(Udata)
  , m_Bface(Bface)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_amrflags(amrflags)
  , m_local_num_octants(local_num_octants)
  , m_mhd_settings(config_map)
  , m_flaParams(config_map)
  , m_level_refine(level_refine)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFieldLoopAdvectionRefineFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                                          FaceDataArrayBlock_t         Bface,
                                                          FieldMap<models::MHD>  fm,
                                                          orchard_key_view_t<device_t> orchard_keys,
                                                          amrflags_view_t              amrflags,
                                                          int32_t           local_num_octants,
                                                          int               level_refine,
                                                          ConfigMap const & config_map)
{
  // iterate functor for refinement
  InitFieldLoopAdvectionRefineFunctor functor(
    Udata, Bface, fm, orchard_keys, amrflags, local_num_octants, level_refine, config_map);


  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitFieldLoopAdvectionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitFieldLoopAdvectionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }
} // InitFieldLoopAdvectionRefineFunctor<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFieldLoopAdvectionRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                               const iOct_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFieldLoopAdvectionRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                               const iOct_t & iOct) const
{
  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  const auto level = orchard_key_t<dim>::level(key);

  // compute block length (in real space units)
  const auto block_length = compute_block_length<dim>(level) * m_scaling_factor;

  // default : do nothing, i.e. neither refine or coarsen
  auto flag = AMRContextBase::KALYPSSO_DO_NOTHING;

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    auto           xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    // compute distance to interface
    // const auto d = fabs(xyz[IX] - m_flaParams.xd);

    auto d2 = (xyz[IX] - m_flaParams.xc) * (xyz[IX] - m_flaParams.xc) +
              (xyz[IY] - m_flaParams.yc) * (xyz[IY] - m_flaParams.yc);

    if (fabs(sqrt(d2) - m_flaParams.radius) < (block_length * KALYPSSO_NUM(1.25)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitFieldLoopAdvectionRefineFunctor::operator ()

// explicit template instantiation
template class InitFieldLoopAdvectionRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitFieldLoopAdvectionRefineFunctor<3, kalypsso::DefaultDevice>;

// ===========================================================
// ===========================================================
template <size_t dim, typename device_t>
void
InitFieldLoopAdvection<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
{
  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitFieldLoopAdvectionDataFunctor<dim, device_t>::apply(solver.U(),
                                                          solver.Bface(),
                                                          solver.model().get_fieldmap(),
                                                          *(solver.mesh_map()),
                                                          solver.amr_mesh()->local_num_quadrants(),
                                                          config_map);

  const auto init_refine_type = core::get_init_indicator(config_map);

  if (init_refine_type == +core::InitConditionsIndicator::SAME_AS_REGULAR_DYNAMICS)
  {

    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {

      //
      // 1. apply amr cycle using regular refine criterion
      //
      solver.do_amr_cycle();

      //
      // 2. update Udata
      //
      InitFieldLoopAdvectionDataFunctor<dim, device_t>::apply(
        solver.U(),
        solver.Bface(),
        solver.model().get_fieldmap(),
        *(solver.mesh_map()),
        solver.amr_mesh()->local_num_quadrants(),
        config_map);

      // update level
      ++level;

    } // end while level<level_max
  }
  else // use custom initial refine criterion (either GEOMETRIC or ALWAYS_REFINE)
  {
    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {
      //
      // 1. create context data for AMR cycle
      //
      AMRContext<dim, device_t> amr_context(amr_mesh->forest()->local_num_quadrants);
      auto                      flags_d = amr_context.m_amrflags_d;
      auto                      flags_h = amr_context.m_amrflags_h;

      //
      // 2. compute refine/coarsen flags
      //
      InitFieldLoopAdvectionRefineFunctor<dim, device_t>::apply(
        solver.U(),
        solver.Bface(),
        solver.model().get_fieldmap(),
        solver.mesh_map()->orchard_keys(),
        flags_d,
        solver.amr_mesh()->local_num_quadrants(),
        level,
        solver.config_map());

      // amr context will adapt mesh on CPU, so we need flags on host up to date
      Kokkos::deep_copy(flags_h, flags_d);

      //
      // 3. apply AMR cycle on device : refine + coarsen + 2:1 balance
      //
      {
        Kokkos::Profiling::ScopedRegion prof("AMR_refinement_device");
        [[maybe_unused]] auto changed = amr_context.adapt_mesh(solver.amr_mesh()->forest());
        KALYPSSO_INFO_ALL("Mesh changed ? {}", static_cast<int>(changed));
      }

      //
      // 4. re-compute update orchard keys
      //
      solver.update_mesh(do_reset_ghosts);

      // 5. resize Udata
      // now we know the size of the mesh, we can allocate memory for
      // heavy data (U, U2, Uhost, ...)
      solver.resize_solver_data();

      //
      // 6. update Udata
      //
      InitFieldLoopAdvectionDataFunctor<dim, device_t>::apply(
        solver.U(),
        solver.Bface(),
        solver.model().get_fieldmap(),
        *(solver.mesh_map()),
        solver.amr_mesh()->local_num_quadrants(),
        config_map);

      // update level
      ++level;

    } // end while level<level_max
  } // end init_refine_type

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // init_field_loop_advection

template class InitFieldLoopAdvection<2, kalypsso::DefaultDevice>;
template class InitFieldLoopAdvection<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
