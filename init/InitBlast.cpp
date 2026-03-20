// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitBlast.cpp
 */

#include <godunov_mhd_ct/init/InitBlast.h>
#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitBlastDataFunctor<dim, device_t>::InitBlastDataFunctor(orchard_key_view_t<device_t> orchard_keys,
                                                          int32_t           local_num_octants,
                                                          HydroParams       params,
                                                          ConfigMap const & config_map,
                                                          FieldMap<core::models::MHD> fm,
                                                          brick_size_t<dim>           brick_sizes,
                                                          DataArrayBlock_t            Udata,
                                                          FaceDataArrayBlock_t        Bface)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_mhd_settings(config_map)
  , m_bParams(config_map)
  , m_fm(fm)
  , m_brick_sizes(brick_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_Udata(Udata)
  , m_Bface(Bface)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
InitBlastDataFunctor<dim, device_t>::apply([[maybe_unused]] ParallelEnv const & par_env,
                                           orchard_key_view_t<device_t>         orchard_keys,
                                           int32_t                              local_num_octants,
                                           HydroParams                          params,
                                           ConfigMap const &                    config_map,
                                           FieldMap<core::models::MHD>          fm,
                                           brick_size_t<dim>                    brick_sizes,
                                           DataArrayBlock_t                     Udata,
                                           FaceDataArrayBlock_t                 Bface)
{
  // data init functor
  InitBlastDataFunctor functor(
    orchard_keys, local_num_octants, params, config_map, fm, brick_sizes, Udata, Bface);

  // compute volume inside ball
  real_t              volume_inside, total_volume_inside = ZERO_F;
  Kokkos::Sum<real_t> reducer(volume_inside);

  // for cell-centered variables init : compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // for face-centered variables init
  const auto nbFacesPerLeaf = Bface.num_elements_per_octant();
  const auto nbFacesTotal = local_num_octants * nbFacesPerLeaf;

  // initialize all hydro variables but not total energy (cell-centered)
  Kokkos::parallel_reduce(
    "kalypsso::godunov_mhd_ct::InitBlastDataFunctor - all hydro variables except total energy",
    Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
    functor,
    reducer);

  // initialize mag field (face-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitBlastDataFunctor - magnetic field",
                       Kokkos::RangePolicy<exec_space, TagInitMagField>(0, nbFacesTotal),
                       functor);

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitBlastDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);

#ifdef KALYPSSO_CORE_USE_MPI
  par_env.comm().MPI_Allreduce<MpiComm::SUM>(&volume_inside, &total_volume_inside, 1);
#else
  total_volume_inside = volume_inside;
#endif // KALYPSSO_CORE_USE_MPI

  return total_volume_inside;

} // InitBlastDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastDataFunctor<dim, device_t>::operator()(TagInitHydroVar const &,
                                                const int32_t & global_index,
                                                real_t &        volume) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  constexpr auto ID = core::models::MHD::ID;
  constexpr auto IE = core::models::MHD::IE;
  constexpr auto IU = core::models::MHD::IU;
  constexpr auto IV = core::models::MHD::IV;
  constexpr auto IW = core::models::MHD::IW;

  // blast problem parameters
  const real_t blast_radius = m_bParams.blast_radius;
  const real_t radius2 = blast_radius * blast_radius;
  const real_t blast_center_x = m_bParams.blast_center_x;
  const real_t blast_center_y = m_bParams.blast_center_y;
  const real_t blast_center_z = m_bParams.blast_center_z;
  const real_t blast_density_in = m_bParams.blast_density_in;
  const real_t blast_density_out = m_bParams.blast_density_out;
  const real_t blast_pressure_in = m_bParams.blast_pressure_in;
  const real_t blast_pressure_out = m_bParams.blast_pressure_out;

  const real_t gamma0 = m_mhd_settings.hydro.gamma0;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, m_Udata.block_size());

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, m_Udata.block_size()[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // if using replicated initial conditions, shift coordinates into tree id 0
  // so that all trees get initialized the same
  if (m_params.replicated_init_cond)
  {
    const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
    xyz[IX] -= tree_coords[IX] * m_scaling_factor;
    xyz[IY] -= tree_coords[IY] * m_scaling_factor;
    if constexpr (dim == 3)
    {
      xyz[IZ] -= tree_coords[IZ] * m_scaling_factor;
    }
  }

  // initialize
  real_t d2 = (xyz[IX] - blast_center_x) * (xyz[IX] - blast_center_x) +
              (xyz[IY] - blast_center_y) * (xyz[IY] - blast_center_y);

  if constexpr (dim == 3)
    d2 += (xyz[IZ] - blast_center_z) * (xyz[IZ] - blast_center_z);

  if (d2 < radius2)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = blast_density_in;
    m_Udata(cell_index, m_fm[IE], iOct) = blast_pressure_in / (gamma0 - ONE_F);
    m_Udata(cell_index, m_fm[IU], iOct) = 0.0;
    m_Udata(cell_index, m_fm[IV], iOct) = 0.0;

    // compute volume of current cell assuming
    // dx=dy=dz, i.e. block_sizes are the same along all directions
    {
      const auto level = orchard_key_t<dim>::level(key);

      // compute cell size
      const auto dx = compute_cell_length<dim>(level, m_Udata.block_size()[IX]) * m_scaling_factor;
      if constexpr (dim == 2)
        volume += dx * dx;
      else if constexpr (dim == 3)
        volume += dx * dx * dx;
    }
  }
  else
  {
    m_Udata(cell_index, m_fm[ID], iOct) = blast_density_out;
    m_Udata(cell_index, m_fm[IE], iOct) = blast_pressure_out / (gamma0 - ONE_F);
    m_Udata(cell_index, m_fm[IU], iOct) = 0.0;
    m_Udata(cell_index, m_fm[IV], iOct) = 0.0;
  }

  if constexpr (dim == 3)
  {
    m_Udata(cell_index, m_fm[IW], iOct) = 0.0;
  }

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

} // end InitBlastDataFunctor::operator () - TagInitHydroVar

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBlastDataFunctor<dim, device_t>::operator()(TagInitMagField const &,
                                                const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - face_flat_index inside block (from 0 to m_nbFacesPerLeaf-1)
  //
  // please remember that is slightly too large, so we need to protect write access to only valid
  // multi-index i,j,k
  const auto & nbFacesPerLeaf = m_Bface.num_elements_per_octant();

  const auto    iOct = global_index / nbFacesPerLeaf;
  const int32_t face_flat_index = static_cast<int32_t>(global_index - iOct * nbFacesPerLeaf);

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes = face_flat_index_unravel<dim>(
    face_flat_index, m_Udata.block_size(), m_Bface.offsets(), m_Bface.shift());

  if constexpr (dim == 2)
  {

    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & ivar = face_indexes[dim];

    if (ivar == IX)
    {
      // Bx on X-face
      m_Bface(i, j, IX, iOct) = m_bParams.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, IY, iOct) = m_bParams.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, IZ, iOct) = m_bParams.bz;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & k = face_indexes[IZ];
    auto const & ivar = face_indexes[dim];

    if (ivar == IX)
    {
      // Bx on X-face
      m_Bface(i, j, k, IX, iOct) = m_bParams.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, k, IY, iOct) = m_bParams.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, k, IZ, iOct) = m_bParams.bz;
    }
  } // end dim == 3

} // end InitBlastDataFunctor::operator() - TagInitMagField

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy const &,
                                                const int32_t & global_index) const
{
  constexpr auto IE = core::models::MHD::IE;

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, m_Udata.block_size());

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

} // end InitBlastDataFunctor::operator () - TagInitTotalEnergy

// explicit template instantiation
template class InitBlastDataFunctor<2, kalypsso::DefaultDevice>;
template class InitBlastDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitBlastRefineFunctor<dim, device_t>::InitBlastRefineFunctor(
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  ConfigMap const &            config_map,
  HydroParams                  params,
  FieldMap<core::models::MHD>  fm,
  brick_size_t<dim>            brick_sizes,
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  amrflags_view_t              amrflags,
  int                          level_refine)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_bParams(config_map)
  , m_fm(fm)
  , m_brick_sizes(brick_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_Udata(Udata)
  , m_Bface(Bface)
  , m_amrflags(amrflags)
  , m_level_refine(level_refine)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBlastRefineFunctor<dim, device_t>::apply(orchard_key_view_t<device_t> orchard_keys,
                                             int32_t                      local_num_octants,
                                             ConfigMap const &            config_map,
                                             HydroParams                  params,
                                             FieldMap<core::models::MHD>  fm,
                                             brick_size_t<dim>            brick_sizes,
                                             DataArrayBlock_t             Udata,
                                             FaceDataArrayBlock_t         Bface,
                                             amrflags_view_t              amrflags,
                                             int                          level_refine)
{
  // iterate functor for refinement
  InitBlastRefineFunctor functor(orchard_keys,
                                 local_num_octants,
                                 config_map,
                                 params,
                                 fm,
                                 brick_sizes,
                                 Udata,
                                 Bface,
                                 amrflags,
                                 level_refine);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitBlastRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitBlastRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }
} // IniBlastRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                  const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                  const size_t & iOct) const
{

  // blast problem parameters
  const auto radius = m_bParams.blast_radius;
  const auto blast_center_x = m_bParams.blast_center_x;
  const auto blast_center_y = m_bParams.blast_center_y;
  const auto blast_center_z = m_bParams.blast_center_z;

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

    // if using replicated initial conditions, shift coordinates into tree id 0
    // so that all trees get initialized the same
    if (m_params.replicated_init_cond)
    {
      const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
      xyz[IX] -= tree_coords[IX] * m_scaling_factor;
      xyz[IY] -= tree_coords[IY] * m_scaling_factor;
      if constexpr (dim == 3)
      {
        xyz[IZ] -= tree_coords[IZ] * m_scaling_factor;
      }
    }

    auto d2 = (xyz[IX] - blast_center_x) * (xyz[IX] - blast_center_x) +
              (xyz[IY] - blast_center_y) * (xyz[IY] - blast_center_y);

    if constexpr (dim == 3)
      d2 += (xyz[IZ] - blast_center_z) * (xyz[IZ] - blast_center_z);

    if (fabs(sqrt(d2) - radius) < (block_length * KALYPSSO_NUM(1.25)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitBlastRefineFunctor::operator() - TagRefineGeometric

// explicit template instantiation
template class InitBlastRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitBlastRefineFunctor<3, kalypsso::DefaultDevice>;

// ===========================================================
// ===========================================================
template <size_t dim, typename device_t>
void
InitBlast<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
{

  using exec_space = typename device_t::execution_space;

  auto                amr_mesh = solver.amr_mesh();
  ConfigMap const &   config_map = solver.config_map();
  HydroParams const & params = solver.hydro_params();
  const int           level_min = solver.hydro_params().level_min;
  const int           level_max = solver.hydro_params().level_max;

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // initialize total volume inside
  auto total_volume_inside = ZERO_F;

  // first init of Udata
  total_volume_inside =
    InitBlastDataFunctor<dim, device_t>::apply(solver.par_env(),
                                               solver.mesh_map()->orchard_keys(),
                                               solver.amr_mesh()->local_num_quadrants(),
                                               params,
                                               config_map,
                                               solver.model().get_fieldmap(),
                                               solver.brick_sizes(),
                                               solver.U(),
                                               solver.Bface());

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
      total_volume_inside =
        InitBlastDataFunctor<dim, device_t>::apply(solver.par_env(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   params,
                                                   config_map,
                                                   solver.model().get_fieldmap(),
                                                   solver.brick_sizes(),
                                                   solver.U(),
                                                   solver.Bface());

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
      InitBlastRefineFunctor<dim, device_t>::apply(solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   solver.config_map(),
                                                   solver.hydro_params(),
                                                   solver.model().get_fieldmap(),
                                                   solver.brick_sizes(),
                                                   solver.U(),
                                                   solver.Bface(),
                                                   flags_d,
                                                   level);
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
      total_volume_inside =
        InitBlastDataFunctor<dim, device_t>::apply(solver.par_env(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   params,
                                                   config_map,
                                                   solver.model().get_fieldmap(),
                                                   solver.brick_sizes(),
                                                   solver.U(),
                                                   solver.Bface());

      // update level
      ++level;

    } // end while level<level_max
  } // end init_refine_type

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

  BlastParams blastParams = BlastParams(config_map);

  // checking if we need rescaling pressure inside the ball
  if (blastParams.total_energy_inside > 0)
  {

    auto       fm = solver.model().get_fieldmap();
    auto       U = solver.U();
    const auto nbCellsPerLeaf = U.num_cells();
    const auto local_num_octants = solver.amr_mesh()->local_num_quadrants();
    const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;
    const auto scaling_factor = get_scaling_factor(config_map);
    const auto xyz_min = get_xyz_min<dim>(config_map);
    const auto orchard_keys = solver.mesh_map()->orchard_keys();

    // rescale pressure
    Kokkos::parallel_for(
      "kalypsso::godunov_mhd_ct::init_blast: rescale pressure",
      Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
      KOKKOS_LAMBDA(const int64_t global_index) {
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        const real_t blast_radius = blastParams.blast_radius;
        const real_t radius2 = blast_radius * blast_radius;
        const real_t blast_center_x = blastParams.blast_center_x;
        const real_t blast_center_y = blastParams.blast_center_y;
        const real_t blast_center_z = blastParams.blast_center_z;

        // compute ix,iy,iz of local cell inside
        // block from index
        auto iCoord = cellindex_to_coord<dim>(static_cast<int32_t>(cell_index), U.block_size());

        // get block orchard key
        const auto key = orchard_keys(iOct);

        // compute physical x,y,z for that cell (cell center)
        const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, U.block_size()[IX]);

        auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

        real_t d2 = (xyz[IX] - blast_center_x) * (xyz[IX] - blast_center_x) +
                    (xyz[IY] - blast_center_y) * (xyz[IY] - blast_center_y);

        if constexpr (dim == 3)
          d2 += (xyz[IZ] - blast_center_z) * (xyz[IZ] - blast_center_z);

        constexpr auto IP = core::models::MHD::IP;

        if (d2 < radius2)
        {
          U(cell_index, fm[IP], iOct) = blastParams.total_energy_inside / total_volume_inside;
        }
      });
  }

} // InitBlast::apply

template class InitBlast<2, kalypsso::DefaultDevice>;
template class InitBlast<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
