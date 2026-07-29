// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRotor.cpp
 */

#include <godunov_mhd_ct/init/InitRotor.h>
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
InitRotorDataFunctor<dim, device_t>::InitRotorDataFunctor(DataArrayBlock_t             Udata,
                                                          FaceDataArrayBlock_t         Bface,
                                                          FieldMap<models::MHD>        fm,
                                                          orchard_key_view_t<device_t> orchard_keys,
                                                          int32_t           local_num_octants,
                                                          ConfigMap const & config_map)
  : m_Udata(Udata)
  , m_Bface(Bface)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_mhd_settings(config_map)
  , m_rparams(config_map)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRotorDataFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                           FaceDataArrayBlock_t         Bface,
                                           FieldMap<models::MHD>        fm,
                                           orchard_key_view_t<device_t> orchard_keys,
                                           int32_t                      local_num_octants,
                                           ConfigMap const &            config_map)
{
  // data init functor
  InitRotorDataFunctor functor(Udata, Bface, fm, orchard_keys, local_num_octants, config_map);

  // for cell-centered variables init : compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // for face-centered variables init
  const auto nbFacesPerLeaf = Bface.num_elements_per_octant();
  const auto nbFacesTotal = local_num_octants * nbFacesPerLeaf;

  // initialize all hydro variables but not total energy (cell-centered)
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::InitRotorDataFunctor - all hydro variables except total energy",
    Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
    functor);

  // initialize mag field (face-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRotorDataFunctor - magnetic field",
                       Kokkos::RangePolicy<exec_space, TagInitMagField>(0, nbFacesTotal),
                       functor);

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRotorDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);

} // InitRotorDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRotorDataFunctor<dim, device_t>::operator()(TagInitHydroVar const &,
                                                const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  constexpr auto ID = models::MHD::ID;
  constexpr auto IE = models::MHD::IE;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;

  // rotor problem parameters
  const auto & r0 = m_rparams.r0;
  const auto & r1 = m_rparams.r1;
  const auto & rho0 = m_rparams.rho0;
  const auto & rho1 = m_rparams.rho1;
  const auto & p0 = m_rparams.p0;
  const auto & u0 = m_rparams.u0;

  // rotor center
  const auto &                  xc = m_rparams.xc;
  const auto &                  yc = m_rparams.yc;
  [[maybe_unused]] const auto & zc = m_rparams.zc;

  const auto & block_sizes = m_Udata.block_size();

  const auto gamma0 = m_mhd_settings.hydro.gamma0;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // compute distance to rotor center
  auto r = (xyz[IX] - xc) * (xyz[IX] - xc) + (xyz[IY] - yc) * (xyz[IY] - yc);

  if constexpr (dim == 3)
  {
    r += (xyz[IZ] - zc) * (xyz[IZ] - zc);
  }
  r = sqrt(r);


  // taper function
  const auto f_r = (r1 - r) / (r1 - r0);

  if (r <= r0)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = rho0;
    m_Udata(cell_index, m_fm[IE], iOct) = p0 / (gamma0 - ONE_F);
    m_Udata(cell_index, m_fm[IU], iOct) = -rho0 * f_r * u0 * (xyz[IY] - yc) / r0;
    m_Udata(cell_index, m_fm[IV], iOct) = rho0 * f_r * u0 * (xyz[IX] - xc) / r0;
    m_Udata(cell_index, m_fm[IW], iOct) = 0.0;
  }
  else if (r <= r1)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = rho1 + (rho0 - rho1) * f_r;
    m_Udata(cell_index, m_fm[IE], iOct) = p0 / (gamma0 - ONE_F);
    m_Udata(cell_index, m_fm[IU], iOct) = -rho0 * f_r * u0 * (xyz[IY] - yc) / r;
    m_Udata(cell_index, m_fm[IV], iOct) = rho0 * f_r * u0 * (xyz[IX] - xc) / r;
    m_Udata(cell_index, m_fm[IW], iOct) = 0.0;
  }
  else
  {
    m_Udata(cell_index, m_fm[ID], iOct) = rho1;
    m_Udata(cell_index, m_fm[IE], iOct) = p0 / (gamma0 - ONE_F);
    m_Udata(cell_index, m_fm[IU], iOct) = 0.0;
    m_Udata(cell_index, m_fm[IV], iOct) = 0.0;
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

} // end InitRotorDataFunctor::operator () - TagInitHydroVar

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRotorDataFunctor<dim, device_t>::operator()(TagInitMagField const &,
                                                const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - face_flat_index inside block (from 0 to m_nbFacesPerLeaf-1)
  //
  // please remember that is slightly too large, so we need to protect write access to only valid
  // multi-index i,j,k
  const auto & nbFacesPerLeaf = m_Bface.num_elements_per_octant();

  const auto iOct = global_index / nbFacesPerLeaf;
  const auto face_flat_index = static_cast<int32_t>(global_index - iOct * nbFacesPerLeaf);

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes =
    face_flat_index_unravel<dim>(face_flat_index, block_sizes, m_Bface.offsets(), m_Bface.shift());

  if constexpr (dim == 2)
  {

    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & ivar = face_indexes[dim];

    if (ivar == IX)
    {
      // Bx on X-face
      m_Bface(i, j, IX, iOct) = m_rparams.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, IY, iOct) = m_rparams.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, IZ, iOct) = m_rparams.bz;
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
      m_Bface(i, j, k, IX, iOct) = m_rparams.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, k, IY, iOct) = m_rparams.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, k, IZ, iOct) = m_rparams.bz;
    }
  } // end dim == 3

} // end InitRotorDataFunctor::operator() - TagInitMagField

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRotorDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy const &,
                                                const int32_t & global_index) const
{
  constexpr auto IE = models::MHD::IE;

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto & block_sizes = m_Udata.block_size();

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

} // end InitRotorDataFunctor::operator () - TagInitTotalEnergy

// explicit template instantiation
template class InitRotorDataFunctor<2, kalypsso::DefaultDevice>;
template class InitRotorDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitRotorRefineFunctor<dim, device_t>::InitRotorRefineFunctor(
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  FieldMap<models::MHD>        fm,
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
  , m_rparams(config_map)
  , m_level_refine(level_refine)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRotorRefineFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                             FaceDataArrayBlock_t         Bface,
                                             FieldMap<models::MHD>        fm,
                                             orchard_key_view_t<device_t> orchard_keys,
                                             amrflags_view_t              amrflags,
                                             int32_t                      local_num_octants,
                                             int                          level_refine,
                                             ConfigMap const &            config_map)
{

  // iterate functor for refinement
  InitRotorRefineFunctor functor(
    Udata, Bface, fm, orchard_keys, amrflags, local_num_octants, level_refine, config_map);


  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRotorRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRotorRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }
} // IniRotorRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRotorRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                  const iOct_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRotorRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                  const iOct_t & iOct) const
{

  // rotor problem parameters
  const auto r0 = m_rparams.r0;
  const auto r1 = m_rparams.r1;
  const auto xc = m_rparams.xc;
  const auto yc = m_rparams.yc;
  const auto zc = m_rparams.zc;

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

    auto r = (xyz[IX] - xc) * (xyz[IX] - xc) + (xyz[IY] - yc) * (xyz[IY] - yc);

    if constexpr (dim == 3)
    {
      r += (xyz[IZ] - zc) * (xyz[IZ] - zc);
    }
    r = sqrt(r);

    if (fabs(r - r0) < (block_length * KALYPSSO_NUM(1.25)) or
        fabs(r - r1) < (block_length * KALYPSSO_NUM(1.25)))
    {
      flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitRotorRefineFunctor::operator ()

// explicit template instantiation
template class InitRotorRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitRotorRefineFunctor<3, kalypsso::DefaultDevice>;

// ===========================================================
// ===========================================================
template <size_t dim, typename device_t>
void
InitRotor<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
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
  InitRotorDataFunctor<dim, device_t>::apply(solver.U(),
                                             solver.Bface(),
                                             solver.model().get_fieldmap(),
                                             solver.mesh_map()->orchard_keys(),
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
      InitRotorDataFunctor<dim, device_t>::apply(solver.U(),
                                                 solver.Bface(),
                                                 solver.model().get_fieldmap(),
                                                 solver.mesh_map()->orchard_keys(),
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
      InitRotorRefineFunctor<dim, device_t>::apply(solver.U(),
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
      InitRotorDataFunctor<dim, device_t>::apply(solver.U(),
                                                 solver.Bface(),
                                                 solver.model().get_fieldmap(),
                                                 solver.mesh_map()->orchard_keys(),
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

} // InitRotor::apply

template class InitRotor<2, kalypsso::DefaultDevice>;
template class InitRotor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
