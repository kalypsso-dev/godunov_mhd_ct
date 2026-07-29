// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitDiamagCavity.cpp
 */

#include <godunov_mhd_ct/init/InitDiamagCavity.h>
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
InitDiamagCavityDataFunctor<dim, device_t>::InitDiamagCavityDataFunctor(
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  HydroParams                  params,
  ConfigMap const &            config_map,
  FieldMap<models::MHD>        fm,
  brick_size_t<dim>            brick_sizes,
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_mhd_settings(config_map)
  , m_dcParams(config_map)
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
InitDiamagCavityDataFunctor<dim, device_t>::apply([[maybe_unused]] ParallelEnv const & par_env,
                                                  orchard_key_view_t<device_t>         orchard_keys,
                                                  int32_t               local_num_octants,
                                                  HydroParams           params,
                                                  ConfigMap const &     config_map,
                                                  FieldMap<models::MHD> fm,
                                                  brick_size_t<dim>     brick_sizes,
                                                  DataArrayBlock_t      Udata,
                                                  FaceDataArrayBlock_t  Bface)
{
  // data init functor
  InitDiamagCavityDataFunctor functor(
    orchard_keys, local_num_octants, params, config_map, fm, brick_sizes, Udata, Bface);

  // for cell-centered variables init : compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // for face-centered variables init
  const auto nbFacesPerLeaf = Bface.num_elements_per_octant();
  const auto nbFacesTotal = local_num_octants * nbFacesPerLeaf;

  // initialize all hydro variables but not total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitDiamagCavityDataFunctor - all hydro "
                       "variables except total energy",
                       Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
                       functor);

  // initialize mag field (face-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitDiamagCavityDataFunctor- magnetic field",
                       Kokkos::RangePolicy<exec_space, TagInitMagField>(0, nbFacesTotal),
                       functor);

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitDiamagCavityDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);

} // InitBlastDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDiamagCavityDataFunctor<dim, device_t>::operator()(TagInitHydroVar const &,
                                                       const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  constexpr auto ID = models::MHD::ID;
  constexpr auto IE = models::MHD::IE;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;

  // Diamagnetic Cavity problem parameters
  // const auto Navg = m_DiamagneticCavityParams.Navg;
  const auto kb = constants::BOLTZMANN;
  // const auto mu0 = m_dcParams.mu0;

  // Diamag cavity parameters
  const auto cavity_center_x = m_dcParams.center_x;
  const auto cavity_center_y = m_dcParams.center_y;
  const auto cavity_center_z = m_dcParams.center_z;

  const auto t0 = m_dcParams.t0;
  const auto Vmin = m_dcParams.Vmin;
  const auto Vmax = m_dcParams.Vmax;
  const auto rmin = m_dcParams.rmin;
  const auto rmax = m_dcParams.rmax;
  const auto dV = m_dcParams.dV;
  // const auto M_d = m_dcParams.M_d;
  const auto m_d = m_dcParams.m_d;
  const auto N_d = m_dcParams.N_d;
  const auto n_a = m_dcParams.n_a;
  const auto N_a = m_dcParams.N_a;
  // const auto M_a = m_dcParams.M_a;
  const auto m_a = m_dcParams.m_a;
  const auto rho_a = m_dcParams.rho_a;
  // const auto B0 = m_dcParams.B0;
  const auto T_d = m_dcParams.T_d;
  const auto rho_profile = m_dcParams.rho_profile;
  const auto shape = m_dcParams.shape;
  const auto amplitude_perturb = m_dcParams.amplitude_perturb;
  const auto wave_number = m_dcParams.wave_number;
  const auto perturb = m_dcParams.perturb;

  const auto gamma0 = m_mhd_settings.hydro.gamma0;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, m_Udata.block_size());

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, m_Udata.block_size()[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // initialize
  real_t r2 = (xyz[IX] - cavity_center_x) * (xyz[IX] - cavity_center_x) +
              (xyz[IY] - cavity_center_y) * (xyz[IY] - cavity_center_y);

  if (dim == 3 && shape == +DiamagCavityShape::SPHERE)
    r2 += (xyz[IZ] - cavity_center_z) * (xyz[IZ] - cavity_center_z);

  real_t r_interface;
  real_t theta = atan2(xyz[IY] - cavity_center_y, xyz[IX] - cavity_center_x);
  if (perturb)
  {
    r_interface = rmax + amplitude_perturb * sin(wave_number * theta);
  }
  else
  {
    r_interface = rmax;
  }


  real_t       r = sqrt(r2);
  const real_t V0 = 0.5 * Vmax;
  const real_t vth = 5 * sqrt(3 * kb * T_d / m_d);
  // const real_t xexp = (r - V0 * t0) / (vth * t0);
  const real_t xexp = (r - HALF_F * r_interface) / (vth * t0);
  const real_t CoeffNorm =
    2 * PI_F * sqrt(2 * PI_F * vth * vth * t0 * t0) * V0 * t0 *
    erf(V0 * t0 / sqrt(2 * vth * vth * t0 * t0)); //! Normalisation coefficient for bounded
                                                  //! integrals in cylindrical coordinates system

  real_t d, u, v, w, p;

  if (r <= r_interface)
  {
    const real_t V = Vmin + dV * (r - rmin); //! linear profile
    if (rho_profile == +RhoProfile::BASE)
    {
      d = rho_a + (N_d * m_d - N_a * m_a) / CoeffNorm * exp(-xexp * xexp / 2.0);
    }
    else
    {
      d = rho_a + (N_d * m_d) / CoeffNorm * exp(-xexp * xexp / 2.0);
    }
    // d = rho_a + (N_d * m_d - N_a * m_a) / CoeffNorm * exp(- xexp * xexp / 2.0 );
    u = V * (xyz[IX] - cavity_center_x) / r;
    v = V * (xyz[IY] - cavity_center_y) / r;
    w = 0.0;
    if constexpr (dim == 3)
    {
      w += V * (xyz[IZ] - cavity_center_z) / r;
    }
    p = d / m_a * kb * T_d; // Check if m_a or m_d has to be used
  }
  else
  {
    d = rho_a;
    u = ZERO_F;
    v = ZERO_F;
    w = ZERO_F;
    p = n_a * kb * T_d; //! Check if we can have T_a << T_d
  }

  m_Udata(cell_index, m_fm[ID], iOct) = d;
  m_Udata(cell_index, m_fm[IU], iOct) = d * u;
  m_Udata(cell_index, m_fm[IV], iOct) = d * v;
  m_Udata(cell_index, m_fm[IW], iOct) = d * w;
  m_Udata(cell_index, m_fm[IE], iOct) = p / (gamma0 - 1.0);

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
InitDiamagCavityDataFunctor<dim, device_t>::operator()(TagInitMagField const &,
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
      m_Bface(i, j, IX, iOct) = m_dcParams.Bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, IY, iOct) = m_dcParams.By;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, IZ, iOct) = m_dcParams.Bz;
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
      m_Bface(i, j, k, IX, iOct) = m_dcParams.Bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, k, IY, iOct) = m_dcParams.By;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, k, IZ, iOct) = m_dcParams.Bz;
    }
  } // end dim == 3

} // end InitBlastDataFunctor::operator() - TagInitMagField

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDiamagCavityDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy const &,
                                                       const int32_t & global_index) const
{
  constexpr auto IE = models::MHD::IE;

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
template class InitDiamagCavityDataFunctor<2, kalypsso::DefaultDevice>;
template class InitDiamagCavityDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitDiamagCavityRefineFunctor<dim, device_t>::InitDiamagCavityRefineFunctor(
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  ConfigMap const &            config_map,
  HydroParams                  params,
  FieldMap<models::MHD>        fm,
  brick_size_t<dim>            brick_sizes,
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  amrflags_view_t              amrflags,
  int                          level_refine)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_dcParams(config_map)
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
InitDiamagCavityRefineFunctor<dim, device_t>::apply(orchard_key_view_t<device_t> orchard_keys,
                                                    int32_t                      local_num_octants,
                                                    ConfigMap const &            config_map,
                                                    HydroParams                  params,
                                                    FieldMap<models::MHD>        fm,
                                                    brick_size_t<dim>            brick_sizes,
                                                    DataArrayBlock_t             Udata,
                                                    FaceDataArrayBlock_t         Bface,
                                                    amrflags_view_t              amrflags,
                                                    int                          level_refine)
{
  // iterate functor for refinement
  InitDiamagCavityRefineFunctor functor(orchard_keys,
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
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitDiamagCavityRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitDiamagCavityRefineFunctor",
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
InitDiamagCavityRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                         const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDiamagCavityRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                         const size_t & iOct) const
{

  // blast problem parameters
  const auto Vmax = m_dcParams.Vmax;
  const auto t0 = m_dcParams.t0;
  const auto cavity_center_x = m_dcParams.center_x;
  const auto cavity_center_y = m_dcParams.center_y;
  const auto cavity_center_z = m_dcParams.center_z;
  const auto shape = m_dcParams.shape;

  const real_t rmax = Vmax * t0;

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

    auto d2 = (xyz[IX] - cavity_center_x) * (xyz[IX] - cavity_center_x) +
              (xyz[IY] - cavity_center_y) * (xyz[IY] - cavity_center_y);

    if (dim == 3 && shape == +DiamagCavityShape::SPHERE)
      d2 += (xyz[IZ] - cavity_center_z) * (xyz[IZ] - cavity_center_z);

    if (fabs(sqrt(d2) - rmax) < (block_length * 1.25))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitBlastRefineFunctor::operator ()

// explicit template instantiation
template class InitDiamagCavityRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitDiamagCavityRefineFunctor<3, kalypsso::DefaultDevice>;

// ===========================================================
// ===========================================================
template <size_t dim, typename device_t>
void
InitDiamagCavity<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
{

  auto                amr_mesh = solver.amr_mesh();
  ConfigMap const &   config_map = solver.config_map();
  HydroParams const & params = solver.hydro_params();
  const int           level_min = solver.hydro_params().level_min;
  const int           level_max = solver.hydro_params().level_max;

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitDiamagCavityDataFunctor<dim, device_t>::apply(solver.par_env(),
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
      InitDiamagCavityDataFunctor<dim, device_t>::apply(solver.par_env(),
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
      InitDiamagCavityRefineFunctor<dim, device_t>::apply(solver.mesh_map()->orchard_keys(),
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
      InitDiamagCavityDataFunctor<dim, device_t>::apply(solver.par_env(),
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

} // InitBlast::apply

template class InitDiamagCavity<2, kalypsso::DefaultDevice>;
template class InitDiamagCavity<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
