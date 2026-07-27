// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRayleighTaylor.cpp
 */

#include <godunov_mhd_ct/init/InitRayleighTaylor.h>
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
InitRayleighTaylorDataFunctor<dim, device_t>::InitRayleighTaylorDataFunctor(
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  HydroParams                  params,
  ConfigMap const &            config_map,
  Kokkos::Array<real_t, dim>   gravity_field,
  FieldMap<models::MHD>  fm,
  brick_size_t<dim>            brick_sizes,
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_mhd_settings(config_map)
  , m_rt_params(config_map)
  , m_grav(gravity_field)
  , m_rand_pool(m_rt_params.seed)
  , m_fm(fm)
  , m_brick_sizes(brick_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_Udata(Udata)
  , m_Bface(Bface)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_xyz_max(get_xyz_max<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
InitRayleighTaylorDataFunctor<dim, device_t>::apply([[maybe_unused]] ParallelEnv const & par_env,
                                                    orchard_key_view_t<device_t> orchard_keys,
                                                    int32_t                      local_num_octants,
                                                    HydroParams                  params,
                                                    ConfigMap const &            config_map,
                                                    FieldMap<models::MHD>  fm,
                                                    brick_size_t<dim>            brick_sizes,
                                                    DataArrayBlock_t             Udata,
                                                    FaceDataArrayBlock_t         Bface)
{
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  // data init functor
  InitRayleighTaylorDataFunctor functor(orchard_keys,
                                        local_num_octants,
                                        params,
                                        config_map,
                                        gravity_field,
                                        fm,
                                        brick_sizes,
                                        Udata,
                                        Bface);

  // for cell-centered variables init : compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // for face-centered variables init
  const auto nbFacesPerLeaf = Bface.num_elements_per_octant();
  const auto nbFacesTotal = local_num_octants * nbFacesPerLeaf;

  // initialize all hydro variables but not total energy (cell-centered)
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::InitRayleighTaylorDataFunctor - all hydro variables "
    "except total energy",
    Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
    functor);

  // initialize mag field (face-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRayleighTaylorDataFunctor - magnetic field",
                       Kokkos::RangePolicy<exec_space, TagInitMagField>(0, nbFacesTotal),
                       functor);

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRayleighTaylorDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);


} // InitRayleighTaylorDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRayleighTaylorDataFunctor<dim, device_t>::operator()(TagInitHydroVar const &,
                                                         const int32_t & global_index) const
{
  /// convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  constexpr auto ID = models::MHD::ID;
  constexpr auto IP = models::MHD::IP;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;

  // Rayleigh-Taylor problem parameters
  const auto                  rho_up = m_rt_params.rho_up;
  const auto                  rho_down = m_rt_params.rho_down;
  const auto                  ampl = m_rt_params.amplitude;
  const auto                  P0 = m_rt_params.P0;
  const auto                  nx = m_rt_params.nx;
  const auto                  ny = m_rt_params.ny;
  [[maybe_unused]] const auto nz = m_rt_params.nz;
  const auto                  perturb_type = m_rt_params.perturb_type;

  const auto & use_isothermal_init = m_rt_params.use_isothermal_init;
  const auto & T_up = m_rt_params.T_up;
  const auto & T_down = m_rt_params.T_down;

  // heat capacity ratio
  const auto gamma0 = m_mhd_settings.hydro.gamma0;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  // const auto level = orchard_key_t<dim>::level(key);

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  // const auto dx = compute_cell_length<dim>(level, block_sizes[IX]);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);
  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);


  if constexpr (dim == 2)
  {
    auto const & x = xyz[IX];
    auto const & y = xyz[IY];

    auto const & xmin = m_xyz_min[IX];
    auto const & xmax = m_xyz_max[IX];
    auto const & ymin = m_xyz_min[IY];
    auto const & ymax = m_xyz_max[IY];

    // center of the box
    const auto xc = (xmin + xmax) / 2;
    const auto yc = (ymin + ymax) / 2;

    // box length
    const auto Lx = xmax - xmin;
    const auto Ly = ymax - ymin;

    real_t d;
    real_t P;
    if (use_isothermal_init)
    {
      real_t boltzmann_factor = (y > yc)
                                  ? exp((m_grav[IX] * (x - xc) + m_grav[IY] * (y - yc)) / T_up)
                                  : exp((m_grav[IX] * (x - xc) + m_grav[IY] * (y - yc)) / T_down);

      d = (y > yc) ? P0 / T_up * boltzmann_factor : P0 / T_down * boltzmann_factor;

      // initialize pressure so that initial state is at hydrostatic equilibrium, under istothermal
      // hypothesis: - dP/dy + rhog = 0, which can be re-written as dP/dy = P/T g which can be
      // integrated into: P(y) = P0 * exp( g(y-y0) / T )
      P = P0 * boltzmann_factor;
    }
    else
    {
      d = (y > yc) ? rho_up : rho_down;

      // initialize pressure so that initial state is at hydrostatic equilibrium, i.e.:
      // -dP/dy + rhog = 0, which can be integrated (under constant uniform gravity) into
      // P = P0 + rho g y
      P = P0 + d * (m_grav[IX] * x + m_grav[IY] * y);
    }

    m_Udata(cell_index, m_fm[ID], iOct) = d;

    m_Udata(cell_index, m_fm[IU], iOct) = ZERO_F;

    if (perturb_type == +RayleighTaylorPerturbationType::SINE)
    {
      m_Udata(cell_index, m_fm[IV], iOct) =
        ampl * d * (1 + cos(2 * PI_F * x * static_cast<real_t>(nx) / Lx)) *
        (1 + cos(2 * PI_F * y * static_cast<real_t>(ny) / Ly)) / 4;
    }
    else if (perturb_type == +RayleighTaylorPerturbationType::RANDOM)
    {
      // get random number state
      rng_state_t rand_gen = m_rand_pool.get_state();

      m_Udata(cell_index, m_fm[IV], iOct) =
        (static_cast<real_t>(rand_gen.drand()) - HALF_F) *
        (ONE_F + cos(2 * PI_F * y * static_cast<real_t>(ny) / Ly));

      // free random number, so that it can be used by other threads later
      m_rand_pool.free_state(rand_gen);
    }

    // clang-format off
    auto ekin = HALF_F * (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                       m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct)) / d;
    // clang-format on

    m_Udata(cell_index, m_fm[IP], iOct) = P / (gamma0 - ONE_F) + ekin;
  }
  else if constexpr (dim == 3)
  {
    auto const & x = xyz[IX];
    auto const & y = xyz[IY];
    auto const & z = xyz[IZ];

    auto const & xmin = m_xyz_min[IX];
    auto const & xmax = m_xyz_max[IX];
    auto const & ymin = m_xyz_min[IY];
    auto const & ymax = m_xyz_max[IY];
    auto const & zmin = m_xyz_min[IZ];
    auto const & zmax = m_xyz_max[IZ];

    // center of the box
    const auto xc = (xmin + xmax) / 2;
    const auto yc = (ymin + ymax) / 2;
    const auto zc = (zmin + zmax) / 2;

    // box length
    const auto Lx = xmax - xmin;
    const auto Ly = ymax - ymin;
    const auto Lz = zmax - zmin;

    real_t d;
    real_t P;

    if (use_isothermal_init)
    {
      real_t boltzmann_factor =
        (z > zc)
          ? exp((m_grav[IX] * (x - xc) + m_grav[IY] * (y - yc) + m_grav[IZ] * (z - zc)) / T_up)
          : exp((m_grav[IX] * (x - xc) + m_grav[IY] * (y - yc) + m_grav[IZ] * (z - zc)) / T_down);

      d = (z > zc) ? P0 / T_up * boltzmann_factor : P0 / T_down * boltzmann_factor;

      // initialize pressure so that initial state is at hydrostatic equilibrium, under istothermal
      // hypothesis: - dP/dz + rhog = 0, which can be re-written as dP/dz = P/T g which can be
      // integrated into: P(z) = P0 * exp( g(z-z0) / T )
      P = P0 * boltzmann_factor;
    }
    else
    {
      d = (z > zc) ? rho_up : rho_down;

      // initialize pressure so that initial state is at hydrostatic equilibrium, i.e.:
      // -dP/dz + rho g = 0, which can be integrated (under constant uniform gravity) into
      // P = P0 + rho g z
      P = P0 + d * (m_grav[IX] * x + m_grav[IY] * y + m_grav[IZ] * z);
    }

    m_Udata(cell_index, m_fm[ID], iOct) = d;

    m_Udata(cell_index, m_fm[IU], iOct) = ZERO_F;
    m_Udata(cell_index, m_fm[IV], iOct) = ZERO_F;

    if (perturb_type == +RayleighTaylorPerturbationType::SINE)
    {
      m_Udata(cell_index, m_fm[IW], iOct) =
        ampl * d * (1 + cos(2 * PI_F * x * static_cast<real_t>(nx) / Lx)) *
        (1 + cos(2 * PI_F * y * static_cast<real_t>(ny) / Ly)) *
        (1 + cos(2 * PI_F * z * static_cast<real_t>(nz) / Lz)) / 8;
    }
    else if (perturb_type == +RayleighTaylorPerturbationType::RANDOM)
    {
      // get random number state
      rng_state_t rand_gen = m_rand_pool.get_state();

      m_Udata(cell_index, m_fm[IW], iOct) =
        (static_cast<real_t>(rand_gen.drand()) - HALF_F) *
        (ONE_F + cos(2 * PI_F * z * static_cast<real_t>(nz) / Lz));

      // free random number, so that it can be used by other threads later
      m_rand_pool.free_state(rand_gen);
    }

    // clang-format off
    auto ekin = HALF_F * (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                       m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct) +
                       m_Udata(cell_index, m_fm[IW], iOct) * m_Udata(cell_index, m_fm[IW], iOct)) / d;
    // clang-format on

    m_Udata(cell_index, m_fm[IP], iOct) = P / (gamma0 - ONE_F) + ekin;

  } // end dim==3

} // end InitRayleighTaylorDataFunctor::operator () - TagInitHydroVar

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRayleighTaylorDataFunctor<dim, device_t>::operator()(TagInitMagField const &,
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
      m_Bface(i, j, IX, iOct) = m_rt_params.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, IY, iOct) = m_rt_params.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, IZ, iOct) = m_rt_params.bz;
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
      m_Bface(i, j, k, IX, iOct) = m_rt_params.bx;
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, k, IY, iOct) = m_rt_params.by;
    }
    else if (ivar == IZ)
    {
      // Bz on Z-face
      m_Bface(i, j, k, IZ, iOct) = m_rt_params.bz;
    }
  } // end dim == 3

} // end InitRayleighTaylorDataFunctor::operator() - TagInitMagField

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRayleighTaylorDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy const &,
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

} // end InitRayleighTaylorDataFunctor::operator () - TagInitTotalEnergy

// explicit template instantiation
template class InitRayleighTaylorDataFunctor<2, kalypsso::DefaultDevice>;
template class InitRayleighTaylorDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitRayleighTaylorRefineFunctor<dim, device_t>::InitRayleighTaylorRefineFunctor(
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  ConfigMap const &            config_map,
  HydroParams                  params,
  FieldMap<models::MHD>  fm,
  brick_size_t<dim>            brick_sizes,
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  amrflags_view_t              amrflags,
  int                          level_refine)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_params(params)
  , m_rt_params(config_map)
  , m_fm(fm)
  , m_brick_sizes(brick_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_Udata(Udata)
  , m_Bface(Bface)
  , m_amrflags(amrflags)
  , m_level_refine(level_refine)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_xyz_max(get_xyz_max<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRayleighTaylorRefineFunctor<dim, device_t>::apply(orchard_key_view_t<device_t> orchard_keys,
                                                      int32_t                     local_num_octants,
                                                      ConfigMap const &           config_map,
                                                      HydroParams                 params,
                                                      FieldMap<models::MHD> fm,
                                                      brick_size_t<dim>           brick_sizes,
                                                      DataArrayBlock_t            Udata,
                                                      FaceDataArrayBlock_t        Bface,
                                                      amrflags_view_t             amrflags,
                                                      int                         level_refine)
{
  // iterate functor for refinement
  InitRayleighTaylorRefineFunctor functor(orchard_keys,
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
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRayleighTaylorRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitRayleighTaylorRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }
} // IniRayleighTaylorRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRayleighTaylorRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                           const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRayleighTaylorRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                           const size_t & iOct) const
{
  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  const auto level = orchard_key_t<dim>::level(key);

  // compute block length (in real space units)
  const auto block_length = compute_block_length<dim>(level) * m_scaling_factor;

  // default : do nothing, i.e. neither refine or coarsen
  auto flag = AMRContextBase::KALYPSSO_DO_NOTHING;

  // Rayleigh-Taylor problem parameters
  auto const & ymin = m_xyz_min[IY];
  auto const & ymax = m_xyz_max[IY];

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    if constexpr (dim == 2)
    {
      // compute distance to interface
      const auto d = fabs(xyz[IY] - (ymin + ymax) / 2);

      if (d < (block_length * KALYPSSO_NUM(0.95)))
        flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }
    else if constexpr (dim == 3)
    {
      auto const & zmin = m_xyz_min[IZ];
      auto const & zmax = m_xyz_max[IZ];

      // compute distance to interface
      const auto d = fabs(xyz[IZ] - (zmin + zmax) / 2);

      if (d < (block_length * KALYPSSO_NUM(0.95)))
        flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;
} // InitRayleighTaylorRefineFunctor::operator ()

// explicit template instantiation
template class InitRayleighTaylorRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitRayleighTaylorRefineFunctor<3, kalypsso::DefaultDevice>;

// ===========================================================
// ===========================================================
template <size_t dim, typename device_t>
void
InitRayleighTaylor<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
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

  InitRayleighTaylorDataFunctor<dim, device_t>::apply(solver.par_env(),
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
      InitRayleighTaylorDataFunctor<dim, device_t>::apply(solver.par_env(),
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
      InitRayleighTaylorRefineFunctor<dim, device_t>::apply(
        solver.mesh_map()->orchard_keys(),
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
      InitRayleighTaylorDataFunctor<dim, device_t>::apply(solver.par_env(),
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

} // InitRayleighTaylor::apply

template class InitRayleighTaylor<2, kalypsso::DefaultDevice>;
template class InitRayleighTaylor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
