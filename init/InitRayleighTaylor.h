// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRayleighTaylor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_RAYLEIGH_TAYLOR_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_RAYLEIGH_TAYLOR_H_

#include <godunov_mhd_ct/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/problems/RayleighTaylorParams.h>

#include <Kokkos_Random.hpp> // for random number drawing on device

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve rayleigh_taylor problem.
 *
 * See http://www.astro.princeton.edu/~jstone/Athena/tests/rayleigh_taylor/rayleigh_taylor.html
 *
 * This functor takes as input a mesh, already refined (after companion functor
 * InitRayleighTaylorRefineFunctor), and initializes user data on host.
 * Copying data from host to device, should be done outside.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * \sa InitRayleighTaylorRefineFunctor
 */
template <size_t dim, typename device_t>
class InitRayleighTaylorDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  using RGPool_t = typename Kokkos::Random_XorShift64_Pool<exec_space>;
  using rng_state_t = typename RGPool_t::generator_type;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroParams m_params;

  //! MHD settings (used on device)
  MHDSettings m_mhd_settings;

  //! RayleighTaylor problem specific parameters (used on device)
  RayleighTaylorParams m_rt_params;

  //! gravity field
  Kokkos::Array<real_t, dim> m_grav;

  //! random number generator pool
  RGPool_t m_rand_pool;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! p4est brick connectivity sizes
  brick_size_t<dim> m_brick_sizes;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

public:
  //! init all hydro var, except total energy (partially initialized)
  struct TagInitHydroVar
  {};

  //! init magnetic field
  struct TagInitMagField
  {};

  //! update total energy with magnetic energy
  struct TagInitTotalEnergy
  {};

  InitRayleighTaylorDataFunctor(orchard_key_view_t<device_t> orchard_keys,
                                int32_t                      local_num_octants,
                                HydroParams                  params,
                                ConfigMap const &            config_map,
                                Kokkos::Array<real_t, dim>   gravity_field,
                                FieldMap<core::models::MHD>  fm,
                                brick_size_t<dim>            brick_sizes,
                                DataArrayBlock_t             Udata,
                                FaceDataArrayBlock_t         Bface);

  // static method which does it all: create and execute functor
  static auto
  apply([[maybe_unused]] ParallelEnv const & par_env,
        orchard_key_view_t<device_t>         orchard_keys,
        int32_t                              local_num_octants,
        HydroParams                          params,
        ConfigMap const &                    config_map,
        FieldMap<core::models::MHD>          fm,
        brick_size_t<dim>                    brick_sizes,
        DataArrayBlock_t                     Udata,
        FaceDataArrayBlock_t                 Bface);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitHydroVar const &, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitMagField const &, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitTotalEnergy const &, const int32_t & global_index) const;

}; // InitRayleighTaylorDataFunctor

// explicit template instantiation
extern template class InitRayleighTaylorDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylorDataFunctor<3, kalypsso::DefaultDevice>;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initial refinement to solve rayleigh_taylor problem.
 *
 * See http://www.astro.princeton.edu/~jstone/Athena/tests/rayleigh_taylor/rayleigh_taylor.html
 *
 * This functor only performs mesh refinement, no user data init.
 * User data init is actually done in InitRayleighTaylorDataFunctor
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitRayleighTaylorDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitRayleighTaylorRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroParams m_params;

  //! RayleighTaylor problem specific parameters (used on device)
  RayleighTaylorParams m_rt_params;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! p4est brick connectivity sizes
  brick_size_t<dim> m_brick_sizes;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

public:
  // ===========================================================
  // ===========================================================
  InitRayleighTaylorRefineFunctor(orchard_key_view_t<device_t> orchard_keys,
                                  int32_t                      local_num_octants,
                                  ConfigMap const &            config_map,
                                  HydroParams                  params,
                                  FieldMap<core::models::MHD>  fm,
                                  brick_size_t<dim>            brick_sizes,
                                  DataArrayBlock_t             Udata,
                                  FaceDataArrayBlock_t         Bface,
                                  amrflags_view_t              amrflags,
                                  int                          level_refine);

  // ===========================================================
  // ===========================================================
  // static method which does it all: create and execute functor
  static void
  apply(orchard_key_view_t<device_t> orchard_keys,
        int32_t                      local_num_octants,
        ConfigMap const &            config_map,
        HydroParams                  params,
        FieldMap<core::models::MHD>  fm,
        brick_size_t<dim>            brick_sizes,
        DataArrayBlock_t             Udata,
        FaceDataArrayBlock_t         Bface,
        amrflags_view_t              amrflags,
        int                          level_refine);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // InitRayleighTaylorRefineFunctor

// explicit template instantiation
extern template class InitRayleighTaylorRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylorRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Hydrodynamical rayleigh_taylor Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/rayleigh_taylor/rayleigh_taylor.html
 *
 * Initial condition is mostly done on host, the final refined initial
 * condition data are uploaded to kokkos device.
 *
 * Different initial refinement strategies are possible:
 * - no refinement at all
 * - geometric refinement, i.e. refine near interface
 * - regular gradient based refinement
 *
 * This is controlled input parameter in ini file : "amr"/"init_condition_refine_criterion"
 */
template <size_t dim, typename device_t>
class InitRayleighTaylor
{
public:
  static void
  apply(SolverGodunovMHD<dim, device_t> & solver);
}; // class InitRayleighTaylor

// explicit template instantiation declaration to prevent implicit instantiation
extern template class InitRayleighTaylor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_RAYLEIGH_TAYLOR_H_
