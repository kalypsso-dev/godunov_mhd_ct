// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitDiamagCavity.h
 *
 * Reference:
 *
 * Combined Radiation and Release Effects Satellite (CRRES) experiment G-10
 * https://ntrs.nasa.gov/api/citations/19960001204/downloads/19960001204.pdf
 *
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_DIAMAG_CAVITY_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_DIAMAG_CAVITY_H_

#include <godunov_mhd_ct/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/problems/DiamagCavityParams.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 *
 */
template <size_t dim, typename device_t>
class InitDiamagCavityDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroParams m_params;

  //! MHD settings
  MHDSettings m_mhd_settings;

  //! Diamagnetic cavity problem specific parameters (used on device)
  DiamagCavityParams m_dcParams;

  //! field manager
  FieldMap<models::MHD> m_fm;

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

  InitDiamagCavityDataFunctor(orchard_key_view_t<device_t> orchard_keys,
                              int32_t                      local_num_octants,
                              HydroParams                  params,
                              ConfigMap const &            config_map,
                              FieldMap<models::MHD>  fm,
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
        FieldMap<models::MHD>          fm,
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

}; // InitDiamagCavityDataFunctor

extern template class InitDiamagCavityDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitDiamagCavityDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 *
 */
template <size_t dim, typename device_t>
class InitDiamagCavityRefineFunctor
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

  //! DiamagCavity problem specific parameters (used on device)
  DiamagCavityParams m_dcParams;

  //! field manager
  FieldMap<models::MHD> m_fm;

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

public:
  // ===========================================================
  // ===========================================================
  InitDiamagCavityRefineFunctor(orchard_key_view_t<device_t> orchard_keys,
                                int32_t                      local_num_octants,
                                ConfigMap const &            config_map,
                                HydroParams                  params,
                                FieldMap<models::MHD>  fm,
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
        FieldMap<models::MHD>  fm,
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

}; // InitDiamagCavityRefineFunctor

extern template class InitDiamagCavityRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitDiamagCavityRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
class InitDiamagCavity
{
public:
  static void
  apply(SolverGodunovMHD<dim, device_t> & solver);
}; // class InitDiamagCavity

// explicit template instantiation declaration to prevent implicit instantiation
extern template class InitDiamagCavity<2, kalypsso::DefaultDevice>;
extern template class InitDiamagCavity<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_DIAMAG_CAVITY_H_
