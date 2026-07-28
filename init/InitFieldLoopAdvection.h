// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitFieldLoopAdvection.h
 *
 * The 2D/3D MHD field loop advection problem.
 *
 * Parameters that can be set in the ini file :
 * - radius        : radius of field loop
 * - B0            : amplitude of magnetic field inside field loop
 * - vx,vy,vz      : flow velocity
 * - density_ratio : density ratio in loop.  Enables density advection and
 *                  thermal conduction tests.
 *
 * The flow is automatically set to run along the diagonal.
 * - direction : integer
 *   direction 0 -> field loop in x-y plane (cylinder in 3D)
 *   direction 1 -> field loop in y-z plane (cylinder in 3D)
 *   direction 2 -> field loop in z-x plane (cylinder in 3D)
 *   direction 3 -> rotated cylindrical field loop in 3D.
 *
 * Reference :
 * - T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
 *   via constrained transport", JCP, 205, 509 (2005)
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_FIELD_LOOP_ADVECTION_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_FIELD_LOOP_ADVECTION_H_

#include <godunov_mhd_ct/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/FieldLoopAdvectionParams.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim>
struct FieldLoopAdvectionVectorPotential
{
  //! FieldLoopAdvection problem specific parameters (used on device)
  FieldLoopAdvectionParams m_flaParams;

  FieldLoopAdvectionVectorPotential(FieldLoopAdvectionParams flaParams)
    : m_flaParams(flaParams)
  {}

  /**
   * Components of vector potential.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  operator()(Kokkos::Array<real_t, dim> xyz, int ivar) const
  {
    if (ivar == IX or ivar == IY)
    {
      return ZERO_F;
    }
    else
    {
      auto const & xc = m_flaParams.xc;
      auto const & yc = m_flaParams.yc;

      auto r = sqrt((xyz[IX] - xc) * (xyz[IX] - xc) + (xyz[IY] - yc) * (xyz[IY] - yc));

      if (r < m_flaParams.radius)
      {
        return m_flaParams.B0 * (m_flaParams.radius - r);
      }
      else
      {
        return ZERO_F;
      }
    }
  } // value
}; // struct FieldLoopAdvectionVectorPotential

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve magnetic field loop advection problem.
 *
 *
 * This functor takes as input a mesh, already refined (after companion functor
 * InitFieldLoopAdvectionRefineFunctor), and initializes user data on host.
 * Copying data from host to device, should be done outside.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * \sa InitFieldLoopAdvectionRefineFunctor
 */
template <size_t dim, typename device_t>
class InitFieldLoopAdvectionDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;
  using EdgeDataArrayBlock_t = EdgeDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! MHD settings
  MHDSettings m_mhd_settings;

  //! FieldLoopAdvection problem specific parameters (used on device)
  FieldLoopAdvectionParams m_flaParams;

  //! FieldLoopAdvection analytical vector potential (used on device)
  FieldLoopAdvectionVectorPotential<dim> m_fla_vector_potential;

  //! vector potential components
  EdgeDataArrayBlock_t m_vectorPotential;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  //! init all hydro var, except total energy (partially initialized)
  struct TagInitHydroVar
  {};

  //! init vector potential (must be done before initialization magnetic field obviously)
  struct TagInitVectorPotential
  {};

  //! update total energy with magnetic energy
  struct TagInitTotalEnergy
  {};

  InitFieldLoopAdvectionDataFunctor(DataArrayBlock_t             Udata,
                                    FaceDataArrayBlock_t         Bface,
                                    FieldMap<models::MHD>        fm,
                                    orchard_key_view_t<device_t> orchard_keys,
                                    int32_t                      local_num_octants,
                                    ConfigMap const &            config_map);

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t         Udata,
        FaceDataArrayBlock_t     Bface,
        FieldMap<models::MHD>    fm,
        MeshMap<dim, device_t> & mesh_map,
        int32_t                  local_num_octants,
        ConfigMap const &        config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitHydroVar, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitVectorPotential, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitTotalEnergy, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  auto
  vectorPotential()
  {
    return m_vectorPotential;
  }

  // ====================================================================
  // ====================================================================
  FaceDataArrayBlock_t &
  magnetic_field()
  {
    return m_Bface;
  }

}; // InitFieldLoopAdvectionDataFunctor

// explicit template instantiation
extern template class InitFieldLoopAdvectionDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitFieldLoopAdvectionDataFunctor<3, kalypsso::DefaultDevice>;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initial refinement to solve FieldLoopAdvection shock tube problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * User data init is actually done in InitFieldLoopAdvectionDataFunctor
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitFieldLoopAdvectionDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitFieldLoopAdvectionRefineFunctor
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
  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  MHDSettings m_mhd_settings;

  //! FieldLoopAdvection problem specific parameters (used on device)
  FieldLoopAdvectionParams m_flaParams;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  // ===========================================================
  // ===========================================================
  InitFieldLoopAdvectionRefineFunctor(DataArrayBlock_t             Udata,
                                      FaceDataArrayBlock_t         Bface,
                                      FieldMap<models::MHD>        fm,
                                      orchard_key_view_t<device_t> orchard_keys,
                                      amrflags_view_t              amrflags,
                                      int32_t                      local_num_octants,
                                      int                          level_refine,
                                      ConfigMap const &            config_map);

  // ===========================================================
  // ===========================================================
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t             Udata,
        FaceDataArrayBlock_t         Bface,
        FieldMap<models::MHD>        fm,
        orchard_key_view_t<device_t> orchard_keys,
        amrflags_view_t              amrflags,
        int32_t                      local_num_octants,
        int                          level_refine,
        ConfigMap const &            config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const iOct_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const iOct_t & iOct) const;

}; // InitFieldLoopAdvectionRefineFunctor

// explicit template instantiation
extern template class InitFieldLoopAdvectionRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitFieldLoopAdvectionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * MHD magnetic field loop advection problem.
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
class InitFieldLoopAdvection
{
public:
  static void
  apply(SolverGodunovMHD<dim, device_t> & solver);
};

// explicit template instantiation declaration to prevent implicit instantiation
extern template class InitFieldLoopAdvection<2, kalypsso::DefaultDevice>;
extern template class InitFieldLoopAdvection<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_FIELD_LOOP_ADVECTION_H_
