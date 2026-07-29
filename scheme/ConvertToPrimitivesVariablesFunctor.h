// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConvertToPrimitivesVariablesFunctor.h
 *
 * MHD variant.
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_CONVERTTOPRIMITIVESVARIABLES_H_
#define KALYPSSO_GODUNOV_MHD_CT_CONVERTTOPRIMITIVESVARIABLES_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/utils_block.h> // for coord_t
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/FaceDataArrayBlock.h>
#include <godunov_mhd_ct/models/mhd_utils.h> // for computePrimitives
#include <kalypsso/core/HydroParams.h>       // for MHDSettings
#include <kalypsso/core/mesh_utils.h>        // for definition of Face::XMIN, etc...

namespace kalypsso
{

namespace godunov_mhd_ct
{

/**
 * \class ConvertToPrimitivesVariablesFunctor
 *
 * This class is a direct adaptation of FillBlockGhostsFunctor modified to the special need of
 * converting conservative variables to primitives ones in MHD.
 *
 * What is specific to MHD is that we store face-centered magnetic field (cell-centered magnetic
 * field will be recomputed as needed from face-centered values).
 *
 */
template <size_t dim, typename device_t>
class ConvertToPrimitivesVariablesFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  //! type alias for face-centered data array at block level (see kalypsso_data_container.h)
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  //! makes enum MHD::VarId available
  using MHD = models::MHD;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard keys that are "mirrors" (in the p4est sense).
  //! only used when we want to solely computed mirror quadrants.
  orchard_key_view_t m_mirror_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! starting octant id.
  //! this is a global octant id offset to the first octant to be processed when computing primitive
  //! variables in a group of owned quadrants.
  //! \note it is not used when computing primitive variables in mirrors quadrants, because mirror
  //! quadrants are process all at once.
  const int32_t m_iOct_begin;

  //! cell-centered conservative variables (no ghosts, sizes= block_x,block_y,block_z)
  DataArrayBlock_t m_userdata_in;

  //! ghosted face-centered magnetic field
  FaceDataArrayBlock_t m_Bface_ghosted;

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayGhostedBlock_t m_userdata_out;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! MHD parameters
  MHDSettings m_mhd_settings;

  //! prolongation parameter
  const ProlongationParam m_prolongation;

  //! MPI comm rank from parallel environment (maybe unused, but useful for debug)
  const int m_mpi_comm_rank;

public:
  struct TagComputeMirrorQuad
  {};
  struct TagComputeAllQuad
  {};

  /**
   *
   * Compute primitives variables in a group of owned quadrants.
   *
   * \param[in] stencil helper
   * \param[in] amr_mesh_info number of octants (owned, ghost, outside, ...)
   * \param[in] iOct_begin is the first octant to process
   * \param[in] userdata_in data array used to fill ghost of userdata_out
   * \param[in] Bface_ghosted ghosted face-centered magnetic field
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   *
   */
  ConvertToPrimitivesVariablesFunctor(StencilHelper_t const &         stencil_helper,
                                      AMRMeshInfo const &             amr_mesh_info,
                                      int32_t                         iOct_begin,
                                      DataArrayBlock_t const &        userdata_in,
                                      FaceDataArrayBlock_t const &    Bface_ghosted,
                                      DataArrayGhostedBlock_t const & userdata_out,
                                      MHDSettings const &             mhd_settings,
                                      ProlongationParam const &       prolongation,
                                      const int                       mpi_comm_rank);

  //! same as above, but specifying also the mirror keys array
  ConvertToPrimitivesVariablesFunctor(StencilHelper_t const &         stencil_helper,
                                      orchard_key_view_t const &      mirror_orchard_keys,
                                      AMRMeshInfo const &             amr_mesh_info,
                                      DataArrayBlock_t const &        userdata_in,
                                      FaceDataArrayBlock_t const &    Bface_ghosted,
                                      DataArrayGhostedBlock_t const & userdata_out,
                                      MHDSettings const &             mhd_settings,
                                      ProlongationParam const &       prolongation,
                                      const int                       mpi_comm_rank);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply_on_group(ConfigMap const &                config_map,
                 amr_hashmap_t const &            amr_hashmap,
                 orchard_key_view_t const &       orchard_keys,
                 AMRMeshInfo const &              amr_mesh_info,
                 int32_t                          iOct_begin,
                 int32_t                          num_octants_in_group,
                 DataArrayBlock_t const &         userdata_in,
                 FaceDataArrayBlock_t const &     Bface_ghosted,
                 DataArrayGhostedBlock_t const &  userdata_out,
                 brick_size_t<dim> const &        brick_sizes,
                 Kokkos::Array<bool, dim> const & is_brick_periodic,
                 MHDSettings const &              mhd_settings,
                 ParallelEnv const &              par_env);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy.
  //!
  //! Use this member when computing primitive only in mirror quadrants.
  static void
  apply_in_mirrors(ConfigMap const &                config_map,
                   amr_hashmap_t const &            amr_hashmap,
                   orchard_key_view_t const &       orchard_keys,
                   orchard_key_view_t const &       mirror_orchard_keys,
                   AMRMeshInfo const &              amr_mesh_info,
                   DataArrayBlock_t const &         userdata_in,
                   FaceDataArrayBlock_t const &     Bface_ghosted,
                   DataArrayGhostedBlock_t const &  userdata_out,
                   brick_size_t<dim> const &        brick_sizes,
                   Kokkos::Array<bool, dim> const & is_brick_periodic,
                   MHDSettings const &              mhd_settings,
                   ParallelEnv const &              par_env);

  // ==============================================================
  // ==============================================================
  /**
   * Read cell-centered conservative hydro variables and convert face-centered into cell-centered
   * magnetic field components.
   */
  KOKKOS_INLINE_FUNCTION
  MHDStateCell
  get_conservative_vars(const int32_t        cellindex,
                        coord_t<dim> const & iCoord,
                        const iOct_t         iOct) const;

  // ==============================================================
  // ==============================================================
  /**
   * Read cell-centered conservative hydro variables and compute cell-centered
   * magnetic field components (from face-centered).
   */
  KOKKOS_INLINE_FUNCTION
  MHDStateCell
  get_conservative_vars(CellLocation_t const & cell_loc_in) const;

  // ==============================================================
  // ==============================================================
  /**
   * Read cell-centered conservative hydro variables and compute cell-centered
   * magnetic field components (from face-centered) and perform a restriction (average from fine to
   * coarse AMR level).
   */
  KOKKOS_INLINE_FUNCTION
  MHDStateCell
  get_conservative_vars_restriction(coord_t<dim> const &   coord_out,
                                    CellLocation_t const & cell_loc_out,
                                    CellLocation_t const & cell_loc_in) const;

  // ==============================================================
  // ==============================================================
  /**
   * Read face-centered magnetic field components from a cell location (inner block).
   *
   */
  KOKKOS_INLINE_FUNCTION
  FaceMagState
  get_face_mag(CellLocation_t const & cell_loc_in) const;

  // ==============================================================
  // ==============================================================
  /**
   * Read face-centered magnetic field components in ghost zones.
   *
   * \param[in] coord_out cell coordinates inside ghost zone
   * \param[in] iOct_out octant index
   */
  KOKKOS_INLINE_FUNCTION
  FaceMagState
  get_face_mag(coord_t<dim> const & coord_out, iOct_t const & iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Write cell-centered primitive hydro variables and face-centered
   * magnetic field components.
   */
  KOKKOS_INLINE_FUNCTION
  void
  set_primitive_vars(const int32_t        cellindex_out_g,
                     coord_t<dim> const & coord_in,
                     const iOct_t         iOct_in,
                     const iOct_t         iOct_out,
                     MHDStateCell const & q,
                     FaceMagState const & face_mag) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] cellindex_in cell index where to read data from
   * \param[in] cellindex_out is the cell index of the ghost cell to fill
   * \param[in] iOct_global is the octant id among all octant owned by current MPI process.
   * \param[in] iOct_out index where to write data
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_inner(coord_t<dim> const & coord_in,
             int32_t              cellindex_in,
             int32_t              cellindex_out,
             iOct_t               iOct_global,
             iOct_t               iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at the same AMR level.
   *
   * \param[in] cell_loc_out
   * \param[in] cell_loc_in
   * \param[in] cellindex_out integer used to map the ghost cell to fill
   * \param[in] iOct_out
   * \param[in] hydro_only is true when only hydrodynamics variables are involved (magnetic field
   * unmodified)
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_ghost_copy(CellLocation_t const & cell_loc_out,
                  CellLocation_t const & cell_loc_in,
                  index_t const &        cellindex_out,
                  coord_t<dim> const &   coord_out,
                  iOct_t const &         iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation of hydrodynamics variables using limited slopes of hydro variables (no
   * magnetic field).
   *
   * \note it is important to note that this routine requires magnetic field is up to date (i.e.
   * already prolongated)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_hydro_vars_using_limited_slopes(CellLocation<2> const & cell_loc_neigh,
                                                     coord_t<2> const &      coord_in,
                                                     iOct_t const &          iOct_global,
                                                     index_t const &         cellindex_out,
                                                     coord_t<2> const &      coord_out,
                                                     iOct_t const &          iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation of hydrodynamics variables using limited slopes of hydro variables (no
   * magnetic field).
   *
   * \note it is important to note that this routine requires magnetic field is up to date (i.e.
   * already prolongated)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_hydro_vars_using_limited_slopes(CellLocation<3> const & cell_loc_neigh,
                                                     coord_t<3> const &      coord_in,
                                                     iOct_t const &          iOct_global,
                                                     index_t const &         cellindex_out,
                                                     coord_t<3> const &      coord_out,
                                                     iOct_t const &          iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data all around current octant (iOct_global).
   *
   * This is (almost) the main entry point of the functor, i.e. directly called inside operator().
   *
   * \param[in] cellindex integer used to map the ghost cell to fill
   * \param[in] coord cartesian coordinates of current cell inside block
   * \param[in] iOct_global is index to current octant
   * \param[in] iOct_out is index to where to write data
   *
   */
  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(index_t const &      cellindex,
              coord_t<dim> const & coord,
              iOct_t const &       iOct_global,
              iOct_t const &       iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor when computing in all group quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeAllQuad const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor when computing only mirror quadrant
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeMirrorQuad const &, const index_t & global_index) const;

}; // class ConvertToPrimitivesVariablesFunctor

// explicit template instantiation
extern template class ConvertToPrimitivesVariablesFunctor<2, kalypsso::DefaultDevice>;
extern template class ConvertToPrimitivesVariablesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_CONVERTTOPRIMITIVESVARIABLES_H_
