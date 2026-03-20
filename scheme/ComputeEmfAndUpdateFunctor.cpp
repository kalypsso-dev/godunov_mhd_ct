// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeEmfAndUpdateFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeEmfAndUpdateFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ComputeEmfAndUpdateFunctor<dim, device_t>::ComputeEmfAndUpdateFunctor(
  ConfigMap const &                   config_map,
  amr_hashmap_t const &               amr_hashmap,
  orchard_key_view_t const &          orchard_keys,
  conformal_status_view_type const &  conformal_status,
  AMRMeshInfo const &                 amr_mesh_info,
  FaceDataArrayBlock_t const &        b_in,
  FaceDataArrayBlock_t const &        b_out,
  DataArrayGhostedBlock_t const &     q,
  DataArrayGhostedBlock_t const &     q2,
  DataArrayGhostedBlock_t const &     slopes_x,
  DataArrayGhostedBlock_t const &     slopes_y,
  DataArrayGhostedBlock_t const &     slopes_z,
  DataArrayGhostedBlock_t const &     sFaceMag,
  FieldMap<core::models::MHD>         fm,
  int32_t                             iOct_begin,
  int32_t                             num_octants,
  Kokkos::Array<uint8_t, dim> const & brick_sizes,
  Kokkos::Array<bool, dim> const &    is_brick_periodic,
  MHDSettings const &                 mhd_settings,
  real_t                              dt)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_conformal_status(conformal_status)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Bin(b_in)
  , m_Bout(b_out)
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
  , m_block_sizes_emf(slopes_x.block_size() + 1)
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_nbEdgeEmfPerLeaf(Kokkos::dim_prod(m_block_sizes_emf))
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeEmfAndUpdateFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                   config_map,
  amr_hashmap_t const &               amr_hashmap,
  orchard_key_view_t const &          orchard_keys,
  conformal_status_view_type const &  conformal_status,
  AMRMeshInfo const &                 amr_mesh_info,
  FaceDataArrayBlock_t const &        Bin,
  FaceDataArrayBlock_t const &        Bout,
  DataArrayGhostedBlock_t const &     q,
  DataArrayGhostedBlock_t const &     q2,
  DataArrayGhostedBlock_t const &     slopes_x,
  DataArrayGhostedBlock_t const &     slopes_y,
  DataArrayGhostedBlock_t const &     slopes_z,
  DataArrayGhostedBlock_t const &     sFaceMag,
  FieldMap<core::models::MHD>         fm,
  int32_t                             iOct_begin,
  int32_t                             num_octants,
  Kokkos::Array<uint8_t, dim> const & brick_sizes,
  Kokkos::Array<bool, dim> const &    is_brick_periodic,
  MHDSettings const &                 mhd_settings,
  real_t                              dt)
{

  ComputeEmfAndUpdateFunctor<dim, device_t> functor(config_map,
                                                    amr_hashmap,
                                                    orchard_keys,
                                                    conformal_status,
                                                    amr_mesh_info,
                                                    Bin,
                                                    Bout,
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

  // we use atomic update, computations is edge-oriented
  const auto nbIterations = num_octants * functor.nb_edge_emf_per_leaf();

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
ComputeEmfAndUpdateFunctor<dim, device_t>::apply_on_ghosts(
  ConfigMap const &                   config_map,
  amr_hashmap_t const &               amr_hashmap,
  orchard_key_view_t const &          orchard_keys,
  conformal_status_view_type const &  conformal_status,
  AMRMeshInfo const &                 amr_mesh_info,
  FaceDataArrayBlock_t const &        Bin,
  FaceDataArrayBlock_t const &        Bout,
  DataArrayGhostedBlock_t const &     q,
  DataArrayGhostedBlock_t const &     q2,
  DataArrayGhostedBlock_t const &     slopes_x,
  DataArrayGhostedBlock_t const &     slopes_y,
  DataArrayGhostedBlock_t const &     slopes_z,
  DataArrayGhostedBlock_t const &     sFaceMag,
  FieldMap<core::models::MHD>         fm,
  Kokkos::Array<uint8_t, dim> const & brick_sizes,
  Kokkos::Array<bool, dim> const &    is_brick_periodic,
  MHDSettings const &                 mhd_settings,
  real_t                              dt)
{

  ComputeEmfAndUpdateFunctor<dim, device_t> functor(
    config_map,
    amr_hashmap,
    orchard_keys,
    conformal_status,
    amr_mesh_info,
    Bin,
    Bout,
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

  // we use atomic update, computations is edge-oriented
  const auto nbIterations = amr_mesh_info.local_num_ghosts() * functor.nb_edge_emf_per_leaf();

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
ComputeEmfAndUpdateFunctor<dim, device_t>::reconstruct_state_2d_at_edge(int32_t         is,
                                                                        int32_t         js,
                                                                        int32_t         iOct_local,
                                                                        MHDEdgeLocation edge_loc,
                                                                        MHDStateCell &  q) const
{
  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  // get current location cell-centered primitive variables state at time t_{n+1/2}
  // note: primitive variables is a ghosted array with ghost width of 1
  get_state(m_q2, is, js, iOct_local, q);

  auto const & smallr = m_mhd_settings.smallr;
  auto const & smallp = m_mhd_settings.smallp;

  int sign_x = 1;
  int sign_y = 1;
  int ia = is;
  int ja = js;
  int ib = 0;
  int jb = 0;
  int sign_a = 1;
  int sign_b = 1;

  if (edge_loc == MHDEdgeLocation::LB)
  {
    ia = is;
    ja = js;
    ib = 0;
    jb = 0;
    sign_x = -1;
    sign_y = -1;
    sign_a = -1;
    sign_b = -1;
  }
  else if (edge_loc == MHDEdgeLocation::RT)
  {
    ia = is + 1;
    ja = js;
    ib = 0;
    jb = 1;
    sign_x = 1;
    sign_y = 1;
    sign_a = 1;
    sign_b = 1;
  }
  else if (edge_loc == MHDEdgeLocation::RB)
  {
    ia = is + 1;
    ja = js;
    ib = 0;
    jb = 0;
    sign_x = 1;
    sign_y = -1;
    sign_a = -1;
    sign_b = 1;
  }
  else if (edge_loc == MHDEdgeLocation::LT)
  {
    ia = 0;
    ja = 0;
    ib = 0;
    jb = 1;
    sign_x = -1;
    sign_y = 1;
    sign_a = 1;
    sign_b = -1;
  }

  // remember that m_q is a ghosted array with ghost width of 2
  // while slopes arrays have a ghost width of 1, hence the shift of 1
  const real_t A = m_q(ia + 1, ja + 1, MHD::IAL, iOct_local) + m_sFaceMag(ia, ja, IX, iOct_local);
  const real_t dAy = compute_limited_slope(ia + 1, ja + 1, IA, iOct_local, IY);

  const real_t B = m_q(ib + 1, jb + 1, MHD::IBL, iOct_local) + m_sFaceMag(ib, jb, IY, iOct_local);
  const real_t dBx = compute_limited_slope(ib + 1, jb + 1, IB, iOct_local, IX);

  // get limited slopes along given direction "dir"
  const MHDStateCell dqX = get_state(get_slopes(IX), is, js, iOct_local);
  const MHDStateCell dqY = get_state(get_slopes(IY), is, js, iOct_local);

  // reconstruct state on edge center

  q[ID] += HALF_F * (sign_x * dqX[ID] + sign_y * dqY[ID]);
  q[IP] += HALF_F * (sign_x * dqX[IP] + sign_y * dqY[IP]);
  q[IU] += HALF_F * (sign_x * dqX[IU] + sign_y * dqY[IU]);
  q[IV] += HALF_F * (sign_x * dqX[IV] + sign_y * dqY[IV]);
  q[IW] += HALF_F * (sign_x * dqX[IW] + sign_y * dqY[IW]);
  // q[IA] += HALF_F * (sign_x * dqX[IA] + sign_y * dqY[IA]);
  // q[IB] += HALF_F * (sign_x * dqX[IB] + sign_y * dqY[IB]);
  // q[IC] += HALF_F * (sign_x * dqX[IC] + sign_y * dqY[IC]);

  q[IA] = A + HALF_F * (sign_a * dAy);
  q[IB] = B + HALF_F * (sign_b * dBx);
  q[IC] += HALF_F * (sign_x * dqX[IC] + sign_y * dqY[IC]);

  q[ID] = fmax(smallr, q[ID]);
  q[IP] = fmax(smallp * q[ID], q[IP]);

  return q;

} // reconstruct_state_2d_at_edge

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeEmfAndUpdateFunctor<dim, device_t>::reconstruct_state_3d_at_edge(int32_t         is,
                                                                        int32_t         js,
                                                                        int32_t         ks,
                                                                        int32_t         iOct_local,
                                                                        MHDEdgeLocation edge_loc,
                                                                        int             dir0,
                                                                        int             dir1,
                                                                        MHDStateCell &  q) const
{

  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  auto const & smallr = m_mhd_settings.smallr;
  auto const & smallp = m_mhd_settings.smallp;

  int                       sign_dq0 = 1;
  int                       sign_dq1 = 1;
  Kokkos::Array<int32_t, 3> ijk0{ is, js, ks };
  Kokkos::Array<int32_t, 3> ijk1{ is, js, ks };
  int                       sign_b0 = 1;
  int                       sign_b1 = 1;

  if (edge_loc == MHDEdgeLocation::LB)
  {
    sign_dq0 = -1;
    sign_dq1 = -1;
    sign_b0 = -1;
    sign_b1 = -1;
  }
  else if (edge_loc == MHDEdgeLocation::RT)
  {
    ijk0[dir0] += 1;
    ijk1[dir1] += 1;
    sign_dq0 = 1;
    sign_dq1 = 1;
    sign_b0 = 1;
    sign_b1 = 1;
  }
  else if (edge_loc == MHDEdgeLocation::RB)
  {
    ijk0[dir0] += 1;
    sign_dq0 = 1;
    sign_dq1 = -1;
    sign_b0 = -1;
    sign_b1 = 1;
  }
  else if (edge_loc == MHDEdgeLocation::LT)
  {
    ijk1[dir1] += 1;
    sign_dq0 = -1;
    sign_dq1 = 1;
    sign_b0 = 1;
    sign_b1 = -1;
  }

  const real_t B0 = m_q(ijk0[IX] + 1, ijk0[IY] + 1, ijk0[IZ] + 1, IA + dir0, iOct_local) +
                    m_sFaceMag(ijk0[IX], ijk0[IY], ijk0[IZ], dir0, iOct_local);
  const real_t dB0d1 =
    compute_limited_slope(ijk0[IX] + 1, ijk0[IY] + 1, ijk0[IZ] + 1, IA + dir0, iOct_local, dir1);

  const real_t B1 = m_q(ijk1[IX] + 1, ijk1[IY] + 1, ijk1[IZ] + 1, IA + dir1, iOct_local) +
                    m_sFaceMag(ijk1[IX], ijk1[IY], ijk1[IZ], dir1, iOct_local);
  const real_t dB1d0 =
    compute_limited_slope(ijk1[IX] + 1, ijk1[IY] + 1, ijk1[IZ] + 1, IA + dir1, iOct_local, dir0);


  // get limited slopes along in direction 0
  MHDStateCell dq0 = get_state(get_slopes(dir0), is, js, ks, iOct_local);
  MHDStateCell dq1 = get_state(get_slopes(dir1), is, js, ks, iOct_local);

  q[ID] += HALF_F * (sign_dq0 * dq0[ID] + sign_dq1 * dq1[ID]);
  q[IP] += HALF_F * (sign_dq0 * dq0[IP] + sign_dq1 * dq1[IP]);
  q[IU] += HALF_F * (sign_dq0 * dq0[IU] + sign_dq1 * dq1[IU]);
  q[IV] += HALF_F * (sign_dq0 * dq0[IV] + sign_dq1 * dq1[IV]);
  q[IW] += HALF_F * (sign_dq0 * dq0[IW] + sign_dq1 * dq1[IW]);
  // q[IA] += HALF_F * (sign_dq0 * dq0[IA] + sign_dq1 * dq1[IA]);
  // q[IB] += HALF_F * (sign_dq0 * dq0[IB] + sign_dq1 * dq1[IB]);
  // q[IC] += HALF_F * (sign_dq0 * dq0[IC] + sign_dq1 * dq1[IC]);

  if (dir0 == IX)
  {
    q[IA] = B0 + HALF_F * (sign_b0 * dB0d1);
  }
  else
  {
    q[IA] += HALF_F * (sign_dq0 * dq0[IA] + sign_dq1 * dq1[IA]);
  }

  if (dir0 == IY)
  {
    q[IB] = B0 + HALF_F * (sign_b0 * dB0d1);
  }
  else
  {
    q[IB] += HALF_F * (sign_dq0 * dq0[IB] + sign_dq1 * dq1[IB]);
  }

  if (dir1 == IY)
  {
    q[IB] = B1 + HALF_F * (sign_b1 * dB1d0);
  }
  else
  {
    q[IB] += HALF_F * (sign_dq0 * dq0[IB] + sign_dq1 * dq1[IB]);
  }

  if (dir1 == IZ)
  {
    q[IC] = B1 + HALF_F * (sign_b1 * dB1d0);
  }
  else
  {
    q[IC] += HALF_F * (sign_dq0 * dq0[IC] + sign_dq1 * dq1[IC]);
  }

  q[ID] = fmax(smallr, q[ID]);
  q[IP] = fmax(smallp * q[ID], q[IP]);

} // reconstruct_state_3d_at_edge

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndUpdateFunctor<dim, device_t>::compute_emf_and_update_2d_group(
  index_t const & cell_index,
  index_t const & iOct_local) const
{
  auto const iOct_global = m_iOct_begin + iOct_local;

  // i,j enumerate block edges
  // i,j are aligned with inner block (non-ghosted block)
  // i is in range [0, bx] ===> bx+1 values
  // j is in range [0, by] ===> by+1 values
  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_emf);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // // coordinates to access m_slopes arrays (which have a ghost width of 1)
  // auto const & is = iq2;
  // auto const & js = jq2;

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dx over dS in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a face area x2 larger than current cell face area
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;
  auto const dtdx_over_dS_cur = m_dt / dx;
  auto const dtdx_over_dS_neigh = dtdx_over_dS_cur / 2;

  // this drawing is a simple helper to remind oneself how edge reconstruction from 4 cell-center is
  // done
  //
  // symbols LB, RB, LT, RT indicate location of reconstructed edge from the cell center.
  // L,R => Left,Right
  // B,T => Bottom,Top
  //
  // the cross "X" locates an edge surrounded by 4 cells
  // hydrodynamics variables are reconstructed on the edge.
  //
  //    _________________________
  //   |           |            |
  //   |           |            |
  //   |     RB    |     LB     |
  //   |  (i-1,j)  |   (i,j)    |
  //   |          \| /          |
  //   |___________X____________|
  //   |          /| \          |
  //   |        /  |  \         |
  //   |     RT    |     LT     |
  //   | (i-1,j-1) |   (i,j-1)  |
  //   |           |            |
  //   |___________|____________|
  //

  // qLB is current cell, qLT, qRT and qRB are direct neighbors surrounding the lower left edge
  // MHDState qLB, qLT, qRB, qRT;
  MHDStateCell   qEdge_emfZ[4];
  MHDStateCell & qRT = qEdge_emfZ[IRT];
  MHDStateCell & qLT = qEdge_emfZ[ILT];
  MHDStateCell & qRB = qEdge_emfZ[IRB];
  MHDStateCell & qLB = qEdge_emfZ[ILB];

  /*
   * reconstruct states on cell edge and update
   *
   * is,js are index to access slopes ghosted array (ghost width of 1)
   */

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const is = i + 1;
  auto const js = j + 1;

  // LB at (i,j)
  reconstruct_state_2d_at_edge(is, js, iOct_local, MHDEdgeLocation::LB, qLB);


  // RT at (i-1, j-1)
  reconstruct_state_2d_at_edge(is - 1, js - 1, iOct_local, MHDEdgeLocation::RT, qRT);


  // RB at (i-1,j)
  reconstruct_state_2d_at_edge(is - 1, js, iOct_local, MHDEdgeLocation::RB, qRB);


  // LT at (i,j-1)
  reconstruct_state_2d_at_edge(is, js - 1, iOct_local, MHDEdgeLocation::LT, qLT);


  const auto emfZ = compute_emf<EMFZ>(qEdge_emfZ, m_mhd_settings);

  const auto emf_cur = emfZ * dtdx_over_dS_cur;


  const auto face_xmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_face::NEIGHBOR_IS_COARSER;

  const auto face_xmin_neighbor_is_finer =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_face::NEIGHBOR_IS_FINER;


} // compute_emf_and_update_2d_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndUpdateFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                      const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  auto const iOct_local = global_index / m_nbEdgeEmfPerLeaf;
  auto const edge_index = global_index - iOct_local * m_nbEdgeEmfPerLeaf;

  if constexpr (dim == 2)
    compute_emf_and_update_2d_group(edge_index, iOct_local);
  // else if constexpr (dim == 3)
  //   compute_emf_and_update_3d_group(edge_index, iOct_local);

} // operator () - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndUpdateFunctor<dim, device_t>::operator()(TagComputeGhostQuad const &,
                                                      const index_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost quadrant
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_nbEdgeEmfPerLeaf;
  const auto edge_index = global_index - iGhost * m_nbEdgeEmfPerLeaf;

  // if constexpr (dim == 2)
  //   compute_emf_and_update_2d_ghost(edge_index, first_ghost, iGhost);
  // else if constexpr (dim == 3)
  //   compute_emf_and_update_3d_ghost(edge_index, first_ghost, iGhost);

} // operator () - TagComputeAllQuadInGroup

// explicit template instantiation
template class ComputeEmfAndUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ComputeEmfAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
