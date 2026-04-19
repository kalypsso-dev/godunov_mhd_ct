// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeEmfAndStoreFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeEmfAndStoreFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ComputeEmfAndStoreFunctor<dim, device_t>::ComputeEmfAndStoreFunctor(
  ConfigMap const &               config_map,
  orchard_key_view_t const &      orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        emf,
  DataArrayGhostedBlock_t const & q,
  DataArrayGhostedBlock_t const & q2,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  DataArrayGhostedBlock_t const & sFaceMag,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_emf_offset,
  int32_t                         num_quads,
  MHDSettings const &             mhd_settings,
  real_t                          dt,
  TimeIntegrator const &          time_integrator)
  : m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_emf(emf)
  , m_q(q)
  , m_q2(q2)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_sFaceMag(sFaceMag)
  , m_fm(fm)
  , m_iOct_emf_offset(iOct_emf_offset)
  , m_num_quads(num_quads)
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_time_integrator(time_integrator)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeEmfAndStoreFunctor<dim, device_t>::apply(ConfigMap const &               config_map,
                                                orchard_key_view_t const &      orchard_keys,
                                                AMRMeshInfo const &             amr_mesh_info,
                                                DataArrayBlock_t const &        emf,
                                                DataArrayGhostedBlock_t const & q,
                                                DataArrayGhostedBlock_t const & q2,
                                                DataArrayGhostedBlock_t const & slopes_x,
                                                DataArrayGhostedBlock_t const & slopes_y,
                                                DataArrayGhostedBlock_t const & slopes_z,
                                                DataArrayGhostedBlock_t const & sFaceMag,
                                                FieldMap<core::models::MHD>     fm,
                                                int32_t                         iOct_emf_offset,
                                                int32_t                         num_quads,
                                                MHDSettings const &             mhd_settings,
                                                real_t                          dt)
{

  ComputeEmfAndStoreFunctor<dim, device_t> functor(
    config_map,
    orchard_keys,
    amr_mesh_info,
    emf,
    q,
    q2,
    slopes_x,
    slopes_y,
    slopes_z,
    sFaceMag,
    fm,
    iOct_emf_offset, // first index to process
    num_quads,       // number of octants to process
    mhd_settings,
    dt,
    TimeIntegratorConfig::get_time_integrator(config_map));

  // we use atomic update, computations is edge-oriented
  const auto nbIterations = num_quads * emf.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::ComputeEmfAndStoreFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeEmfAndStoreFunctor<dim, device_t>::reconstruct_state_2d_at_edge(int32_t         is,
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

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

  int sign_x = 1;
  int sign_y = 1;
  int ia = is;
  int ja = js;
  int ib = is;
  int jb = js;
  int sign_a = 1;
  int sign_b = 1;

  if (edge_loc == MHDEdgeLocation::LB)
  {
    ia = is;
    ja = js;
    ib = is;
    jb = js;
    sign_x = -1;
    sign_y = -1;
    sign_a = -1;
    sign_b = -1;
  }
  else if (edge_loc == MHDEdgeLocation::RT)
  {
    ia = is + 1;
    ja = js;
    ib = is;
    jb = js + 1;
    sign_x = 1;
    sign_y = 1;
    sign_a = 1;
    sign_b = 1;
  }
  else if (edge_loc == MHDEdgeLocation::RB)
  {
    ia = is + 1;
    ja = js;
    ib = is;
    jb = js;
    sign_x = 1;
    sign_y = -1;
    sign_a = -1;
    sign_b = 1;
  }
  else if (edge_loc == MHDEdgeLocation::LT)
  {
    ia = is;
    ja = js;
    ib = is;
    jb = js + 1;
    sign_x = -1;
    sign_y = 1;
    sign_a = 1;
    sign_b = -1;
  }

  // just to be clear here, A and B are face variables (magnetic field components)
  // updated in time (half time-step)
  real_t A = m_q(ia, ja, MHD::IAL, iOct_local);
  real_t B = m_q(ib, jb, MHD::IBL, iOct_local);

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    A += m_sFaceMag(ia, ja, IX, iOct_local);
    B += m_sFaceMag(ib, jb, IY, iOct_local);
  }

  const real_t dAy = compute_limited_slope(ia, ja, IA, iOct_local, IY);
  const real_t dBx = compute_limited_slope(ib, jb, IB, iOct_local, IX);

  // get limited slopes along given direction "dir"
  const MHDStateCell dqX = get_state(get_slopes(IX), is, js, iOct_local);
  const MHDStateCell dqY = get_state(get_slopes(IY), is, js, iOct_local);

  // reconstruct state on edge center

  q[ID] += HALF_F * (static_cast<real_t>(sign_x) * dqX[ID] + static_cast<real_t>(sign_y) * dqY[ID]);
  q[IP] += HALF_F * (static_cast<real_t>(sign_x) * dqX[IP] + static_cast<real_t>(sign_y) * dqY[IP]);
  q[IU] += HALF_F * (static_cast<real_t>(sign_x) * dqX[IU] + static_cast<real_t>(sign_y) * dqY[IU]);
  q[IV] += HALF_F * (static_cast<real_t>(sign_x) * dqX[IV] + static_cast<real_t>(sign_y) * dqY[IV]);
  q[IW] += HALF_F * (static_cast<real_t>(sign_x) * dqX[IW] + static_cast<real_t>(sign_y) * dqY[IW]);

  q[IA] = A + HALF_F * (static_cast<real_t>(sign_a) * dAy);
  q[IB] = B + HALF_F * (static_cast<real_t>(sign_b) * dBx);
  q[IC] += HALF_F * (static_cast<real_t>(sign_x) * dqX[IC] + static_cast<real_t>(sign_y) * dqY[IC]);

  q[ID] = fmax(smallr, q[ID]);
  q[IP] = fmax(smallp * q[ID], q[IP]);

  return q;

} // reconstruct_state_2d_at_edge

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeEmfAndStoreFunctor<dim, device_t>::reconstruct_state_3d_at_edge(int32_t         is,
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

  // get current location cell-centered primitive variables state at time t_{n+1/2}
  // note: primitive variables is a ghosted array with ghost width of 1
  get_state(m_q2, is, js, ks, iOct_local, q);

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

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
    ijk0[static_cast<size_t>(dir0)] += 1;
    ijk1[static_cast<size_t>(dir1)] += 1;
    sign_dq0 = 1;
    sign_dq1 = 1;
    sign_b0 = 1;
    sign_b1 = 1;
  }
  else if (edge_loc == MHDEdgeLocation::RB)
  {
    ijk0[static_cast<size_t>(dir0)] += 1;
    sign_dq0 = 1;
    sign_dq1 = -1;
    sign_b0 = -1;
    sign_b1 = 1;
  }
  else if (edge_loc == MHDEdgeLocation::LT)
  {
    ijk1[static_cast<size_t>(dir1)] += 1;
    sign_dq0 = -1;
    sign_dq1 = 1;
    sign_b0 = 1;
    sign_b1 = -1;
  }

  // just to be clear here, B0 and B1 are face variables (magnetic field components) updated in time
  // (half time-step)
  real_t B0 = m_q(ijk0[IX], ijk0[IY], ijk0[IZ], IA + dir0, iOct_local);
  real_t B1 = m_q(ijk1[IX], ijk1[IY], ijk1[IZ], IA + dir1, iOct_local);

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    B0 += m_sFaceMag(ijk0[IX], ijk0[IY], ijk0[IZ], dir0, iOct_local);
    B1 += m_sFaceMag(ijk1[IX], ijk1[IY], ijk1[IZ], dir1, iOct_local);
  }

  const real_t dB0d1 =
    compute_limited_slope(ijk0[IX], ijk0[IY], ijk0[IZ], IA + dir0, iOct_local, dir1);
  const real_t dB1d0 =
    compute_limited_slope(ijk1[IX], ijk1[IY], ijk1[IZ], IA + dir1, iOct_local, dir0);


  // get limited slopes along in direction 0
  MHDStateCell dq0 = get_state(get_slopes(dir0), is, js, ks, iOct_local);
  MHDStateCell dq1 = get_state(get_slopes(dir1), is, js, ks, iOct_local);

  q[ID] +=
    HALF_F * (static_cast<real_t>(sign_dq0) * dq0[ID] + static_cast<real_t>(sign_dq1) * dq1[ID]);
  q[IP] +=
    HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IP] + static_cast<real_t>(sign_dq1) * dq1[IP]);
  q[IU] +=
    HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IU] + static_cast<real_t>(sign_dq1) * dq1[IU]);
  q[IV] +=
    HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IV] + static_cast<real_t>(sign_dq1) * dq1[IV]);
  q[IW] +=
    HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IW] + static_cast<real_t>(sign_dq1) * dq1[IW]);

  if (dir0 == IX)
  {
    q[IA] = B0 + HALF_F * (static_cast<real_t>(sign_b0) * dB0d1);
  }
  else
  {
    q[IA] +=
      HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IA] + static_cast<real_t>(sign_dq1) * dq1[IA]);
  }

  if (dir0 == IY)
  {
    q[IB] = B0 + HALF_F * (static_cast<real_t>(sign_b0) * dB0d1);
  }
  else
  {
    q[IB] +=
      HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IB] + static_cast<real_t>(sign_dq1) * dq1[IB]);
  }

  if (dir1 == IY)
  {
    q[IB] = B1 + HALF_F * (static_cast<real_t>(sign_b1) * dB1d0);
  }
  else
  {
    q[IB] +=
      HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IB] + static_cast<real_t>(sign_dq1) * dq1[IB]);
  }

  if (dir1 == IZ)
  {
    q[IC] = B1 + HALF_F * (static_cast<real_t>(sign_b1) * dB1d0);
  }
  else
  {
    q[IC] +=
      HALF_F * (static_cast<real_t>(sign_dq0) * dq0[IC] + static_cast<real_t>(sign_dq1) * dq1[IC]);
  }

  q[ID] = fmax(smallr, q[ID]);
  q[IP] = fmax(smallp * q[ID], q[IP]);

} // reconstruct_state_3d_at_edge

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndStoreFunctor<dim, device_t>::compute_emf_and_store_2d(index_t const & cell_index,
                                                                   index_t const & iOct_local) const
{
  auto const iOct_global = m_iOct_emf_offset + iOct_local;

  // i,j enumerate block edges
  // i,j are aligned with inner block (non-ghosted block)
  // i is in range [0, bx] ===> bx+1 values
  // j is in range [0, by] ===> by+1 values
  auto const   coords = cellindex_to_coord<2>(cell_index, m_emf.block_size());
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dx over dS in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a face area x2 larger than current cell face area
  auto const dx = compute_cell_length<2>(level, m_q.block_size()[IX]) * m_scaling_factor;
  auto const dtdx_over_dS_cur = m_dt / dx;

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
   * reconstruct states on cell edge and store
   *
   * is,js are index to access slopes ghosted array (ghost width of 1)
   */

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const & is = i;
  auto const & js = j;

  // clang-format off

  // LB at (i,j)
  reconstruct_state_2d_at_edge(is    , js    , iOct_local, MHDEdgeLocation::LB, qLB);


  // RT at (i-1, j-1)
  reconstruct_state_2d_at_edge(is - 1, js - 1, iOct_local, MHDEdgeLocation::RT, qRT);


  // RB at (i-1,j)
  reconstruct_state_2d_at_edge(is - 1, js    , iOct_local, MHDEdgeLocation::RB, qRB);


  // LT at (i,j-1)
  reconstruct_state_2d_at_edge(is    , js - 1, iOct_local, MHDEdgeLocation::LT, qLT);

  // clang-format on

  const auto emfZ = compute_emf<EMFZ>(qEdge_emfZ, m_mhd_settings);

  m_emf(i, j, ALONG_Z, iOct_global) = emfZ * dtdx_over_dS_cur;

} // compute_emf_and_store_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndStoreFunctor<dim, device_t>::compute_emf_and_store_3d(index_t const & cell_index,
                                                                   index_t const & iOct_local) const
{
  auto const iOct_global = m_iOct_emf_offset + iOct_local;

  // i,j,k enumerate block edges
  // i,j,k are aligned with inner block (non-ghosted block)
  // i is in range [0, bx] ===> bx+1 values
  // j is in range [0, by] ===> by+1 values
  // k is in range [0, bz] ===> bz+1 values
  auto const   coords = cellindex_to_coord<3>(cell_index, m_emf.block_size());
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dx over dS in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a face area x2 larger than current cell face area
  auto const dx = compute_cell_length<3>(level, m_q.block_size()[IX]) * m_scaling_factor;
  auto const dtdx_over_dS_cur = m_dt / dx;

  // qLB is current cell, qLT, qRT and qRB are direct neighbors surrounding the lower left edge
  // MHDState qLB, qLT, qRB, qRT;
  MHDStateCell   qEdge_emf[4];
  MHDStateCell & qRT = qEdge_emf[IRT];
  MHDStateCell & qLT = qEdge_emf[ILT];
  MHDStateCell & qRB = qEdge_emf[IRB];
  MHDStateCell & qLB = qEdge_emf[ILB];

  /*
   * reconstruct states on cell edge and store
   *
   * is,js,ks are index to access slopes ghosted array (ghost width of 1)
   */

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const & is = i;
  auto const & js = j;
  auto const & ks = k;

  // clang-format off

  // along Z
  if (k<m_emf.block_size()[IZ]-1)
  {
    // LB at (i,j)
    reconstruct_state_3d_at_edge(is    , js    , ks, iOct_local, MHDEdgeLocation::LB, IX, IY, qLB);

    // RT (i-1, j-1, k)
    reconstruct_state_3d_at_edge(is - 1, js - 1, ks, iOct_local, MHDEdgeLocation::RT, IX, IY, qRT);

    // RB (i-1, j, k)
    reconstruct_state_3d_at_edge(is - 1, js    , ks, iOct_local, MHDEdgeLocation::RB, IX, IY, qRB);

    // LT (i, j-1, k)
    reconstruct_state_3d_at_edge(is    , js - 1, ks, iOct_local, MHDEdgeLocation::LT, IX, IY, qLT);

    const real_t emfZ = compute_emf<EMFZ>(qEdge_emf, m_mhd_settings);

    m_emf(i, j, k, IZ, iOct_global) = emfZ * dtdx_over_dS_cur;

  }

  // along Y
  if (j<m_emf.block_size()[IY]-1)
  {
    // LB at (i,j)
    reconstruct_state_3d_at_edge(is    , js, ks    , iOct_local, MHDEdgeLocation::LB, IX, IZ, qLB);

    // RT (i-1, j, k-1)
    reconstruct_state_3d_at_edge(is - 1, js, ks - 1, iOct_local, MHDEdgeLocation::RT, IX, IZ, qRT);

    // RB (i-1, j, k)
    reconstruct_state_3d_at_edge(is - 1, js, ks    , iOct_local, MHDEdgeLocation::RB, IX, IZ, qRB);

    // LT (i, j, k-1)
    reconstruct_state_3d_at_edge(is    , js, ks - 1, iOct_local, MHDEdgeLocation::LT, IX, IZ, qLT);

    // exchange RB and LT
    {
      my_swap(qRB[MHD::ID], qLT[MHD::ID]);
      my_swap(qRB[MHD::IU], qLT[MHD::IU]);
      my_swap(qRB[MHD::IV], qLT[MHD::IV]);
      my_swap(qRB[MHD::IW], qLT[MHD::IW]);
      my_swap(qRB[MHD::IP], qLT[MHD::IP]);
      my_swap(qRB[MHD::IA], qLT[MHD::IA]);
      my_swap(qRB[MHD::IB], qLT[MHD::IB]);
      my_swap(qRB[MHD::IC], qLT[MHD::IC]);
    }
    const real_t emfY = compute_emf<EMFY>(qEdge_emf, m_mhd_settings);

    m_emf(i, j, k, IY, iOct_global) = emfY * dtdx_over_dS_cur;

  }

  // along X
  if (i<m_emf.block_size()[IX]-1)
  {
    // LB at (i,j)
    reconstruct_state_3d_at_edge(is, js    , ks    , iOct_local, MHDEdgeLocation::LB, IY, IZ, qLB);

    // RT (i, j-1, k-1)
    reconstruct_state_3d_at_edge(is, js - 1, ks - 1, iOct_local, MHDEdgeLocation::RT, IY, IZ, qRT);

    // RB (i, j-1, k)
    reconstruct_state_3d_at_edge(is, js - 1, ks    , iOct_local, MHDEdgeLocation::RB, IY, IZ, qRB);

    // LT (i, j, k-1)
    reconstruct_state_3d_at_edge(is, js    , ks - 1, iOct_local, MHDEdgeLocation::LT, IY, IZ, qLT);

    const real_t emfX = compute_emf<EMFX>(qEdge_emf, m_mhd_settings);

    m_emf(i, j, k, IX, iOct_global) = emfX * dtdx_over_dS_cur;

  }

  // clang-format on

} // compute_emf_and_store_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeEmfAndStoreFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  auto const iOct_local = global_index / m_emf.num_cells();
  auto const edge_index = global_index - iOct_local * m_emf.num_cells();

  if constexpr (dim == 2)
  {
    compute_emf_and_store_2d(edge_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    compute_emf_and_store_3d(edge_index, iOct_local);
  }

} // operator ()

// explicit template instantiation
template class ComputeEmfAndStoreFunctor<2, kalypsso::DefaultDevice>;
template class ComputeEmfAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
