// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeHydroFluxesAndStoreFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeHydroFluxesAndStoreFunctor.h>

#include <godunov_mhd_ct/models/RiemannSolvers_MHD.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::ComputeHydroFluxesAndStoreFunctor(
  orchard_key_view_t const &      orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        fluxes,
  DataArrayGhostedBlock_t const & q_ghosted,
  DataArrayGhostedBlock_t const & q2_ghosted,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  DataArrayGhostedBlock_t const & sFaceMag,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_flux_offset,
  int32_t                         num_quads,
  int                             direction,
  MHDSettings const &             mhd_settings,
  real_t                          dt,
  real_t                          scaling_factor,
  TimeIntegrator const &          time_integrator)
  : m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Fluxes(fluxes)
  , m_q(q_ghosted)
  , m_q2(q2_ghosted)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_sFaceMag(sFaceMag)
  , m_fm(fm)
  , m_iOct_flux_offset(iOct_flux_offset)
  , m_num_quads(num_quads)
  , m_direction(direction)
  , m_block_sizes(slopes_x.block_size())
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(scaling_factor)
  , m_time_integrator(time_integrator)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::apply(ConfigMap const &          config_map,
                                                        orchard_key_view_t const & orchard_keys,
                                                        AMRMeshInfo const &        amr_mesh_info,
                                                        DataArrayBlock_t const &   fluxes,
                                                        DataArrayGhostedBlock_t const & q_ghosted,
                                                        DataArrayGhostedBlock_t const & q2_ghosted,
                                                        DataArrayGhostedBlock_t const & slopes_x,
                                                        DataArrayGhostedBlock_t const & slopes_y,
                                                        DataArrayGhostedBlock_t const & slopes_z,
                                                        DataArrayGhostedBlock_t const & sFaceMag,
                                                        FieldMap<core::models::MHD>     fm,
                                                        int32_t             iOct_flux_offset,
                                                        int32_t             num_quads,
                                                        int                 direction,
                                                        MHDSettings const & mhd_settings,
                                                        real_t              dt)
{
  // Important note: the caller is responsible for provide a flux array with right shape.
  {
    [[maybe_unused]] auto flux_block_sizes = q_ghosted.block_size();
    flux_block_sizes[direction]++;
    assertm(flux_block_sizes == fluxes.shape(), "Flux array has incompatible shape.");
  }

  ComputeHydroFluxesAndStoreFunctor<dim, device_t> functor(
    orchard_keys,
    amr_mesh_info,
    fluxes,
    q_ghosted,
    q2_ghosted,
    slopes_x,
    slopes_y,
    slopes_z,
    sFaceMag,
    fm,
    iOct_flux_offset,
    num_quads,
    direction,
    mhd_settings,
    dt,
    get_scaling_factor(config_map),
    TimeIntegratorConfig::get_time_integrator(config_map));

  const auto nbIterations = num_quads * fluxes.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::ComputeHydroFluxesAndStoreFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::reconstruct_state_2d(MHDStateCell const & q,
                                                                       int32_t              is,
                                                                       int32_t              js,
                                                                       int32_t     iOct_local,
                                                                       int         dir,
                                                                       face_type_t face) const
{

  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

  const auto offset = face == face_type_t::LEFT ? KALYPSSO_NUM(-0.5) : KALYPSSO_NUM(0.5);

  // get limited slopes along given direction "dir"
  const MHDStateCell dq = get_state(get_slopes(dir), is, js, iOct_local);

  // reconstruct state on face center
  MHDStateCell qr;

  qr[ID] = q[ID] + offset * dq[ID];
  qr[IP] = q[IP] + offset * dq[IP];
  qr[IU] = q[IU] + offset * dq[IU];
  qr[IV] = q[IV] + offset * dq[IV];
  qr[IW] = q[IW] + offset * dq[IW];
  qr[IA] = q[IA] + offset * dq[IA];
  qr[IB] = q[IB] + offset * dq[IB];
  qr[IC] = q[IC] + offset * dq[IC];

  qr[ID] = fmax(smallr, qr[ID]);
  qr[IP] = fmax(smallp * qr[ID], qr[IP]);

  return qr;

} // reconstruct_state_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::reconstruct_state_3d(MHDStateCell const & q,
                                                                       int32_t              is,
                                                                       int32_t              js,
                                                                       int32_t              ks,
                                                                       int32_t     iOct_local,
                                                                       int         dir,
                                                                       face_type_t face) const
{

  constexpr auto ID = MHD::ID;
  constexpr auto IP = MHD::IP;
  constexpr auto IU = MHD::IU;
  constexpr auto IV = MHD::IV;
  constexpr auto IW = MHD::IW;
  constexpr auto IA = MHD::IA;
  constexpr auto IB = MHD::IB;
  constexpr auto IC = MHD::IC;

  auto const & smallr = m_mhd_settings.hydro.smallr;
  auto const & smallp = m_mhd_settings.hydro.smallp;

  const auto offset = face == face_type_t::LEFT ? KALYPSSO_NUM(-0.5) : KALYPSSO_NUM(0.5);

  // get limited slopes along given direction "dir"
  const MHDStateCell dq = get_state(get_slopes(dir), is, js, ks, iOct_local);

  // reconstruct state on interface
  MHDStateCell qr;

  qr[ID] = q[ID] + offset * dq[ID];
  qr[IP] = q[IP] + offset * dq[IP];
  qr[IU] = q[IU] + offset * dq[IU];
  qr[IV] = q[IV] + offset * dq[IV];
  qr[IW] = q[IW] + offset * dq[IW];
  qr[IA] = q[IA] + offset * dq[IA];
  qr[IB] = q[IB] + offset * dq[IB];
  qr[IC] = q[IC] + offset * dq[IC];

  qr[ID] = fmax(smallr, qr[ID]);
  qr[IP] = fmax(smallp * qr[ID], qr[IP]);

  return qr;

} // reconstruct_state_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::compute_fluxes_and_store_2d(
  int32_t const & cell_index,
  int32_t const & iOct_local) const
{

  auto const coords = cell_index_unravel<2>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const & iq2 = i;
  auto const & jq2 = j;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 1
  auto qprim = get_state(m_q2, iq2, jq2, iOct_local);

  /*
   * compute flux from left face along X dir and update both sides
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_state(m_q2, iq2 - 1, jq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in left neighbor
    auto qL = reconstruct_state_2d(qprim_n, is - 1, js, iOct_local, IX, face_type_t::RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_2d(qprim, is, js, iOct_local, IX, face_type_t::LEFT);

    // fix normal component of B
    {
      const auto AL = m_q(i, j, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, IX, iOct_local);
      qR[MHD::IA] = AL;
      qL[MHD::IA] = AL;
    }

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);
  }

  /*
   * compute flux from left face along Y dir and update both sides
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_state(m_q2, iq2, jq2 - 1, iOct_local);

    // step 1 :
    // reconstruct state at right face in neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(qprim_n, is, js - 1, iOct_local, IY, face_type_t::RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_2d(qprim, is, js, iOct_local, IY, face_type_t::LEFT);

    // fix normal component of B
    {
      const auto BL = m_q(i, j, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, IY, iOct_local);
      qR[MHD::IB] = BL;
      qL[MHD::IB] = BL;
    }

    // swap IU / IV
    my_swap(qL[MHD::IU], qL[MHD::IV]);
    my_swap(qR[MHD::IU], qR[MHD::IV]);

    // swap IA / IB
    my_swap(qL[MHD::IA], qL[MHD::IB]);
    my_swap(qR[MHD::IA], qR[MHD::IB]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IV]);
    my_swap(flux[MHD::IA], flux[MHD::IB]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);
  }

} // compute_fluxes_and_store_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::compute_fluxes_and_store_3d(
  const int32_t & cell_index,
  const int32_t & iOct_local) const
{
  auto const coords = cell_index_unravel<3>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // coordinates to access m_q2 array (which have a ghost width of 1)
  auto const & iq2 = i;
  auto const & jq2 = j;
  auto const & kq2 = k;

  // coordinates to access m_slopes arrays (which have a ghost width of 1)
  auto const & is = iq2;
  auto const & js = jq2;
  auto const & ks = kq2;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 8 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 1
  auto qprim = get_state(m_q2, iq2, jq2, kq2, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_state(m_q2, iq2 - 1, jq2, kq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is - 1, js, ks, iOct_local, IX, face_type_t::RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IX, face_type_t::LEFT);

    // fix normal component of B
    {
      const auto AL =
        m_q(i, j, k, MHD::IAL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IX, iOct_local);
      qR[MHD::IA] = AL;
      qL[MHD::IA] = AL;
    }

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end update along X

  /*
   * compute flux from left face along Y dir
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_state(m_q2, iq2, jq2 - 1, kq2, iOct_local);

    // step 1 :
    // reconstruct state at right face in the left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is, js - 1, ks, iOct_local, IY, face_type_t::RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IY, face_type_t::LEFT);

    // fix normal component of B
    {
      const auto BL =
        m_q(i, j, k, MHD::IBL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IY, iOct_local);
      qR[MHD::IB] = BL;
      qL[MHD::IB] = BL;
    }

    // swap IU / IV
    my_swap(qL[MHD::IU], qL[MHD::IV]);
    my_swap(qR[MHD::IU], qR[MHD::IV]);

    // swap IA / IB
    my_swap(qL[MHD::IA], qL[MHD::IB]);
    my_swap(qR[MHD::IA], qR[MHD::IB]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IV]);
    my_swap(flux[MHD::IA], flux[MHD::IB]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end update along Y

  /*
   * compute flux from left face along Z dir
   */
  if (m_direction == IZ)
  {
    // get state in neighbor along Z
    auto qprim_n = get_state(m_q2, iq2, jq2, kq2 - 1, iOct_local);

    // step 1 :
    // reconstruct state at right face in the left neighbor
    auto qL = reconstruct_state_3d(qprim_n, is, js, ks - 1, iOct_local, IZ, face_type_t::RIGHT);

    // step 2 :
    // reconstruct state at left face in current cell
    auto qR = reconstruct_state_3d(qprim, is, js, ks, iOct_local, IZ, face_type_t::LEFT);

    // fix normal component of B
    {
      const auto CL =
        m_q(i, j, k, MHD::ICL, iOct_local) + m_sFaceMag(iq2, jq2, kq2, IZ, iOct_local);
      qR[MHD::IC] = CL;
      qL[MHD::IC] = CL;
    }

    // swap IU / IW
    my_swap(qL[MHD::IU], qL[MHD::IW]);
    my_swap(qR[MHD::IU], qR[MHD::IW]);

    // swap IA / IC
    my_swap(qL[MHD::IA], qL[MHD::IC]);
    my_swap(qR[MHD::IA], qR[MHD::IC]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_mhd(qL, qR, m_mhd_settings);

    my_swap(flux[MHD::IU], flux[MHD::IW]);
    my_swap(flux[MHD::IA], flux[MHD::IC]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end update along Z

} // compute_fluxes_and_store_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeHydroFluxesAndStoreFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_Fluxes.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_Fluxes.num_cells());

  if constexpr (dim == 2)
  {
    compute_fluxes_and_store_2d(cell_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    compute_fluxes_and_store_3d(cell_index, iOct_local);
  }

} // operator ()

// explicit template instantiation
template class ComputeHydroFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
template class ComputeHydroFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
