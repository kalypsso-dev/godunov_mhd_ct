// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeViscousFluxesAndStoreFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeViscousFluxesAndStoreFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::ComputeViscousFluxesAndStoreFunctor(
  orchard_key_view_t const &      orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        fluxes,
  DataArrayGhostedBlock_t const & q_ghosted,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_flux_offset,
  int32_t                         num_quads,
  int                             direction,
  ViscosityParams const &         viscosity,
  real_t                          dt,
  real_t                          scaling_factor)
  : m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Fluxes(fluxes)
  , m_q(q_ghosted)
  , m_fm(fm)
  , m_iOct_flux_offset(iOct_flux_offset)
  , m_num_quads(num_quads)
  , m_direction(direction)
  , m_block_sizes(q_ghosted.block_size())
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_viscosity(viscosity)
  , m_dt(dt)
  , m_scaling_factor(scaling_factor)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::apply(ConfigMap const &          config_map,
                                                          orchard_key_view_t const & orchard_keys,
                                                          AMRMeshInfo const &        amr_mesh_info,
                                                          DataArrayBlock_t const &   fluxes,
                                                          DataArrayGhostedBlock_t const & q_ghosted,
                                                          FieldMap<core::models::MHD>     fm,
                                                          int32_t                 iOct_flux_offset,
                                                          int32_t                 num_quads,
                                                          int                     direction,
                                                          ViscosityParams const & viscosity,
                                                          real_t                  dt)
{
  // Important note: the caller is responsible for provide a flux array with right shape.
  {
    [[maybe_unused]] auto flux_block_sizes = q_ghosted.block_size();
    flux_block_sizes[static_cast<size_t>(direction)]++;
    assertm(flux_block_sizes == fluxes.shape(), "Flux array has incompatible shape.");
  }

  ComputeViscousFluxesAndStoreFunctor<dim, device_t> functor(orchard_keys,
                                                             amr_mesh_info,
                                                             fluxes,
                                                             q_ghosted,
                                                             fm,
                                                             iOct_flux_offset,
                                                             num_quads,
                                                             direction,
                                                             viscosity,
                                                             dt,
                                                             get_scaling_factor(config_map));

  const auto nbIterations = num_quads * fluxes.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::ComputeViscousFluxesAndStoreFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::compute_velocity_gradient_2d(int32_t i,
                                                                                 int32_t j,
                                                                                 int32_t iOct_local,
                                                                                 real_t  dx) const
{
  GradTensor<dim * dim> g;

  auto qL = get_velocity(i - 1, j, iOct_local);
  auto qR = get_velocity(i, j, iOct_local);

  // finite difference gradient
  g[Grad::IUX] = (qR[IX] - qL[IX]) / dx;
  g[Grad::IVX] = (qR[IY] - qL[IY]) / dx;

  qL = get_velocity(i, j - 1, iOct_local);

  // finite difference gradient
  g[Grad::IUY] = (qR[IX] - qL[IX]) / dx;
  g[Grad::IVY] = (qR[IY] - qL[IY]) / dx;

  return g;

} // compute_velocity_gradient_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::compute_velocity_gradient_3d(int32_t i,
                                                                                 int32_t j,
                                                                                 int32_t k,
                                                                                 int32_t iOct_local,
                                                                                 real_t  dx) const
{
  GradTensor<dim * dim> g;

  auto qL = get_velocity(i - 1, j, k, iOct_local);
  auto qR = get_velocity(i, j, k, iOct_local);

  g[Grad::IUX] = (qR[IX] - qL[IX]) / dx;
  g[Grad::IVX] = (qR[IY] - qL[IY]) / dx;
  g[Grad::IWX] = (qR[IZ] - qL[IZ]) / dx;

  qL = get_velocity(i, j - 1, k, iOct_local);

  // finite difference gradient
  g[Grad::IUY] = (qR[IX] - qL[IX]) / dx;
  g[Grad::IVY] = (qR[IY] - qL[IY]) / dx;
  g[Grad::IWY] = (qR[IZ] - qL[IZ]) / dx;

  qL = get_velocity(i, j, k - 1, iOct_local);

  // finite difference gradient
  g[Grad::IUZ] = (qR[IX] - qL[IX]) / dx;
  g[Grad::IVZ] = (qR[IY] - qL[IY]) / dx;
  g[Grad::IWZ] = (qR[IZ] - qL[IZ]) / dx;

  return g;

} // compute_velocity_gradient_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::compute_viscous_fluxes_and_store_2d(
  int32_t const & cell_index,
  int32_t const & iOct_local) const
{
  auto const & mu = m_viscosity.mu;

  auto const coords = cell_index_unravel<2>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];

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
  auto velocity = get_velocity(i, j, iOct_local);

  GradTensor<dim * dim> g = compute_velocity_gradient_2d(i, j, iOct_local, dx);

  MHDStateCell flux;

  flux[MHD::ID] = ZERO_F;
  flux[MHD::IA] = ZERO_F;
  flux[MHD::IB] = ZERO_F;
  flux[MHD::IC] = ZERO_F;

  /*
   * compute flux from left face along X dir and update both sides
   */
  if (m_direction == IX)
  {
    real_t tau_xx = 2 * mu * (g[Grad::IUX] - HALF_F * (g[Grad::IUX] + g[Grad::IVY]));
    real_t tau_xy = mu * (g[Grad::IVX] + g[Grad::IUY]);

    flux[MHD::IU] = -tau_xx;
    flux[MHD::IV] = -tau_xy;
    flux[MHD::IE] = -velocity[IX] * tau_xx - velocity[IY] * tau_xy;

    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);

  } // end direction IX

  /*
   * compute flux from left face along Y dir and update both sides
   */
  if (m_direction == IY)
  {
    real_t tau_yy = 2 * mu * (g[Grad::IVY] - HALF_F * (g[Grad::IUX] + g[Grad::IVY]));
    real_t tau_xy = mu * (g[Grad::IVX] + g[Grad::IUY]);

    flux[MHD::IU] = -tau_xy;
    flux[MHD::IV] = -tau_yy;
    flux[MHD::IE] = -velocity[IX] * tau_xy - velocity[IY] * tau_yy;

    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);
  } // end direction IY

} // compute_viscous_fluxes_and_store_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::compute_viscous_fluxes_and_store_3d(
  const int32_t & cell_index,
  const int32_t & iOct_local) const
{
  auto const & mu = m_viscosity.mu;

  auto const coords = cell_index_unravel<3>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

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
  auto velocity = get_velocity(i, j, k, iOct_local);

  GradTensor<dim * dim> g = compute_velocity_gradient_3d(i, j, k, iOct_local, dx);

  MHDStateCell flux;

  flux[MHD::ID] = ZERO_F;
  flux[MHD::IA] = ZERO_F;
  flux[MHD::IB] = ZERO_F;
  flux[MHD::IC] = ZERO_F;

  /*
   * compute flux from left face along X dir
   */
  if (m_direction == IX)
  {
    real_t tau_xx =
      2 * mu * (g[Grad::IUX] - ONE_THIRD_F * (g[Grad::IUX] + g[Grad::IVY] + g[Grad::IWZ]));
    real_t tau_yx = mu * (g[Grad::IVX] + g[Grad::IUY]);
    real_t tau_zx = mu * (g[Grad::IWX] + g[Grad::IUZ]);

    flux[MHD::IU] = -tau_xx;
    flux[MHD::IV] = -tau_yx;
    flux[MHD::IW] = -tau_zx;
    flux[MHD::IE] = -velocity[IX] * tau_xx - velocity[IY] * tau_yx - velocity[IZ] * tau_zx;

    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end direction IX

  /*
   * compute flux from left face along Y dir
   */
  if (m_direction == IY)
  {
    real_t tau_xy = mu * (g[Grad::IUY] + g[Grad::IVX]);
    real_t tau_yy =
      2 * mu * (g[Grad::IVY] - ONE_THIRD_F * (g[Grad::IUX] + g[Grad::IVY] + g[Grad::IWZ]));
    real_t tau_zy = mu * (g[Grad::IWY] + g[Grad::IVZ]);

    flux[MHD::IU] = -tau_xy;
    flux[MHD::IV] = -tau_yy;
    flux[MHD::IW] = -tau_zy;
    flux[MHD::IE] = -velocity[IX] * tau_xy - velocity[IY] * tau_yy - velocity[IZ] * tau_zy;

    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end direction IY

  /*
   * compute flux from left face along Z dir
   */
  if (m_direction == IZ)
  {

    real_t tau_xz = mu * (g[Grad::IUZ] + g[Grad::IWX]);
    real_t tau_yz = mu * (g[Grad::IVZ] + g[Grad::IWY]);
    real_t tau_zz =
      2 * mu * (g[Grad::IWZ] - ONE_THIRD_F * (g[Grad::IUX] + g[Grad::IVY] + g[Grad::IWZ]));

    flux[MHD::IU] = -tau_xz;
    flux[MHD::IV] = -tau_yz;
    flux[MHD::IW] = -tau_zz;
    flux[MHD::IE] = -velocity[IX] * tau_xz - velocity[IY] * tau_yz - velocity[IZ] * tau_zz;

    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);

  } // end direction IZ

} // compute_viscous_fluxes_and_store_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeViscousFluxesAndStoreFunctor<dim, device_t>::operator()(const index_t & global_index) const
{
  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_Fluxes.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_Fluxes.num_cells());

  if constexpr (dim == 2)
  {
    compute_viscous_fluxes_and_store_2d(cell_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    compute_viscous_fluxes_and_store_3d(cell_index, iOct_local);
  }

} // operator()

// explicit template instantiation
template class ComputeViscousFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
template class ComputeViscousFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
