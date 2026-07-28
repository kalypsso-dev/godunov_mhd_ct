// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeUpdatedPrimitiveVariablesFunctor.cpp
 *
 * \brief Perform half a time step (time integration) of the cell-centered primitive variables
 * using a ghosted blocked array.
 */
#include <godunov_mhd_ct/scheme/ComputeUpdatedPrimitiveVariablesFunctor.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::ComputeUpdatedPrimitiveVariablesFunctor(
  ConfigMap const &                config_map,
  orchard_key_view_t const &       orchard_keys,
  DataArrayGhostedBlock_t const &  prim_var,
  DataArrayGhostedBlock_t const &  updated_prim_var,
  DataArrayGhostedBlock_t const &  slopes_x,
  DataArrayGhostedBlock_t const &  slopes_y,
  DataArrayGhostedBlock_t const &  slopes_z,
  FieldMap<models::MHD>            fm,
  int32_t                          iOct_begin,
  int32_t                          num_octants,
  MHDSettings const &              mhd_settings,
  real_t                           dt,
  bool                             gravity_enabled,
  UniformGravityField<dim> const & gravity_field,
  ViscosityParams const &          viscosity,
  TimeIntegrator const &           time_integrator)
  : m_orchard_keys_device(orchard_keys)
  , m_q(prim_var)
  , m_q2(updated_prim_var)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field)
  , m_viscosity(viscosity)
  , m_time_integrator(time_integrator)
{}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &               config_map,
  orchard_key_view_t const &      orchard_keys,
  DataArrayGhostedBlock_t const & primitive_vars,
  DataArrayGhostedBlock_t const & updated_primitive_vars,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  FieldMap<models::MHD>           fm,
  int32_t                         num_octants,
  MHDSettings const &             mhd_settings,
  ViscosityParams const &         viscosity,
  real_t                          dt)
{

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t> functor(
    config_map,
    orchard_keys,
    primitive_vars,
    updated_primitive_vars,
    slopes_x,
    slopes_y,
    slopes_z,
    fm,
    0,           // first index to compute
    num_octants, // number of octants to process
    mhd_settings,
    dt,
    gravity_enabled,
    gravity_field,
    viscosity,
    TimeIntegratorConfig::get_time_integrator(config_map));

  const auto nbCellsPerGhostedLeaf = slopes_x.num_cells();
  const auto nbCellsTotal = num_octants * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeUpdatedPrimitiveVariablesFunctor - All quadrants",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::apply_on_ghosts(
  ConfigMap const &               config_map,
  orchard_key_view_t const &      orchard_keys,
  DataArrayGhostedBlock_t const & primitive_vars,
  DataArrayGhostedBlock_t const & updated_primitive_vars,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  FieldMap<models::MHD>           fm,
  int32_t                         num_mirrors,
  int32_t                         num_ghosts,
  MHDSettings const &             mhd_settings,
  ViscosityParams const &         viscosity,
  real_t                          dt)
{

  // we expect primitive_vars to be of size num_mirrors + num_ghosts
  KOKKOS_ASSERT(primitive_vars.num_quadrants() == num_mirrors + num_ghosts &&
                "[ComputeLimitedSlopesFunctor] primitive_vars has wrong sizes");

  // we expect slopes_x, slopes_y and slopes_z to be of size num_ghosts
  KOKKOS_ASSERT(slopes_x.num_quadrants() == num_ghosts &&
                "[ComputeLimitedSlopesFunctor] slopes_x has wrong sizes");
  KOKKOS_ASSERT(slopes_y.num_quadrants() == num_ghosts &&
                "[ComputeLimitedSlopesFunctor] slopes_y has wrong sizes");
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT(slopes_z.num_quadrants() == num_ghosts &&
                  "[ComputeLimitedSlopesFunctor] slopes_z has wrong sizes");
  }

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t> functor(
    config_map,
    orchard_keys,
    primitive_vars,
    updated_primitive_vars,
    slopes_x,
    slopes_y,
    slopes_z,
    fm,
    num_mirrors, // first index to compute
    num_ghosts,  // number of octant to compute
    mhd_settings,
    dt,
    gravity_enabled,
    gravity_field,
    viscosity,
    TimeIntegratorConfig::get_time_integrator(config_map));

  const auto nbCellsPerGhostedLeaf = slopes_x.num_cells();
  const auto nbCellsTotal = num_ghosts * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeLimitedSlopesFunctor - ghost quadrants only",
                       Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::add_viscous_predictor_2d(
  real_t &        u,
  real_t &        v,
  int32_t const & iq,
  int32_t const & jq,
  int32_t const & iOct,
  real_t const &  dtdx,
  real_t const &  dtdy,
  real_t const &  dx,
  real_t const &  dy) const
{
  // kinematic viscosity
  const auto nu = m_viscosity.mu / m_q(iq, jq, m_fm[MHD::ID], iOct);

  // IU
  const auto d2udx2 = m_q(iq + 1, jq, m_fm[MHD::IU], iOct) + m_q(iq - 1, jq, m_fm[MHD::IU], iOct) -
                      2 * m_q(iq, jq, m_fm[MHD::IU], iOct);
  u += HALF_F * nu * d2udx2 / dx * dtdx;

  const auto d2udy2 = m_q(iq, jq + 1, m_fm[MHD::IU], iOct) + m_q(iq, jq - 1, m_fm[MHD::IU], iOct) -
                      2 * m_q(iq, jq, m_fm[MHD::IU], iOct);
  u += HALF_F * nu * d2udy2 / dy * dtdy;

  // IV
  const auto d2vdx2 = m_q(iq + 1, jq, m_fm[MHD::IV], iOct) + m_q(iq - 1, jq, m_fm[MHD::IV], iOct) -
                      2 * m_q(iq, jq, m_fm[MHD::IV], iOct);
  v += HALF_F * nu * d2vdx2 / dx * dtdx;

  const auto d2vdy2 = m_q(iq, jq + 1, m_fm[MHD::IV], iOct) + m_q(iq, jq - 1, m_fm[MHD::IV], iOct) -
                      2 * m_q(iq, jq, m_fm[MHD::IV], iOct);
  v += HALF_F * nu * d2vdy2 / dy * dtdy;
} // add_viscous_predictor_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::add_viscous_predictor_3d(
  real_t &        u,
  real_t &        v,
  real_t &        w,
  int32_t const & iq,
  int32_t const & jq,
  int32_t const & kq,
  int32_t const & iOct,
  real_t const &  dtdx,
  real_t const &  dtdy,
  real_t const &  dtdz,
  real_t const &  dx,
  real_t const &  dy,
  real_t const &  dz) const
{
  // kinematic viscosity
  const auto nu = m_viscosity.mu / m_q(iq, jq, kq, m_fm[MHD::ID], iOct);

  // IU
  const auto d2udx2 = m_q(iq + 1, jq, kq, m_fm[MHD::IU], iOct) +
                      m_q(iq - 1, jq, kq, m_fm[MHD::IU], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IU], iOct);
  u += HALF_F * nu * d2udx2 / dx * dtdx;

  const auto d2udy2 = m_q(iq, jq + 1, kq, m_fm[MHD::IU], iOct) +
                      m_q(iq, jq - 1, kq, m_fm[MHD::IU], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IU], iOct);
  u += HALF_F * nu * d2udy2 / dy * dtdy;

  const auto d2udz2 = m_q(iq, jq, kq + 1, m_fm[MHD::IU], iOct) +
                      m_q(iq, jq, kq - 1, m_fm[MHD::IU], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IU], iOct);
  u += HALF_F * nu * d2udz2 / dz * dtdz;

  // IV
  const auto d2vdx2 = m_q(iq + 1, jq, kq, m_fm[MHD::IV], iOct) +
                      m_q(iq - 1, jq, kq, m_fm[MHD::IV], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IV], iOct);
  v += HALF_F * nu * d2vdx2 / dx * dtdx;

  const auto d2vdy2 = m_q(iq, jq + 1, kq, m_fm[MHD::IV], iOct) +
                      m_q(iq, jq - 1, kq, m_fm[MHD::IV], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IV], iOct);
  v += HALF_F * nu * d2vdy2 / dy * dtdy;

  const auto d2vdz2 = m_q(iq, jq, kq + 1, m_fm[MHD::IV], iOct) +
                      m_q(iq, jq, kq - 1, m_fm[MHD::IV], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IV], iOct);
  v += HALF_F * nu * d2vdz2 / dz * dtdz;

  // IW
  const auto d2wdx2 = m_q(iq + 1, jq, kq, m_fm[MHD::IW], iOct) +
                      m_q(iq - 1, jq, kq, m_fm[MHD::IW], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IW], iOct);
  w += HALF_F * nu * d2wdx2 / dx * dtdx;

  const auto d2wdy2 = m_q(iq, jq + 1, kq, m_fm[MHD::IW], iOct) +
                      m_q(iq, jq - 1, kq, m_fm[MHD::IW], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IW], iOct);
  w += HALF_F * nu * d2wdy2 / dy * dtdy;

  const auto d2wdz2 = m_q(iq, jq, kq + 1, m_fm[MHD::IW], iOct) +
                      m_q(iq, jq, kq - 1, m_fm[MHD::IW], iOct) -
                      2 * m_q(iq, jq, kq, m_fm[MHD::IW], iOct);
  w += HALF_F * nu * d2wdz2 / dz * dtdz;
} // add_viscous_predictor_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::update_primitive_variables(
  index_t const & cell_index,
  index_t const   iOct_in,
  index_t const   iOct_out) const
{
  // compute cartesian coordinates inside ghosted block
  const auto coord = cellindex_to_coord<dim>(cell_index, m_q2.ghosted_block_size(), m_q2.shift());

  auto const & gamma = m_mhd_settings.hydro.gamma0;

  // get AMR level
  auto const level = orchard_key_t<dim>::level(m_orchard_keys_device(iOct_in));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<dim>(level, m_q2.block_size()[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  if constexpr (dim == 2)
  {

    auto const & dtdx = dtdS_over_dV_cur;
    auto const & dtdy = dtdS_over_dV_cur;

    // coords to access slopes and q2
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    // coords to access q
    auto const & iq = coord[IX];
    auto const & jq = coord[IY];

    // cell centered values
    auto const r = m_q(iq, jq, MHD::ID, iOct_in);
    auto const p = m_q(iq, jq, MHD::IP, iOct_in);
    auto       u = m_q(iq, jq, MHD::IU, iOct_in);
    auto       v = m_q(iq, jq, MHD::IV, iOct_in);
    auto const w = m_q(iq, jq, MHD::IW, iOct_in);
    auto const A = (m_q(iq, jq, MHD::IAL, iOct_in) + m_q(iq, jq, MHD::IAR, iOct_in)) * HALF_F;
    auto const B = (m_q(iq, jq, MHD::IBL, iOct_in) + m_q(iq, jq, MHD::IBR, iOct_in)) * HALF_F;
    auto const C = (m_q(iq, jq, MHD::ICL, iOct_in) + m_q(iq, jq, MHD::ICR, iOct_in)) * HALF_F;

    if (m_time_integrator == +TimeIntegrator::HANCOCK)
    {
      // Cell centered TVD slopes in X direction
      auto const drx = m_slopes_x(i, j, MHD::ID, iOct_in);
      auto const dpx = m_slopes_x(i, j, MHD::IP, iOct_in);
      auto const dux = m_slopes_x(i, j, MHD::IU, iOct_in);
      auto const dvx = m_slopes_x(i, j, MHD::IV, iOct_in);
      auto const dwx = m_slopes_x(i, j, MHD::IW, iOct_in);
      auto const dAx = m_slopes_x(i, j, MHD::IA, iOct_in);
      auto const dBx = m_slopes_x(i, j, MHD::IB, iOct_in);
      auto const dCx = m_slopes_x(i, j, MHD::IC, iOct_in);

      // Cell centered TVD slopes in Y direction
      auto const dry = m_slopes_y(i, j, MHD::ID, iOct_in);
      auto const dpy = m_slopes_y(i, j, MHD::IP, iOct_in);
      auto const duy = m_slopes_y(i, j, MHD::IU, iOct_in);
      auto const dvy = m_slopes_y(i, j, MHD::IV, iOct_in);
      auto const dwy = m_slopes_y(i, j, MHD::IW, iOct_in);
      auto const dAy = m_slopes_y(i, j, MHD::IA, iOct_in);
      auto const dBy = m_slopes_y(i, j, MHD::IB, iOct_in);
      auto const dCy = m_slopes_y(i, j, MHD::IC, iOct_in);

      /*
       * compute Hancock half time-step update terms
       */
      // clang-format off
      const real_t sr0 = (-u * drx - dux * r) * dtdx +
                         (-v * dry - dvy * r) * dtdy;
      const real_t su0 = (-u * dux - dpx / r - B * dBx / r - C * dCx / r) * dtdx +
                         (-v * duy           + B * dAy / r              ) * dtdy;
      const real_t sv0 = (-u * dvx           + A * dBx / r              ) * dtdx +
                         (-v * dvy - dpy / r - A * dAy / r - C * dCy / r) * dtdy;
      const real_t sw0 = (-u * dwx + A * dCx / r) * dtdx +
                         (-v * dwy + B * dCy / r) * dtdy;
      const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx +
                         (-v * dpy - dvy * gamma * p) * dtdy;
      const real_t sA0 = (u * dBy + B * duy - v * dAy - A * dvy) * dtdy;
      const real_t sB0 = (-u * dBx - B * dux + v * dAx + A * dvx) * dtdx;
      const real_t sC0 = ( w * dAx + A * dwx - u * dCx - C * dux) * dtdx +
                         (-v * dCy - C * dvy + w * dBy + B * dwy) * dtdy;
      // clang-format on

      // add gravity time predictor, half time step (only needed when using Hancock integration)
      if (m_gravity_enabled)
      {
        u += m_gravity_field[IX] * HALF_F * m_dt;
        v += m_gravity_field[IY] * HALF_F * m_dt;
      }

      // add viscous force predictor
      if (m_viscosity.enabled and m_viscosity.hancock_predictor_enabled)
      {
        add_viscous_predictor_2d(u, v, iq, jq, iOct_in, dtdx, dtdy, dx, dx);
      }

      // Update in time the primitive variables (Hancock half time step)
      m_q2(i, j, MHD::ID, iOct_out) = r + HALF_F * sr0;
      m_q2(i, j, MHD::IU, iOct_out) = u + HALF_F * su0;
      m_q2(i, j, MHD::IV, iOct_out) = v + HALF_F * sv0;
      m_q2(i, j, MHD::IW, iOct_out) = w + HALF_F * sw0;
      m_q2(i, j, MHD::IP, iOct_out) = p + HALF_F * sp0;
      m_q2(i, j, MHD::IA, iOct_out) = A + HALF_F * sA0;
      m_q2(i, j, MHD::IB, iOct_out) = B + HALF_F * sB0;
      m_q2(i, j, MHD::IC, iOct_out) = C + HALF_F * sC0;
    } // end time Hancock

    else if (m_time_integrator == +TimeIntegrator::RK2_SSP)
    {
      // Update in time the primitive variables (Hancock half time step)
      m_q2(i, j, MHD::ID, iOct_out) = r;
      m_q2(i, j, MHD::IU, iOct_out) = u;
      m_q2(i, j, MHD::IV, iOct_out) = v;
      m_q2(i, j, MHD::IW, iOct_out) = w;
      m_q2(i, j, MHD::IP, iOct_out) = p;
      m_q2(i, j, MHD::IA, iOct_out) = A;
      m_q2(i, j, MHD::IB, iOct_out) = B;
      m_q2(i, j, MHD::IC, iOct_out) = C;
    } // end time RK2

  } // end dim = 2
  else if constexpr (dim == 3)
  {
    auto const & dtdx = dtdS_over_dV_cur;
    auto const & dtdy = dtdS_over_dV_cur;
    auto const & dtdz = dtdS_over_dV_cur;

    // coords to access slopes and q2
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    // coords to access q
    auto const & iq = coord[IX];
    auto const & jq = coord[IY];
    auto const & kq = coord[IZ];

    // cell centered values
    auto const r = m_q(iq, jq, kq, MHD::ID, iOct_in);
    auto const p = m_q(iq, jq, kq, MHD::IP, iOct_in);
    auto       u = m_q(iq, jq, kq, MHD::IU, iOct_in);
    auto       v = m_q(iq, jq, kq, MHD::IV, iOct_in);
    auto       w = m_q(iq, jq, kq, MHD::IW, iOct_in);
    auto const A =
      (m_q(iq, jq, kq, MHD::IAL, iOct_in) + m_q(iq, jq, kq, MHD::IAR, iOct_in)) * HALF_F;
    auto const B =
      (m_q(iq, jq, kq, MHD::IBL, iOct_in) + m_q(iq, jq, kq, MHD::IBR, iOct_in)) * HALF_F;
    auto const C =
      (m_q(iq, jq, kq, MHD::ICL, iOct_in) + m_q(iq, jq, kq, MHD::ICR, iOct_in)) * HALF_F;

    if (m_time_integrator == +TimeIntegrator::HANCOCK)
    {
      // Cell centered TVD slopes in X direction
      auto const drx = m_slopes_x(i, j, k, MHD::ID, iOct_in);
      auto const dpx = m_slopes_x(i, j, k, MHD::IP, iOct_in);
      auto const dux = m_slopes_x(i, j, k, MHD::IU, iOct_in);
      auto const dvx = m_slopes_x(i, j, k, MHD::IV, iOct_in);
      auto const dwx = m_slopes_x(i, j, k, MHD::IW, iOct_in);
      auto const dAx = m_slopes_x(i, j, k, MHD::IA, iOct_in);
      auto const dBx = m_slopes_x(i, j, k, MHD::IB, iOct_in);
      auto const dCx = m_slopes_x(i, j, k, MHD::IC, iOct_in);

      // Cell centered TVD slopes in Y direction
      auto const dry = m_slopes_y(i, j, k, MHD::ID, iOct_in);
      auto const dpy = m_slopes_y(i, j, k, MHD::IP, iOct_in);
      auto const duy = m_slopes_y(i, j, k, MHD::IU, iOct_in);
      auto const dvy = m_slopes_y(i, j, k, MHD::IV, iOct_in);
      auto const dwy = m_slopes_y(i, j, k, MHD::IW, iOct_in);
      auto const dAy = m_slopes_y(i, j, k, MHD::IA, iOct_in);
      auto const dBy = m_slopes_y(i, j, k, MHD::IB, iOct_in);
      auto const dCy = m_slopes_y(i, j, k, MHD::IC, iOct_in);

      // Cell centered TVD slopes in Y direction
      auto const drz = m_slopes_z(i, j, k, MHD::ID, iOct_in);
      auto const dpz = m_slopes_z(i, j, k, MHD::IP, iOct_in);
      auto const duz = m_slopes_z(i, j, k, MHD::IU, iOct_in);
      auto const dvz = m_slopes_z(i, j, k, MHD::IV, iOct_in);
      auto const dwz = m_slopes_z(i, j, k, MHD::IW, iOct_in);
      auto const dAz = m_slopes_z(i, j, k, MHD::IA, iOct_in);
      auto const dBz = m_slopes_z(i, j, k, MHD::IB, iOct_in);
      auto const dCz = m_slopes_z(i, j, k, MHD::IC, iOct_in);

      /*
       * compute Hancock half time-step update terms
       */
      // clang-format off
      const real_t sr0 = (-u * drx - dux * r) * dtdx +
                         (-v * dry - dvy * r) * dtdy +
                         (-w * drz - dwz * r) * dtdz;

      const real_t su0 = (-u * dux - (dpx + B * dBx + C * dCx) / r) * dtdx +
                         (-v * duy                  + B * dAy  / r) * dtdy +
                         (-w * duz                  + C * dAz  / r) * dtdz;

      const real_t sv0 = (-u * dvx                  + A * dBx  / r) * dtdx +
                         (-v * dvy - (dpy + A * dAy + C * dCy) / r) * dtdy +
                         (-w * dvz                  + C * dBz  / r) * dtdz;

      const real_t sw0 = (-u * dwx                  + A * dCx  / r) * dtdx +
                         (-v * dwy                  + B * dCy  / r) * dtdy +
                         (-w * dwz - (dpz + A * dAz + B * dBz) / r) * dtdz;

      const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx +
                         (-v * dpy - dvy * gamma * p) * dtdy +
                         (-w * dpz - dwz * gamma * p) * dtdz;

      const real_t sA0 = (u * dBy + B * duy - v * dAy - A * dvy) * dtdy +
                         (u * dCz + C * duz - w * dAz - A * dwz) * dtdz;

      const real_t sB0 = (v * dAx + A * dvx - u * dBx - B * dux) * dtdx +
                         (v * dCz + C * dvz - w * dBz - B * dwz) * dtdz;

      const real_t sC0 = (w * dAx + A * dwx - u * dCx - C * dux) * dtdx +
                         (w * dBy + B * dwy - v * dCy - C * dvy) * dtdy;
      // clang-format on

      // add gravity time predictor, half time step (only needed when using Hancock integration)
      if (m_gravity_enabled)
      {
        u += m_gravity_field[IX] * HALF_F * m_dt;
        v += m_gravity_field[IY] * HALF_F * m_dt;
        w += m_gravity_field[IZ] * HALF_F * m_dt;
      }

      // add viscous force predictor
      if (m_viscosity.enabled and m_viscosity.hancock_predictor_enabled)
      {
        add_viscous_predictor_3d(u, v, w, iq, jq, kq, iOct_in, dtdx, dtdy, dtdz, dx, dx, dx);
      }

      // Update in time the primitive variables (Hancock half time step)
      m_q2(i, j, k, MHD::ID, iOct_out) = r + HALF_F * sr0;
      m_q2(i, j, k, MHD::IU, iOct_out) = u + HALF_F * su0;
      m_q2(i, j, k, MHD::IV, iOct_out) = v + HALF_F * sv0;
      m_q2(i, j, k, MHD::IW, iOct_out) = w + HALF_F * sw0;
      m_q2(i, j, k, MHD::IP, iOct_out) = p + HALF_F * sp0;
      m_q2(i, j, k, MHD::IA, iOct_out) = A + HALF_F * sA0;
      m_q2(i, j, k, MHD::IB, iOct_out) = B + HALF_F * sB0;
      m_q2(i, j, k, MHD::IC, iOct_out) = C + HALF_F * sC0;

    } // end time Hancock

    else if (m_time_integrator == +TimeIntegrator::RK2_SSP)
    {
      // Update in time the primitive variables (Hancock half time step)
      m_q2(i, j, k, MHD::ID, iOct_out) = r;
      m_q2(i, j, k, MHD::IU, iOct_out) = u;
      m_q2(i, j, k, MHD::IV, iOct_out) = v;
      m_q2(i, j, k, MHD::IW, iOct_out) = w;
      m_q2(i, j, k, MHD::IP, iOct_out) = p;
      m_q2(i, j, k, MHD::IA, iOct_out) = A;
      m_q2(i, j, k, MHD::IB, iOct_out) = B;
      m_q2(i, j, k, MHD::IC, iOct_out) = C;
    }

  } // end dim == 3

} // update_primitive_variables

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::operator()(
  TagComputeAllQuadInGroup const &,
  const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  const auto iOct_local = global_index / m_q2.num_cells();
  const auto cell_index = global_index - iOct_local * m_q2.num_cells();

  update_primitive_variables(cell_index, iOct_local, iOct_local);

} // operator() - TagComputeAllQuad

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeUpdatedPrimitiveVariablesFunctor<dim, device_t>::operator()(
  TagComputeGhostQuad const &,
  const index_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_q2.num_cells();
  const auto cell_index = global_index - iGhost * m_q2.num_cells();

  update_primitive_variables(cell_index, first_ghost + iGhost, iGhost);

} // operator() - TagComputeGhostQuad

// explicit template instantiation
template class ComputeUpdatedPrimitiveVariablesFunctor<2, kalypsso::DefaultDevice>;
template class ComputeUpdatedPrimitiveVariablesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
