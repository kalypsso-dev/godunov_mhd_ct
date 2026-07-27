// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDtMHDFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ComputeDtMHDFunctor.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeDtMHDFunctor<dim, device_t>::ComputeDtMHDFunctor(
  ConfigMap const &                config_map,
  orchard_key_view_t const &       orchard_keys,
  int32_t                          local_num_octants,
  MHDSettings const &              mhd_settings,
  FieldMap<models::MHD>            fm,
  block_size_t<dim> const &        block_sizes,
  DataArrayBlock_t const &         Udata,
  FaceDataArrayBlock_t const &     Bface,
  bool                             gravity_enabled,
  UniformGravityField<dim> const & gravity_field)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_mhd_settings(mhd_settings)
  , m_viscosity_params(config_map)
  , m_fm(fm)
  , m_block_sizes(block_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_Udata(Udata)
  , m_Bface(Bface)
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeDtMHDFunctor<dim, device_t>::apply(ConfigMap const &            config_map,
                                          orchard_key_view_t const &   orchard_keys,
                                          int32_t                      local_num_octants,
                                          MHDSettings const &          mhd_settings,
                                          FieldMap<models::MHD>        fm,
                                          block_size_t<dim> const &    block_sizes,
                                          DataArrayBlock_t const &     Udata,
                                          FaceDataArrayBlock_t const & Bface,
                                          real_t &                     invDt)
{
  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeDtMHDFunctor functor(config_map,
                              orchard_keys,
                              local_num_octants,
                              mhd_settings,
                              fm,
                              block_sizes,
                              Udata,
                              Bface,
                              gravity_enabled,
                              gravity_field);

  Kokkos::Max<real_t> reducer(invDt);

  const auto nbCellsPerLeaf = Udata.num_cells();

  // compute total number of cells
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_reduce("kalypsso::godunov_mhd_ct::ComputeDtMHDFunctor",
                          Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                          functor,
                          reducer);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtMHDFunctor<dim, device_t>::compute_cfl(int32_t const & iOct,
                                                int32_t const & cell_index,
                                                real_t &        invDt) const
{
  MHDStateCell            uLoc;       // cell-centered conservative variables in current cell
  [[maybe_unused]] real_t c = ZERO_F; // speed of sound

  // get block level
  const auto level = orchard_key_t<dim>::level(m_orchard_keys(iOct));

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  const auto dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;

  // compute i,j,k coordinates of current cell inside block
  const auto iCoord = cellindex_to_coord<dim>(cell_index, m_block_sizes);

  // get conservative variable in current cell
  uLoc[MHD::ID] = m_Udata(cell_index, m_fm[MHD::ID], iOct);
  uLoc[MHD::IP] = m_Udata(cell_index, m_fm[MHD::IP], iOct);
  uLoc[MHD::IU] = m_Udata(cell_index, m_fm[MHD::IU], iOct);
  uLoc[MHD::IV] = m_Udata(cell_index, m_fm[MHD::IV], iOct);
  uLoc[MHD::IW] = m_Udata(cell_index, m_fm[MHD::IW], iOct);

  if constexpr (dim == 2)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    uLoc[MHD::IA] = HALF_F * (m_Bface(i, j, IX, iOct) + m_Bface(i + 1, j, IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface(i, j, IY, iOct) + m_Bface(i, j + 1, IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface(i, j, IZ, iOct) + m_Bface(i, j, IZ, iOct));
  }
  else if constexpr (dim == 3)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    auto const & k = iCoord[IZ];
    uLoc[MHD::IA] = HALF_F * (m_Bface(i, j, k, IX, iOct) + m_Bface(i + 1, j, k, IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface(i, j, k, IY, iOct) + m_Bface(i, j + 1, k, IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface(i, j, k, IZ, iOct) + m_Bface(i, j, k + 1, IZ, iOct));
  }

  // compute primitive variables and speed of sound in current cell
  const auto qLoc = models::mhd::compute_primitives(uLoc, m_mhd_settings);

  // compute fastest information speeds
  const auto v = models::mhd::find_speed_info<dim>(qLoc, m_mhd_settings);

  // update cfl
  if constexpr (dim == 2)
  {
    invDt = fmax(invDt, v[IX] / dx + v[IY] / dx);
  }
  else if constexpr (dim == 3)
  {
    invDt = fmax(invDt, v[IX] / dx + v[IY] / dx + v[IZ] / dx);
  }

  if (m_viscosity_params.enabled)
  {
    const auto nu = m_viscosity_params.mu / uLoc[MHD::ID];
    invDt = fmax(invDt, 4 * nu / (dx * dx));
  }

} // compute_cfl

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtMHDFunctor<dim, device_t>::compute_cfl_with_gravity(int32_t const & iOct,
                                                             int32_t const & cell_index,
                                                             real_t &        invDt) const
{
  MHDStateCell            uLoc;       // cell-centered conservative variables in current cell
  [[maybe_unused]] real_t c = ZERO_F; // speed of sound

  // get block level
  const auto level = orchard_key_t<dim>::level(m_orchard_keys(iOct));

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  const auto dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;

  // compute i,j,k coordinates of current cell inside block
  const auto iCoord = cellindex_to_coord<dim>(cell_index, m_block_sizes);

  // get conservative variable in current cell
  uLoc[MHD::ID] = m_Udata(cell_index, m_fm[MHD::ID], iOct);
  uLoc[MHD::IP] = m_Udata(cell_index, m_fm[MHD::IP], iOct);
  uLoc[MHD::IU] = m_Udata(cell_index, m_fm[MHD::IU], iOct);
  uLoc[MHD::IV] = m_Udata(cell_index, m_fm[MHD::IV], iOct);
  uLoc[MHD::IW] = m_Udata(cell_index, m_fm[MHD::IW], iOct);

  if constexpr (dim == 2)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    uLoc[MHD::IA] = HALF_F * (m_Bface(i, j, IX, iOct) + m_Bface(i + 1, j, IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface(i, j, IY, iOct) + m_Bface(i, j + 1, IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface(i, j, IZ, iOct) + m_Bface(i, j, IZ, iOct));
  }
  else if constexpr (dim == 3)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    auto const & k = iCoord[IZ];
    uLoc[MHD::IA] = HALF_F * (m_Bface(i, j, k, IX, iOct) + m_Bface(i + 1, j, k, IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface(i, j, k, IY, iOct) + m_Bface(i, j + 1, k, IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface(i, j, k, IZ, iOct) + m_Bface(i, j, k + 1, IZ, iOct));
  }

  // compute primitive variables and speed of sound in current cell
  const auto qLoc = models::mhd::compute_primitives(uLoc, m_mhd_settings);

  // compute fastest information speeds
  const auto v = models::mhd::find_speed_info<dim>(qLoc, m_mhd_settings);

  // compute square
  real_t v2 = v[IX] * v[IX] + v[IY] * v[IY];
  if constexpr (dim == 3)
  {
    v2 += v[IZ] * v[IZ];
  }

  /* Due to the gravitational acceleration, the CFL condition
   * can be written as
   * g dt^2 / (2 dx) + u dt / dx <= cfl
   * where u = |v| and g = sum(|g_i|)
   *
   * u / dx has to be corrected by a factor k / (sqrt(1 + 2k) - 1)
   * in order to satisfy the new CFL, where k = g dx cfl / u^2
   */
  real_t k = fabs(m_gravity_field[IX]) + fabs(m_gravity_field[IY]);
  if constexpr (dim == 3)
  {
    k += fabs(m_gravity_field[IZ]);
  }

  k *= dx / v2;

  // prevent numerical errors due to very low gravity
  k = fmax(k, KALYPSSO_NUM(1e-4));

  real_t factor = k / (sqrt(KALYPSSO_NUM(1.0) + KALYPSSO_NUM(2.0) * k) - KALYPSSO_NUM(1.0));

  real_t invDt_local = ZERO_F;
  invDt_local += (v[IX] * factor) / dx;
  invDt_local += (v[IY] * factor) / dx;
  if constexpr (dim == 3)
  {
    invDt_local += (v[IZ] * factor) / dx;
  }

  // update cfl
  invDt = fmax(invDt, invDt_local);

  if (m_viscosity_params.enabled)
  {
    const auto nu = m_viscosity_params.mu / uLoc[MHD::ID];
    invDt = fmax(invDt, 4 * nu / (dx * dx));
  }

} // compute_cfl_with_gravity

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtMHDFunctor<dim, device_t>::operator()(const index_t & global_index, real_t & invDt) const
{
  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  // compute cfl in current cell and update invDt
  if (m_gravity_enabled)
  {
    compute_cfl_with_gravity(iOct, cell_index, invDt);
  }
  else
  {
    compute_cfl(iOct, cell_index, invDt);
  }
} // operator()

// explicit template instantiation
template class ComputeDtMHDFunctor<2, kalypsso::DefaultDevice>;
template class ComputeDtMHDFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
