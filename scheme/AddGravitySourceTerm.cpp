// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AddGravitySourceTerm.cpp
 * \brief \copybrief AddGravitySourceTerm.h
 */

#include <godunov_mhd_ct/scheme/AddGravitySourceTerm.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
AddGravitySourceTerm<dim, device_t>::apply(ConfigMap const &        config_map,
                                           DataArrayBlock_t const & Uold,
                                           DataArrayBlock_t const & Unew,
                                           FieldMap<models::MHD>    fm,
                                           int32_t                  local_num_octants,
                                           real_t                   dt)
{

  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  AddGravitySourceTerm functor(gravity_field, Uold, Unew, fm, dt);

  // compute total number of cells
  const auto nbCellsPerLeaf = Uold.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::AddGravitySourceTerm",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);
} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
AddGravitySourceTerm<dim, device_t>::operator()(const index_t & global_index) const
{
  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  const auto rho_old = m_Uold(cell_index, m_fm[MHD::ID], iOct);
  const auto rho_new = m_Unew(cell_index, m_fm[MHD::ID], iOct);

  // read momentum before gravity update
  auto rhou = m_Unew(cell_index, m_fm[MHD::IU], iOct);
  auto rhov = m_Unew(cell_index, m_fm[MHD::IV], iOct);
  auto rhow = ZERO_F;
  if constexpr (dim == 3)
  {
    rhow = m_Unew(cell_index, m_fm[MHD::IW], iOct);
  }

  // compute kinetic energy before updating momentum
  auto ekin_old = HALF_F * (rhou * rhou + rhov * rhov) / rho_new;
  if constexpr (dim == 3)
  {
    ekin_old += HALF_F * (rhow * rhow) / rho_new;
  }

  // update momentum
  rhou += m_dt * m_grav[IX] * (rho_old + rho_new) / 2;
  rhov += m_dt * m_grav[IY] * (rho_old + rho_new) / 2;
  if constexpr (dim == 3)
  {
    rhow += m_dt * m_grav[IZ] * (rho_old + rho_new) / 2;
  }

  m_Unew(cell_index, m_fm[MHD::IU], iOct) = rhou;
  m_Unew(cell_index, m_fm[MHD::IV], iOct) = rhov;
  if constexpr (dim == 3)
  {
    m_Unew(cell_index, m_fm[MHD::IW], iOct) = rhow;
  }

  // compute kinetic energy after updating momentum
  auto ekin_new = HALF_F * (rhou * rhou + rhov * rhov) / rho_new;
  if constexpr (dim == 3)
  {
    ekin_new += HALF_F * (rhow * rhow) / rho_new;
  }

  // update total energy
  m_Unew(cell_index, m_fm[MHD::IP], iOct) += (ekin_new - ekin_old);

} // operator()

template class AddGravitySourceTerm<2, kalypsso::DefaultDevice>;
template class AddGravitySourceTerm<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
