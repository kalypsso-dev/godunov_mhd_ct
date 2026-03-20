// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AddGravitySourceTerm.
 * \brief Add gravity source term.
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_ADD_GRAVITY_SOURCE_TERM_H_
#define KALYPSSO_GODUNOV_MHD_CT_ADD_GRAVITY_SOURCE_TERM_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
// #include <kalypsso/core/models/mhd_utils.h> // for computePrimitives
#include <kalypsso/core/GravityField.h>


namespace kalypsso
{

namespace godunov_mhd_ct
{

//*************************************************
//*************************************************
//*************************************************
/**
 * This functor is aimed at being applied right after the hydro time step.
 *
 * So we need :
 * - Uold (conservative variables at the beginning of time step)
 * - Unew (conservative variables already updated with hydro flux)
 */
template <size_t dim, typename device_t>
class AddGravitySourceTerm
{
public:
  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  //! global cell index
  using index_t = int32_t;

private:
  //! gravitational field acceleration
  Kokkos::Array<real_t, dim> m_grav;

  //! conservative variables at beginning of time step (not modified here)
  DataArrayBlock_t m_Uold;

  //! conservative variables right after hydro update (will be modified here)
  DataArrayBlock_t m_Unew;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  //! time step
  const real_t m_dt;

public:
  AddGravitySourceTerm(Kokkos::Array<real_t, dim> const & grav,
                       DataArrayBlock_t const &           Uold,
                       DataArrayBlock_t const &           Unew,
                       FieldMap<core::models::MHD>        fm,
                       real_t                             dt)
    : m_grav(grav)
    , m_Uold(Uold)
    , m_Unew(Unew)
    , m_fm(fm)
    , m_nbCellsPerLeaf(Uold.num_cells())
    , m_dt(dt){};

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  //! \param[in] Uold is conservative variables array at beginning of time step
  //! \param[in,out] Unew is conservative variables array after hydro step
  //! \param[in] fm is the field map
  //! \param[in] local_num_octant is the local (current MPI proc) number of octants
  //! \param[in] dt is time step
  static void
  apply(ConfigMap const &           config_map,
        DataArrayBlock_t const &    Uold,
        DataArrayBlock_t const &    Unew,
        FieldMap<core::models::MHD> fm,
        int32_t                     local_num_octants,
        real_t                      dt);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // class AddGravitySourceTerm

extern template class AddGravitySourceTerm<2, kalypsso::DefaultDevice>;
extern template class AddGravitySourceTerm<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_ADD_GRAVITY_SOURCE_TERM_H_
