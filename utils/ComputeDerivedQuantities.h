// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDerivedQuantities.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_DERIVED_QUANTITIES_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_DERIVED_QUANTITIES_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock

#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/models/mhd_utils.h> // for perfect gas EOS

namespace kalypsso
{

namespace godunov_mhd_ct
{

/**
 * Define a better enum to list supported derived quantities.
 *
 * Derived quantities are quantities that are not solved in the PDE systems but can be
 * deduced from conservative variables.
 *
 * Derived quantities can be scalar or vector valued.
 *
 * - thermal pressure
 * - speed of sound
 * - specific kinetic energy
 * - magnetic pressure
 * - local Mach number : M =|u|/c where c is local speed of sound
 */
// clang-format off
BETTER_ENUM(DERIVED_QUANTITY, uint32_t,
            THERMAL_PRESSURE,
            SPEED_OF_SOUND,
            SPECIFIC_EKIN,
            MAGNETIC_PRESSURE,
            LOCAL_MACH_NUMBER
  )
// clang-format on

// ========================================================
// ========================================================
// ========================================================
/**
 * A simple helper structure to compute a derived quantity (thermal pressure, specific kinetic
 * energy, ...)
 */
template <size_t dim, typename device_t>
struct ComputeDerivedQuantities
{
  //! type alias for cell-centered data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! type alias for face-centered data array at block level (see kalypsso_data_container.h)
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  //! makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  // ==========================================================================
  // ==========================================================================
  static void
  check_args_validity(DataArrayBlock_t const & Udata,
                      int64_t const &          iOct_begin,
                      int64_t const &          num_octs)
  {

    if (iOct_begin < 0 or iOct_begin >= Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for iOct_begin.");
    }
    if (iOct_begin + num_octs < 0 or (iOct_begin + num_octs) > Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for num_octs.");
    }

  } // check_args_validity

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t            Udata,
      FaceDataArrayBlock_t        Bface,
      FieldMap<core::models::MHD> fm,
      DERIVED_QUANTITY            quantity,
      MHDSettings const &         mhd_settings,
      int64_t                     iOct_begin,
      int64_t                     num_octs)
  {
    const auto label = std::string("compute derived quantity ") + quantity._to_string();

    auto res = DataArrayBlock_t(label, Udata.block_size(), 1, Udata.num_quadrants());

    const auto    nbCellsPerLeaf = Udata.num_cells();
    const int64_t total_num_cells = nbCellsPerLeaf * num_octs;

    Kokkos::parallel_for(
      label,
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        /// convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = iOct_begin + global_index / nbCellsPerLeaf;
        const auto cell_index =
          static_cast<int32_t>(global_index - (iOct - iOct_begin) * nbCellsPerLeaf);

        // compute i,j,k coordinates of current cell inside block
        const auto iCoord = cellindex_to_coord<dim>(cell_index, Udata.block_size());

        MHDStateCell uLoc; // cell-centered conservative variables in current cell

        // get conservative variable in current cell
        uLoc[MHD::ID] = Udata(cell_index, fm[MHD::ID], iOct);
        uLoc[MHD::IP] = Udata(cell_index, fm[MHD::IP], iOct);
        uLoc[MHD::IU] = Udata(cell_index, fm[MHD::IU], iOct);
        uLoc[MHD::IV] = Udata(cell_index, fm[MHD::IV], iOct);
        uLoc[MHD::IW] = Udata(cell_index, fm[MHD::IW], iOct);

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support lambda capturing variables inside inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (Bface.num_quadrants() == 0)
          dummy++;
#endif

        if constexpr (dim == 2)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          // clang-format off
          uLoc[MHD::IA] = HALF_F * (Bface(i, j, IX, iOct) + Bface(i + 1, j    , IX, iOct));
          uLoc[MHD::IB] = HALF_F * (Bface(i, j, IY, iOct) + Bface(i    , j + 1, IY, iOct));
          uLoc[MHD::IC] = HALF_F * (Bface(i, j, IZ, iOct) + Bface(i    , j    , IZ, iOct));
          // clang-format on
        }
        else if constexpr (dim == 3)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          auto const & k = iCoord[IZ];
          // clang-format off
          uLoc[MHD::IA] = HALF_F * (Bface(i, j, k, IX, iOct) + Bface(i + 1, j    , k    , IX, iOct));
          uLoc[MHD::IB] = HALF_F * (Bface(i, j, k, IY, iOct) + Bface(i    , j + 1, k    , IY, iOct));
          uLoc[MHD::IC] = HALF_F * (Bface(i, j, k, IZ, iOct) + Bface(i    , j    , k + 1, IZ, iOct));
          // clang-format on
        }

        // compute primitive variables and speed of sound in current cell
        const auto qLoc = core::models::mhd::computePrimitives(uLoc, mhd_settings);

        if (quantity._to_integral() == +DERIVED_QUANTITY::THERMAL_PRESSURE)
        {
          res(cell_index, 0, iOct) = qLoc[MHD::IP];
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::SPEED_OF_SOUND)
        {
          // ideal gas
          res(cell_index, 0, iOct) =
            sqrt(mhd_settings.hydro.gamma0 * qLoc[MHD::IP] / qLoc[MHD::ID]);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::SPECIFIC_EKIN)
        {
          if constexpr (dim == 2)
          {
            res(cell_index, 0, iOct) =
              HALF_F * (qLoc[MHD::IU] * qLoc[MHD::IU] + qLoc[MHD::IV] * qLoc[MHD::IV]);
          }
          else if constexpr (dim == 3)
          {
            res(cell_index, 0, iOct) =
              HALF_F * (qLoc[MHD::IU] * qLoc[MHD::IU] + qLoc[MHD::IV] * qLoc[MHD::IV] +
                        qLoc[MHD::IW] * qLoc[MHD::IW]);
          }
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::MAGNETIC_PRESSURE)
        {
          res(cell_index, 0, iOct) =
            HALF_F * (qLoc[MHD::IA] * qLoc[MHD::IA] + qLoc[MHD::IB] * qLoc[MHD::IB] +
                      qLoc[MHD::IC] * qLoc[MHD::IC]);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::LOCAL_MACH_NUMBER)
        {
          auto u_norm = qLoc[MHD::IU] * qLoc[MHD::IU] + qLoc[MHD::IV] * qLoc[MHD::IV];
          if constexpr (dim == 3)
            u_norm += qLoc[MHD::IW] * qLoc[MHD::IW];
          u_norm = sqrt(u_norm);

          // speed of sound (perfect gas)
          const auto cs = sqrt(mhd_settings.hydro.gamma0 * qLoc[MHD::IP] / qLoc[MHD::ID]);

          res(cell_index, 0, iOct) = u_norm / cs;
        }
      });

    return res;

  } // run

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t            Udata,
      FaceDataArrayBlock_t        Bface,
      FieldMap<core::models::MHD> fm,
      std::string                 quantity,
      MHDSettings                 mhd_settings,
      int64_t                     iOct_begin,
      int64_t                     num_octs)
  {
    if (quantity == "thermal_pressure")
    {
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::THERMAL_PRESSURE, mhd_settings, iOct_begin, num_octs);
    }
    else if (quantity == "speed_of_sound")
    {
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::SPEED_OF_SOUND, mhd_settings, iOct_begin, num_octs);
    }
    else if (quantity == "specific_ekin")
    {
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::SPECIFIC_EKIN, mhd_settings, iOct_begin, num_octs);
    }
    else if (quantity == "magnetic_pressure")
    {
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::MAGNETIC_PRESSURE, mhd_settings, iOct_begin, num_octs);
    }
    else if (quantity == "local_mach_number")
    {
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::LOCAL_MACH_NUMBER, mhd_settings, iOct_begin, num_octs);
    }
    else
    {
      KALYPSSO_ERROR(
        "ComputeDerivedQuantity: unknow quantity (check your input parameter file) - use "
        "thermal pressure instead.");
      return run(
        Udata, Bface, fm, DERIVED_QUANTITY::THERMAL_PRESSURE, mhd_settings, iOct_begin, num_octs);
    }
  } // run

}; // struct ComputeDerivedQuantities

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_DERIVED_QUANTITIES_H_
