// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_mhd_ct authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroSettings.h
 * \brief Hydrodynamics solver parameters.
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_MODELS_HYDROSETTINGS_H_
#define KALYPSSO_GODUNOV_MHD_CT_MODELS_HYDROSETTINGS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#include <godunov_mhd_ct/models/riemann_solver_types.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ===========================================================================
// ===========================================================================
/**
 * Parameters that can be passed by copy to a Kokkos device.
 */
struct HydroSettings
{

  real_t gamma0;         /*!< specific heat capacity ratio (adiabatic index) - should be removed if
                            refactored for using EosWrapper instead) */
  real_t            cfl; /*!< Courant-Friedrich-Lewy parameter.*/
  real_t            slope_type;        /*!< type of slope computation (2 for second order scheme).*/
  real_t            smallr;            /*!< small density cut-off*/
  real_t            smallc;            /*!< small speed of sound cut-off*/
  real_t            smallp;            /*!< small pressure cut-off*/
  real_t            smallpp;           /*!< smallp times smallr*/
  real_t            cIso;              /*!< if non zero, isothermal */
  int               niter_riemann;     /*!< number of iteration used in quasi-exact Riemann solver*/
  RiemannSolverType riemannSolverType; /*!< type of Riemann solver (LLF, HLLC, ...) */
  bool              abort_when_negative_eint;     /*!< abort if negative internal energy */
  bool              abort_when_negative_pressure; /*!< abort if negative pressure detected */

  KALYPSSO_STATIC_MATH_CONSTANT(SMALLR, 1e-8);
  KALYPSSO_STATIC_MATH_CONSTANT(SMALLP, 1e-6);

  HydroSettings(ConfigMap const & config_map);
  ~HydroSettings() = default;

  void
  print() const;

}; // struct HydroSettings

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_MODELS_HYDROSETTINGS_H_
