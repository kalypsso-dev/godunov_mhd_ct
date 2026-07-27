// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_mhd_ct authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDSettings.h
 * \brief MHD solver parameters.
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_MODELS_MHDSETTINGS_H_
#define KALYPSSO_GODUNOV_MHD_CT_MODELS_MHDSETTINGS_H_

#include <godunov_mhd_ct/models/HydroSettings.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ===========================================================================
// ===========================================================================
/**
 * Parameters that can be passed by copy to a Kokkos device.
 */
struct MHDSettings
{
  //! Hydrodynamics settings
  HydroSettings hydro;

  //! Lower threshold value for plasma beta
  real_t small_beta;

  MHDSettings(ConfigMap const & config_map);
  ~MHDSettings() = default;

  void
  print() const;

}; // struct MHDSettings

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_MODELS_MHDSETTINGS_H_
