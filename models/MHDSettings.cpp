// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_mhd_ct authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDSettings.cpp
 */
#include <godunov_mhd_ct/models/MHDSettings.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// =======================================================
// =======================================================
MHDSettings::MHDSettings(ConfigMap const & config_map)
  : hydro(config_map)
  , small_beta(config_map.getReal("mhd", "small_beta", KALYPSSO_NUM(1e-3)))
{}

// =======================================================
// =======================================================
void
MHDSettings::print() const
{
  hydro.print();

  KALYPSSO_INFO("small_beta                     : {}", small_beta);

} // MHDSettings::print

} // namespace godunov_mhd_ct

} // namespace kalypsso
