// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_mhd_ct authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHD.cpp
 */
#include <godunov_mhd_ct/models/MHD.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

namespace models
{

// clang-format off
const id2names_t MHD::m_id2names_all = {
  { ID, "rho" },
  { IE, "e_tot" },
  { IU, "rho_vx" },
  { IV, "rho_vy" },
  { IW, "rho_vz" },
  { IBX, "Bx" },
  { IBY, "By" },
  { IBZ, "Bz" },
  { IGX, "grav_x" },
  { IGY, "grav_y" },
  { IGZ, "grav_z" }
};
// clang-format on

} // namespace models

} // namespace godunov_mhd_ct

} // namespace kalypsso
