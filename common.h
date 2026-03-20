// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file common.h
 *
 * \brief Contains a shared set of values and types.
 */

#ifndef KALYPSSO_GODUNOV_MHD_CT_COMMON_H_
#define KALYPSSO_GODUNOV_MHD_CT_COMMON_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/models/MHD.h>
#include <kalypsso/core/models/MHDSettings.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/brick_utils.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

template <size_t dim, typename device_t>
class SolverGodunovMHD;

//! Array of Orchard keys
template <typename device_t>
using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

//! Shorthand for the hydrodynamic model
using MHD = core::models::MHD;

//! Hashmap type from Orchard keys to octant index
template <typename device_t>
using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

//! AMR flags array type
template <size_t dim, typename device_t>
using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_COMMON_H_
