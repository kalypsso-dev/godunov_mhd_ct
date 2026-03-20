// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file init.h
 *
 * \brief Contains the function that initialises a solver.
 */

#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_H_

#include <godunov_mhd_ct/common.h>

#include <kalypsso/core/SolverBase.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// =======================================================
// =======================================================
/**
 * \brief Initializes conservative variables at time t=0 using analytical values.
 *
 * \param solver The solver itself.
 *
 */
template <size_t dim, typename device_t>
void
init(SolverGodunovMHD<dim, device_t> & solver);

// =======================================================
// =======================================================
/**
 * \brief Initializes the solver and its conservative values data using a previous run.
 *
 * The input parameter file must contain the of the file to read.
 */
template <size_t dim, typename device_t>
void
init_restart([[maybe_unused]] SolverGodunovMHD<dim, device_t> & solver);

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_H_
