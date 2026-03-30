// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file solver.h
 *
 *  @brief Functions of the Solver module.
 *
 *  This file contains Solver functions that are needed to solve a given
 *  problem.
 *
 *  @author S. Biplab Raut
 */

#include "core/solvers/solver.h"
#include "solver.h"

// Table of solvers that is populated with applicable solvers at setup time
// ct, direct, nDim, buf, permKer, batched, bluestein, PFA, rader, permCopy,
// trans
dft_solver_ solvers_table[NUM_SOLVERS_END] = { 0x0, };

INT32 register_solvers(INT32 dt, INT32 is_real, INT32 cpu_flags)
{
    aoclfftz_solver_type solv_idx;

    for (solv_idx = SOLVER_DIRECT; solv_idx < NUM_SOLVERS_END; solv_idx++)
    {
        solvers_table[solv_idx] = NULL;
    }

    // Add all the available solvers
    if (is_real)
    {
        solvers_table[SOLVER_REAL_DIRECT] = register_execute_real_direct_solver();
        solvers_table[SOLVER_REAL_DIRECT_TWIDDLE] =
            register_execute_real_direct_solver();
        solvers_table[SOLVER_REAL_CT] = register_execute_real_ct_solver();
        solvers_table[SOLVER_REAL_BATCHED] =
            register_execute_real_batched_solver();
        solvers_table[SOLVER_REAL_BUFFERED] =
            register_execute_real_buffered_solver();
        solvers_table[SOLVER_REAL_NDIM] = register_execute_real_ndim_solver();
        solvers_table[SOLVER_REAL_SIZEONE] =
            register_execute_real_sizeone_solver();
#ifdef MULTI_THREADING
        solvers_table[SOLVER_REAL_MT_DIRECT] =
            register_execute_real_mt_direct_solver();
        solvers_table[SOLVER_REAL_MT_DIRECT_TWIDDLE] =
            register_execute_real_mt_direct_solver();
        solvers_table[SOLVER_REAL_MT_BATCHED] =
            register_execute_real_mt_batched_solver();
#endif
    }
    solvers_table[SOLVER_DIRECT] = register_execute_direct_solver();
    solvers_table[SOLVER_DIRECT_BATCHED_COLMAJOR] =
        register_execute_direct_batched_colmajor_solver();
    solvers_table[SOLVER_CT] = register_execute_ct_solver();
    solvers_table[SOLVER_CT_TWIDDLE] = register_execute_ct_twiddle_solver();
    solvers_table[SOLVER_BATCHED_CT_L1_DIRECT] =
        register_execute_batched_ct_l1_direct_solver();
    solvers_table[SOLVER_BATCHED] = register_execute_batched_solver();
    solvers_table[SOLVER_BUFFERED] = register_execute_buffered_solver();
    solvers_table[SOLVER_BLUESTEIN] = register_execute_bluestein_solver();
    solvers_table[SOLVER_NDIM] = register_execute_ndim_solver();
    solvers_table[SOLVER_SIZEONE] = register_execute_sizeone_solver();
    // SR is currently limited only to scalar execution as CT outperforms SR
    // in AVX mode. Skip registration of SR solver.
    if (cpu_flags == 0)
    {
        solvers_table[SOLVER_SR] = register_execute_sr_solver();
    }
    solvers_table[SOLVER_TRANSPOSE] = register_execute_transpose_solver();
#ifdef MULTI_THREADING
    solvers_table[SOLVER_MT_DIRECT] = register_execute_mt_direct_solver();
    solvers_table[SOLVER_MT_BATCHED] = register_execute_mt_batched_solver();
    solvers_table[SOLVER_MT_DIRECT_BATCHED_COLMAJOR] =
        register_execute_mt_direct_batched_colmajor_solver();
    solvers_table[SOLVER_MT_DIRECT_BATCHED_ROWMAJOR] =
        register_execute_mt_direct_batched_rowmajor_solver();
#endif

    return SOLVER_SUCCESS;
}

INT32 is_solver_registered(aoclfftz_solver_type solver_type)
{
    if (solvers_table[solver_type] == NULL)
    {
        return SOLVER_FAILURE;
    }
    return SOLVER_SUCCESS;
}

INT32 set_solver_fp(aoclfftz_generic_solver_t *solver_obj)
{
    if (is_solver_registered(solver_obj->solver_type) != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    solver_obj->execute_solver = solvers_table[solver_obj->solver_type];
    return SOLVER_SUCCESS;
}

INT64 compute_kernel_cost(const kernel_t *ker, UINT8 precision,
                          UINT8 direction, INTP batch)
{
    ops_cycles_t oc = ker->k_ops_cnt(precision, direction);
    INT64 ops = (oc.fma   * AMD_ZEN_FP_FMA_CYCLES) +
                (oc.mul   * AMD_ZEN_FP_MUL_CYCLES) +
                (oc.add   * AMD_ZEN_FP_ADD_CYCLES) +
                (oc.move  * AMD_ZEN_FP_MOVE_CYCLES) +
                (oc.perm  * AMD_ZEN_FP_PERM_CYCLES) +
                (oc.other * AMD_ZEN_FP_OTHER_CYCLES);
    UINT8 sets = ker->sets[precision - 2];
    if (batch >= sets)
    {
        ops = (ops + sets - 1) / sets;
    }
    return ops * batch;
}
