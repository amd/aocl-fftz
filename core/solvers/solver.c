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

// Table of (solver_type -> execute function pointer). Re-populated by
// register_solvers() on every aoclfftz_setup_* call. Each write stores
// the same pointer value the slot already holds (idempotent), so readers
// observing a partially-written table still see a valid function
// pointer; no synchronisation is needed on the read side.
dft_solver_ solvers_table[NUM_SOLVERS_END] = { 0x0, };

// Populates solvers_table. Called from every aoclfftz_setup_* entry
// point, so it runs once per setup rather than once per process. The
// writes are idempotent (same pointers every time), which makes
// concurrent setup calls benign even though they technically race.
FFTZ_INT32 register_solvers(FFTZ_VOID)
{
    // Real solvers
    solvers_table[SOLVER_REAL_DIRECT_R2C] =
        register_execute_real_direct_r2c();
    solvers_table[SOLVER_REAL_DIRECT_R2C_BATCHED] =
        register_execute_real_direct_r2c_batched();
    solvers_table[SOLVER_REAL_DIRECT_C2R] =
        register_execute_real_direct_c2r();
    solvers_table[SOLVER_REAL_DIRECT_CT_R2C] =
        register_execute_real_direct_ct_r2c();
    solvers_table[SOLVER_REAL_DIRECT_CT_C2R] =
        register_execute_real_direct_ct_c2r();
    solvers_table[SOLVER_REAL_CT] = register_execute_real_ct_solver();
    solvers_table[SOLVER_REAL_BATCHED] = register_execute_real_batched_solver();
    solvers_table[SOLVER_REAL_BUFFERED] =
        register_execute_real_buffered_solver();
    solvers_table[SOLVER_REAL_NDIM] = register_execute_real_ndim_solver();
    solvers_table[SOLVER_REAL_SIZEONE] = register_execute_real_sizeone_solver();
#ifdef MULTI_THREADING
    solvers_table[SOLVER_REAL_MT_DIRECT_R2C] =
        register_execute_real_mt_direct_r2c();
    solvers_table[SOLVER_REAL_MT_DIRECT_R2C_BATCHED] =
        register_execute_real_mt_direct_r2c_batched();
    solvers_table[SOLVER_REAL_MT_DIRECT_C2R] =
        register_execute_real_mt_direct_c2r();
    solvers_table[SOLVER_REAL_MT_DIRECT_CT_R2C] =
        register_execute_real_mt_direct_ct_r2c();
    solvers_table[SOLVER_REAL_MT_DIRECT_CT_C2R] =
        register_execute_real_mt_direct_ct_c2r();
    solvers_table[SOLVER_REAL_MT_BATCHED] =
        register_execute_real_mt_batched_solver();
#endif

    // Complex solvers
    solvers_table[SOLVER_DIRECT] = register_execute_direct_solver();
    solvers_table[SOLVER_DIRECT_BATCHED_COLMAJOR] =
        register_execute_direct_batched_colmajor_solver();
    solvers_table[SOLVER_CT] = register_execute_ct_solver();
    solvers_table[SOLVER_BATCHED_CT_L1_DIRECT] =
        register_execute_batched_ct_l1_direct_solver();
    solvers_table[SOLVER_BATCHED] = register_execute_batched_solver();
    solvers_table[SOLVER_BUFFERED] = register_execute_buffered_solver();
    solvers_table[SOLVER_BLUESTEIN] = register_execute_bluestein_solver();
    solvers_table[SOLVER_NDIM] = register_execute_ndim_solver();
    solvers_table[SOLVER_SIZEONE] = register_execute_sizeone_solver();
    // SR is registered unconditionally here. The selector decides whether
    // to actually dispatch SR based on per-call cpu_flags.
    solvers_table[SOLVER_SR] = register_execute_sr_solver();
    // POW2 iterative fast path is registered unconditionally; the selector gate
    // decides whether to dispatch it.
    solvers_table[SOLVER_POW2_ITERATIVE] =
        register_execute_pow2_iterative_solver();
    solvers_table[SOLVER_TRANSPOSE] = register_execute_transpose_solver();
#ifdef MULTI_THREADING
    solvers_table[SOLVER_MT_DIRECT] = register_execute_mt_direct_solver();
    solvers_table[SOLVER_MT_BATCHED] = register_execute_mt_batched_solver();
    solvers_table[SOLVER_MT_DIRECT_BATCHED_COLMAJOR] =
        register_execute_mt_direct_batched_colmajor_solver();
    solvers_table[SOLVER_MT_DIRECT_BATCHED_ROWMAJOR] =
        register_execute_mt_direct_batched_rowmajor_solver();
    solvers_table[SOLVER_MT_BLUESTEIN] =
        register_execute_mt_bluestein_solver();
#endif

    return SOLVER_SUCCESS;
}

FFTZ_INT32 is_solver_registered(aoclfftz_solver_type solver_type)
{
    if (solvers_table[solver_type] == NULL)
    {
        return SOLVER_FAILURE;
    }
    return SOLVER_SUCCESS;
}

FFTZ_INT32 set_solver_fp(aoclfftz_generic_solver_t *solver_obj)
{
    if (is_solver_registered(solver_obj->solver_type) != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    solver_obj->execute_solver = solvers_table[solver_obj->solver_type];
    return SOLVER_SUCCESS;
}

FFTZ_INT64 compute_kernel_cost(const kernel_t *ker, FFTZ_UINT8 precision,
                          FFTZ_UINT8 direction, FFTZ_INTP batch)
{
    ops_cycles_t oc = ker->k_ops_cnt(precision, direction);
    FFTZ_INT64 ops = (oc.fma   * AMD_ZEN_FP_FMA_CYCLES) +
                (oc.mul   * AMD_ZEN_FP_MUL_CYCLES) +
                (oc.add   * AMD_ZEN_FP_ADD_CYCLES) +
                (oc.move  * AMD_ZEN_FP_MOVE_CYCLES) +
                (oc.perm  * AMD_ZEN_FP_PERM_CYCLES) +
                (oc.other * AMD_ZEN_FP_OTHER_CYCLES);
    FFTZ_UINT8 sets = ker->sets[precision - 2];
    if (batch >= sets)
    {
        ops = (ops + sets - 1) / sets;
    }
    return ops * batch;
}

FFTZ_INTP find_radix_base_idx(kernel_t *kertab, FFTZ_INTP radix)
{
    for (FFTZ_INTP base_idx = 0; base_idx < NUM_KERNELS_IN_EACH_CATEGORY;
         base_idx++)
    {
        if (kertab[base_idx].radix == 0) // End of suitable kernels in the list
        {
            break;
        }

        if ((FFTZ_INTP)kertab[base_idx].radix == radix)
        {
            return base_idx;
        }
    }
    return -1;
}

kernel_choice_t find_best_kernel(kernel_t *kertab,
                                 FFTZ_INTP base_idx,
                                 FFTZ_UINT8 precision,
                                 FFTZ_UINT8 direction,
                                 FFTZ_INTP batch)
{
    kernel_choice_t optimal = {-1, INT64_MAX};

    for (FFTZ_INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
    {
        FFTZ_INTP kloc = kcat * NUM_KERNELS_IN_EACH_CATEGORY + base_idx;
        if (kertab[kloc].kfft[direction] == NULL)
        {
            continue;
        }

        FFTZ_INT64 cost = compute_kernel_cost(&kertab[kloc], precision,
                                         direction, batch);
        if (cost < optimal.cost)
        {
            optimal.idx  = kloc;
            optimal.cost = cost;
        }
    }
    return optimal;
}
