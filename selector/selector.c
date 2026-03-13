/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file selector.c
 *
 *  @brief Selects and returns a suitable solution for the input problem.
 *
 *  This file contains the implementation of functions that are used to
 *  select a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/common/twiddle.h"
#include "core/kernels/kernel_list.h"
#include "utils/utils.h"

// Function pointers to communicate the exact selector model for executing the
// problem.
// Not thread-safe
typedef INT32(*selector_model_func_)(aoclfftz_selector_t*);
typedef INT32 (*selector_model_rdft_func_)(aoclfftz_selector_t *,
                                           aoclfftz_realhelper_t *);
selector_model_func_ sel_fp = NULL;
selector_model_rdft_func_ sel_rdft_fp = NULL;

// Register all applicable solvers and kernels into the respective tables
// based on the input problem and CPU opt level
INT32 register_solvers_kernels(kernel_tables_t *kernel_tables, INT32 dt,
                               INT32 dir, INT32 is_real, INT32 cpu_flags)
{
    INT32 ret = SELECTOR_FAILURE;

    // Register Solvers
    ret = register_solvers(dt, is_real, cpu_flags);
    if (ret != SOLVER_SUCCESS)
    {
        return SELECTOR_FAILURE;
    }

    // Register Kernels
    if (is_real)
    {
        // Register standard kernels for C2R (real-backward) since the twiddle
        // kernels are not supported
        if (dir == BACKWARD_FFT_DIR)
        {
            ret |= register_kernels_real(kernel_tables->kt_rdft,
                                         kernels_real, dt, dir, cpu_flags);
        }
        else
        {
            ret |= register_kernels_real(kernel_tables->kt_rdft,
                                         kernels_twid_real, dt, dir, cpu_flags);
        }
    }
    ret |= register_kernels_complex(kernel_tables->kt_dft, kernels_c2c, dt,
                                    dir, cpu_flags);
    ret |= register_kernels_complex(kernel_tables->kt_twid_dft,
                                    kernels_twid_c2c, dt, dir, cpu_flags);

    return ret;
}

INT32 check_FFT_kernel_support(INTP n, kernel_t *kernels_table, INT32 is_innermost_dim)
{
    INT32 is_supported = 0;
    INTP num_kernels_to_check = NUM_KERNELS_IN_EACH_CATEGORY;

    // The "special" (large) kernels are not well suited to run for problems
    // whose strides are large. This is typically the case for non-innermost
    // dimensions of an ND problem. Hence, we skip checking those kernels in
    // such cases.
    if (!is_innermost_dim)
    {
        num_kernels_to_check -= NUMBER_OF_HIGHER_RADIX_KERNELS;
    }

    // It is enough to check for the existence of a suitable C kernel
    for (INTP i = 0; i < num_kernels_to_check; i++)
    {
        if (kernels_table[i].radix == 0)
        {
            break;
        }
        if (kernels_table[i].radix == n)
        {
            is_supported = 1;
            break;
        }
    }

    return is_supported;
}

// Check if the problem is col-major or row-major.
// col-major: vec-strides < elemental-strides
// row-major: elemental-strides < vec-strides
UINT8 check_col_major(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    UINT8 is_col_major = (decomp_scheme->vecs[0].in_stride <
        decomp_scheme->dims[0].in_stride) &&
       (decomp_scheme->vecs[0].out_stride < decomp_scheme->dims[0].out_stride);
    return is_col_major;
}

static INT32 is_split_radix_applicable(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    INTP n = decomp_scheme->dims[0].n;
    UINT8 is_col_major = check_col_major(decomp_scheme);
    // Split-radix is applicable for complex, 1D, row-major, non-batched transforms,
    // innermost dimension where the problem size is divisible by 4 and
    // greater than or equal to 4096.
    return (!IS_REAL(decomp_scheme->flags) && (n >= 4096) && ((n % 4) == 0)
            && (decomp_scheme->vec_rank == 1) && (decomp_scheme->dim_rank == 1)
            && (decomp_scheme->batched_vecs == NULL) && (!is_col_major)
            && (!IS_NOT_INNERMOST_DIM(decomp_scheme->flags)));
}

INTP check_CT_solvability(INTP n, kernel_t *kertab)
{
    // It is enough to check for the existence of a suitable C kernel
    for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        UINT32 radix = kertab[i].radix;
        if (radix == 0) // End of suitable kernels in the list
        {
            break;
        }
        // Check if this radix can factorize the problem
        if ((n % (INTP)radix) == 0)
        {
            return 1;
        }
    }
    return 0;
}

INT32 check_prime_solvability_bluestein(aoclfftz_decomp_scheme_t *decomp_scheme,
                                        INT32 is_FFT_ker_supported,
                                        kernel_t *kertab)
{
    INTP n = decomp_scheme->dims[0].n;
    INTP batch = decomp_scheme->vecs[0].n;
    INT32 dim_rank = decomp_scheme->dim_rank;
    INT32 vec_rank = decomp_scheme->vec_rank;

    if (n == 1 || dim_rank != 1 || vec_rank != 1 || batch != 1)
    {
        return 0;
    }

    // n is solvable by bluestein solver if is not solvable by direct
    // and CT solvers
    INT32 is_solvable_by_CT = check_CT_solvability(n, kertab);
    if (is_FFT_ker_supported == 0 && is_solvable_by_CT == 0)
    {
        return 1;
    }

    return 0;
}

INT32 is_prime(INT32 n)
{
    for (INT32 i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
        {
            return 0;
        }
    }

    return 1;
}

// Check if the input problem contains a prime factor that requires a
// bluestein solver to solve.
INT32 check_bluestein_problem(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    INT32 dim_rank = decomp_scheme->dim_rank;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP n = decomp_scheme->dims[i].n;
        // Check for prime factors greater than 13 in (n)
        for (INT32 i = 17; i <= n; i++)
        {
            if (n % i == 0 && is_prime(i))
            {
                return 1;
            }
        }
    }

    return 0;
}

// Check whether n can be factored as radix_r * radix_m where radix_r
// has twiddle kernel support and radix_m has plain FFT kernel support.
INT32 check_batched_ct_l1_direct_solvability(INTP n, kernel_t *kertab_twid,
                                   kernel_t *kertab_dft)
{
    for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        INTP radix_r = (INTP)kertab_twid[i].radix;
        
        if (radix_r == 0) // End of suitable kernels in the list
        {
            break;
        }

        // Check if this radix can factorize the problem
        if ((n % radix_r) != 0)
        {
            continue;
        }
        
        if (check_FFT_kernel_support(n / radix_r, kertab_dft, 1))
        {
            return 1;
        }
    }
    return 0;
}

// Fixed decision logic and CPI based selector mode execution for the
// input problem based on the applicable tables of solvers and kernels
INT32 selector_fixed_mode_dft_(aoclfftz_selector_t *sel)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = sel->kernel_tables->kt_dft;
    INT32 ret = SELECTOR_FAILURE;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 avl_threads = sel->solution->decomp_scheme->thread_info->avl_threads;

    // TODO: Should be removed after supporting MT in bluestein solver
    if (check_bluestein_problem(sel->solution->decomp_scheme) &&
        (avl_threads > 1))
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Multi Threaded execution is"
        " not supported for Bluestein solver, falling back to single"
        " threaded execution");
        sel->solution->decomp_scheme->thread_info->avl_threads = 1;
        avl_threads = 1;
    }
    INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));
    INT32 is_solvable_by_bluestein = check_prime_solvability_bluestein(
        sel->solution->decomp_scheme, is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;
    INT32 standalone_transpose_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }
    // SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* non-size-one */
         ((sel->solution->decomp_scheme->vec_rank > 1) ||  /* ND Batched */
          /* 1D Batched 1D Non-direct cases*/
          ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
           !is_FFT_ker_supported) ||
          /* 1D Batched ND case*/
          (dim_rank > 1 && sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    // Future work: GENERALIZE THE SCOPE OF BUFFERED SOLVER
    // level1_cond2 = !(IS_OUT_OF_PLACE(sel->solution->decomp_scheme->flags));
    // level1_cond2 &= ((sel->solution->decomp_scheme->dims[0].n >
    //                   MAX_GUARANTEED_CACHEABLE_SIZE) &&
    //                  (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
    //                   AOCLFFTZ_AUTO_SELECTOR));
    level1_cond2 = IS_BUFFERED(sel->solution->decomp_scheme->flags);
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_SR
    level2_cond |=
        (is_split_radix_applicable(sel->solution->decomp_scheme) << 1);
    // SOLVER_PFA
    // SOLVER_RADER

    // SOLVER_TRANSPOSE
    standalone_transpose_cond =
        GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags);

    /** Level 1 decisions : Solvers **/
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call buffered solver master
        return selector_buffered_dft(sel, kertab);
    }
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call batched solver master
        return selector_batched_dft(sel, kertab);
    }
    // Transpose Solver (Standalone)
    if (standalone_transpose_cond)
    {
        solver_obj->solver_type = SOLVER_TRANSPOSE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        // call the transpose solver selector
        return selector_transpose(sel);
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call ndim solver master
        return selector_ndim_dft(sel, kertab);
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        solver_obj->solver_type = SOLVER_BLUESTEIN;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call bluestein solver master
        return selector_bluestein_dft(sel, kertab);
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        solver_obj->solver_type = SOLVER_PERM_KER;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call permuted solver master
        return selector_permuted_dft(sel, kertab);
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // SizeOne FFT Solver
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call SizeOne Solver master
        return selector_sizeone_dft(sel, kertab);
    }
    else if (level2_cond & 0x1)
    {
        if (avl_threads <= 1)
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_DIRECT;
            }
            else
            {
                solver_obj->solver_type = SOLVER_DIRECT_BATCHED_COLMAJOR;
            }
        }
        else
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_MT_DIRECT;
            }
            else
            {
                if (should_use_colmajor_batched_solver(sel->solution,
                                                       kertab, avl_threads))
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_COLMAJOR;
                }
                else
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_ROWMAJOR;
                }
            }
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call Direct Solver master
        return selector_direct_dft(sel, kertab);
    }
    // Split-Radix FFT Solver
    else if (level2_cond & 0x2)
    {
        solver_obj->solver_type = SOLVER_SR;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }
        // Call Split-Radix Selector
        return selector_sr_dft(sel, kertab);
    }
    else
    {
        solver_obj->solver_type = SOLVER_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call CT Solver master
        return selector_ct_dft(sel, kertab);
    }

    return ret;
}

// Fixed decision logic and CPI based selector mode execution for the
// input problem based on the applicable tables of solvers and kernels
// wherein the successive stage dfts are fused with twiddle multiplications
INT32 selector_fixed_mode_fused_twid_dft_(aoclfftz_selector_t *sel)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = sel->kernel_tables->kt_dft;
    INT32 ret = SELECTOR_FAILURE;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 avl_threads = sel->solution->decomp_scheme->thread_info->avl_threads;

    // TODO: Should be removed after supporting MT in bluestein solver
    if (check_bluestein_problem(sel->solution->decomp_scheme) &&
        avl_threads > 1)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Multi Threaded execution is"
        " not supported for Bluestein solver, falling back to single"
        " threaded execution");
        sel->solution->decomp_scheme->thread_info->avl_threads = 1;
        avl_threads = 1;
    }
    INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));

    INT32 is_solvable_by_bluestein = check_prime_solvability_bluestein(
        sel->solution->decomp_scheme, is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;
    INT32 standalone_transpose_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }
    // SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* non-size-one */
         ((sel->solution->decomp_scheme->vec_rank > 1) ||  /* ND Batched */
          /* 1D Batched 1D Non-direct cases*/
          ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
           !is_FFT_ker_supported) ||
          /* 1D Batched ND case*/
          (dim_rank > 1 && sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    // Future work: GENERALIZE THE SCOPE OF BUFFERED SOLVER
    // level1_cond2 = !(IS_OUT_OF_PLACE(sel->solution->decomp_scheme->flags));
    // level1_cond2 &= ((sel->solution->decomp_scheme->dims[0].n >
    //                   MAX_GUARANTEED_CACHEABLE_SIZE) &&
    //                  (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
    //                   AOCLFFTZ_AUTO_SELECTOR));
    level1_cond2 = IS_BUFFERED(sel->solution->decomp_scheme->flags);
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_SR
    level2_cond |=
        (is_split_radix_applicable(sel->solution->decomp_scheme) << 1);
    // SOLVER_PFA
    // SOLVER_RADER

    // SOLVER_TRANSPOSE
    standalone_transpose_cond =
        GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags);

    // SOLVER_BATCHED_CT_L1_DIRECT
    INT32 is_batched_ct_l1_direct = check_batched_ct_l1_direct_solvability(
        sel->solution->decomp_scheme->dims[0].n,
        sel->kernel_tables->kt_twid_dft,
        sel->kernel_tables->kt_dft);

    UINT8 is_col_major = check_col_major(sel->solution->decomp_scheme);
    
    /** Level 1 decisions : Solvers **/
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call buffered solver master
        ret = selector_buffered_dft(sel, kertab);
        return ret;
    }

    // Batched CT one level direct: fuses the batch loop and one
    // CT-twiddle level (both radix_m and radix_r directly supported)
    // into a single flat function with direct kernel calls.
    // Applicable for single-threaded, non-direct sizes, row-major,
    // single batch dimension (vec_rank == 1) and innermost ndim dimension.
    if (avl_threads <= 1 &&
        !is_FFT_ker_supported && dim_rank == 1 &&
        sel->solution->decomp_scheme->vec_rank == 1 &&
        sel->solution->decomp_scheme->batched_vecs == NULL &&
        !is_col_major &&
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags) &&
        is_batched_ct_l1_direct)
    {
        solver_obj->solver_type = SOLVER_BATCHED_CT_L1_DIRECT;
        if (set_solver_fp(solver_obj) == SOLVER_SUCCESS)
        {
            ret = selector_batched_ct_l1_direct_dft(sel);
            if (ret == SELECTOR_SUCCESS)
            {
                return ret;
            }
        }
    }

    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call batched solver master
        return selector_batched_dft(sel, kertab);
    }
    // Transpose Solver (Standalone)
    if (standalone_transpose_cond)
    {
        solver_obj->solver_type = SOLVER_TRANSPOSE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
            return SELECTOR_FAILURE;

        // call the transpose solver selector
        return selector_transpose(sel);
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call ndim solver master
        return selector_ndim_dft(sel, kertab);
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        solver_obj->solver_type = SOLVER_BLUESTEIN;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call bluestein solver master
        return selector_bluestein_dft(sel, kertab);
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        solver_obj->solver_type = SOLVER_PERM_KER;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // call permuted solver master
        return selector_permuted_dft(sel, kertab);
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // SizeOne FFT Solver
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call SizeOne Solver master
        return selector_sizeone_dft(sel, kertab);
    }
    else if (level2_cond & 0x1)
    {
        if (avl_threads <= 1)
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_DIRECT;
            }
            else
            {
                solver_obj->solver_type = SOLVER_DIRECT_BATCHED_COLMAJOR;
            }
        }
        else
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_MT_DIRECT;
            }
            else
            {
                if (should_use_colmajor_batched_solver(sel->solution,
                                                       kertab, avl_threads))
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_COLMAJOR;
                }
                else
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_ROWMAJOR;
                }
            }
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        //Direct sub-problems will use normal dft kernels
        //since they are leaf sub-problems always
        kertab = sel->kernel_tables->kt_dft;

        // Call Direct Solver master
        return selector_direct_dft(sel, kertab);
    }
    // Split-Radix FFT Solver
    else if (level2_cond & 0x2)
    {
        solver_obj->solver_type = SOLVER_SR;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }
        // Call Split-Radix Selector
        return selector_sr_dft(sel, kertab);
    }
    else
    {
        solver_obj->solver_type = SOLVER_CT_TWIDDLE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        //All the CT sub-problems will use fused twiddle dft kernels
        //since they are non-leaf sub-problems always
        kertab = sel->kernel_tables->kt_twid_dft;

        // Call CT Solver master
        return selector_ct_dft(sel, kertab);
    }

    return ret;
}

// Fixed decision logic and CPI based selector mode execution for the
// real input problem based on the applicable tables
// of real solvers and real kernels
INT32 selector_fixed_mode_rdft_(aoclfftz_selector_t *sel,
                                aoclfftz_realhelper_t *realhelper)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = sel->kernel_tables->kt_rdft;
    INT32 ret = SELECTOR_FAILURE;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 batch = sel->solution->decomp_scheme->vecs[0].n;
    INT32 avl_threads =
            sel->solution->decomp_scheme->thread_info->avl_threads;

    INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));
    INT32 is_solvable_by_bluestein =
            check_prime_solvability_bluestein(sel->solution->decomp_scheme,
                                              is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }
    // SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* non-size-one */
         ((vec_rank > 1) ||                                /* ND Batched */
          /* 1D Batched 1D Non-direct cases*/
          ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
           !is_FFT_ker_supported &&
           (sel->solution->decomp_scheme->dims[0].n ==
            realhelper->problem_size)) ||
          /* 1D Batched ND case*/
          (dim_rank > 1 && sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    // Buffered solver will be used for all CT problems as of now
    level1_cond2 = !realhelper->is_buffered_invoked && !is_FFT_ker_supported &&
                    sel->solution->decomp_scheme->dims[0].n != 1 /* non-size-one */;
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_PFA
    // SOLVER_RADER

    /** Level 1 decisions : Solvers **/
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_REAL_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_REAL_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_batched_rdft(sel, kertab, realhelper);
    }
    // Multi-dimentional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_REAL_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_ndim_rdft(sel, kertab, realhelper);
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        AOCLFFTZ_ERROR("SELECTOR_FAILURE : "
                                   "Large Prime RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_REAL_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_buffered_rdft(sel, kertab, realhelper);
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        AOCLFFTZ_ERROR("SELECTOR_FAILURE : "
                                   "Permuted RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // Size one problem
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_REAL_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_sizeone_dft(sel, kertab);
    }
    else if (level2_cond & 0x1)
    {
        // Single threaded direct solver to be executed for batch <= 1,
        // irrespective of avl_threads
        if ((avl_threads <= 1) || batch <= 1)
        {
            solver_obj->solver_type = SOLVER_REAL_DIRECT;
        }
        else
        {
            solver_obj->solver_type = SOLVER_REAL_MT_DIRECT;
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call Direct Solver master
        return selector_direct_rdft(sel, kertab, realhelper);
    }
    else
    {
        solver_obj->solver_type = SOLVER_REAL_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call CT Solver master
        return selector_ct_rdft(sel, kertab, realhelper);
    }

    return ret;
}

// Fixed decision logic and CPI based selector mode execution for the
// real input problem based on the applicable tables
// of real solvers and real kernels
INT32 selector_fixed_mode_fused_twid_rdft_(aoclfftz_selector_t *sel,
                                           aoclfftz_realhelper_t *realhelper)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;

    //All the CT sub-problems will use fused twiddle dft kernels
    kernel_t *kertab = sel->kernel_tables->kt_rdft;
    INT32 ret = SELECTOR_FAILURE;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 batch = sel->solution->decomp_scheme->vecs[0].n;
    INT32 avl_threads =
            sel->solution->decomp_scheme->thread_info->avl_threads;

    INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));

    INT32 is_solvable_by_bluestein =
            check_prime_solvability_bluestein(sel->solution->decomp_scheme,
                                              is_FFT_ker_supported, kertab);
    INT32 level1_cond1 = 0;
    INT32 level1_cond2 = 0;
    INT32 level2_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution);
    }

    // SOLVER_BATCHED
    level1_cond1 =
        ((sel->solution->decomp_scheme->dims[0].n != 1) && /* non-size-one */
         ((vec_rank > 1) ||                                /* ND Batched */
          /* 1D Batched 1D Non-direct cases*/
          ((sel->solution->decomp_scheme->vecs[0].n > 1) &&
           !is_FFT_ker_supported &&
           (sel->solution->decomp_scheme->dims[0].n ==
            realhelper->problem_size)) ||
          /* 1D Batched ND case*/
          (dim_rank > 1 && sel->solution->decomp_scheme->vecs[0].n > 1)));
    // SOLVER_NDIM
    level1_cond1 |= ((dim_rank > 1) << 1);
    // SOLVER_BLUESTEIN
    level1_cond1 |= (is_solvable_by_bluestein << 2);
    // SOLVER_BUFFERED
    // Buffered solver will be used for all CT problems as of now
    level1_cond2 = !realhelper->is_buffered_invoked && !is_FFT_ker_supported &&
                   sel->solution->decomp_scheme->dims[0].n != 1 /* non-size-one */;
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_PFA
    // SOLVER_RADER

    /** Level 1 decisions : Solvers **/
    // Batched/vector FFT Solver
    if (level1_cond1 & 0x1)
    {
        if (avl_threads <= 1)
        {
            solver_obj->solver_type = SOLVER_REAL_BATCHED;
        }
        else
        {
            solver_obj->solver_type = SOLVER_REAL_MT_BATCHED;
        }

        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_batched_rdft(sel, kertab, realhelper);
    }
    // Multi-dimensional FFT Solver
    if (level1_cond1 & 0x2)
    {
        solver_obj->solver_type = SOLVER_REAL_NDIM;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_ndim_rdft(sel, kertab, realhelper);
    }
    // Large Primes - Bluestein FFT Solver
    if (level1_cond1 & 0x4)
    {
        AOCLFFTZ_ERROR("SELECTOR_FAILURE : "
                                   "Large Prime RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    // Buffered FFT Solver
    if (level1_cond2 & 0x1)
    {
        solver_obj->solver_type = SOLVER_REAL_BUFFERED;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_buffered_rdft(sel, kertab, realhelper);
    }
    // Permuted (out-of-order output) FFT Solver
    if (level1_cond2 & 0x2)
    {
        AOCLFFTZ_ERROR("SELECTOR_FAILURE : "
                                   "Permuted RealFFT is not supported");
        return SELECTOR_FAILURE;
    }
    /** Level 2 decisions : CT Solver and Kernels **/
    // Size one problem
    if (sel->solution->decomp_scheme->dims[0].n == 1)
    {
        solver_obj->solver_type = SOLVER_REAL_SIZEONE;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        return selector_sizeone_dft(sel, kertab);
    }
    else if (level2_cond & 0x1)
    {
        // Single threaded direct solver to be executed for batch <= 1,
        // irrespective of avl_threads
        if ((avl_threads <= 1) || batch <= 1)
        {
            solver_obj->solver_type = SOLVER_REAL_DIRECT_TWIDDLE;
        }
        else
        {
            solver_obj->solver_type = SOLVER_REAL_MT_DIRECT_TWIDDLE;
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call Direct Solver master
        return selector_direct_rdft(sel, kertab, realhelper);
    }
    else
    {
        solver_obj->solver_type = SOLVER_REAL_CT;
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            return SELECTOR_FAILURE;
        }

        // Call CT Solver master
        return selector_ct_rdft(sel, kertab, realhelper);
    }

    return ret;
}

INT32 selector_autotuner_mode_dft_(aoclfftz_selector_t* sel)
{
    AOCLFFTZ_LOG(INFO, global_logger_mode,
        "Autotuner selector is not yet available for evaluation");
    return SELECTOR_FAILURE;
}

INT32 selector_autotuner_mode_rdft_(aoclfftz_selector_t* sel)
{
    AOCLFFTZ_LOG(INFO, global_logger_mode,
        "Autotuner selector for RealFFT is not yet available for evaluation");
    return SELECTOR_FAILURE;
}

// Main selector driver that invokes the complementary/alternate selector
// algorithms/models and decides on the final selector based on its suitability
// and performance.
// Also provides a cleaner approach to init and handle various related
// solvers and kernel tables
INT32 selector_driver_dft_(aoclfftz_selector_t* sel)
{
    INT32 ret = SELECTOR_FAILURE;

    // if bit reproducibility is requested, directly take its associated
    // code path and return
    if (GET_BIT_REPRODUCIBLE(sel->solution->decomp_scheme->flags))
    {
        sel_fp = selector_fixed_mode_fused_twid_dft_;
        return selector_model_dft_(sel);
    }

    aoclfftz_selector_t *sel_models[AOCLFFTZ_SELECTOR_MODELS] = { 0x0, };
    UINT32 best_model_id = 0;
    cost_analysis_t best_cost = {INT64_MAX, INT64_MAX};
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;

    // FIXED SELECTOR BASED MODEL : start
#ifdef AOCLFFTZ_FIXED_SELECTOR_MODE
    // Allocate selector object
    sel_models[AOCLFFTZ_FIXED_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->inplace_buffer =
            sel->solution->dft_bufs->inplace_buffer;
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->inplace_ndim_buffer =
            sel->solution->dft_bufs->inplace_ndim_buffer;

        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        sel_fp = selector_fixed_mode_dft_;
        ret = selector_model_dft_(sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR]->cost_analysis) =
                *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // FIXED SELECTOR BASED MODEL : end

    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT_MODE
    sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] != NULL)
    {
        COPY_DECOMP_SCHEME(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                               ->solution->decomp_scheme,
                           sel->solution->decomp_scheme);
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                          ->solution->decomp_scheme->flags,
                      DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                ->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));
        sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
            ->solution->dft_bufs->ct_buffer =
            sel->solution->dft_bufs->ct_buffer;
        sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
            ->solution->dft_bufs->ct_buf_real =
            sel->solution->dft_bufs->ct_buf_real;
        sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
            ->solution->dft_bufs->ct_buf_imag =
            sel->solution->dft_bufs->ct_buf_imag;
        sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
            ->solution->decomp_scheme->decomp_level =
            sel->solution->decomp_scheme->decomp_level;
        sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
            ->solution->dft_bufs->ct_buf_size =
            sel->solution->dft_bufs->ct_buf_size;

        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_fused_twid_dft_(
        //     sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        sel_fp = selector_fixed_mode_fused_twid_dft_;
        ret = selector_model_dft_(
            sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                  ->cost_analysis) = *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : end

    // AUTO TUNED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
    sel_models[AOCLFFTZ_AUTO_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_AUTO_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));
        sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->dft_bufs->ct_buffer =
            sel->solution->dft_bufs->ct_buffer;
        sel_models[AOCLFFTZ_AUTO_SELECTOR]
            ->solution->dft_bufs->ct_buf_real =
            sel->solution->dft_bufs->ct_buf_real;
        sel_models[AOCLFFTZ_AUTO_SELECTOR]
            ->solution->dft_bufs->ct_buf_imag =
            sel->solution->dft_bufs->ct_buf_imag;
        sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->decomp_level =
            sel->solution->decomp_scheme->decomp_level;
        sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->dft_bufs->ct_buf_size =
            sel->solution->dft_bufs->ct_buf_size;

        // Fixed decision logic and CPI based selector mode
        // ret = selector_autotuner_mode_dft_(
        //         sel_models[AOCLFFTZ_AUTO_SELECTOR]);
        sel_fp = selector_autotuner_mode_dft_;
        ret = selector_model_dft_(sel_models[AOCLFFTZ_AUTO_SELECTOR]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_AUTO_SELECTOR]->cost_analysis) =
                *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // AUTO TUNED SELECTOR BASED MODEL : end

    // Compare and find the best performant model for the input problem
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (sel_models[model_id] != NULL)
        {
            if (sel_models[model_id]->cost_analysis->ops < best_cost.ops)
            {
                best_cost = *(sel_models[model_id]->cost_analysis);
                best_model_id = model_id;
            }
        }
    }

    // Update the primary selector object with the best selector model based
    // solution
    aoclfftz_solution_t *org_sol = sel->solution;
    sel->solution = sel_models[best_model_id]->solution;
    *(sel->cost_analysis) = *(sel_models[best_model_id]->cost_analysis);

    // Destroy and Free unnecessary objects
    destroy_solution(org_sol, 0 /*destroy_buffers*/);
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (model_id != best_model_id)
            destroy_selector_without_scratch_space(sel_models[model_id]);
        else
            destroy_selector_without_solution(sel_models[model_id]);
    }



    return ret;
}

// Specific Selector model function that invokes the internal recursive
// selector function pointer passing the selector object along.
// Common function for both single-precision and double-precision
INT32 selector_model_dft_(aoclfftz_selector_t *sel)
{
    return sel_fp(sel);
}

// Main selector driver that invokes the complementary/alternate selector
// algorithms/models and decides on the final selector based on its suitability
// and performance for real-fft problems.
// Also provides a cleaner approach to init and handle various related
// solvers and kernel tables
INT32 selector_driver_rdft_(aoclfftz_selector_t *sel,
                            aoclfftz_realhelper_t *realhelper)
{
    INT32 ret = SELECTOR_FAILURE;

    // if bit reproducibility is requested, directly take its associated
    // code path and return
    if (GET_BIT_REPRODUCIBLE(sel->solution->decomp_scheme->flags))
    {
        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_fused_twid_rdft_(
        //         sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        // TODO: Enable twiddle kernels for C2R problems
        if (FFT_DIR(sel->solution->decomp_scheme->flags) ==
            BACKWARD_FFT_DIR)
        {
            AOCLFFTZ_LOG(
                INFO,
                global_logger_mode,
                "Twiddle kernels are not supported for C2R problems, so "
                "using non-twiddle kernels + twiddle multiplier approach.");

            sel_rdft_fp = selector_fixed_mode_rdft_;
        }
        else
        {
            sel_rdft_fp = selector_fixed_mode_fused_twid_rdft_;
        }
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_fixed_mode_fused_twid_dft_;

        return selector_model_rdft_(sel, realhelper);
    }

    aoclfftz_selector_t *sel_models[AOCLFFTZ_SELECTOR_MODELS] = { 0x0, };
    UINT32 best_model_id = 0;
    cost_analysis_t best_cost = {INT64_MAX, INT64_MAX};
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;

    // FIXED SELECTOR BASED MODEL : start
#ifdef AOCLFFTZ_FIXED_SELECTOR_MODE
    // Allocate selector object
    sel_models[AOCLFFTZ_FIXED_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_rdft_(sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        sel_rdft_fp = selector_fixed_mode_rdft_;
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_fixed_mode_dft_;
        ret = selector_model_rdft_(sel_models[AOCLFFTZ_FIXED_SELECTOR],
                                   realhelper);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR]->cost_analysis) =
                *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // FIXED SELECTOR BASED MODEL : end

    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT_MODE
    sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT] != NULL)
    {
        COPY_DECOMP_SCHEME(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                               ->solution->decomp_scheme,
                           sel->solution->decomp_scheme);
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                          ->solution->decomp_scheme->flags,
                      DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                ->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_fused_twid_rdft_(
        //         sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]);
        // TODO: Enable twiddle kernels for C2R problems
        if (FFT_DIR(sel->solution->decomp_scheme->flags) == BACKWARD_FFT_DIR)
        {
            AOCLFFTZ_LOG(
                INFO, global_logger_mode,
                "Twiddle kernels are not supported for C2R problems, so using "
                "non-twiddle kernels + twiddle multiplier approach.");
            sel_rdft_fp = selector_fixed_mode_rdft_;
        }
        else
        {
            sel_rdft_fp = selector_fixed_mode_fused_twid_rdft_;
        }
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_fixed_mode_fused_twid_dft_;
        ret = selector_model_rdft_(
            sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT], realhelper);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR_FUSED_TWID_DFT]
                  ->cost_analysis) = *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // FIXED SELECTOR + FUSED TWIDDLE DFT BASED MODEL : end

    // AUTO TUNED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
    sel_models[AOCLFFTZ_AUTO_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                       sel->kernel_tables, 0 /*unused*/);
    if (sel_models[AOCLFFTZ_AUTO_SELECTOR] != NULL)
    {
        COPY_DECOMP_SCHEME(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        SET_PRECISION(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // TODO: Autotuner mode for RDFT is not implemented yet
        AOCLFFTZ_LOG(
            INFO, global_logger_mode,
            "Autotuner selector mode for RDFT is not implemented yet");
        // ret = selector_autotuner_mode_rdft_(
        //     sel_models[AOCLFFTZ_AUTO_SELECTOR]);
        sel_rdft_fp = selector_autotuner_mode_rdft_;
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_autotuner_mode_dft_;
        ret = selector_model_rdft_(sel_models[AOCLFFTZ_AUTO_SELECTOR],
                                   realhelper);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_AUTO_SELECTOR]->cost_analysis) =
                *(sel->cost_analysis);
        }
    }
    else
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return ret;
    }
#endif
    // AUTO TUNED SELECTOR BASED MODEL : end

    // Compare and find the best performant model for the input problem
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (sel_models[model_id] != NULL)
        {
            if (sel_models[model_id]->cost_analysis->ops < best_cost.ops)
            {
                best_cost = *(sel_models[model_id]->cost_analysis);
                best_model_id = model_id;
            }
        }
    }
    // Update the primary selector object with the best selector model based
    // solution
    aoclfftz_solution_t *org_sol = sel->solution;
    sel->solution = sel_models[best_model_id]->solution;
    *(sel->cost_analysis) = *(sel_models[best_model_id]->cost_analysis);

    // Destroy and Free unnecessary objects
    destroy_solution(org_sol, 0 /*destroy_buffers*/);
    for (UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS; model_id++)
    {
        if (model_id != best_model_id)
            destroy_selector_without_scratch_space(sel_models[model_id]);
        else
            destroy_selector_without_solution(sel_models[model_id]);
    }

    return ret;
}

// Specific Selector model function that invokes the internal recursive
// selector function pointer (for real-fft) passing the selector object along.
// Common function for both single-precision and double-precision
INT32 selector_model_rdft_(aoclfftz_selector_t *sel,
                           aoclfftz_realhelper_t *realhelper)
{
    INT32 ret = SELECTOR_FAILURE;

    ret = sel_rdft_fp(sel, realhelper);

    return ret;
}

// Selector interface function that performs setup for finding solution for a
// single-precision LP64 problem
VOID *setup_dft_f(aoclfftz_prob_desc_f *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    INT32 num_threads = problem->pthr_fft.num_threads;
    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL, &kertab_tables,
                             num_threads);
    if (sel_obj == NULL)
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return NULL;
    }

    // Determine CPU optimization level to be used by the dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level);
    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU opt level and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(sel_obj->kernel_tables, DT_FLOAT,
                                   flags.fft_direction, flags.fft_type,
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_f;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    // Note: Currently the 8th bit of the flags member of both, the problem and
    // the selector represent the same thing -> a standalone transpose
    // operation. Hence the reuse of the same macro is valid. If this no longer
    // holds for whatever reason, change this "GET_STANDALONE_TRANSPOSE" macro
    // accordingly.
    if (flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        if (ret == SELECTOR_FAILURE)
        {
            ret = AOCLFFTZ_SETUP_FAILURE;
        }
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(ret));
        goto exit_setup_dft_f;
    }
#ifdef MULTI_THREADING
    UINT32 scratch_buf_idx = 0;
    UINT32 ct_buf_idx = 0;
    UINT32 num_ct_buf = 0;
    post_process_solution(sel_obj->solution, &scratch_buf_idx, &ct_buf_idx,
                          &num_ct_buf);
#endif
    return sel_obj;

exit_setup_dft_f:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision LP64 problem
VOID *setup_dft_d(aoclfftz_prob_desc_d *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    INT32 num_threads = problem->pthr_fft.num_threads;
    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL, &kertab_tables,
                             num_threads);
    if (sel_obj == NULL)
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return NULL;
    }

    // Determine CPU optimization level to be used by the dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level);

    // Register solvers and kernels for solving the problem based on
    // input problem datatype, CPU opt level and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(sel_obj->kernel_tables, DT_DOUBLE,
                                   flags.fft_direction, flags.fft_type,
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_d;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_DOUBLE);

    if (problem->flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        if (ret == SELECTOR_FAILURE)
        {
            ret = AOCLFFTZ_SETUP_FAILURE;
        }
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(ret));
        goto exit_setup_dft_d;
    }
#ifdef MULTI_THREADING
    UINT32 scratch_buf_idx = 0;
    UINT32 ct_buf_idx = 0;
    UINT32 num_ct_buf = 0;
    post_process_solution(sel_obj->solution, &scratch_buf_idx, &ct_buf_idx,
                          &num_ct_buf);
#endif
    return sel_obj;

exit_setup_dft_d:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// single-precision ILP64 problem
VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    INT32 num_threads = problem->pthr_fft.num_threads;
    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL, &kertab_tables,
                             num_threads);
    if (sel_obj == NULL)
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return NULL;
    }

    // Determine CPU optimization level to be used by the dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU opt level and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(sel_obj->kernel_tables, DT_FLOAT,
                                   flags.fft_direction, flags.fft_type,
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_f_64_;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_FLOAT);

    if (flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        if (ret == SELECTOR_FAILURE)
        {
            ret = AOCLFFTZ_SETUP_FAILURE;
        }
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(ret));
        goto exit_setup_dft_f_64_;
    }
#ifdef MULTI_THREADING
    UINT32 scratch_buf_idx = 0;
    UINT32 ct_buf_idx = 0;
    UINT32 num_ct_buf = 0;
    post_process_solution(sel_obj->solution, &scratch_buf_idx, &ct_buf_idx,
                          &num_ct_buf);
#endif
    return sel_obj;

exit_setup_dft_f_64_:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision ILP64 problem
VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    INT32 ret = 0;
    INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    INT32 num_threads = problem->pthr_fft.num_threads;
    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {{0x0}};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, NULL, &kertab_tables,
                             num_threads);
    if (sel_obj == NULL)
    {
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return NULL;
    }

    // Determine CPU optimization level to be used by the dynamic dispatcher
    cpu_flags = setup_dynamic_dispatcher(cntrl_params.opt_off,
                                         cntrl_params.opt_level);

    //Register solvers and kernels for solving the problem based on
    //input problem datatype, CPU opt level and dynamic dispatcher FMV selection
    ret = register_solvers_kernels(sel_obj->kernel_tables, DT_DOUBLE,
                                   flags.fft_direction, flags.fft_type,
                                   cpu_flags);
    if (ret != 0)
    {
        goto exit_setup_dft_d_64_;
    }

    // Initialize decomposition scheme data object
    INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank);
    SET_PRECISION(sel_obj->solution->decomp_scheme->flags, DT_DOUBLE);

    if (flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    PREPARE_AND_SETUP_DFT(sel_obj, ret);
    if (ret != SELECTOR_SUCCESS)
    {
        if (ret == SELECTOR_FAILURE)
        {
            ret = AOCLFFTZ_SETUP_FAILURE;
        }
        AOCLFFTZ_ERROR("Setup failure with %s",
                                  get_status_string(ret));
        goto exit_setup_dft_d_64_;
    }
#ifdef MULTI_THREADING
    UINT32 scratch_buf_idx = 0;
    UINT32 ct_buf_idx = 0;
    UINT32 num_ct_buf = 0;
    post_process_solution(sel_obj->solution, &scratch_buf_idx, &ct_buf_idx,
                          &num_ct_buf);
#endif
    return sel_obj;

exit_setup_dft_d_64_:
    destroy_selector(sel_obj);
    return NULL;
}

VOID destroy_handle(VOID *handle)
{
    destroy_selector((aoclfftz_selector_t *)handle);
}

VOID fuse_vecs(aoclfftz_solution_t *sol)
{
    INT32 last_fused_idx = 0, is_fusable = 0, fused_rank = 0;
    INTP fused_size = sol->decomp_scheme->vecs[0].n;
    INT32 vec_rank = sol->decomp_scheme->vec_rank;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;
    for (INT32 i = 1; i < vec_rank; i++)
    {
        // expected stride is the regular stride we obtain by n * stride of
        // prev dim
        INTP expected_in_stride = vecs[i-1].n * vecs[i-1].in_stride;
        INTP expected_out_stride = vecs[i-1].n * vecs[i-1].out_stride;
        INTP actual_in_stride = vecs[i].in_stride;
        INTP actual_out_stride = vecs[i].out_stride;
        // Do not fuse the first vector
        if ((expected_in_stride == actual_in_stride &&
             expected_out_stride == actual_out_stride) && i != 1)
        {
            // mark the vector for fusing and compute the new size for the fused
            // vector
            is_fusable = 1;
            fused_size = fused_size * vecs[i].n;
        }
        else
        {
            if (is_fusable)
            {
                vecs[fused_rank].n = fused_size;
            }
            else
            {
                vecs[fused_rank].n = vecs[last_fused_idx].n;
            }
            vecs[fused_rank].in_stride = vecs[last_fused_idx].in_stride;
            vecs[fused_rank].out_stride = vecs[last_fused_idx].out_stride;
            fused_size = vecs[i].n;
            last_fused_idx = i;
            fused_rank++;
            is_fusable = 0;
        }
    }
    //initalize last vector
    vecs[fused_rank].n = fused_size;
    vecs[fused_rank].out_stride = vecs[last_fused_idx].out_stride;
    vecs[fused_rank].in_stride = vecs[last_fused_idx].in_stride;
    sol->decomp_scheme->vec_rank = fused_rank + 1;
}

VOID setup_twiddle_buffer_complex(aoclfftz_solution_t *solution)
{
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    aoclfftz_solution_t *curr = solution;
    UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
    aoclfftz_solution_t *nd_sol = NULL;
    do // for ND support
    {
        if (nd_sol != NULL)
        {
            curr = nd_sol;
            nd_sol = NULL;
        }
        while (curr != NULL)
        {
            if (curr->solver->solver_type == SOLVER_CT ||
                curr->solver->solver_type == SOLVER_CT_TWIDDLE)
            {
                INTP r = curr->next_sol[0]->decomp_scheme->dims[0].n;
                INTP m =
                    curr->next_sol[0]->next_sol[0]->decomp_scheme->dims[0].n;

                VOID *TW = alloc_twiddle_buffer(r * m, dt_prec);
                if (TW != NULL)
                {
                    compute_twiddle_buffer(TW, r, m, dt_prec);
                    curr->next_sol[0]->twiddle->cols = m;
                    curr->next_sol[0]->twiddle->TW = TW;
                    curr->next_sol[0]->twiddle->twiddle_buf_ptr = TW;
                }
            }
            else if (curr->solver->solver_type == SOLVER_SR)
            {
                INTP n = curr->decomp_scheme->dims[0].n;
                // Number of twiddle pairs(W^k, W^(3k)) for the split-radix twiddle multiplication
                INTP sr_tw_pairs = n / 4;

                // Allocate twiddle buffer for split-radix: N/4 complex pairs (W^k and W^(3k))
                VOID *TW = alloc_twiddle_buffer(sr_tw_pairs * 2, dt_prec);
                if (TW != NULL)
                {
                    compute_sr_twiddle_buffer(TW, n, dt_prec);
                    curr->twiddle->cols = sr_tw_pairs;
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                }

                // Recursively set up twiddles for ALL 3 sub-problems
                // Note: next_sol[0] (even) will be handled by the loop continuation
                // So we only need to explicitly recurse for odd1 and odd3 from dft_bufs
                if (curr->dft_bufs && curr->dft_bufs->sr->odd1_sol
                    && curr->dft_bufs->sr->odd3_sol)
                {
                    setup_twiddle_buffer_complex(curr->dft_bufs->sr->odd1_sol);
                    setup_twiddle_buffer_complex(curr->dft_bufs->sr->odd3_sol);
                }
            }
            else if (curr->solver->solver_type == SOLVER_BATCHED_CT_L1_DIRECT)
            {
                INTP r = (INTP)curr->solver->kernel_c2c->count;
                INTP m = (INTP)curr->solver->kernel_c2c_r->count;

                VOID *TW = alloc_twiddle_buffer(r * m, dt_prec);
                if (TW != NULL)
                {
                    compute_twiddle_buffer(TW, r, m, dt_prec);
                    curr->twiddle->cols = m;
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                }
            }
            // Process N-D solution after the current solution
            if (curr->solver->solver_type == SOLVER_NDIM)
            {
                nd_sol = curr->dft_bufs->nd_sol;
            }
            curr = curr->next_sol ? curr->next_sol[0] : NULL;
        }
    } while (nd_sol != NULL);
#endif
}

VOID setup_twiddle_buffer_real(aoclfftz_solution_t *solution)
{
    if (solution == NULL)
    {
        return;
    }
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
    if (FFT_DIR(solution->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        // aoclfftz_solution_t *curr = solution;
        aoclfftz_solution_t *prev = solution;
        FOR_EACH_SOLUTION(curr, solution)
        {
            // Always inherit the parent's twiddle buffer reference.
            curr->twiddle->TW = prev->twiddle->TW;
            if (curr->solver->solver_type == SOLVER_REAL_CT)
            {
                // goto next direct node to setup twiddle buffer
                while (
                    curr != NULL &&
                    curr->solver->solver_type != SOLVER_REAL_DIRECT &&
                    curr->solver->solver_type != SOLVER_REAL_DIRECT_TWIDDLE &&
                    curr->solver->solver_type != SOLVER_REAL_MT_DIRECT &&
                    curr->solver->solver_type != SOLVER_REAL_MT_DIRECT_TWIDDLE)
                {
                    curr->twiddle->TW = prev->twiddle->TW;
                    curr = curr->next_sol[0];
                }
                if (curr == NULL)
                {
                    return;
                }
                INTP radix = curr->decomp_scheme->dims[0].n;
                INTP num_groups = NUM_RFFT_GROUPS(curr->solver);
                INTP num_c2c_per_group = curr->solver->kernel_c2c->count / num_groups;
                INTP tw_buf_size =
                    radix * num_c2c_per_group * DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle values for every
                // radix-n c2c kernel of a group
                VOID *TW = alloc_twiddle_buffer(tw_buf_size, dt_prec);
                if (TW != NULL)
                {
                    INTP p = (curr->decomp_scheme->vecs[0].n *
                              curr->decomp_scheme->dims[0].n) /
                             num_groups;
                    compute_twiddle_buffer_real(TW, radix, num_c2c_per_group,
                                                num_groups, p,
                                                FORWARD_FFT_DIR, dt_prec);
                    curr->twiddle->cols = num_c2c_per_group;
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                }
            }
            prev = curr;
        }
    }
    else
    {
        aoclfftz_solution_t *prev = solution;
        FOR_EACH_SOLUTION(curr,
                          (solution->next_sol) ? solution->next_sol[0] : NULL)
        {
            if (curr->solver->solver_type == SOLVER_REAL_CT &&
                (prev->solver->solver_type == SOLVER_REAL_DIRECT ||
                 prev->solver->solver_type == SOLVER_REAL_DIRECT_TWIDDLE ||
                 prev->solver->solver_type == SOLVER_REAL_MT_DIRECT ||
                 prev->solver->solver_type == SOLVER_REAL_MT_DIRECT_TWIDDLE))
            {
                INTP radix = prev->decomp_scheme->dims[0].n;
                INTP num_groups = NUM_RFFT_GROUPS(prev->solver);
                // No. of c2c kernels per group that require twiddle computation
                INTP num_c2c_per_group =
                    prev->solver->kernel_c2c->count / num_groups;
                INTP tw_buf_sz = radix * num_c2c_per_group * DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle factors for every
                // radix-n c2c kernel of a group
                VOID *TW = alloc_twiddle_buffer(tw_buf_sz, dt_prec);
                if (TW != NULL)
                {
                    INTP p = (prev->decomp_scheme->vecs[0].n *
                              prev->decomp_scheme->dims[0].n) /
                             num_groups;
                    compute_twiddle_buffer_real(TW, radix,
                                                num_c2c_per_group,
                                                num_groups, p,
                                                BACKWARD_FFT_DIR, dt_prec);
                    prev->twiddle->cols = num_c2c_per_group;
                    prev->twiddle->TW = TW;
                    prev->twiddle->twiddle_buf_ptr = TW;
                }
            }
            // Always inherit the parent's twiddle buffer reference.
            curr->twiddle->TW = prev->twiddle->TW;
            prev = curr;
        }
    }
#endif
}

/**
 * @brief Deep-copy a solution subtree for a specific thread.
 *
 * Recursively copies each node, assigning unique scratch and ct_buf
 * slices via scratch_buf_idx and ct_buf_idx.  MT_BATCHED nodes replicate
 * next_sol[0] across thread slots, advancing both counters. nd_sol and
 * next_sol share indices (saved/restored around the nd_sol copy).
 *
 * BATCHED_CT_L1_DIRECT nodes that own their ct_buffer get a fresh allocation per
 * copy; those reusing a parent buffer share it via ct_buf_idx offsets.
 */
aoclfftz_solution_t *deep_copy_solution_tree(aoclfftz_solution_t *src,
                                             UINT32 *scratch_buf_idx,
                                             UINT32 *ct_buf_idx)
{
    if (!src)
    {
        return NULL;
    }

    // Copy the solution object and its strides
    INT32 vec_rank = src->decomp_scheme->vec_rank;
    INT32 dim_rank = src->decomp_scheme->dim_rank;
    aoclfftz_solution_t *dst = alloc_solution(vec_rank, dim_rank);
    COPY_SOLUTION_OBJ(dst, src);
    if (src->solver->solver_type == SOLVER_BATCHED_CT_L1_DIRECT)
    {
        COPY_STRIDES_BATCHED_CT_L1_DIRECT(dst, src);
    }
    else
    {
        COPY_STRIDES(dst, src);
    }
    dst->dft_bufs->nd_sol = NULL;

    // Assign relevant scratch space to each thread
    dst->dft_bufs->scratch_space = MOVE_ADDR(src->dft_bufs->scratch_space,
                                (*scratch_buf_idx) * scratch_space_capacity);

    // If the src BATCHED_CT_L1_DIRECT node owns its ct_buffer (ct_buf_allocated),
    // allocate a separate copy for dst and mark it for cleanup.
    // Otherwise, assign ct_buf pointers offset by ct_buf_idx.
    if (src->solver->solver_type == SOLVER_BATCHED_CT_L1_DIRECT &&
        src->dft_bufs->ct_buf_allocated)
    {
        // mark ct_buf_allocated so destroy_solution frees it correctly.
        dst->dft_bufs->ct_buf_allocated = 1;
        ALLOC_ALIGN_UNINIT(dst->dft_bufs->ct_buffer, VOID, src->dft_bufs->ct_buf_size);
        dst->dft_bufs->ct_buf_real = dst->dft_bufs->ct_buffer;
        UINT32 dt_bytes = SOL_DT_SIZE(src);
        dst->dft_bufs->ct_buf_imag = MOVE_ADDR(dst->dft_bufs->ct_buffer, dt_bytes);
    }
    else
    {
        dst->dft_bufs->ct_buf_real = MOVE_ADDR(src->dft_bufs->ct_buf_real,
                                       (*ct_buf_idx) * src->dft_bufs->ct_buf_size);
        dst->dft_bufs->ct_buf_imag = MOVE_ADDR(src->dft_bufs->ct_buf_imag,
                                   (*ct_buf_idx) * src->dft_bufs->ct_buf_size);
    }

    // Hold the current scratch buffer index and restore after copy of each ND-subtree
    // as both nd_sol and next_sol of ND node share the same scratch space
    UINT32 temp = *scratch_buf_idx;
    UINT32 temp_ct_idx = *ct_buf_idx;
    // Recursively copy nd_sol if present (for NDIM solvers)
    if (src->dft_bufs->nd_sol)
    {
        dst->dft_bufs->nd_sol = deep_copy_solution_tree(
                            src->dft_bufs->nd_sol, scratch_buf_idx, ct_buf_idx);
        *scratch_buf_idx = temp;
        *ct_buf_idx = temp_ct_idx;
    }

    // Recursively copy sr odd1_sol and odd3_sol if present (for SR solvers)
    if (src->dft_bufs->sr->odd1_sol)
    {
           dst->dft_bufs->sr->odd1_sol = deep_copy_solution_tree(
                               src->dft_bufs->sr->odd1_sol,
                               scratch_buf_idx, ct_buf_idx);
           *scratch_buf_idx = temp;
           *ct_buf_idx = temp_ct_idx;
    }
    if (src->dft_bufs->sr->odd3_sol)
    {
           dst->dft_bufs->sr->odd3_sol = deep_copy_solution_tree(
                               src->dft_bufs->sr->odd3_sol,
                               scratch_buf_idx, ct_buf_idx);
           *scratch_buf_idx = temp;
           *ct_buf_idx = temp_ct_idx;
    }

    // Copy SR input copy buffer for multi-threading safety
    if (src->dft_bufs->sr->input_copy != NULL)
    {
        INTP sr_input_copy_size = src->dft_bufs->sr->input_copy_size;

        ALLOC_ALIGN_UNINIT(dst->dft_bufs->sr->input_copy, VOID, sr_input_copy_size);
        if (dst->dft_bufs->sr->input_copy == NULL)
        {
            /* SR executor will memcpy into sr->input_copy for in-place
             * transforms; a NULL buffer would cause a crash.
             * Propagate the failure so the caller can handle it. */
            destroy_solution(dst, 1);
            return NULL;
        }
        dst->dft_bufs->sr->input_copy_size = sr_input_copy_size;
    }

    // Initiate deep copy of next_sol recursively until leaf node
    if (src->next_sol)
    {
        INT32 n = (src->solver->solver_type == SOLVER_MT_BATCHED ||
                    src->solver->solver_type == SOLVER_REAL_MT_BATCHED) ?
                    src->decomp_scheme->thread_info->n_threads : 1;
        dst->next_sol = alloc_sol_array(n);

        // For MT_BATCHED: need to count ct_bufs in subtree to properly offset
        // First copy gets current ct_buf_idx, subsequent copies need incremented indices
        if (n > 1)
        {
            // Save base ct_buf_idx for this MT_BATCHED
            UINT32 base_ct_buf_idx = *ct_buf_idx;

            // First copy - also counts how many ct_bufs are in subtree
            dst->next_sol[0] = deep_copy_solution_tree(
                src->next_sol[0], scratch_buf_idx, ct_buf_idx);

            // Calculate ct_bufs_per_thread based on how much ct_buf_idx
            // advanced. Note: nd_sol's ct_buf usage is not counted here
            // because ct_buf_idx is reset after nd_sol processing (nd_sol
            // and next_sol share indices since they run sequentially).
            // For nested NDIM cases (e.g., 3D problems), this can result
            // in ct_bufs_used = 0 even when nd_sol used buffers, so we
            // default to 1 to ensure unique indices for each thread.
            UINT32 ct_bufs_used = (*ct_buf_idx > base_ct_buf_idx)
                                    ? (*ct_buf_idx - base_ct_buf_idx + 1) : 1;

            for (INT32 i = 1; i < n; i++)
            {
                (*scratch_buf_idx)++;
                // Each subsequent thread gets next contiguous ct_buf index
                *ct_buf_idx = base_ct_buf_idx + (i * ct_bufs_used);
                dst->next_sol[i] = deep_copy_solution_tree(
                    src->next_sol[0], scratch_buf_idx, ct_buf_idx);
            }
        }
        else
        {
            dst->next_sol[0] = deep_copy_solution_tree(
                src->next_sol[0], scratch_buf_idx, ct_buf_idx);
        }
    }
    else
    {
        dst->next_sol = NULL;
    }
    return dst;
}

/**
 * @brief Post-processes FFT solution trees for multi-threaded execution
 *
 * This function traverses the solution tree and performs post-processing
 * for multi-threaded (MT_BATCHED and REAL_MT_BATCHED) solvers:
 *
 * 1. Identifies MT_BATCHED solutions in the tree
 * 2. Recursively processes nested solution trees to find innermost MT_BATCHED nodes
 * 3. Replicates the primary solution (next_sol[0]) across all thread slots
 * 4. Assigns unique scratch buffer indices to each thread's solution copy
 * 5. Maintains same scratch buffer for both nd-sol and next_sol of a single N-D node
 * 6. Handles both N-dimensional (nd_sol) and sequential (next_sol) solution paths
 *
 * The function ensures thread-safe execution by providing each thread with:
 * - Its own deep copy of the solution tree
 * - Unique scratch space allocation indexed by scratch_buf_idx
 * - Proper memory isolation to prevent race conditions
 *
 * @param sol             Pointer to the root solution tree to post-process
 * @param scratch_buf_idx Pointer to scratch buffer index counter for unique allocation
 * @param ct_buf_idx      Pointer to CT (Cooley-Tukey) buffer index counter.
 *                        Used to assign unique auxiliary buffer indices to each
 *                        thread's Buffered solver nodes.
 * @param num_ct_buf      Pointer to counter tracking total number of CT buffers
 *                        used in the solution subtree. Updated during traversal
 *                        to account for buffer requirements of MT_BATCHED nodes.
 *
 * @note The scratch_buf_idx is incremented for each thread to ensure unique scratch
 *       space allocation across the entire solution hierarchy.
 *
 * @example
 * Consider a batched 2D problem's solution tree before post-processing with 4 threads:
 * ```
 * Input Tree:
 * MT_BATCHED(2) -> next_sol[0](N-Dim) -> MT_BATCHED(2) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 0)
 *       |                    |                 |--------> next_sol[1](NULL)
 *       |                    |
 *       |                    V
 *       |             nd_sol(MT_BATCHED(2)) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 0)
 *       |                           |--------> next_sol[1](NULL)
 *       |
 *       |--------> next_sol[1](NULL)
 *
 * Initial state: scratch_buf_idx = 0
 * ```
 * After post_process_solution():
 * ```
 * Output Tree (Fully expanded for all threads):
 * MT_BATCHED(2) -> next_sol[0](N-Dim) -> MT_BATCHED(2) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 0)
 *       |                    |                 |--------> next_sol[1] -> CT -> Direct -> Direct (scratch_idx: 1)
 *       |                    |
 *       |                    V
 *       |             nd_sol(MT_BATCHED(2)) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 0)
 *       |                           |--------> next_sol[1] -> CT -> Direct -> Direct (scratch_idx: 1)
 *       |
 *       |--------> next_sol[1](N-Dim) -> MT_BATCHED(2) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 2)
 *                            |                 |--------> next_sol[1] -> CT -> Direct -> Direct (scratch_idx: 3)
 *                            |
 *                            V
 *                     nd_sol(MT_BATCHED(2)) -> next_sol[0] -> CT -> Direct -> Direct (scratch_idx: 2)
 *                                   |--------> next_sol[1] -> CT -> Direct -> Direct (scratch_idx: 4)
 */

VOID post_process_solution(aoclfftz_solution_t *sol, UINT32 *scratch_buf_idx,
                           UINT32 *ct_buf_idx, UINT32 *num_ct_buf)
{
    // Track max ct_buf count from nd_sol paths (nd_sol and next_sol share indices)
    UINT32 max_nd_sol_ct_count = 0;

    while (sol != NULL)
    {
        if ((sol->solver->solver_type == SOLVER_MT_BATCHED) ||
            (sol->solver->solver_type == SOLVER_REAL_MT_BATCHED))
        {
            // Save base ct_buf_idx before processing subtree
            UINT32 base_ct_buf_idx = *ct_buf_idx;

            // Save current num_ct_buf and reset to count buffers in subtree
            UINT32 saved_num_ct_buf = *num_ct_buf;
            *num_ct_buf = 0;

            // Process first thread's subtree to count ct buffers needed per thread
            post_process_solution(sol->next_sol[0], scratch_buf_idx, ct_buf_idx, num_ct_buf);

            // num_ct_buf now contains count of ct buffers used by one thread's subtree
            UINT32 ct_bufs_per_thread = *num_ct_buf;

            INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
            // Replicate the solution in next_sol[0] to array of next_sols in
            // each MT batched solution
            for (INT32 i = 1; i < n_threads; i++)
            {
                // Increment the scratch buffer index for MT batched solutions
                (*scratch_buf_idx)++;
                // Each thread gets contiguous ct_buf indices starting from base
                // Thread 0: base_idx + 0 * ct_bufs_per_thread
                // Thread 1: base_idx + 1 * ct_bufs_per_thread
                // Thread i: base_idx + i * ct_bufs_per_thread
                *ct_buf_idx = base_ct_buf_idx + (i * ct_bufs_per_thread);
                sol->next_sol[i] = deep_copy_solution_tree(sol->next_sol[0],
                                                           scratch_buf_idx, ct_buf_idx);
            }

            // Update ct_buf_idx to point past all used indices
            *ct_buf_idx = base_ct_buf_idx + (n_threads * ct_bufs_per_thread) - 1;

            // Total ct buffers used by this MT = n_threads * ct_bufs_per_thread
            // Update num_ct_buf to reflect total for parent's accounting
            *num_ct_buf = n_threads * ct_bufs_per_thread;

            // Add back any buffers counted before this MT
            *num_ct_buf += saved_num_ct_buf;
            break;
        }
        // Hold the current scratch buffer index and restore after copy of each ND-subtree
        // as both nd_sol and next_sol of ND node share the same scratch space and ct buffers
        UINT32 temp = *scratch_buf_idx;
        UINT32 temp1 = *ct_buf_idx;
        if (sol->dft_bufs->nd_sol)
        {
            // Process nd_sol separately - it may have MT_BATCHED with its own buffer needs
            // nd_sol and next_sol share indices (they run sequentially), so track max
            UINT32 nd_sol_ct_count = 0;
            post_process_solution(sol->dft_bufs->nd_sol, scratch_buf_idx, ct_buf_idx, &nd_sol_ct_count);
            *scratch_buf_idx = temp;
            *ct_buf_idx = temp1;

            // Track maximum across all nd_sol paths
            if (nd_sol_ct_count > max_nd_sol_ct_count)
            {
                max_nd_sol_ct_count = nd_sol_ct_count;
            }
        }

        // Count nodes that share the parent's ct_buffer via the global
        // ct_buf_idx scheme: BUFFERED always, and BATCHED_CT_L1_DIRECT when it
        // reuses the parent's buffer instead of allocating its own.
        if (sol->solver->solver_type == SOLVER_BUFFERED ||
            (sol->solver->solver_type == SOLVER_BATCHED_CT_L1_DIRECT &&
             !sol->dft_bufs->ct_buf_allocated))
        {
            (*num_ct_buf)++;
        }

        // Process SR children if present (share scratch space like nd_sol)
        if (sol->dft_bufs->sr->odd1_sol)
        {
            UINT32 sr_odd1_ct_count = 0;
            post_process_solution(sol->dft_bufs->sr->odd1_sol, scratch_buf_idx, ct_buf_idx, &sr_odd1_ct_count);
            *scratch_buf_idx = temp;
            *ct_buf_idx = temp1;
            if (sr_odd1_ct_count > max_nd_sol_ct_count)
            {
                max_nd_sol_ct_count = sr_odd1_ct_count;
            }
        }
        if (sol->dft_bufs->sr->odd3_sol)
        {
            UINT32 sr_odd3_ct_count = 0;
            post_process_solution(sol->dft_bufs->sr->odd3_sol, scratch_buf_idx, ct_buf_idx, &sr_odd3_ct_count);
            *scratch_buf_idx = temp;
            *ct_buf_idx = temp1;
            if (sr_odd3_ct_count > max_nd_sol_ct_count)
            {
                max_nd_sol_ct_count = sr_odd3_ct_count;
            }
        }

        sol = sol->next_sol ? sol->next_sol[0] : NULL;
    }

    // After traversing next_sol path, take max with nd_sol paths
    // since nd_sol and next_sol share buffer indices (sequential execution)
    if (max_nd_sol_ct_count > *num_ct_buf)
    {
        *num_ct_buf = max_nd_sol_ct_count;
    }
}

/**
 * @brief Check if col-major processing MT batched solver should be used based
 * on workload and stride heuristics
 *
 * This function implements a heuristic to choose between row-major and col-major
 * processing for multi-threaded batched solvers by analyzing workload distribution
 * and memory access patterns to avoid performance degradation from false sharing.
 *
 * @param solution Pointer to the solution object
 * @param kertab Kernel table
 * @param avl_threads Available threads count
 * @return UINT8 1 if col-major should be used, 0 otherwise
 */
UINT8 should_use_colmajor_batched_solver(aoclfftz_solution_t *solution,
                                                kernel_t *kertab,
                                                INT32 avl_threads)
{
    INTP radix = solution->decomp_scheme->dims[0].n;
    DOUBLE kernel_weightage = get_kernel_weightage(radix, kertab, solution);
    // Compute workload distribution across available threads
    // Formula: (batch_count * CT vector_size * computational_intensity of the kernel) / thread_count
    DOUBLE workload_per_thread =
        (DOUBLE)(solution->decomp_scheme->batched_vecs[0].n *
                 solution->decomp_scheme->vecs[0].n * kernel_weightage) /
        (DOUBLE)(avl_threads);

    return (workload_per_thread > MIN_OPCNT_PER_THREAD &&
            solution->decomp_scheme->dims[0].in_stride >=
                MIN_DIM_STRIDE_FOR_COLMAJOR);
}

/**
 * @brief Calculate kernel weightage based on compute operation cycles for given
 * radix kernel
 *
 * @param radix The radix value to search for in the kernel table
 * @param kertab Pointer to the kernel table containing available kernels
 * @param sol Pointer to the solution object
 *
 * @return DOUBLE Computed weightage value based on operation cycles, scaled by
 * 1/KERNEL_WEIGHTAGE_SCALE_FACTOR. Returns 1.0 if no matching kernel radix is found.
 *
 */
DOUBLE get_kernel_weightage(INTP radix, kernel_t *kertab,
                            aoclfftz_solution_t *sol)
{
    DOUBLE weightage = 1.0;
    UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    ops_cycles_t ops_cycles = {0};
    for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        if ((INTP)kertab[i].radix == radix)
        {
            ops_cycles = kertab[i].k_ops_cnt(precision, direction);
            weightage = (DOUBLE)((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                                 (ops_cycles.mul * AMD_ZEN_FP_MUL_CYCLES) +
                                 (ops_cycles.add * AMD_ZEN_FP_ADD_CYCLES)) /
                        KERNEL_WEIGHTAGE_SCALE_FACTOR;
            break;
        }
    }
    return weightage;
}
