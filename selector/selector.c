// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#include "utils/dispatcher.h"
#include "utils/utils.h"

// Function pointers to communicate the exact selector model for executing the
// problem.
// Not thread-safe
typedef FFTZ_INT32(*selector_model_func_)(aoclfftz_selector_t*);
typedef FFTZ_INT32 (*selector_model_rdft_func_)(aoclfftz_selector_t *,
                                           aoclfftz_realhelper_t *);
selector_model_func_ sel_fp = NULL;
selector_model_rdft_func_ sel_rdft_fp = NULL;

// Register all applicable solvers and kernels into the respective tables.
FFTZ_INT32 register_solvers_kernels(kernel_tables_t *kernel_tables,
                                    FFTZ_INT32 dt, FFTZ_INT32 dir,
                                    FFTZ_INT32 is_real, FFTZ_INT32 cpu_flags)
{
    FFTZ_INT32 ret = SELECTOR_SUCCESS;

    // Register Solvers
    ret = register_solvers();
    if (ret != SOLVER_SUCCESS)
    {
        return SELECTOR_FAILURE;
    }
    // Register Kernels
    if (is_real)
    {
        if (dir == FORWARD_FFT_DIR)
        {
            ret |= register_kernels_real(kernel_tables->kt_rdft,
                                         kernels_twid_real_r2c, dt, dir,
                                         cpu_flags);
        }
        else // BACKWARD_FFT_DIR
        {
            ret |= register_kernels_real(kernel_tables->kt_rdft,
                                         kernels_twid_real_c2r, dt, dir,
                                         cpu_flags);
        }
    }
    // Standared C2C kernels (kernels_c2c) are bidirectional,
    // both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to
    // kernels_c2c
    ret |= register_kernels_complex(kernel_tables->kt_dft, kernels_c2c, NULL,
                                    dt, dir, cpu_flags);
    // Twiddle C2C kernels are direction specific,
    // kfft[FORWARD_FFT_DIR] points to kernels_twid_c2c_fwd and
    // kfft[BACKWARD_FFT_DIR] points to kernels_twid_c2c_bwd
    ret |= register_kernels_complex(kernel_tables->kt_twid_dft,
                                    kernels_twid_c2c_fwd,
                                    kernels_twid_c2c_bwd, dt, dir, cpu_flags);

    kernel_tables->ele_mul[FORWARD_FFT_DIR] =
        register_elementwise_mul_kernel(cpu_flags, dt, FORWARD_FFT_DIR);
    kernel_tables->ele_mul[BACKWARD_FFT_DIR] =
        register_elementwise_mul_kernel(cpu_flags, dt, BACKWARD_FFT_DIR);
    kernel_tables->normalize = register_normalize_kernel(cpu_flags, dt);
    if (kernel_tables->ele_mul[FORWARD_FFT_DIR] == NULL ||
        kernel_tables->ele_mul[BACKWARD_FFT_DIR] == NULL ||
        kernel_tables->normalize == NULL)
    {
        return SELECTOR_FAILURE;
    }

    return ret;
}

FFTZ_INT32 check_FFT_kernel_support(FFTZ_INTP n, kernel_t *kernels_table,
                                    FFTZ_INT32 is_innermost_dim)
{
    FFTZ_INT32 is_supported = 0;
    FFTZ_INTP num_kernels_to_check = NUM_KERNELS_IN_EACH_CATEGORY;

    // The "special" (large) kernels are not well suited to run for problems
    // whose strides are large. This is typically the case for non-innermost
    // dimensions of an ND problem. Hence, we skip checking those kernels in
    // such cases.
    if (!is_innermost_dim)
    {
        num_kernels_to_check -= NUMBER_OF_HIGHER_RADIX_KERNELS;
    }

    // It is enough to check for the existence of a suitable C kernel
    for (FFTZ_INTP i = 0; i < num_kernels_to_check; i++)
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

// Column-major iff multiple batches exist and vec-stride < dim-stride
// for both input and output.
FFTZ_UINT8 check_col_major(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    FFTZ_UINT8 is_col_major = (decomp_scheme->vecs[0].in_stride <
        decomp_scheme->dims[0].in_stride) &&
        (decomp_scheme->vecs[0].out_stride < decomp_scheme->dims[0].out_stride)
        && (decomp_scheme->vecs[0].n > 1);
    return is_col_major;
}

static FFTZ_INT32
is_split_radix_applicable(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    FFTZ_INTP n = decomp_scheme->dims[0].n;
    FFTZ_UINT8 is_col_major = check_col_major(decomp_scheme);
    // Split-radix is applicable for complex, 1D, row-major, non-batched
    // transforms, innermost dimension where the problem size is a power of 2,
    // >= 4096, and single-threaded only.
    // TODO: Replace the hardcoded threshold value (4096)

    return (!IS_REAL(decomp_scheme->flags) && (n >= 4096) &&
            ((n & (n - 1)) == 0) && (decomp_scheme->vec_rank == 1) &&
            (decomp_scheme->dim_rank == 1) &&
            (decomp_scheme->batched_vecs == NULL) && (!is_col_major) &&
            !IS_NOT_INNERMOST_DIM(decomp_scheme->flags) &&
            (decomp_scheme->thread_info->pthr_fft->num_threads == 1));
}

FFTZ_INTP check_CT_solvability(FFTZ_INTP n, kernel_t *kertab)
{
    // It is enough to check for the existence of a suitable C kernel
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_UINT32 radix = kertab[i].radix;
        if (radix == 0) // End of suitable kernels in the list
        {
            break;
        }
        // Check if this radix can factorize the problem
        if ((n % (FFTZ_INTP)radix) == 0)
        {
            return 1;
        }
    }
    return 0;
}

FFTZ_INT32
check_prime_solvability_bluestein(aoclfftz_decomp_scheme_t *decomp_scheme,
                                  FFTZ_INT32 is_FFT_ker_supported,
                                  kernel_t *kertab)
{
    FFTZ_INTP n = decomp_scheme->dims[0].n;
    FFTZ_INTP batch = decomp_scheme->vecs[0].n;
    FFTZ_INT32 dim_rank = decomp_scheme->dim_rank;
    FFTZ_INT32 vec_rank = decomp_scheme->vec_rank;

    if (n == 1 || dim_rank != 1 || vec_rank != 1 || batch != 1)
    {
        return 0;
    }

    // n is solvable by bluestein solver if is not solvable by direct
    // and CT solvers
    FFTZ_INT32 is_solvable_by_CT = check_CT_solvability(n, kertab);
    if (is_FFT_ker_supported == 0 && is_solvable_by_CT == 0)
    {
        return 1;
    }

    return 0;
}

FFTZ_INT32 is_prime(FFTZ_INT32 n)
{
    for (FFTZ_INT32 i = 2; i * i <= n; i++)
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
FFTZ_INT32 check_bluestein_problem(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    FFTZ_INT32 dim_rank = decomp_scheme->dim_rank;
    for (FFTZ_INT32 i = 0; i < dim_rank; i++)
    {
        FFTZ_INTP n = decomp_scheme->dims[i].n;
        // Check for prime factors greater than 13 in (n)
        for (FFTZ_INT32 i = 17; i <= n; i++)
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
FFTZ_INT32 check_batched_ct_l1_direct_solvability(FFTZ_INTP n,
                                                  kernel_t *kertab_twid,
                                                  kernel_t *kertab_dft)
{
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_INTP radix_r = (FFTZ_INTP)kertab_twid[i].radix;

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
// wherein the successive stage dfts are fused with twiddle multiplications
FFTZ_INT32 selector_fixed_mode_dft_(aoclfftz_selector_t *sel)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    kernel_t *kertab = sel->kernel_tables->kt_dft;
    FFTZ_INT32 ret = SELECTOR_FAILURE;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 avl_threads =
        sel->solution->decomp_scheme->thread_info->avl_threads;

    FFTZ_INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));

    FFTZ_INT32 is_solvable_by_bluestein = check_prime_solvability_bluestein(
        sel->solution->decomp_scheme, is_FFT_ker_supported, kertab);
    FFTZ_INT32 level1_cond1 = 0;
    FFTZ_INT32 level1_cond2 = 0;
    FFTZ_INT32 level2_cond = 0;
    FFTZ_INT32 standalone_transpose_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution, is_FFT_ker_supported);
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
    //                  (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags)
    //                  ==
    //                   AOCLFFTZ_AUTO_SELECTOR));
    level1_cond2 = IS_BUFFERED(sel->solution->decomp_scheme->flags);
    // SOLVER_PERM_KER
    level1_cond2 |= (IS_OUT_OF_ORDER(sel->solution->decomp_scheme->flags) << 1);
    // SOLVER_DIRECT
    level2_cond = is_FFT_ker_supported;
    // SOLVER_SR
    aoclfftz_cntrl_params_t *cp = sel->solution->decomp_scheme->cntrl_params;
    FFTZ_INT32 is_scalar_only =
        (cp != NULL) && (cp->opt_level == optlevel_scalar);
    level2_cond |=
        ((is_scalar_only
         && is_split_radix_applicable(sel->solution->decomp_scheme)) << 1);
    // SOLVER_PFA
    // SOLVER_RADER

    // SOLVER_TRANSPOSE
    standalone_transpose_cond =
        GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags);

    // SOLVER_BATCHED_CT_L1_DIRECT
    FFTZ_INT32 is_batched_ct_l1_direct = check_batched_ct_l1_direct_solvability(
        sel->solution->decomp_scheme->dims[0].n,
        sel->kernel_tables->kt_twid_dft,
        sel->kernel_tables->kt_dft);

    FFTZ_UINT8 is_col_major = check_col_major(sel->solution->decomp_scheme);

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
    if (avl_threads == 1 &&
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
        if (avl_threads == 1)
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
    // Large Primes - Bluestein FFT Solver. Pick the MT variant whenever this
    // node still has > 1 available thread (avl_threads already reflects any
    // parent MT_BATCHED slicing). MT_BLUESTEIN parallelizes the
    // ele_mul / normalize kernels across that per-node thread budget.
    // SOLVER_BLUESTEIN is the single-thread fallback when only one is left.
    if (level1_cond1 & 0x4)
    {
#ifdef MULTI_THREADING
        solver_obj->solver_type = (avl_threads > 1)
                                ? SOLVER_MT_BLUESTEIN
                                : SOLVER_BLUESTEIN;
#else
        solver_obj->solver_type = SOLVER_BLUESTEIN;
#endif
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
        if (avl_threads == 1)
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
        solver_obj->solver_type = SOLVER_CT;
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
FFTZ_INT32 selector_fixed_mode_rdft_(aoclfftz_selector_t *sel,
                                aoclfftz_realhelper_t *realhelper)
{
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;

    //All the CT sub-problems will use fused twiddle dft kernels
    kernel_t *kertab = sel->kernel_tables->kt_rdft;
    FFTZ_INT32 ret = SELECTOR_FAILURE;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 avl_threads =
            sel->solution->decomp_scheme->thread_info->avl_threads;

    FFTZ_INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        sel->solution->decomp_scheme->dims[0].n, kertab,
        !IS_NOT_INNERMOST_DIM(sel->solution->decomp_scheme->flags));

    FFTZ_INT32 is_solvable_by_bluestein =
            check_prime_solvability_bluestein(sel->solution->decomp_scheme,
                                              is_FFT_ker_supported, kertab);
    FFTZ_INT32 level1_cond1 = 0;
    FFTZ_INT32 level1_cond2 = 0;
    FFTZ_INT32 level2_cond = 0;

    SET_SELECTOR_MODE(sel->solution->decomp_scheme->flags,
                      AOCLFFTZ_FIXED_SELECTOR);

    if (sel->solution->decomp_scheme->vec_rank > 1)
    {
        fuse_vecs(sel->solution, is_FFT_ker_supported);
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
    level1_cond2 =
        !realhelper->is_buffered_invoked && !is_FFT_ker_supported &&
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
        if (avl_threads == 1)
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
        if (avl_threads == 1)
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

FFTZ_INT32 selector_autotuner_mode_dft_(aoclfftz_selector_t* sel)
{
    AOCLFFTZ_LOG(INFO, global_logger_mode,
        "Autotuner selector is not yet available for evaluation");
    return SELECTOR_FAILURE;
}

FFTZ_INT32 selector_autotuner_mode_rdft_(aoclfftz_selector_t* sel)
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
FFTZ_INT32 selector_driver_dft_(aoclfftz_selector_t* sel)
{
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    // if bit reproducibility is requested, directly take its associated
    // code path and return
    if (GET_BIT_REPRODUCIBLE(sel->solution->decomp_scheme->flags))
    {
        sel_fp = selector_fixed_mode_dft_;
        return selector_model_dft_(sel);
    }

    aoclfftz_selector_t *sel_models[AOCLFFTZ_SELECTOR_MODELS] = { 0x0, };
    FFTZ_UINT32 best_model_id = 0;
    cost_analysis_t best_cost = {INT64_MAX, INT64_MAX};
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;

    // FIXED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_FIXED_SELECTOR_MODE
    sel_models[AOCLFFTZ_FIXED_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR] != NULL)
    {
        ret = copy_decomp_scheme(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            AOCLFFTZ_ERROR("copy_decomp_scheme failed: %s",
                           get_status_string(ret));
            return ret;
        }
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR]
                          ->solution->decomp_scheme->flags,
                      DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]
                ->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->ct_buffer =
            sel->solution->dft_bufs->ct_buffer;
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->ct_buf_real =
            sel->solution->dft_bufs->ct_buf_real;
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->ct_buf_imag =
            sel->solution->dft_bufs->ct_buf_imag;
        sel_models[AOCLFFTZ_FIXED_SELECTOR]
            ->solution->dft_bufs->ct_buf_size =
            sel->solution->dft_bufs->ct_buf_size;

        // Fixed decision logic and CPI based selector mode
        sel_fp = selector_fixed_mode_dft_;
        ret = selector_model_dft_(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR]
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
    // FIXED SELECTOR BASED MODEL : end

    // AUTO TUNED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
    sel_models[AOCLFFTZ_AUTO_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (sel_models[AOCLFFTZ_AUTO_SELECTOR] != NULL)
    {
        ret = copy_decomp_scheme(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            AOCLFFTZ_ERROR("copy_decomp_scheme failed: %s",
                           get_status_string(ret));
            return ret;
        }
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
    for (FFTZ_UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS;
         model_id++)
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
    destroy_solution(org_sol);
    for (FFTZ_UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS;
         model_id++)
    {
        if (model_id != best_model_id)
        {
            destroy_selector(sel_models[model_id]);
        }
        else
        {
            destroy_selector_without_solution(sel_models[model_id]);
        }
    }



    return ret;
}

// Specific Selector model function that invokes the internal recursive
// selector function pointer passing the selector object along.
// Common function for both single-precision and double-precision
FFTZ_INT32 selector_model_dft_(aoclfftz_selector_t *sel)
{
    return sel_fp(sel);
}

// Main selector driver that invokes the complementary/alternate selector
// algorithms/models and decides on the final selector based on its suitability
// and performance for real-fft problems.
// Also provides a cleaner approach to init and handle various related
// solvers and kernel tables
FFTZ_INT32 selector_driver_rdft_(aoclfftz_selector_t *sel,
                            aoclfftz_realhelper_t *realhelper)
{
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    // if bit reproducibility is requested, directly take its associated
    // code path and return
    if (GET_BIT_REPRODUCIBLE(sel->solution->decomp_scheme->flags))
    {
        // Fixed decision logic and CPI based selector mode
        sel_rdft_fp = selector_fixed_mode_rdft_;
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_fixed_mode_dft_;

        return selector_model_rdft_(sel, realhelper);
    }

    aoclfftz_selector_t *sel_models[AOCLFFTZ_SELECTOR_MODELS] = { 0x0, };
    FFTZ_UINT32 best_model_id = 0;
    cost_analysis_t best_cost = {INT64_MAX, INT64_MAX};
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;

    // FIXED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_FIXED_SELECTOR_MODE
    sel_models[AOCLFFTZ_FIXED_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (sel_models[AOCLFFTZ_FIXED_SELECTOR] != NULL)
    {
        ret = copy_decomp_scheme(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            AOCLFFTZ_ERROR("copy_decomp_scheme failed: %s",
                           get_status_string(ret));
            return ret;
        }
        SET_PRECISION(sel_models[AOCLFFTZ_FIXED_SELECTOR]
                          ->solution->decomp_scheme->flags,
                      DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags));
        SET_STANDALONE_TRANSPOSE(
            sel_models[AOCLFFTZ_FIXED_SELECTOR]
                ->solution->decomp_scheme->flags,
            GET_STANDALONE_TRANSPOSE(sel->solution->decomp_scheme->flags));

        // Fixed decision logic and CPI based selector mode
        // ret = selector_fixed_mode_rdft_(
        //         sel_models[AOCLFFTZ_FIXED_SELECTOR]);
        sel_rdft_fp = selector_fixed_mode_rdft_;
        // Registering complex selector for real problems also since the real
        // selector will invoke complex selector internally for ND and
        // Bluestein problems
        sel_fp = selector_fixed_mode_dft_;
        ret = selector_model_rdft_(
            sel_models[AOCLFFTZ_FIXED_SELECTOR], realhelper);
        if (ret != SELECTOR_FAILURE)
        {
            *(sel_models[AOCLFFTZ_FIXED_SELECTOR]
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
    // FIXED SELECTOR BASED MODEL : end

    // AUTO TUNED SELECTOR BASED MODEL : start
    // Allocate selector object
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
    sel_models[AOCLFFTZ_AUTO_SELECTOR] =
        alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (sel_models[AOCLFFTZ_AUTO_SELECTOR] != NULL)
    {
        ret = copy_decomp_scheme(
            sel_models[AOCLFFTZ_AUTO_SELECTOR]->solution->decomp_scheme,
            sel->solution->decomp_scheme);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            AOCLFFTZ_ERROR("copy_decomp_scheme failed: %s",
                           get_status_string(ret));
            return ret;
        }
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
    for (FFTZ_UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS;
         model_id++)
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
    destroy_solution(org_sol);
    for (FFTZ_UINT32 model_id = 0; model_id < AOCLFFTZ_SELECTOR_MODELS;
         model_id++)
    {
        if (model_id != best_model_id)
            destroy_selector(sel_models[model_id]);
        else
            destroy_selector_without_solution(sel_models[model_id]);
    }

    return ret;
}

// Specific Selector model function that invokes the internal recursive
// selector function pointer (for real-fft) passing the selector object along.
// Common function for both single-precision and double-precision
FFTZ_INT32 selector_model_rdft_(aoclfftz_selector_t *sel,
                           aoclfftz_realhelper_t *realhelper)
{
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    ret = sel_rdft_fp(sel, realhelper);

    return ret;
}

static FFTZ_VOID compute_exec_metadata(aoclfftz_solution_t *sol,
                                  aoclfftz_immutable_metadata_t *out);

#ifdef MULTI_THREADING
static FFTZ_INT32 post_process_real_solution(aoclfftz_solution_t *sol,
                                             FFTZ_INT32 *aux_buf_slots);
#endif

static FFTZ_INT32 setup_chirp_fft(aoclfftz_solution_t *sol,
                                  aoclfftz_mutable_ctx_t *ctx);

static inline FFTZ_INT32 prepare_and_setup_dft(aoclfftz_selector_t *sel_obj)
{
    FFTZ_INT32 ret;
    sel_obj->execute = register_execute_dft();
    if (IS_REAL(sel_obj->solution->decomp_scheme->flags))
    {
        aoclfftz_realhelper_t *realhelper;
        ALLOC_ALIGN_UNINIT(realhelper, aoclfftz_realhelper_t,
            sizeof(aoclfftz_realhelper_t));
        if (realhelper == NULL)
        {
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        realhelper->stage = 0;
        realhelper->is_CT = 0;
        realhelper->is_buffered_invoked = 0;
        realhelper->problem_size = sel_obj->solution->decomp_scheme->dims[0].n;
        if (FFT_DIR(sel_obj->solution->decomp_scheme->flags) ==
            FORWARD_FFT_DIR)
        {
            realhelper->freq_factor = 1;
        }
        else
        {
            realhelper->freq_factor = realhelper->problem_size;
        }
        ret = selector_driver_rdft_(sel_obj, realhelper);
        swap_real_ct_solutions(sel_obj);
        setup_twiddle_buffer_real(sel_obj->solution);
        FREE_ALIGN_ALLOCATED_MEM(realhelper);
    }
    else
    {
        ret = selector_driver_dft_(sel_obj);
        setup_twiddle_buffer_complex(sel_obj->solution);
    }
    if (ret != SELECTOR_SUCCESS)
    {
        ret = AOCLFFTZ_SETUP_FAILURE;
        goto exit_prepare_and_setup_dft;
    }

    ALLOC_ALIGN_UNINIT(sel_obj->exec_metadata,
                       aoclfftz_immutable_metadata_t,
                       sizeof(aoclfftz_immutable_metadata_t));
    if (sel_obj->exec_metadata == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_prepare_and_setup_dft;
    }
    *sel_obj->exec_metadata = (aoclfftz_immutable_metadata_t){0};

#ifdef MULTI_THREADING
    if (IS_REAL(sel_obj->solution->decomp_scheme->flags))
    {
        // Real plans still rely on the legacy per-thread next_sol[i]
        // clones populated by post_process_real_solution: the real
        // solvers mutate decomp_scheme->in/out per batch.
        // The C2C subtree is left untouched and uses the per-call
        // exec ctx for its concurrency-safety.
        ret = post_process_real_solution(sel_obj->solution, NULL);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            goto exit_prepare_and_setup_dft;
        }
    }
#endif

    // Populate per-call scratch sizes and plan-time scratch pointers
    // directly into exec_metadata->base_ctx. This also assigns each Bluestein node a
    // disjoint slice (bs_dim_offset) within the shared bs pool.
    compute_exec_metadata(sel_obj->solution, sel_obj->exec_metadata);

    // Allocate the shared bs_in_base / bs_out_base scratch pools.
    // Each Bluestein node carves out its own slice using bs_dim_offset.
    if (sel_obj->exec_metadata->bs_buffer_size > 0)
    {
        ALLOC_ALIGN_UNINIT(sel_obj->exec_metadata->base_ctx.bs_in_base,
                           FFTZ_VOID,
                           sel_obj->exec_metadata->bs_buffer_size);
        ALLOC_ALIGN_UNINIT(sel_obj->exec_metadata->base_ctx.bs_out_base,
                           FFTZ_VOID,
                           sel_obj->exec_metadata->bs_buffer_size);
        if (sel_obj->exec_metadata->base_ctx.bs_in_base == NULL ||
            sel_obj->exec_metadata->base_ctx.bs_out_base == NULL)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_prepare_and_setup_dft;
        }
    }

    aoclfftz_mutable_ctx_t *base = &sel_obj->exec_metadata->base_ctx;
    base->in_real  = sel_obj->solution->decomp_scheme->in_real;
    base->in_imag  = sel_obj->solution->decomp_scheme->in_imag;
    base->out_real = sel_obj->solution->decomp_scheme->out_real;
    base->out_imag = sel_obj->solution->decomp_scheme->out_imag;
    base->flags    = sel_obj->solution->decomp_scheme->flags;

    ret = setup_chirp_fft(sel_obj->solution, base);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_prepare_and_setup_dft;
    }
    return ret;

exit_prepare_and_setup_dft:
    AOCLFFTZ_ERROR("Setup failure with %s", get_status_string(ret));

    return ret;
}

// Selector interface function that performs setup for finding solution for a
// single-precision LP64 problem
FFTZ_VOID *setup_dft_f(aoclfftz_prob_desc_f *problem)
{
    FFTZ_INT32 ret = 0;
    FFTZ_INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    FFTZ_INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft, {NULL},
                                     NULL};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, &kertab_tables);
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
    SET_EFFECTIVE_OPT_LEVEL(sel_obj, cpu_flags);

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
    ret = prepare_and_setup_dft(sel_obj);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_f;
    }
    return sel_obj;

exit_setup_dft_f:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision LP64 problem
FFTZ_VOID *setup_dft_d(aoclfftz_prob_desc_d *problem)
{
    FFTZ_INT32 ret = 0;
    FFTZ_INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    FFTZ_INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft, {NULL},
                                     NULL};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, &kertab_tables);
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
    // input problem datatype, CPU opt level and dynamic dispatcher FMV
    // selection
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
    SET_EFFECTIVE_OPT_LEVEL(sel_obj, cpu_flags);

    if (problem->flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    ret = prepare_and_setup_dft(sel_obj);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_d;
    }
    return sel_obj;

exit_setup_dft_d:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// single-precision ILP64 problem
FFTZ_VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem)
{
    FFTZ_INT32 ret = 0;
    FFTZ_INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    FFTZ_INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft, {NULL},
                                     NULL};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, &kertab_tables);
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
    SET_EFFECTIVE_OPT_LEVEL(sel_obj, cpu_flags);

    if (flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    ret = prepare_and_setup_dft(sel_obj);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_f_64_;
    }
    return sel_obj;

exit_setup_dft_f_64_:
    destroy_selector(sel_obj);
    return NULL;
}

// Selector interface function that performs setup for finding solution for a
// double-precision ILP64 problem
FFTZ_VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem)
{
    FFTZ_INT32 ret = 0;
    FFTZ_INT32 cpu_flags = 0;
    aoclfftz_cntrl_params_t cntrl_params = problem->cntrl_params;
    aoclfftz_flags_t flags = problem->flags;
    aoclfftz_selector_t *sel_obj = NULL;

    // shrink dim_rank
    // used in n dim case where size one problems are removed
    FFTZ_INT32 dim_rank = 1;
    SHRINK_DIM_RANK(problem->dims, problem->dim_rank, dim_rank);

    kernel_t kt_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_twid_dft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_t kt_rdft[MAX_NUM_KERNELS_IN_TABLE] = {0};
    kernel_tables_t kertab_tables = {kt_dft, kt_twid_dft, kt_rdft, {NULL},
                                     NULL};

    // allocate selector object
    sel_obj = alloc_selector(problem->vec_rank, dim_rank, &kertab_tables);
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
    SET_EFFECTIVE_OPT_LEVEL(sel_obj, cpu_flags);

    if (flags.transpose_mode)
    {
        SET_STANDALONE_TRANSPOSE(sel_obj->solution->decomp_scheme->flags, 1);
    }

    // Select the best solution for the given input problem
    ret = prepare_and_setup_dft(sel_obj);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_setup_dft_d_64_;
    }
    return sel_obj;

exit_setup_dft_d_64_:
    destroy_selector(sel_obj);
    return NULL;
}

FFTZ_VOID destroy_handle(FFTZ_VOID *handle)
{
    destroy_selector((aoclfftz_selector_t *)handle);
}

FFTZ_VOID fuse_vecs(aoclfftz_solution_t *sol, FFTZ_INT32 is_FFT_ker_supported)
{
    FFTZ_INT32 last_fused_idx = 0, is_fusable = 0, fused_rank = 0;
    FFTZ_INTP fused_size = sol->decomp_scheme->vecs[0].n;
    FFTZ_INT32 vec_rank = sol->decomp_scheme->vec_rank;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;
    for (FFTZ_INT32 i = 1; i < vec_rank; i++)
    {
        // expected stride is the regular stride we obtain by n * stride of
        // prev dim
        FFTZ_INTP expected_in_stride = vecs[i-1].n * vecs[i-1].in_stride;
        FFTZ_INTP expected_out_stride = vecs[i-1].n * vecs[i-1].out_stride;
        FFTZ_INTP actual_in_stride = vecs[i].in_stride;
        FFTZ_INTP actual_out_stride = vecs[i].out_stride;
        // Do not fuse the first vector to enable memory-efficient batch
        // processing for NDim problems where the outerdimension is CT.
        // ie., problems like AxBxC that decomposes to BxCvA and A is CT,
        // its efficient if we do not fuse BxC
        if ((expected_in_stride == actual_in_stride &&
             expected_out_stride == actual_out_stride) &&
             (i != 1 || is_FFT_ker_supported ||
                        !IS_NOT_INNERMOST_DIM(sol->decomp_scheme->flags)))
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

// Only for column major problems, we set load_multi_cols to 0
static inline FFTZ_INTP detect_load_multi_cols(aoclfftz_solution_t *sol)
{
    return (sol != NULL && sol->decomp_scheme != NULL
            && sol->decomp_scheme->batched_vecs != NULL)
               ? 0
               : 1;
}

FFTZ_VOID setup_twiddle_buffer_complex(aoclfftz_solution_t *solution)
{
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    aoclfftz_solution_t *curr = solution;
    FFTZ_UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
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
            if (curr->solver->solver_type == SOLVER_CT)
            {
                FFTZ_INTP r = curr->next_sol[0]->decomp_scheme->dims[0].n;
                FFTZ_INTP m =
                    curr->next_sol[0]->next_sol[0]->decomp_scheme->dims[0].n;
                FFTZ_INTP rw =
                    (FFTZ_INTP)curr->next_sol[0]->solver->kernel_c2c->sets;
                FFTZ_INTP lmc = detect_load_multi_cols(curr);
                FFTZ_VOID *TW = alloc_twiddle_buffer((r - 1) * m, dt_prec);
                if (TW != NULL)
                {
                    compute_twiddle_buffer(TW, r, m, rw, lmc, dt_prec);
                    curr->next_sol[0]->twiddle->TW = TW;
                    curr->next_sol[0]->twiddle->twiddle_buf_ptr = TW;
                    curr->next_sol[0]->twiddle->load_multi_cols = lmc;
                }
            }
            else if (curr->solver->solver_type == SOLVER_SR)
            {
                FFTZ_INTP n = curr->decomp_scheme->dims[0].n;
                // Number of twiddle pairs(W^k, W^(3k)) for the split-radix
                // twiddle multiplication
                FFTZ_INTP sr_tw_pairs = n / 4;

                // Allocate twiddle buffer for split-radix: N/4 complex pairs
                // (W^k and W^(3k))
                FFTZ_VOID *TW = alloc_twiddle_buffer(sr_tw_pairs * 2, dt_prec);
                if (TW != NULL)
                {
                    compute_sr_twiddle_buffer(TW, n, dt_prec);
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                }

                // Recursively set up twiddles for ALL 3 sub-problems
                // Note: next_sol[0] (even) will be handled by the loop
                // continuation So we only need to explicitly recurse for odd1
                // and odd3 from dft_bufs
                if (curr->dft_bufs && curr->dft_bufs->sr->odd1_sol
                    && curr->dft_bufs->sr->odd3_sol)
                {
                    setup_twiddle_buffer_complex(curr->dft_bufs->sr->odd1_sol);
                    setup_twiddle_buffer_complex(curr->dft_bufs->sr->odd3_sol);
                }
            }
            else if (curr->solver->solver_type == SOLVER_BATCHED_CT_L1_DIRECT)
            {
                FFTZ_INTP r = (FFTZ_INTP)curr->solver->kernel_c2c->count;
                FFTZ_INTP m = (FFTZ_INTP)curr->solver->kernel_c2c_r->count;
                // In this solver the twiddle (radix-r) kernel lives in
                // kernel_c2c_r.
                FFTZ_INTP rw = (FFTZ_INTP)curr->solver->kernel_c2c_r->sets;
                FFTZ_INTP lmc = detect_load_multi_cols(curr);
                FFTZ_VOID *TW = alloc_twiddle_buffer((r - 1) * m, dt_prec);
                if (TW != NULL)
                {
                    compute_twiddle_buffer(TW, r, m, rw, lmc, dt_prec);
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                    curr->twiddle->load_multi_cols = lmc;
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

FFTZ_VOID setup_twiddle_buffer_real(aoclfftz_solution_t *solution)
{
    if (solution == NULL)
    {
        return;
    }
#if IN_MEMORY_TWIDDLE_FACTORS == 1
    FFTZ_UINT32 dt_prec = DT_PRECISION_FLAG(solution->decomp_scheme->flags);
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
                    curr->solver->solver_type != SOLVER_REAL_MT_DIRECT)
                {
                    curr->twiddle->TW = prev->twiddle->TW;
                    curr = curr->next_sol ? curr->next_sol[0] : NULL;
                }
                if (curr == NULL)
                {
                    return;
                }
                FFTZ_INTP radix = curr->decomp_scheme->dims[0].n;
                FFTZ_INTP num_groups = NUM_RFFT_GROUPS(curr->solver);
                FFTZ_INTP num_c2c_per_group =
                    curr->solver->kernel_c2c->count / num_groups;
                // The twiddle packing must match how execute_c2c_kernels()
                // consumes it: the asymmetric branch (num_c2c_per_group >=
                // num_groups) feeds sol->twiddle straight to the kernel and so
                // needs the tile-packed (lmc == 1) layout; the non-asymmetric
                // branch forces broadcast and walks the buffer (radix - 1)
                // cpairs per column, i.e. the plain j-major (lmc == 0) layout.
                FFTZ_INTP rw = (FFTZ_INTP)curr->solver->kernel_c2c->sets;
                FFTZ_INTP lmc = (num_c2c_per_group >= num_groups) ? 1 : 0;
                FFTZ_INTP tw_buf_size = (radix - 1) * num_c2c_per_group *
                                        DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle values for every
                // radix-n c2c kernel of a group
                FFTZ_VOID *TW = alloc_twiddle_buffer(tw_buf_size, dt_prec);
                if (TW != NULL)
                {
                    FFTZ_INTP p = (curr->decomp_scheme->vecs[0].n *
                              curr->decomp_scheme->dims[0].n) /
                             num_groups;
                    compute_twiddle_buffer_real(TW, radix, num_c2c_per_group, p,
                                                FORWARD_FFT_DIR, rw, lmc,
                                                dt_prec);
                    curr->twiddle->TW = TW;
                    curr->twiddle->twiddle_buf_ptr = TW;
                    curr->twiddle->load_multi_cols = lmc;
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
                 prev->solver->solver_type == SOLVER_REAL_MT_DIRECT))
            {
                FFTZ_INTP radix = prev->decomp_scheme->dims[0].n;
                FFTZ_INTP num_groups = NUM_RFFT_GROUPS(prev->solver);
                // No. of c2c kernels per group that require twiddle computation
                FFTZ_INTP num_c2c_per_group =
                    prev->solver->kernel_c2c->count / num_groups;
                // Layout must follow execute_c2c_kernels()'s branch: asymmetric
                // -> tile-packed (lmc == 1); non-asymmetric -> j-major
                // broadcast (lmc == 0). See the forward branch for details.
                FFTZ_INTP rw = (FFTZ_INTP)prev->solver->kernel_c2c->sets;
                FFTZ_INTP lmc = (num_c2c_per_group >= num_groups) ? 1 : 0;
                FFTZ_INTP tw_buf_sz = (radix - 1) * num_c2c_per_group *
                                        DATA_STRIDE;
                // Allocate Twiddle buffer to store twiddle factors for every
                // radix-n c2c kernel of a group
                FFTZ_VOID *TW = alloc_twiddle_buffer(tw_buf_sz, dt_prec);
                if (TW != NULL)
                {
                    FFTZ_INTP p = (prev->decomp_scheme->vecs[0].n *
                              prev->decomp_scheme->dims[0].n) /
                             num_groups;
                    compute_twiddle_buffer_real(TW, radix, num_c2c_per_group, p,
                                                BACKWARD_FFT_DIR, rw, lmc,
                                                dt_prec);
                    prev->twiddle->TW = TW;
                    prev->twiddle->twiddle_buf_ptr = TW;
                    prev->twiddle->load_multi_cols = lmc;
                }
            }
            // Always inherit the parent's twiddle buffer reference.
            curr->twiddle->TW = prev->twiddle->TW;
            prev = curr;
        }
    }
#endif
}

#ifdef MULTI_THREADING

/* Configures static aux_buffer_1/2 routing through BUFFERED -> DIRECT -> CT
 * chain. aux_buffer_1 and aux_buffer_2 are spaced by aux_buf_size_per_thread
 * (64-byte aligned) so MT deep-copies get non-overlapping cache lines per
 * thread.
 *
 * Solver chain of CT problem after buffered sol
 * ... -> buffered -> direct -> CT -> direct -> ... -> CT -> direct
 *
 * Here, the buffered solver will have in & out of the current batch
 * Buffered solver will change the input/output buffers of direct & CT
 * solution in the following way:
 *
 * buffered    [in -> out]
 * |--> direct   [in -> aux1]
 * |----> CT & Direct [aux1 -> aux2]
 * |----> CT & Direct [aux2 -> aux1]
 * |----> CT & Direct [aux1 -> out]
 * this example is for a 3 level CT problem
 *
 * the input sol points to the first solution
 * move the sol pointer to buffered_solver and modify the input & output
 * address of buffered struct to point the updated problem input & output
 */
static FFTZ_INT32 setup_buffered_chain_structure(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sol == NULL)
    {
        AOCLFFTZ_ERROR("sol is NULL");
        return SOLVER_FAILURE;
    }
    FFTZ_INT32 dt_bytes = SOL_DT_SIZE(sol);

    // Find the buffered solver in the chain (sol may be REAL_BUFFERED or
    // REAL_NDIM)
    aoclfftz_solution_t *cur_sol = sol;
    while ((cur_sol != NULL) &&
           (cur_sol->solver->solver_type != SOLVER_REAL_BUFFERED))
    {
        if (!cur_sol->next_sol || !cur_sol->next_sol[0])
        {
            AOCLFFTZ_ERROR("No REAL_BUFFERED solver found in chain");
            return SOLVER_FAILURE;
        }
        cur_sol = cur_sol->next_sol[0];
    }

    if (cur_sol == NULL)
    {
        AOCLFFTZ_ERROR("No REAL_BUFFERED solver found in chain");
        return SOLVER_FAILURE;
    }

    aoclfftz_solution_t *buffered_sol = cur_sol;

    if (buffered_sol->dft_bufs->buffered->aux_buffer_1 == NULL ||
        buffered_sol->dft_bufs->buffered->aux_buffer_2 == NULL)
    {
        AOCLFFTZ_ERROR("REAL_BUFFERED aux buffers are not allocated");
        return SOLVER_FAILURE;
    }

    if (buffered_sol->dft_bufs->buffered->aux_buf_size_per_thread == 0)
    {
        buffered_sol->dft_bufs->buffered->aux_buf_size_per_thread =
            GET_PADDED_SIZE(buffered_sol->decomp_scheme->dims[0].n * dt_bytes);
    }

    // Get aux buffers (per-thread region base; pool slots use
    // aux_buf_size_per_thread)
    FFTZ_VOID *aux_in = buffered_sol->dft_bufs->buffered->aux_buffer_1;
    FFTZ_VOID *aux_out = buffered_sol->dft_bufs->buffered->aux_buffer_2;

    // Move to the first direct solution of CT problem
    cur_sol = buffered_sol->next_sol[0];

    // Update first direct solution's output to aux_out
    cur_sol->decomp_scheme->out_real = aux_out;
    cur_sol->decomp_scheme->out_imag = MOVE_ADDR(aux_out, dt_bytes);
    // Swap aux buffers so current output becomes next input
    SWAP_BUFFERS(aux_in, aux_out);
    cur_sol = cur_sol->next_sol[0];

    // Update all intermediate CT + direct solutions' in/out
    // (except first direct and last CT + direct)
    while (cur_sol && cur_sol->next_sol &&
           cur_sol->next_sol[0] && cur_sol->next_sol[0]->next_sol)
    {
        cur_sol->decomp_scheme->in_real = aux_in;
        cur_sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);
        cur_sol->decomp_scheme->out_real = aux_out;
        cur_sol->decomp_scheme->out_imag = MOVE_ADDR(aux_out, dt_bytes);
        // Swap aux buffers after every direct solution
        if (cur_sol->solver->solver_type == SOLVER_REAL_DIRECT ||
            cur_sol->solver->solver_type == SOLVER_REAL_MT_DIRECT)
        {
            SWAP_BUFFERS(aux_in, aux_out);
        }
        cur_sol = cur_sol->next_sol[0];
    }

    if (cur_sol == NULL)
    {
        AOCLFFTZ_ERROR("Invalid solution chain: CT solution node not found");
        return SOLVER_FAILURE;
    }

    // Update last CT solution's input (output will be set per-batch)
    cur_sol->decomp_scheme->in_real = aux_in;
    cur_sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);

    // Update last direct solution's input
    if (cur_sol->next_sol == NULL || cur_sol->next_sol[0] == NULL)
    {
        AOCLFFTZ_ERROR("Invalid solution chain: direct solution node not found");
        return SOLVER_FAILURE;
    }
    cur_sol = cur_sol->next_sol[0];
    cur_sol->decomp_scheme->in_real = aux_in;
    cur_sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);

    // Store the address where per-batch output pointer will be updated
    // This is used by buffered executor to update the final output location
    buffered_sol->dft_bufs->buffered->out_ptr =
                    &cur_sol->decomp_scheme->out_real;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Recursively deep-copies a real solution subtree for one MT thread.
 *
 * Creates an independent copy of every node in the real subtree rooted at
 * src. The C2C tree is concurrent-safe via per-call ctx and does not need
 * per-thread clones.
 *
 * Traversal order mirrors post_process_real_solution():
 *   REAL_NDIM         — share nd_sol (C2C boundary), copy next_sol[0].
 *   REAL_MT_BATCHED   — copy thread 0 first, then threads 1..N.
 *
 * @param src     Root of the source subtree to copy. If NULL, returns NULL.
 * @param aux_buf_base Linear slot into REAL_BUFFERED stretched aux pool
 *                (stride aux_buf_size_per_thread per slot, 64-byte aligned;
 *                 n_slots = active_threads).
 * @param aux_bufs Output: REAL_BUFFERED slot demand for this subtree
 * (max/aggregate).
 * @param aux_ndim_pool_slot_idx For MT duplicate subtrees (post_process thread
 *                i>0), REAL_NDIM multi-slot aux_buffer_1 is offset by this
 * index into the pool; pass 0 for template / thread-0 trees. Under nested
 *                REAL_MT_BATCHED, use parent_slot * n_threads + inner index.
 *
 * @return Pointer to the newly allocated copy of the subtree, or NULL on
 *         allocation failure (partial allocations are freed before returning).
 */
static aoclfftz_solution_t *deep_copy_real_solution_tree(
                                            aoclfftz_solution_t *src,
                                            FFTZ_INT32 aux_buf_base,
                                            FFTZ_INT32 *aux_bufs,
                                            FFTZ_INT32 aux_ndim_pool_slot_idx)
{
    if (!src)
    {
        return NULL;
    }

    FFTZ_INT32 ret = AOCLFFTZ_SUCCESS;
    // Copy the solution object and its strides
    FFTZ_INT32 vec_rank = src->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = src->decomp_scheme->dim_rank;
    aoclfftz_solution_t *dst = alloc_solution(vec_rank, dim_rank);
    if (!dst)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        AOCLFFTZ_ERROR("deep_copy_real_solution_tree failed,"
                       " alloc_solution failed");
        goto exit_deep_copy;
    }
    ret = copy_solution_obj(dst, src);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("deep_copy_real_solution_tree failed,"
                       " copy_solution_obj failed: %s",
                       get_status_string(ret));
        goto exit_deep_copy;
    }
    ret = copy_strides(dst, src);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("deep_copy_real_solution_tree failed,"
                       " copy_strides failed: %s",
                       get_status_string(ret));
        goto exit_deep_copy;
    }
    dst->dft_bufs->nd_sol = NULL;

    if (src->solver->solver_type == SOLVER_REAL_NDIM)
    {
        // Share the C2C inner subtree across all clones; only the real-dim
        // sub-solution chain (next_sol[0]) is real.
        dst->dft_bufs->nd_sol = src->dft_bufs->nd_sol;
    }

    if (src->solver->solver_type == SOLVER_REAL_BUFFERED)
    {
        FFTZ_INT32 dt_bytes = SOL_DT_SIZE(dst);
        FFTZ_INTP aux_buf_size_per_thread =
                src->dft_bufs->buffered->aux_buf_size_per_thread;
        if (aux_buf_size_per_thread == 0)
        {
            FFTZ_INTP n = dst->decomp_scheme->dims[0].n;
            aux_buf_size_per_thread = GET_PADDED_SIZE(n * dt_bytes);
        }
        FFTZ_INTP aux_buf_offset = aux_buf_base * aux_buf_size_per_thread;
        dst->dft_bufs->buffered->aux_buffer_1 =
            MOVE_ADDR(src->dft_bufs->buffered->aux_buffer_1, aux_buf_offset);
        dst->dft_bufs->buffered->aux_buffer_2 =
            MOVE_ADDR(src->dft_bufs->buffered->aux_buffer_2, aux_buf_offset);
        dst->dft_bufs->buffered->aux_buf_size_per_thread =
            aux_buf_size_per_thread;
        dst->dft_bufs->buffered->is_aux_buffer_allocated = 0;
    }

    if (src->solver->solver_type == SOLVER_REAL_NDIM &&
        src->dft_bufs->buffered &&
        src->dft_bufs->buffered->aux_buffer_1 != NULL &&
        src->dft_bufs->buffered->aux_buf_size_per_thread > 0 &&
        aux_ndim_pool_slot_idx > 0 &&
        aux_ndim_pool_slot_idx <
            dst->decomp_scheme->thread_info->active_threads)
    {
        FFTZ_VOID *base = src->dft_bufs->buffered->aux_buffer_1;
        FFTZ_INTP offset = aux_ndim_pool_slot_idx *
                       src->dft_bufs->buffered->aux_buf_size_per_thread;
        dst->dft_bufs->buffered->aux_buffer_1 = MOVE_ADDR(base, offset);
    }

    if (src->solver->solver_type == SOLVER_REAL_NDIM)
    {
        // nd_sol (C2C subtree) is already shared at the top of this function;
        // only next_sol[0] (the real-dim chain) needs a per-thread clone.
        FFTZ_INT32 ns_aux_bufs = 0;

        dst->next_sol = alloc_sol_array(1);
        if (!dst->next_sol)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            AOCLFFTZ_ERROR(
                "deep_copy_real_solution_tree failed,"
                " alloc_sol_array failed (NDIM)");
            goto exit_deep_copy;
        }
        dst->next_sol[0] = deep_copy_real_solution_tree(
                            src->next_sol[0],
                            aux_buf_base, &ns_aux_bufs,
                            aux_ndim_pool_slot_idx);
        if (!dst->next_sol[0])
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_deep_copy;
        }

        if (aux_bufs)
        {
            *aux_bufs = ns_aux_bufs;
        }
        return dst;
    }

    FFTZ_INT32 is_real_mt = src->solver->solver_type == SOLVER_REAL_MT_BATCHED;
    FFTZ_INT32 ns_aux_bufs = 0;
    if (src->next_sol)
    {
        FFTZ_INT32 n = is_real_mt ? src->decomp_scheme->thread_info->n_threads
                                  : 1;
        dst->next_sol = alloc_sol_array(n);
        if (!dst->next_sol)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            AOCLFFTZ_ERROR("deep_copy_real_solution_tree failed,"
                           " alloc_sol_array failed (REAL_MT_BATCHED)");
            goto exit_deep_copy;
        }

        // Copy thread 0 first; per_thread_ct_bufs drives the ct_base stride
        // for threads 1..N. When n == 1, the loop is skipped entirely.
        FFTZ_INT32 per_thread_aux_buff = 0;
        FFTZ_INT32 child_ndim_slot0 = is_real_mt
                               ? aux_ndim_pool_slot_idx * n
                               : aux_ndim_pool_slot_idx;
        dst->next_sol[0] = deep_copy_real_solution_tree(
            src->next_sol[0],
            aux_buf_base, &per_thread_aux_buff,
            child_ndim_slot0);
        if (!dst->next_sol[0])
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_deep_copy;
        }

        for (FFTZ_INT32 i = 1; i < n; i++)
        {
            FFTZ_INT32 child_ndim_slot_i = is_real_mt
                                    ? aux_ndim_pool_slot_idx * n + i
                                    : aux_ndim_pool_slot_idx;
            dst->next_sol[i] = deep_copy_real_solution_tree(
                src->next_sol[0],
                aux_buf_base + i * per_thread_aux_buff, NULL,
                child_ndim_slot_i);
            if (!dst->next_sol[i])
            {
                ret = AOCLFFTZ_MEMORY_FAILURE;
                goto exit_deep_copy;
            }
        }
        // MT spawns `n` concurrent executions, each needing per_thread slots;
        // linear chain runs sequentially and reuses a single shared slice.
        ns_aux_bufs = is_real_mt ? n * per_thread_aux_buff
                                 : per_thread_aux_buff;
    }
    else
    {
        dst->next_sol = NULL;
    }

    FFTZ_INT32 aux_buff_idx = 0;
    if (src->solver->solver_type == SOLVER_REAL_BUFFERED)
    {
        aux_buff_idx++;
        if (setup_buffered_chain_structure(dst) != SOLVER_SUCCESS)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_deep_copy;
        }
    }

    if (aux_bufs)
    {
        *aux_bufs = aux_buff_idx + ns_aux_bufs;
    }

exit_deep_copy:
    if (ret != AOCLFFTZ_SUCCESS)
    {
        destroy_solution(dst);
        return NULL;
    }
    return dst;
}

/**
 * @brief Post-processes the real FFT solution tree for multi-threaded execution.
 *
 * At each REAL_MT_BATCHED node: deep-copies next_sol[0] for threads 1..N
 * so that each OMP-parallel batch has its own writable solution clone.
 *
 * Traversal rules:
 *  REAL_MT_BATCHED — recurse into next_sol[0], then deep-copy for threads 1..N.
 *  REAL_NDIM       — recurse into next_sol[0] (nd_sol is C2C, skipped).
 *
 * @param sol           Root of the solution (sub-)tree to process.
 * @param aux_buf_slots Output: REAL_BUFFERED aux slot demand (same max rules as ct).
 *                      May be NULL if the caller does not need the count.
 *
 * @return  AOCLFFTZ_SUCCESS (0) on success, AOCLFFTZ_MEMORY_FAILURE
 *          if any deep_copy_real_solution_tree call fails.
 */
static FFTZ_INT32 post_process_real_solution(aoclfftz_solution_t *sol,
                                        FFTZ_INT32 *aux_buf_slots)
{
    FFTZ_INT32 total_aux_buff = 0;

    while (sol != NULL)
    {
        if (sol->solver->solver_type == SOLVER_REAL_MT_BATCHED)
        {
            FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

            FFTZ_INT32 per_thread_aux_buff = 0;
            if (post_process_real_solution(sol->next_sol[0],
                                           &per_thread_aux_buff) !=
                SELECTOR_SUCCESS)
            {
                return AOCLFFTZ_MEMORY_FAILURE;
            }

            for (FFTZ_INT32 i = 1; i < n_threads; i++)
            {
                sol->next_sol[i] = deep_copy_real_solution_tree(
                                            sol->next_sol[0],
                                            i * per_thread_aux_buff, NULL, i);
                if (!sol->next_sol[i])
                {
                    return AOCLFFTZ_MEMORY_FAILURE;
                }
            }
            if (aux_buf_slots)
            {
                FFTZ_INT32 mt_aux_buff = n_threads * per_thread_aux_buff;
                *aux_buf_slots = total_aux_buff > mt_aux_buff ? total_aux_buff
                                                               : mt_aux_buff;
            }
            return SELECTOR_SUCCESS;
        }

        if (sol->solver->solver_type == SOLVER_REAL_NDIM)
        {
            FFTZ_INT32 nd_aux_bufs = 0, ns_aux_bufs = 0;
            if (post_process_real_solution(sol->dft_bufs->nd_sol,
                                           &nd_aux_bufs) !=
                SELECTOR_SUCCESS)
            {
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            if (post_process_real_solution(sol->next_sol[0], &ns_aux_bufs) !=
                                           SELECTOR_SUCCESS)
            {
                return AOCLFFTZ_MEMORY_FAILURE;
            }

            if (aux_buf_slots)
            {
                FFTZ_INT32 branch_max = nd_aux_bufs > ns_aux_bufs ? nd_aux_bufs
                                                                  : ns_aux_bufs;
                *aux_buf_slots = total_aux_buff > branch_max ? total_aux_buff
                                                             : branch_max;
            }
            return SELECTOR_SUCCESS;
        }

        if (sol->solver->solver_type == SOLVER_REAL_BUFFERED)
        {
            if (total_aux_buff < 1)
            {
                total_aux_buff = 1;
            }
        }

        sol = sol->next_sol ? sol->next_sol[0] : NULL;
    }

    if (aux_buf_slots)
    {
        *aux_buf_slots = total_aux_buff;
    }

    return SELECTOR_SUCCESS;
}
#endif /* MULTI_THREADING */


/**
 * @brief Walk the (post-setup) solution tree and record scratch sizes and
 * setup-time allocated scratch addresses into the handle's exec_metadata.
 *
 * For real plans, the walk enters the inner C2C sub-tree via nd_sol
 * and populates bluestein/ct_buf/sr pointers used by those solvers.
 *
 * aoclfftz_execute always uses these setup-time buffers. In execute_io only the
 * owning thread reuses them; other concurrent callers allocate a fresh per-call
 * slab from the sizes so threads sharing one handle don't trample each other.
 *
 * Sizing rules:
 *   - bs_buffer_size / bluestein: sum of each Bluestein node's padded, disjoint
 *     slice (active_threads_at_level * bs_buf_size); each node records its start
 *     in bs_dim_offset. Once the total size is known, the end of setup allocates
 *     the shared bs_in/bs_out pool (bs_buffer_size each), and each per-call
 *     execute_io allocates two matching regions of bs_buffer_size each.
 *   - sr_input_copy_size: max SR input_copy_size across all SR nodes.
 *   - ct_buffer_total_size: registers the largest CT buffer size across the CT
 *     scratch owners NDIM/BUFFERED/CTL1D only one among them allocates the pool.
 *
 * @param sol       Pointer to the root solution node of the plan tree to walk
 * @param metadata  Pointer to the execution metadata struct to populate with scratch sizes
 */
static FFTZ_VOID compute_exec_metadata(
                                    aoclfftz_solution_t *sol,
                                    aoclfftz_immutable_metadata_t *metadata)
{
    if (sol == NULL)
    {
        return;
    }

    aoclfftz_solver_type stype = sol->solver->solver_type;

    if (sol->dft_bufs->nd_sol != NULL)
    {
        compute_exec_metadata(sol->dft_bufs->nd_sol, metadata);
    }
    if (sol->dft_bufs->sr != NULL)
    {
        compute_exec_metadata(sol->dft_bufs->sr->odd1_sol, metadata);
        compute_exec_metadata(sol->dft_bufs->sr->odd3_sol, metadata);
    }

    if ((stype == SOLVER_BLUESTEIN || stype == SOLVER_MT_BLUESTEIN)
        && sol->dft_bufs->bluestein != NULL
        && sol->dft_bufs->bluestein->bs_buf_size > 0)
    {
        // Every Bluestein node shares one bs_[in/out]_base pool but gets a DISJOINT
        // slice, so different-sized nodes running concurrently never overlap.
        // Assign this node a region at the running total and grow the pool by
        // its full size (slots * per-slot bytes), 64-byte aligned for SIMD.
        //
        // The node slices its region by bs_slot_idx, a dense index in
        // [0, active_threads_at_level), so it needs active_threads_at_level slots.
        thread_info_t *thread_info = sol->decomp_scheme->thread_info;
        FFTZ_UINTP n_slots = (FFTZ_UINTP)thread_info->active_threads;
        FFTZ_UINTP node_size = n_slots
                          * (FFTZ_UINTP)sol->dft_bufs->bluestein->bs_buf_size;
        sol->dft_bufs->bluestein->bs_dim_offset = metadata->bs_buffer_size;
        metadata->bs_buffer_size += GET_PADDED_SIZE(node_size);
    }

    if (stype == SOLVER_SR && sol->dft_bufs->sr != NULL)
    {
        FFTZ_UINTP sr_size = (FFTZ_UINTP)sol->dft_bufs->sr->input_copy_size;
        if (sr_size > metadata->sr_input_copy_size)
        {
            metadata->sr_input_copy_size = sr_size;
            metadata->base_ctx.sr_input_copy_base =
                            sol->dft_bufs->sr->input_copy;
        }
    }

    // Replicate the ct_buffer allocation size used by every CT scratch owner.
    // A CTL1D owns its pool only when no parent pool exists, so all owners share
    // the single ct_buf_base region. Setup-time allocation sizes:
    //   - BUFFERED:    active_threads_at_level * ct_buf_size  (buffered_solver_dft.c)
    //   - CTL1D-owner: active_threads_at_level * ct_buf_size  (batched_ct_l1_direct_solver_dft.c)
    //   - NDIM:        active_threads_at_level * n_threads * ct_buf_size  (alloc_ndim_buffer)
    if (sol->dft_bufs->ct_buf_allocated && sol->dft_bufs->ct_buf_size > 0)
    {
        thread_info_t *thread_info = sol->decomp_scheme->thread_info;
        FFTZ_UINTP n_above = (FFTZ_UINTP)thread_info->active_threads;
        FFTZ_UINTP total = n_above * (FFTZ_UINTP)sol->dft_bufs->ct_buf_size;
        if (stype == SOLVER_NDIM)
        {
            FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->avl_threads;
            total *= (FFTZ_UINTP)n_threads;
        }
        if (total > metadata->ct_buffer_total_size)
        {
            metadata->ct_buffer_total_size = total;
            metadata->base_ctx.ct_buf_base = sol->dft_bufs->ct_buf_real;
        }
    }

    if (sol->next_sol != NULL && sol->next_sol[0] != NULL)
    {
        compute_exec_metadata(sol->next_sol[0], metadata);
    }
}

/**
 * @brief Walks the solution tree and computes the chirp FFT for every
 * Bluestein node.
 *
 * Called from prepare_and_setup_dft once the plan is fully built: all twiddles
 * are ready and the shared Bluestein pool is allocated.
 * A Bluestein node's inner FFT(M) subtree is complex (C2C): it is fully wired
 * at plan-build time and relies on the per-call execution ctx for
 * concurrency-safety. Computing the chirp actually executes each Bluestein
 * node's inner FFT(M), so this walk must run last - after the rest of the plan
 * is set up.
 *
 * Traversal rules:
 *  BLUESTEIN/     — compute chirp FFT for this node; next_sol[0] (the inner
 *  MT_BLUESTEIN     FFT(M) subtree) is then descended by the iterative step
 *                   below in case it itself contains another Bluestein node.
 *  NDIM/REAL_NDIM — recurse into nd_sol explicitly; its next_sol[0] is then
 *                   descended by the iterative step below.
 *  SR             — skip entirely; split-radix is power-of-2 only, so its
 *                   descendants cannot contain Bluestein nodes.
 *  others         — fall through to the iterative step (descend next_sol[0]).
 *
 * @param sol Root of the solution (sub-)tree to walk.
 * @param ctx Base execution context used to run each chirp FFT.
 * @return    SELECTOR_SUCCESS on success, AOCLFFTZ_SETUP_FAILURE if
 *            compute_chirp_fft fails.
 */
static FFTZ_INT32 setup_chirp_fft(aoclfftz_solution_t *sol,
                                  aoclfftz_mutable_ctx_t *ctx)
{
    while (sol != NULL)
    {
        if (sol->solver->solver_type == SOLVER_BLUESTEIN ||
            sol->solver->solver_type == SOLVER_MT_BLUESTEIN)
        {
            FFTZ_INT32 ret = compute_chirp_fft(sol, sol->next_sol[0], ctx);
            if (ret != SOLVER_SUCCESS)
            {
                AOCLFFTZ_ERROR(
                    "compute_chirp_fft failed for Bluestein node "
                    "(n=%ld, m=%ld)",
                    (long)sol->decomp_scheme->dims[0].n,
                    (long)sol->next_sol[0]->decomp_scheme->dims[0].n);
                return AOCLFFTZ_SETUP_FAILURE;
            }
        }

        if ((sol->solver->solver_type == SOLVER_NDIM) ||
            (sol->solver->solver_type == SOLVER_REAL_NDIM))
        {
            FFTZ_INT32 ret = setup_chirp_fft(sol->dft_bufs->nd_sol, ctx);
            if (ret != SELECTOR_SUCCESS)
            {
                return ret;
            }
        }

        if (sol->solver->solver_type == SOLVER_SR)
        {
            return SELECTOR_SUCCESS;
        }

        sol = sol->next_sol ? sol->next_sol[0] : NULL;
    }

    return SELECTOR_SUCCESS;
}

/**
 * @brief Check if col-major processing MT batched solver should be used based
 * on workload and stride heuristics
 *
 * This function implements a heuristic to choose between row-major and
 * col-major processing for multi-threaded batched solvers by analyzing workload
 * distribution and memory access patterns to avoid performance degradation from
 * false sharing.
 *
 * @param solution Pointer to the solution object
 * @param kertab Kernel table
 * @param avl_threads Available threads count
 * @return FFTZ_UINT8 1 if col-major should be used, 0 otherwise
 */
FFTZ_UINT8 should_use_colmajor_batched_solver(aoclfftz_solution_t *solution,
                                                kernel_t *kertab,
                                                FFTZ_INT32 avl_threads)
{
    FFTZ_INTP radix = solution->decomp_scheme->dims[0].n;
    FFTZ_DOUBLE kernel_weightage =
        get_kernel_weightage(radix, kertab, solution);
    // Compute workload distribution across available threads
    // Formula: (batch_count * CT vector_size * computational_intensity of the
    // kernel) / thread_count
    FFTZ_DOUBLE workload_per_thread =
        (FFTZ_DOUBLE)(solution->decomp_scheme->batched_vecs[0].n *
                 solution->decomp_scheme->vecs[0].n * kernel_weightage) /
        (FFTZ_DOUBLE)(avl_threads);

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
 * @return FFTZ_DOUBLE Computed weightage value based on operation cycles,
 * scaled by 1/KERNEL_WEIGHTAGE_SCALE_FACTOR. Returns 1.0 if no matching kernel
 * radix is found.
 *
 */
FFTZ_DOUBLE get_kernel_weightage(FFTZ_INTP radix, kernel_t *kertab,
                            aoclfftz_solution_t *sol)
{
    FFTZ_DOUBLE weightage = 1.0;
    FFTZ_UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    ops_cycles_t ops_cycles = {0};
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        if ((FFTZ_INTP)kertab[i].radix == radix)
        {
            ops_cycles = kertab[i].k_ops_cnt(precision, direction);
            weightage = (FFTZ_DOUBLE)((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                                 (ops_cycles.mul * AMD_ZEN_FP_MUL_CYCLES) +
                                 (ops_cycles.add * AMD_ZEN_FP_ADD_CYCLES)) /
                        KERNEL_WEIGHTAGE_SCALE_FACTOR;
            break;
        }
    }
    return weightage;
}
