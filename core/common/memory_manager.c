// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file memory_manager.c
 *
 *  @brief Declares memory allocation and management functions of AOCL-FFTZ.
 *
 *  This file contains the function declarations for performing memory
 *  allocation management of different modules of the AOCL-FFTZ library.
 *
 *  @author S. Biplab Raut
 */

#include "core/common/memory_manager.h"
#include "api/aoclfftz_internal.h"

/**
 * Compute the maximum buffer size needed for an N-dimensional real FFT
 * - For dimension 0:   (n0 / 2 + 1) * stride_0
 * - For other dims:    (ni - 1) * stride_i
 * Strides are chosen based on FFT direction (forward or backward).
 */
FFTZ_UINTP calculate_max_buffer_size(aoclfftz_solution_t *sol)
{
    FFTZ_UINTP max_size = 1;

    // Compute max buffer size for ND real FFT using half-complex for first dim
    // Uses output stride for forward, input stride for backward
    FFTZ_UINT8 is_forward =
        (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR);
    FFTZ_INTP dim0_size = sol->decomp_scheme->dims[0].n / 2 + 1;
    FFTZ_INTP dim0_stride = is_forward ? sol->decomp_scheme->dims[0].out_stride
                                  : sol->decomp_scheme->dims[0].in_stride;
    max_size += ((dim0_size - 1) * dim0_stride);
    for (FFTZ_INT32 i = 1; i < sol->decomp_scheme->dim_rank; i++)
    {
        FFTZ_INTP dimi_size = sol->decomp_scheme->dims[i].n;
        FFTZ_INTP dimi_stride =
            is_forward ? sol->decomp_scheme->dims[i].out_stride
                       : sol->decomp_scheme->dims[i].in_stride;
        max_size += ((dimi_size - 1) * dimi_stride);
    }
    return max_size;
}

aoclfftz_solution_t *alloc_solution(FFTZ_INT32 vec_rank, FFTZ_INT32 dim_rank)
{
    aoclfftz_solution_t *sol = NULL;
    FFTZ_UINT32 alloc_bytes = sizeof(aoclfftz_solution_t) +
        sizeof(aoclfftz_generic_solver_t) +
        sizeof(kernel_info_t) +
        sizeof(kernel_info_t) +
        sizeof(kernel_info_t) +
        sizeof(kernel_info_t) +
        sizeof(aoclfftz_decomp_scheme_t) +
        //vec_rank and dim_rank can be large numbers, and so should better be
        //allocated separately ? This may avoid a discontinuity in solution_t.
        /*(dim_rank * sizeof(aoclfftz_dim_t_64_)) +
        (vec_rank * sizeof(aoclfftz_dim_t_64_)) +
        sizeof(aoclfftz_cntrl_params_t) +
        sizeof(thread_info_t) +
        sizeof(aoclfftz_smp_pfft_t) + */
        sizeof(aoclfftz_strides_grp_t) +
        (4 * sizeof(aoclfftz_strides_t)) +
        sizeof(aoclfftz_dft_bufs_t) +
        sizeof(aoclfftz_bluestein_t) +
        sizeof(aoclfftz_buffered_t) +
        sizeof(aoclfftz_sr_t) +
        sizeof(aoclfftz_transpose_t) +
        sizeof(aoclfftz_transpose_aux_mem_t) +
        sizeof(aoclfftz_twiddle_t);

    ALLOC_ALIGN_UNINIT(sol, aoclfftz_solution_t, alloc_bytes);
    if (sol)
    {
        sol->solver = (aoclfftz_generic_solver_t*)((FFTZ_UINT8*)sol + sizeof(aoclfftz_solution_t));
        sol->solver->kernel_c2c = (kernel_info_t*)((FFTZ_UINT8*)sol->solver + sizeof(aoclfftz_generic_solver_t));
        sol->solver->kernel_c2c_r = (kernel_info_t*)((FFTZ_UINT8*)sol->solver->kernel_c2c + sizeof(kernel_info_t));
        sol->solver->kernel_r2hc = (kernel_info_t*)((FFTZ_UINT8*)sol->solver->kernel_c2c_r + sizeof(kernel_info_t));
        sol->solver->kernel_r2hcf = (kernel_info_t*)((FFTZ_UINT8*)sol->solver->kernel_r2hc + sizeof(kernel_info_t));

        sol->solver->solver_type = SOLVER_NULL;
        sol->solver->execute_solver = NULL;
        sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] = NULL;
        sol->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] = NULL;
        sol->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR] = NULL;
        sol->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR] = NULL;
        sol->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR] = NULL;
        sol->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR] = NULL;
        sol->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR] = NULL;
        sol->solver->kernel_r2hcf->kfft[BACKWARD_FFT_DIR] = NULL;
        sol->solver->kernel_c2c->sets = 1;
        sol->solver->kernel_c2c_r->sets = 1;
        sol->solver->kernel_r2hc->sets = 1;
        sol->solver->kernel_r2hcf->sets = 1;
        sol->solver->destroy_solver = NULL;

        sol->decomp_scheme = (aoclfftz_decomp_scheme_t*)((FFTZ_UINT8*)sol->solver->kernel_r2hcf + sizeof(kernel_info_t));
        sol->strides_grp = (aoclfftz_strides_grp_t*)((FFTZ_UINT8*)sol->decomp_scheme + sizeof(aoclfftz_decomp_scheme_t));
        sol->strides_grp->strides = (aoclfftz_strides_t*)((FFTZ_UINT8*)sol->strides_grp + sizeof(aoclfftz_strides_grp_t));
        sol->strides_grp->strides_c2c = (aoclfftz_strides_t*)((FFTZ_UINT8*)sol->strides_grp->strides + sizeof(aoclfftz_strides_t));
        sol->strides_grp->strides_r2hc = (aoclfftz_strides_t*)((FFTZ_UINT8*)sol->strides_grp->strides_c2c + sizeof(aoclfftz_strides_t));
        sol->strides_grp->strides_r2hcf = (aoclfftz_strides_t*)((FFTZ_UINT8*)sol->strides_grp->strides_r2hc + sizeof(aoclfftz_strides_t));

        sol->dft_bufs = (aoclfftz_dft_bufs_t*)((FFTZ_UINT8*)sol->strides_grp->strides_r2hcf + sizeof(aoclfftz_strides_t));
        sol->dft_bufs->bluestein = (aoclfftz_bluestein_t*)((FFTZ_UINT8*)sol->dft_bufs + sizeof(aoclfftz_dft_bufs_t));
        sol->dft_bufs->buffered = (aoclfftz_buffered_t*)((FFTZ_UINT8*)sol->dft_bufs->bluestein + sizeof(aoclfftz_bluestein_t));
        sol->dft_bufs->sr = (aoclfftz_sr_t*)((FFTZ_UINT8*)sol->dft_bufs->buffered + sizeof(aoclfftz_buffered_t));
        sol->dft_bufs->transpose = (aoclfftz_transpose_t*)((FFTZ_UINT8*)sol->dft_bufs->sr + sizeof(aoclfftz_sr_t));
        sol->dft_bufs->transpose->aux_mem = (aoclfftz_transpose_aux_mem_t*)((FFTZ_UINT8*)sol->dft_bufs->transpose + sizeof(aoclfftz_transpose_t));

        sol->twiddle = (aoclfftz_twiddle_t*)((FFTZ_UINT8*)sol->dft_bufs->transpose->aux_mem + sizeof(aoclfftz_transpose_aux_mem_t));

        /* Initialize SR struct fields */
        sol->dft_bufs->sr->odd1_sol = NULL;
        sol->dft_bufs->sr->odd3_sol = NULL;
        sol->dft_bufs->sr->input_copy = NULL;
        sol->dft_bufs->sr->input_copy_size = 0;
        sol->next_sol = NULL;
        sol->decomp_scheme->nyquist_im_offset_direct = 1;
        sol->decomp_scheme->nyquist_im_offset_ct = 1;

        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->dims, aoclfftz_dim_t_64_,
            dim_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->vecs, aoclfftz_dim_t_64_,
            vec_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->cntrl_params, aoclfftz_cntrl_params_t,
            sizeof(aoclfftz_cntrl_params_t));
        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->thread_info, thread_info_t,
            sizeof(thread_info_t));
        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->thread_info->pthr_fft,
            aoclfftz_smp_pfft_t, sizeof(aoclfftz_smp_pfft_t));

        if (sol->decomp_scheme->dims == NULL || sol->decomp_scheme->vecs == NULL ||
            sol->decomp_scheme->cntrl_params == NULL ||
            sol->decomp_scheme->thread_info == NULL ||
            sol->decomp_scheme->thread_info->pthr_fft == NULL)
        {
            destroy_decomp_scheme(sol->decomp_scheme);
            FREE_ALIGN_ALLOCATED_MEM(sol);
            return NULL;
        }
        sol->decomp_scheme->thread_info->active_threads = 1;
        sol->next_sol = NULL;
        sol->decomp_scheme->batched_vecs = NULL;
        sol->dft_bufs->nd_sol = NULL;
        sol->dft_bufs->pow2_iterative = NULL;
        sol->dft_bufs->pow2_fourstep = NULL;
        sol->strides_grp->strides->in_strides = NULL;
        sol->strides_grp->strides->out_strides = NULL;
        sol->strides_grp->strides->v_in_stride = 0;
        sol->strides_grp->strides->v_out_stride = 0;
        sol->strides_grp->strides->v_in_sym_stride = 0;
        sol->strides_grp->strides->v_out_sym_stride = 0;
        sol->strides_grp->strides_c2c->in_strides = NULL;
        sol->strides_grp->strides_c2c->out_strides = NULL;
        sol->strides_grp->strides_c2c->v_in_stride = 0;
        sol->strides_grp->strides_c2c->v_out_stride = 0;
        sol->strides_grp->strides_c2c->v_in_sym_stride = 0;
        sol->strides_grp->strides_c2c->v_out_sym_stride = 0;
        sol->strides_grp->strides_r2hc->in_strides = NULL;
        sol->strides_grp->strides_r2hc->out_strides = NULL;
        sol->strides_grp->strides_r2hc->v_in_stride = 0;
        sol->strides_grp->strides_r2hc->v_out_stride = 0;
        sol->strides_grp->strides_r2hc->v_in_sym_stride = 0;
        sol->strides_grp->strides_r2hc->v_out_sym_stride = 0;
        sol->strides_grp->strides_r2hcf->in_strides = NULL;
        sol->strides_grp->strides_r2hcf->out_strides = NULL;
        sol->strides_grp->strides_r2hcf->v_in_stride = 0;
        sol->strides_grp->strides_r2hcf->v_out_stride = 0;
        sol->strides_grp->strides_r2hcf->v_in_sym_stride = 0;
        sol->strides_grp->strides_r2hcf->v_out_sym_stride = 0;
        sol->twiddle->load_multi_cols = 1; // true by default
        sol->twiddle->TW = NULL;
        sol->twiddle->twiddle_buf_ptr = NULL;
        sol->dft_bufs->bluestein->B = NULL;
        sol->dft_bufs->bluestein->B_out = NULL;
        sol->dft_bufs->bluestein->bs_buf_size = 0;
        sol->dft_bufs->bluestein->bs_dim_offset = 0;
        sol->dft_bufs->bluestein->pre_mul[FORWARD_FFT_DIR]  = NULL;
        sol->dft_bufs->bluestein->pre_mul[BACKWARD_FFT_DIR] = NULL;
        sol->dft_bufs->bluestein->mul[FORWARD_FFT_DIR]  = NULL;
        sol->dft_bufs->bluestein->mul[BACKWARD_FFT_DIR] = NULL;
        sol->dft_bufs->bluestein->post_mul[FORWARD_FFT_DIR]  = NULL;
        sol->dft_bufs->bluestein->post_mul[BACKWARD_FFT_DIR] = NULL;
        sol->dft_bufs->bluestein->cast_to_complex = NULL;
        sol->dft_bufs->bluestein->cast_from_complex = NULL;
        sol->dft_bufs->buffered->aux_buffer_1 = NULL;
        sol->dft_bufs->buffered->aux_buffer_2 = NULL;
        sol->dft_bufs->buffered->is_aux_buffer_allocated = 0;
        sol->dft_bufs->buffered->aux_buf_size_per_thread = 0;
        sol->decomp_scheme->thread_info->ndim_concurrency = 0;
        sol->decomp_scheme->real_in_role = REAL_USE_IO_BUF;
        sol->decomp_scheme->real_out_role = REAL_USE_IO_BUF;
        sol->dft_bufs->transpose->row_info = (aoclfftz_dim_t_64_){0};
        sol->dft_bufs->transpose->col_info = (aoclfftz_dim_t_64_){0};
        sol->dft_bufs->transpose->aux_mem->size = 0;
        sol->dft_bufs->transpose->aux_mem->data = NULL;
        sol->dft_bufs->ct_buffer = NULL;
        sol->dft_bufs->ct_buf_real = NULL;
        sol->dft_bufs->ct_buf_imag = NULL;
        sol->dft_bufs->ct_buf_size = 0;
        sol->dft_bufs->ct_buf_allocated = 0;
        sol->solver->kernel_c2c->count = 0;
        sol->solver->kernel_c2c_r->count = 0;
        sol->solver->kernel_r2hc->count = 0;
        sol->solver->kernel_r2hcf->count = 0;
        return sol;
    }
    else
    {
        return NULL;
    }
}

// Allocates memory for the input and output stride arrays within the
// provided strides structure if they are not already allocated.
FFTZ_INT32 alloc_stride_arrays(aoclfftz_strides_t *strides, FFTZ_INTP radix)
{
    if (strides->in_strides == NULL)
    {
        ALLOC_ALIGN_UNINIT(strides->in_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));
        if (strides->in_strides == NULL)
        {
            AOCLFFTZ_ERROR("alloc_stride_arrays (in_strides) failed: %s",
                           get_status_string(AOCLFFTZ_MEMORY_FAILURE));
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        ALLOC_ALIGN_UNINIT(strides->out_strides, FFTZ_INTP,
                           radix * sizeof(FFTZ_INTP));
        if (strides->out_strides == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(strides->in_strides);
            AOCLFFTZ_ERROR("alloc_stride_arrays (out_strides) failed: %s",
                           get_status_string(AOCLFFTZ_MEMORY_FAILURE));
            return AOCLFFTZ_MEMORY_FAILURE;
        }
    }
    return AOCLFFTZ_SUCCESS;
}

// Allocates (if not already allocated) and fills stride arrays with a
// uniform element-stride pattern: strides[i] = i * stride * DATA_STRIDE.
FFTZ_INT32 alloc_and_fill_stride_arrays(aoclfftz_strides_t *strides,
                                        FFTZ_INTP radix, FFTZ_INTP in_stride,
                                        FFTZ_INTP out_stride)
{
    FFTZ_INT32 ret = alloc_stride_arrays(strides, radix);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    for (FFTZ_INTP i = 0; i < radix; i++)
    {
        strides->in_strides[i]  = i * in_stride * DATA_STRIDE;
        strides->out_strides[i] = i * out_stride * DATA_STRIDE;
    }
    return AOCLFFTZ_SUCCESS;
}

aoclfftz_selector_t *alloc_selector(FFTZ_INT32 vec_rank, FFTZ_INT32 dim_rank,
                                    kernel_tables_t *kernel_tables,
                                    FFTZ_UINT8 *has_nested)
{
    aoclfftz_selector_t *selector = NULL;

    ALLOC_ALIGN_UNINIT(selector, aoclfftz_selector_t,
                       sizeof(aoclfftz_selector_t));
    if (selector)
    {
        selector->kernel_tables = NULL;
        selector->exec_metadata = NULL;
        selector->has_nested = has_nested;

        selector->solution = alloc_solution(vec_rank, dim_rank);
        ALLOC_ALIGN_UNINIT(selector->cost_analysis, cost_analysis_t,
                           sizeof(cost_analysis_t));
        // Allocate kernel_tables if provided
        if (kernel_tables != NULL)
        {
            ALLOC_ALIGN_UNINIT(selector->kernel_tables, kernel_tables_t,
                               sizeof(kernel_tables_t));
        }

        if (selector->solution == NULL || selector->cost_analysis == NULL ||
            (kernel_tables != NULL && selector->kernel_tables == NULL))
        {
            destroy_selector(selector);
            return NULL;
        }

        selector->cost_analysis->ops = 0;
        selector->cost_analysis->time = 0;

        if (kernel_tables != NULL && selector->kernel_tables != NULL)
        {
            // copy kernel table
            selector->kernel_tables->kt_dft = kernel_tables->kt_dft;
            selector->kernel_tables->kt_twid_dft = kernel_tables->kt_twid_dft;
            selector->kernel_tables->kt_rdft = kernel_tables->kt_rdft;
            selector->kernel_tables->ele_mul[FORWARD_FFT_DIR] =
                kernel_tables->ele_mul[FORWARD_FFT_DIR];
            selector->kernel_tables->ele_mul[BACKWARD_FFT_DIR] =
                kernel_tables->ele_mul[BACKWARD_FFT_DIR];
            selector->kernel_tables
                ->ele_mul_strided_in[FORWARD_FFT_DIR] =
                kernel_tables->ele_mul_strided_in[FORWARD_FFT_DIR];
            selector->kernel_tables
                ->ele_mul_strided_in[BACKWARD_FFT_DIR] =
                kernel_tables->ele_mul_strided_in[BACKWARD_FFT_DIR];
            selector->kernel_tables->ele_mul_fused_norm[FORWARD_FFT_DIR] =
                kernel_tables->ele_mul_fused_norm[FORWARD_FFT_DIR];
            selector->kernel_tables->ele_mul_fused_norm[BACKWARD_FFT_DIR] =
                kernel_tables->ele_mul_fused_norm[BACKWARD_FFT_DIR];
            selector->kernel_tables->ele_mul_fused_norm_strided_out[FORWARD_FFT_DIR] =
                kernel_tables->ele_mul_fused_norm_strided_out[FORWARD_FFT_DIR];
            selector->kernel_tables->ele_mul_fused_norm_strided_out[BACKWARD_FFT_DIR] =
                kernel_tables->ele_mul_fused_norm_strided_out[BACKWARD_FFT_DIR];
            selector->kernel_tables->type_convert_r2c =
                kernel_tables->type_convert_r2c;
            selector->kernel_tables->type_convert_c2hc =
                kernel_tables->type_convert_c2hc;
            selector->kernel_tables->type_convert_hc2c =
                kernel_tables->type_convert_hc2c;
            selector->kernel_tables->type_convert_c2r =
                kernel_tables->type_convert_c2r;
        }

        return selector;
    }
    else
    {
        return NULL;
    }
}

/**
 * Allocates the 64-byte aligned shared Bluestein chirp buffers B/B_out
 * (bs_buf_size bytes each). Frees any partial allocation on failure.
 *
 * @return AOCLFFTZ_SUCCESS on success, or AOCLFFTZ_MEMORY_FAILURE if any
 *         allocation fails.
 */
FFTZ_INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein,
                                   FFTZ_INTP bs_buf_size)
{
    bluestein->B = NULL;
    bluestein->B_out = NULL;

    ALLOC_ALIGN_UNINIT(bluestein->B, FFTZ_VOID, bs_buf_size);
    if (bluestein->B == NULL)
    {
        goto exit_alloc_bluestein_buffers;
    }

    ALLOC_ALIGN_UNINIT(bluestein->B_out, FFTZ_VOID, bs_buf_size);
    if (bluestein->B_out == NULL)
    {
        goto exit_alloc_bluestein_buffers;
    }

    bluestein->bs_buf_size = bs_buf_size;
    return AOCLFFTZ_SUCCESS;

exit_alloc_bluestein_buffers:
    FREE_ALIGN_ALLOCATED_MEM(bluestein->B_out);
    FREE_ALIGN_ALLOCATED_MEM(bluestein->B);
    return AOCLFFTZ_MEMORY_FAILURE;
}

FFTZ_VOID *alloc_twiddle_buffer(FFTZ_UINTP size, FFTZ_UINT32 dt_prec)
{
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_VOID *buffer = NULL;
    ALLOC_ALIGN_UNINIT(buffer, FFTZ_VOID, DATA_STRIDE * size * dt_bytes);
    return buffer;
}

/**
 * Allocates ct_buffer for complex NDIM (see ndim_solver_dft.c).
 * Uses active_threads if set by parent (e.g. REAL_NDIM);
 * otherwise one slot.
 */
FFTZ_INT32 alloc_ndim_buffer(aoclfftz_solution_t *solution,
                             FFTZ_VOID **buffer_ptr)
{
    AOCLFFTZ_LOG(INFO, global_logger_mode,
                 "Allocating ct_buffer for complex NDIM");

    if (solution->dft_bufs->ct_buffer)
    {
        return AOCLFFTZ_SUCCESS;
    }
    // allocate buffer for the entire problem
    // this can be optimized by :
    //      1. making the buffer unit strided even if input is strided

    FFTZ_INT32 dim_rank = solution->decomp_scheme->dim_rank;
    aoclfftz_dim_t_64_ *dims = solution->decomp_scheme->dims;

    FFTZ_UINTP buffer_length = 1;
    FFTZ_UINTP buffer_size = 0;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(solution);

    // Approach: if the problem is ND where N>2, then create ct_buffer of (N-1)D
    // by removing the smallest dim for. e.g. problem size of 30x40x50 ->
    // ct_buffer of 40x50 for multi-threaded problems, this 2D buffer will be
    // created per thread.
    FFTZ_INTP min_dim_size = dims[0].n;
    for (FFTZ_INT32 i = 0; i < dim_rank; i++)
    {
        buffer_length += ((dims[i].n - 1) * (dims[i].out_stride));
        if (dims[i].n < min_dim_size)
        {
            min_dim_size = dims[i].n;
        }
    }
    FFTZ_INT32 n_threads = solution->decomp_scheme->thread_info->avl_threads;
    if (solution->decomp_scheme->dim_rank > 2)
    {
        buffer_length = buffer_length / min_dim_size;
    }
    buffer_size = GET_PADDED_SIZE(buffer_length * DATA_STRIDE * dt_bytes);

    // for 2D problems, a 2D sized buffer is sufficient for avl_threads to run
    // MT.
    if (dim_rank == 2 && n_threads > 1)
    {
        n_threads = 1;
    }

    FFTZ_INT32 active_threads =
        solution->decomp_scheme->thread_info->active_threads;
    ALLOC_ALIGN_UNINIT(*buffer_ptr, FFTZ_VOID,
            buffer_size * active_threads * n_threads);
    if (*buffer_ptr == NULL)
    {
        AOCLFFTZ_ERROR("alloc_ndim_buffer (ct_buffer) failed: %s",
                       get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    solution->dft_bufs->ct_buf_size = buffer_size;
    solution->dft_bufs->ct_buf_real = *buffer_ptr;
    solution->dft_bufs->ct_buf_imag = MOVE_ADDR(*buffer_ptr, dt_bytes);
    solution->dft_bufs->ct_buf_allocated = 1;
    return AOCLFFTZ_SUCCESS;
}

FFTZ_VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    if (decomp_scheme != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->dims);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->vecs);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->batched_vecs);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->cntrl_params);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->thread_info->pthr_fft);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->thread_info);
    }
    return;
}

FFTZ_VOID destroy_transpose(aoclfftz_transpose_t* transpose)
{
    if (transpose)
    {
        if (transpose->aux_mem)
        {
            FREE_ALIGN_ALLOCATED_MEM(transpose->aux_mem->data);
        }
    }
}

FFTZ_VOID destroy_bluestein(aoclfftz_bluestein_t* bluestein)
{
    // Only the owning node has non-NULL B/B_out (next_sol is copied
    // before allocation, so its pointers stay NULL)
    if (bluestein != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B_out);
    }
}

// Frees the pow2 iterative solver state: the ping-pong pool (both buffers, one
// pair per slot), the per-stage stride arrays, the stage array and the struct.
// The twiddle block is freed by the generic solution teardown.
FFTZ_VOID destroy_pow2_iterative(aoclfftz_pow2_iterative_t *pow2_iterative)
{
    if (pow2_iterative == NULL)
    {
        return;
    }
    if (pow2_iterative->stages != NULL)
    {
        for (FFTZ_INT32 s = 0; s < pow2_iterative->num_stages; s++)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                pow2_iterative->stages[s].stage_info.strides.in_strides);
            FREE_ALIGN_ALLOCATED_MEM(
                pow2_iterative->stages[s].stage_info.strides.out_strides);
        }
    }
    FREE_ALIGN_ALLOCATED_MEM(pow2_iterative->pingpong_buf);
    FREE_ALIGN_ALLOCATED_MEM(pow2_iterative->stages);
    FREE_ALIGN_ALLOCATED_MEM(pow2_iterative);
}

// Frees the four-step state: scratch pool, per-sub-FFT stage and stride arrays,
// and the struct. Twiddle tables are freed by the generic solution teardown.
FFTZ_VOID destroy_pow2_fourstep(aoclfftz_pow2_fourstep_t *pow2_fourstep)
{
    if (pow2_fourstep == NULL)
    {
        return;
    }
    aoclfftz_pow2_fourstep_subfft_t *subffts[2] = {&pow2_fourstep->sub1,
                                                   &pow2_fourstep->sub2};
    for (FFTZ_INT32 sf = 0; sf < 2; sf++)
    {
        if (subffts[sf]->stages == NULL)
        {
            continue;
        }
        for (FFTZ_INT32 s = 0; s < subffts[sf]->num_stages; s++)
        {
            FREE_ALIGN_ALLOCATED_MEM(subffts[sf]->stages[s].strides.in_strides);
            FREE_ALIGN_ALLOCATED_MEM(subffts[sf]->stages[s].strides.out_strides);
        }
    }
    FREE_ALIGN_ALLOCATED_MEM(pow2_fourstep->sub1.stages);
    FREE_ALIGN_ALLOCATED_MEM(pow2_fourstep->sub2.stages);
    FREE_ALIGN_ALLOCATED_MEM(pow2_fourstep->scratch);
    FREE_ALIGN_ALLOCATED_MEM(pow2_fourstep);
}

FFTZ_VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp)
{
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2c->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2c->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hc->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hc->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hcf->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hcf->out_strides);
}

/**
 * Free aux scratch that this node allocated.
 *
 * In a real plan, one node mallocs the aux pool and everyone else borrows it.
 * Child CT/Direct nodes just hold pointers into that pool, so we do nothing
 * for them. We only free when is_aux_buffer_allocated is set on this node.
 *
 * REAL_BUFFERED gets two buffers. The buffered solver uses them as a
 * ping-pong pair while data moves through the CT direct chain
 * (see update_ct_buffers()). Both must be freed when the plan is torn down.
 *
 * REAL_BATCHED_CT_L1_DIRECT gets one buffer. Fused execute passes data from
 * stage_r to stage_m through aux_buffer_1 only; aux_buffer_2 is never used.
 *
 * REAL_NDIM may allocate aux_buffer_1 for inplace/C2R cases (no
 * is_aux_buffer_allocated flag; free when aux_buffer_1 is non-NULL).
 *
 */

FFTZ_VOID release_owned_real_buffered_aux(aoclfftz_solution_t *sol)
{
    if (sol == NULL || sol->solver == NULL || sol->dft_bufs == NULL ||
        sol->dft_bufs->buffered == NULL ||
        sol->dft_bufs->buffered->is_aux_buffer_allocated == 0)
    {
        return;
    }

    FFTZ_INT32 solver_type = sol->solver->solver_type;
    FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
    if (solver_type == SOLVER_REAL_BUFFERED)
    {
        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_2);
    }
    sol->dft_bufs->buffered->aux_buffer_2 = NULL;
    sol->dft_bufs->buffered->is_aux_buffer_allocated = 0;
}

FFTZ_VOID destroy_solution(aoclfftz_solution_t* sol)
{
    if (sol != NULL)
    {
        destroy_decomp_scheme(sol->decomp_scheme);
        destroy_strides_grp(sol->strides_grp);

        FREE_ALIGN_ALLOCATED_MEM(sol->twiddle->twiddle_buf_ptr);
        sol->twiddle->TW = NULL;

        destroy_bluestein(sol->dft_bufs->bluestein);
        destroy_transpose(sol->dft_bufs->transpose);

        if (sol->dft_bufs->ct_buf_allocated)
        {
            FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->ct_buffer);
            sol->dft_bufs->ct_buf_allocated = 0;
        }
        destroy_pow2_iterative(sol->dft_bufs->pow2_iterative);
        destroy_pow2_fourstep(sol->dft_bufs->pow2_fourstep);
        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->sr->input_copy);

        release_owned_real_buffered_aux(sol);
        destroy_solution(sol->dft_bufs->nd_sol);
        destroy_solution(sol->dft_bufs->sr->odd1_sol);
        destroy_solution(sol->dft_bufs->sr->odd3_sol);
        destroy_solution(sol->next_sol);

        FREE_ALIGN_ALLOCATED_MEM(sol);
    }
    return;
}

// Note: Since the use case for this function is primarily to free "temporary"
//       selectors, and since temporary selectors are expected to be allocated
//       using `alloc_selector_without_scratch_space`, this function does not
//       free the scratch space.
FFTZ_VOID destroy_selector_without_solution(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(sel->cost_analysis);
        FREE_ALIGN_ALLOCATED_MEM(sel->kernel_tables);
        if (sel->exec_metadata != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(sel->exec_metadata->base_ctx.bs_in_base);
            FREE_ALIGN_ALLOCATED_MEM(sel->exec_metadata->base_ctx.bs_out_base);
            FREE_ALIGN_ALLOCATED_MEM(
                sel->exec_metadata->base_ctx.c2c_strides_base);
        }
        FREE_ALIGN_ALLOCATED_MEM(sel->exec_metadata);
        FREE_ALIGN_ALLOCATED_MEM(sel);
    }
    return;
}

FFTZ_VOID destroy_selector(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        destroy_solution(sel->solution);
        destroy_selector_without_solution(sel);
    }
    return;
}
