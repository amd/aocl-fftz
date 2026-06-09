// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file memory_manager.h
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

aoclfftz_decomp_scheme_t *alloc_decomp_scheme(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = NULL;

    ALLOC_ALIGN_UNINIT(decomp_scheme, aoclfftz_decomp_scheme_t,
                       sizeof(aoclfftz_decomp_scheme_t));
    if (decomp_scheme)
    {
        ALLOC_ALIGN_UNINIT(decomp_scheme->dims, aoclfftz_dim_t_64_,
                           dim_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(decomp_scheme->vecs, aoclfftz_dim_t_64_,
                           vec_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(decomp_scheme->cntrl_params, aoclfftz_cntrl_params_t,
                           sizeof(aoclfftz_cntrl_params_t));
        ALLOC_ALIGN_UNINIT(decomp_scheme->thread_info, thread_info_t,
                           sizeof(thread_info_t));
        ALLOC_ALIGN_UNINIT(decomp_scheme->thread_info->pthr_fft,
                           aoclfftz_smp_pfft_t, sizeof(aoclfftz_smp_pfft_t));
        decomp_scheme->batched_vecs = NULL;
        if (decomp_scheme->dims == NULL || decomp_scheme->vecs == NULL ||
            decomp_scheme->cntrl_params == NULL ||
            decomp_scheme->thread_info == NULL ||
            decomp_scheme->thread_info->pthr_fft == NULL)
        {
            destroy_decomp_scheme(decomp_scheme);
            FREE_ALIGN_ALLOCATED_MEM(decomp_scheme);
            return NULL;
        }
        return decomp_scheme;
    }
    else
    {
        return NULL;
    }
}

aoclfftz_solution_t *alloc_solution(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_solution_t *sol = NULL;
    UINT32 alloc_bytes = sizeof(aoclfftz_solution_t) +
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
        sol->solver = (aoclfftz_generic_solver_t*)((UINT8*)sol + sizeof(aoclfftz_solution_t));
        sol->solver->kernel_c2c = (kernel_info_t*)((UINT8*)sol->solver + sizeof(aoclfftz_generic_solver_t));
        sol->solver->kernel_c2c_r = (kernel_info_t*)((UINT8*)sol->solver->kernel_c2c + sizeof(kernel_info_t));
        sol->solver->kernel_r2hc = (kernel_info_t*)((UINT8*)sol->solver->kernel_c2c_r + sizeof(kernel_info_t));
        sol->solver->kernel_r2hcf = (kernel_info_t*)((UINT8*)sol->solver->kernel_r2hc + sizeof(kernel_info_t));

        sol->solver->solver_type = SOLVER_NULL;
        sol->solver->execute_solver = NULL;
        sol->solver->kernel_c2c->kfft = NULL;
        sol->solver->kernel_c2c_r->kfft = NULL;
        sol->solver->kernel_r2hc->kfft = NULL;
        sol->solver->kernel_r2hcf->kfft = NULL;
        sol->solver->kernel_c2c->sets = 1;
        sol->solver->kernel_c2c_r->sets = 1;
        sol->solver->kernel_r2hc->sets = 1;
        sol->solver->kernel_r2hcf->sets = 1;
        sol->solver->destroy_solver = NULL;

        sol->decomp_scheme = (aoclfftz_decomp_scheme_t*)((UINT8*)sol->solver->kernel_r2hcf + sizeof(kernel_info_t));
        sol->strides_grp = (aoclfftz_strides_grp_t*)((UINT8*)sol->decomp_scheme + sizeof(aoclfftz_decomp_scheme_t));
        sol->strides_grp->strides = (aoclfftz_strides_t*)((UINT8*)sol->strides_grp + sizeof(aoclfftz_strides_grp_t));
        sol->strides_grp->strides_c2c = (aoclfftz_strides_t*)((UINT8*)sol->strides_grp->strides + sizeof(aoclfftz_strides_t));
        sol->strides_grp->strides_r2hc = (aoclfftz_strides_t*)((UINT8*)sol->strides_grp->strides_c2c + sizeof(aoclfftz_strides_t));
        sol->strides_grp->strides_r2hcf = (aoclfftz_strides_t*)((UINT8*)sol->strides_grp->strides_r2hc + sizeof(aoclfftz_strides_t));

        sol->dft_bufs = (aoclfftz_dft_bufs_t*)((UINT8*)sol->strides_grp->strides_r2hcf + sizeof(aoclfftz_strides_t));
        sol->dft_bufs->bluestein = (aoclfftz_bluestein_t*)((UINT8*)sol->dft_bufs + sizeof(aoclfftz_dft_bufs_t));
        sol->dft_bufs->buffered = (aoclfftz_buffered_t*)((UINT8*)sol->dft_bufs->bluestein + sizeof(aoclfftz_bluestein_t));
        sol->dft_bufs->sr = (aoclfftz_sr_t*)((UINT8*)sol->dft_bufs->buffered + sizeof(aoclfftz_buffered_t));
        sol->dft_bufs->transpose = (aoclfftz_transpose_t*)((UINT8*)sol->dft_bufs->sr + sizeof(aoclfftz_sr_t));
        sol->dft_bufs->transpose->aux_mem = (aoclfftz_transpose_aux_mem_t*)((UINT8*)sol->dft_bufs->transpose + sizeof(aoclfftz_transpose_t));

        sol->twiddle = (aoclfftz_twiddle_t*)((UINT8*)sol->dft_bufs->transpose->aux_mem + sizeof(aoclfftz_transpose_aux_mem_t));

        /* Initialize SR struct fields */
        sol->dft_bufs->sr->odd1_sol = NULL;
        sol->dft_bufs->sr->odd3_sol = NULL;
        sol->dft_bufs->sr->input_copy = NULL;
        sol->dft_bufs->sr->input_copy_size = 0;
        sol->next_sol = NULL;

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

        sol->next_sol = NULL;
        sol->decomp_scheme->batched_vecs = NULL;
        sol->dft_bufs->nd_sol = NULL;
        sol->strides_grp->strides->in_strides = NULL;
        sol->strides_grp->strides->out_strides = NULL;
        sol->strides_grp->strides->v_in_stride = 0;
        sol->strides_grp->strides->v_out_stride = 0;
        sol->strides_grp->strides_c2c->in_strides = NULL;
        sol->strides_grp->strides_c2c->out_strides = NULL;
        sol->strides_grp->strides_c2c->v_in_stride = 0;
        sol->strides_grp->strides_c2c->v_out_stride = 0;
        sol->strides_grp->strides_r2hc->in_strides = NULL;
        sol->strides_grp->strides_r2hc->out_strides = NULL;
        sol->strides_grp->strides_r2hc->v_in_stride = 0;
        sol->strides_grp->strides_r2hc->v_out_stride = 0;
        sol->strides_grp->strides_r2hcf->in_strides = NULL;
        sol->strides_grp->strides_r2hcf->out_strides = NULL;
        sol->strides_grp->strides_r2hcf->v_in_stride = 0;
        sol->strides_grp->strides_r2hcf->v_out_stride = 0;
        sol->strides_grp->strides_c2r_ct_op = NULL;
        sol->twiddle->load_multi_cols = 1; // true by default
        sol->twiddle->cols = 0;
        sol->twiddle->TW = NULL;
        sol->twiddle->twiddle_buf_ptr = NULL;
        sol->dft_bufs->bluestein->B = NULL;
        sol->dft_bufs->bluestein->B_out = NULL;
        sol->dft_bufs->bluestein->in = NULL;
        sol->dft_bufs->bluestein->out = NULL;
        sol->dft_bufs->bluestein->is_chirp_fft_computed = 0;
        sol->dft_bufs->bluestein->ele_mul[FORWARD_FFT_DIR]  = NULL;
        sol->dft_bufs->bluestein->ele_mul[BACKWARD_FFT_DIR] = NULL;
        sol->dft_bufs->bluestein->normalize = NULL;
        sol->dft_bufs->buffered->aux_buffer_1 = NULL;
        sol->dft_bufs->buffered->aux_buffer_2 = NULL;
        sol->dft_bufs->buffered->out_ptr = NULL;
        sol->dft_bufs->transpose->row_info = (aoclfftz_dim_t_64_){0};
        sol->dft_bufs->transpose->col_info = (aoclfftz_dim_t_64_){0};
        sol->dft_bufs->transpose->aux_mem->size = 0;
        sol->dft_bufs->transpose->aux_mem->data = NULL;
        sol->dft_bufs->ct_buffer = NULL;
        sol->dft_bufs->ct_buf_real = NULL;
        sol->dft_bufs->ct_buf_imag = NULL;
        sol->dft_bufs->ct_buf_real_in = NULL;
        sol->dft_bufs->ct_buf_size = 0;
        sol->dft_bufs->num_ct_buf = 0;
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
VOID alloc_stride_arrays(aoclfftz_strides_t *strides, INTP radix)
{
    if (strides->in_strides == NULL)
    {
        ALLOC_ALIGN_UNINIT(strides->in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(strides->out_strides, INTP, radix * sizeof(INTP));
    }
}

// Allocates (if not already allocated) and fills stride arrays with a
// uniform element-stride pattern: strides[i] = i * stride * DATA_STRIDE.
INT32 alloc_and_fill_stride_arrays(aoclfftz_strides_t *strides, INTP radix,
                                   INTP in_stride, INTP out_stride)
{
    alloc_stride_arrays(strides, radix);
    if (strides->in_strides == NULL || strides->out_strides == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    
    for (INTP i = 0; i < radix; i++)
    {
        strides->in_strides[i]  = i * in_stride * DATA_STRIDE;
        strides->out_strides[i] = i * out_stride * DATA_STRIDE;
    }
    return SOLVER_SUCCESS;
}

// Allocate n placeholders for next solution
aoclfftz_solution_t **alloc_sol_array(INT32 n)
{
    aoclfftz_solution_t **sol = NULL;
    if (n > 0)
    {
        ALLOC_ALIGN_UNINIT(sol, aoclfftz_solution_t*,
                        n * sizeof(aoclfftz_solution_t*));
        for (INT32 i = 0; i < n; i++)
        {
            sol[i] = NULL;
        }
    }
    return sol;
}

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank,
                                    kernel_tables_t *kernel_tables)
{
    aoclfftz_selector_t *selector = NULL;

    ALLOC_ALIGN_UNINIT(selector, aoclfftz_selector_t,
                       sizeof(aoclfftz_selector_t));
    if (selector)
    {
        selector->kernel_tables = NULL;

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
            selector->kernel_tables->normalize = kernel_tables->normalize;
        }

        return selector;
    }
    else
    {
        return NULL;
    }
}

INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein, INTP size)
{
    // Allocate bluestein sequence buffers
    ALLOC_ALIGN_UNINIT(bluestein->B, VOID, size);
    if (bluestein->B == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_UNINIT(bluestein->B_out, VOID, size);
    if (bluestein->B_out == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    // Allocate bluestein in and out buffers
    ALLOC_ALIGN_UNINIT(bluestein->in, VOID, size);
    if (bluestein->in == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_UNINIT(bluestein->out, VOID, size);
    if (bluestein->out == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    return AOCLFFTZ_SUCCESS;
}

VOID *alloc_twiddle_buffer(UINTP size, UINT32 dt_prec)
{
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    VOID *buffer = NULL;
    ALLOC_ALIGN_UNINIT(buffer, VOID, DATA_STRIDE * size * dt_bytes);
    return buffer;
}

VOID alloc_ndim_buffer(aoclfftz_solution_t *solution, VOID **buffer_ptr)
{
    if (solution->dft_bufs->ct_buffer)
    {
        return;
    }
    // allocate buffer for the entire problem
    // this can be optimized by :
    //      1. making the buffer unit strided even if input is strided

    INT32 dim_rank = solution->decomp_scheme->dim_rank;
    aoclfftz_dim_t_64_ *dims = solution->decomp_scheme->dims;

    UINTP buffer_length = 1;
    UINTP buffer_size = 0;

    UINT32 dt_bytes = SOL_DT_SIZE(solution);

    // Approach: if the problem is ND where N>2, then create ct_buffer of (N-1)D
    // by removing the smallest dim for. e.g. problem size of 30x40x50 ->
    // ct_buffer of 40x50 for multi-threaded problems, this 2D buffer will be
    // created per thread.
    INTP min_dim_size = dims[0].n;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        buffer_length += ((dims[i].n - 1) * (dims[i].out_stride));
        if (dims[i].n < min_dim_size)
        {
            min_dim_size = dims[i].n;
        }
    }
    INT32 n_threads = solution->decomp_scheme->thread_info->avl_threads;
    INT32 num_buffer =
        solution->dft_bufs->num_ct_buf > 0 ? solution->dft_bufs->num_ct_buf : 1;
    if (solution->decomp_scheme->dim_rank > 2)
    {
        buffer_length = buffer_length / min_dim_size;
    }
    buffer_size = buffer_length * DATA_STRIDE * dt_bytes;
    ALLOC_ALIGN_UNINIT(*buffer_ptr, VOID, buffer_size * num_buffer * n_threads);
    solution->dft_bufs->ct_buf_size = buffer_size;
    solution->dft_bufs->ct_buf_real = *buffer_ptr;
    solution->dft_bufs->ct_buf_imag = MOVE_ADDR(*buffer_ptr, dt_bytes);
    solution->dft_bufs->ct_buf_allocated = 1;
}

VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme)
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

VOID destroy_transpose(aoclfftz_transpose_t* transpose)
{
    if (transpose)
    {
        if (transpose->aux_mem)
        {
            FREE_ALIGN_ALLOCATED_MEM(transpose->aux_mem->data);
        }
    }
}

VOID destroy_bluestein(aoclfftz_bluestein_t* bluestein)
{
    if (bluestein != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B_out);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->in);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->out);
    }
}

VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp)
{
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2c->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2c->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hc->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hc->out_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hcf->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_r2hcf->out_strides);
    if (strides_grp->strides_c2r_ct_op != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2r_ct_op->in_strides);
        FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2r_ct_op->out_strides);
        FREE_ALIGN_ALLOCATED_MEM(strides_grp->strides_c2r_ct_op);
    }
}

VOID destroy_solution(aoclfftz_solution_t* sol)
{
    if (sol != NULL)
    {
        INT32 solver_type = sol->solver->solver_type;
        INT32 n_sols = sol->decomp_scheme->thread_info->n_threads;
        n_sols = ((solver_type == SOLVER_MT_BATCHED) ||
                  (solver_type == SOLVER_REAL_MT_BATCHED)) ? n_sols : 1;
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
        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->sr->input_copy);

        // Free auxiliary buffers based on solver type:
        // 1. For 1D real problems, real buffered solver will create
        //    aux_buffers and the same address will be used in other solvers
        //    So free the aux_buffers only for real buffered solver.
        // 2. For in-place ND real problem, real ND solver will create
        //    aux_buffer, so free it.
        //
        // Clearing these buffers will happen only once (which will be from
        // destroy_handle).
        if (solver_type == SOLVER_BUFFERED ||
            solver_type == SOLVER_REAL_BUFFERED ||
            solver_type == SOLVER_REAL_NDIM)
        {
            FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
            FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_2);
        }
        destroy_solution(sol->dft_bufs->nd_sol);
        destroy_solution(sol->dft_bufs->sr->odd1_sol);
        destroy_solution(sol->dft_bufs->sr->odd3_sol);
        destroy_solutions(sol->next_sol, n_sols);

        FREE_ALIGN_ALLOCATED_MEM(sol);
    }
    return;
}

VOID destroy_solutions(aoclfftz_solution_t **sol, INT32 n)
{
    if (sol != NULL)
    {
        for (INT32 i = 0; i < n; i++)
        {
            aoclfftz_solution_t *cur_sol = sol[i];
            if (cur_sol != NULL)
            {
                INT32 solver_type = cur_sol->solver->solver_type;
                INT32 n_sols = cur_sol->decomp_scheme->thread_info->n_threads;
                // All other solvers (CT, SR, BLUESTEIN, NDIM, etc.) use next_sol[0] only
                n_sols = ((solver_type == SOLVER_MT_BATCHED) ||
                          (solver_type == SOLVER_REAL_MT_BATCHED)) ? n_sols : 1;

                destroy_solutions(cur_sol->next_sol, n_sols);
                destroy_decomp_scheme(cur_sol->decomp_scheme);
                destroy_strides_grp(cur_sol->strides_grp);

                FREE_ALIGN_ALLOCATED_MEM(cur_sol->twiddle->twiddle_buf_ptr);
                cur_sol->twiddle->TW = NULL;
                if (cur_sol->dft_bufs->ct_buf_allocated)
                {
                    FREE_ALIGN_ALLOCATED_MEM(cur_sol->dft_bufs->ct_buffer);
                    cur_sol->dft_bufs->ct_buf_allocated = 0;
                }

                destroy_bluestein(cur_sol->dft_bufs->bluestein);
                destroy_transpose(cur_sol->dft_bufs->transpose);

                FREE_ALIGN_ALLOCATED_MEM(cur_sol->dft_bufs->sr->input_copy);

                // Buffered solver will create aux_buffers and the same address
                // will be used in other solvers.
                // So free the aux_buffers only for buffered solver.
                //
                // Clearing these buffers will happen only once (which will be
                // from destroy_handle).
                if ((i == 0) && (solver_type == SOLVER_BUFFERED ||
                     solver_type == SOLVER_REAL_BUFFERED ||
                     solver_type == SOLVER_REAL_NDIM))
                {
                    FREE_ALIGN_ALLOCATED_MEM(
                        cur_sol->dft_bufs->buffered->aux_buffer_1);
                    FREE_ALIGN_ALLOCATED_MEM(
                        cur_sol->dft_bufs->buffered->aux_buffer_2);
                }
                destroy_solution(cur_sol->dft_bufs->nd_sol);
                destroy_solution(cur_sol->dft_bufs->sr->odd1_sol);
                destroy_solution(cur_sol->dft_bufs->sr->odd3_sol);

                FREE_ALIGN_ALLOCATED_MEM(cur_sol);
            }
        }
        FREE_ALIGN_ALLOCATED_MEM(sol);
    }
}
// Note: Since the use case for this function is primarily to free "temporary"
//       selectors, and since temporary selectors are expected to be allocated
//       using `alloc_selector_without_scratch_space`, this function does not
//       free the scratch space.
VOID destroy_selector_without_solution(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(sel->cost_analysis);
        FREE_ALIGN_ALLOCATED_MEM(sel->kernel_tables);
        FREE_ALIGN_ALLOCATED_MEM(sel);
    }
    return;
}

VOID destroy_selector(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        destroy_solution(sel->solution);
        destroy_selector_without_solution(sel);
    }
    return;
}

