// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_utils.c
 *
 *  @brief Utility functions for copying solution objects, decomposition
 *         schemes, and stride information between selector nodes.
 *
 *  These functions were originally implemented as macros in selector.h
 *  and have been refactored into functions for better debuggability,
 *  type safety, and error handling.
 *
 *  @author Varaprasad, Malothu
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

// to_ds, from_ds: short for to/from decomp_scheme
FFTZ_INT32 copy_decomp_scheme( aoclfftz_decomp_scheme_t *to_ds,
                          aoclfftz_decomp_scheme_t *from_ds)
{
    to_ds->vec_rank = from_ds->vec_rank;
    to_ds->dim_rank = from_ds->dim_rank;
    FFTZ_INT32 cnt, idx = 0;
    for (cnt = 0; cnt < from_ds->dim_rank; cnt++)
    {
        if (from_ds->dims[cnt].n != 1)
        {
            to_ds->dims[idx].n =
                from_ds->dims[cnt].n;
            to_ds->dims[idx].in_stride =
                from_ds->dims[cnt].in_stride;
            to_ds->dims[idx].out_stride =
                from_ds->dims[cnt].out_stride;
            idx++;
        }
    }
    /* Gets Executed in scenario where the shrinked dim_rank is one
       and the problem size is also one.
       Example: 1x1x1 or 1 */
    if (idx == 0)
    {
        to_ds->dims[0].n =
            from_ds->dims[0].n;
        to_ds->dims[0].in_stride =
            from_ds->dims[0].in_stride;
        to_ds->dims[0].out_stride =
            from_ds->dims[0].out_stride;
    }
    for (cnt = 0; cnt < from_ds->vec_rank; cnt++)
    {
        to_ds->vecs[cnt].n =
            from_ds->vecs[cnt].n;
        to_ds->vecs[cnt].in_stride =
            from_ds->vecs[cnt].in_stride;
        to_ds->vecs[cnt].out_stride =
            from_ds->vecs[cnt].out_stride;
    }
    if (from_ds->batched_vecs != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(
            to_ds->batched_vecs);
        ALLOC_ALIGN_UNINIT(to_ds->batched_vecs,
            aoclfftz_dim_t_64_,
            sizeof(aoclfftz_dim_t_64_));
        if (to_ds->batched_vecs == NULL)
        {
            AOCLFFTZ_ERROR("Memory allocation failed.");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        memcpy(to_ds->batched_vecs,
               from_ds->batched_vecs,
               sizeof(aoclfftz_dim_t_64_));
    }
    to_ds->in_real = from_ds->in_real;
    to_ds->in_imag = from_ds->in_imag;
    to_ds->out_real = from_ds->out_real;
    to_ds->out_imag = from_ds->out_imag;
    to_ds->cntrl_params->opt_level =
        from_ds->cntrl_params->opt_level;
    to_ds->cntrl_params->opt_off =
        from_ds->cntrl_params->opt_off;
    to_ds->cntrl_params->logger_mode =
        from_ds->cntrl_params->logger_mode;
    to_ds->cntrl_params->measure_stats =
        from_ds->cntrl_params->measure_stats;
    to_ds->thread_info->pthr_fft->
        dynamic_load_model =
        from_ds->thread_info->pthr_fft->
            dynamic_load_model;
    to_ds->thread_info->pthr_fft->num_threads =
        from_ds->thread_info->pthr_fft->
            num_threads;
    to_ds->thread_info->avl_threads =
        from_ds->thread_info->avl_threads;
    to_ds->thread_info->n_threads = 1;
    to_ds->thread_info->active_threads = from_ds->thread_info->active_threads;
    to_ds->flags = from_ds->flags;
    to_ds->real_in_role = from_ds->real_in_role;
    to_ds->real_out_role = from_ds->real_out_role;
    return AOCLFFTZ_SUCCESS;
}

FFTZ_INT32 copy_solution_obj( aoclfftz_solution_t *to_sol_obj,
                         aoclfftz_solution_t *from_sol_obj )
{
    // solver
    to_sol_obj->solver->solver_type = from_sol_obj->solver->solver_type;
    to_sol_obj->solver->execute_solver = from_sol_obj->solver->execute_solver;
    to_sol_obj->solver->destroy_solver = from_sol_obj->solver->destroy_solver;
    to_sol_obj->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c->sets =
        from_sol_obj->solver->kernel_c2c->sets;
    to_sol_obj->solver->kernel_c2c->count =
        from_sol_obj->solver->kernel_c2c->count;
    to_sol_obj->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c_r->sets =
        from_sol_obj->solver->kernel_c2c_r->sets;
    to_sol_obj->solver->kernel_c2c_r->count =
        from_sol_obj->solver->kernel_c2c_r->count;
    to_sol_obj->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hc->sets =
        from_sol_obj->solver->kernel_r2hc->sets;
    to_sol_obj->solver->kernel_r2hc->count =
        from_sol_obj->solver->kernel_r2hc->count;
    to_sol_obj->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hcf->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hcf->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hcf->sets =
        from_sol_obj->solver->kernel_r2hcf->sets;
    to_sol_obj->solver->kernel_r2hcf->count =
        from_sol_obj->solver->kernel_r2hcf->count;

    // decomp_scheme
    to_sol_obj->decomp_scheme->vec_rank =
        from_sol_obj->decomp_scheme->vec_rank;
    to_sol_obj->decomp_scheme->dim_rank =
        from_sol_obj->decomp_scheme->dim_rank;
    FFTZ_INT32 cnt;
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->dim_rank; cnt++)
    {
        to_sol_obj->decomp_scheme->dims[cnt].n =
            from_sol_obj->decomp_scheme->dims[cnt].n;
        to_sol_obj->decomp_scheme->dims[cnt].in_stride =
            from_sol_obj->decomp_scheme->dims[cnt].in_stride;
        to_sol_obj->decomp_scheme->dims[cnt].out_stride =
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;
    }
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->vec_rank; cnt++)
    {
        to_sol_obj->decomp_scheme->vecs[cnt].n =
            from_sol_obj->decomp_scheme->vecs[cnt].n;
        to_sol_obj->decomp_scheme->vecs[cnt].in_stride =
            from_sol_obj->decomp_scheme->vecs[cnt].in_stride;
        to_sol_obj->decomp_scheme->vecs[cnt].out_stride =
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;
    }
    if (from_sol_obj->decomp_scheme->batched_vecs != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->decomp_scheme->batched_vecs);
        ALLOC_ALIGN_UNINIT(to_sol_obj->decomp_scheme->batched_vecs,
                           aoclfftz_dim_t_64_, sizeof(aoclfftz_dim_t_64_));
        if (to_sol_obj->decomp_scheme->batched_vecs == NULL)
        {
            AOCLFFTZ_ERROR("Memory allocation failed.");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        memcpy(to_sol_obj->decomp_scheme->batched_vecs,
               from_sol_obj->decomp_scheme->batched_vecs,
               sizeof(aoclfftz_dim_t_64_));
    }
    to_sol_obj->decomp_scheme->in_real = from_sol_obj->decomp_scheme->in_real;
    to_sol_obj->decomp_scheme->in_imag = from_sol_obj->decomp_scheme->in_imag;
    to_sol_obj->decomp_scheme->out_real =
        from_sol_obj->decomp_scheme->out_real;
    to_sol_obj->decomp_scheme->out_imag =
        from_sol_obj->decomp_scheme->out_imag;
    to_sol_obj->decomp_scheme->cntrl_params->opt_level =
        from_sol_obj->decomp_scheme->cntrl_params->opt_level;
    to_sol_obj->decomp_scheme->cntrl_params->opt_off =
        from_sol_obj->decomp_scheme->cntrl_params->opt_off;
    to_sol_obj->decomp_scheme->cntrl_params->logger_mode =
        from_sol_obj->decomp_scheme->cntrl_params->logger_mode;
    to_sol_obj->decomp_scheme->cntrl_params->measure_stats =
        from_sol_obj->decomp_scheme->cntrl_params->measure_stats;
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads =
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads;
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model =
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model;
    to_sol_obj->decomp_scheme->thread_info->avl_threads =
        from_sol_obj->decomp_scheme->thread_info->avl_threads;
    to_sol_obj->decomp_scheme->thread_info->n_threads =
        from_sol_obj->decomp_scheme->thread_info->n_threads;
    to_sol_obj->decomp_scheme->flags = from_sol_obj->decomp_scheme->flags;
    setup_rdft_dc_nyquist_offsets_ds(to_sol_obj->decomp_scheme);
    to_sol_obj->decomp_scheme->real_in_role =
        from_sol_obj->decomp_scheme->real_in_role;
    to_sol_obj->decomp_scheme->real_out_role =
        from_sol_obj->decomp_scheme->real_out_role;

    // twiddle
    to_sol_obj->twiddle->TW = from_sol_obj->twiddle->TW;
    to_sol_obj->twiddle->load_multi_cols =
        from_sol_obj->twiddle->load_multi_cols;

    // dft_bufs
    to_sol_obj->dft_bufs->bluestein->B =
        from_sol_obj->dft_bufs->bluestein->B;
    to_sol_obj->dft_bufs->bluestein->B_out =
        from_sol_obj->dft_bufs->bluestein->B_out;
    to_sol_obj->dft_bufs->bluestein->bs_buf_size =
        from_sol_obj->dft_bufs->bluestein->bs_buf_size;
    to_sol_obj->dft_bufs->bluestein->bs_dim_offset =
        from_sol_obj->dft_bufs->bluestein->bs_dim_offset;
    to_sol_obj->dft_bufs->bluestein->pre_mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->pre_mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->pre_mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->pre_mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->post_mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->post_mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->post_mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->post_mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->cast_to_complex =
        from_sol_obj->dft_bufs->bluestein->cast_to_complex;
    to_sol_obj->dft_bufs->bluestein->cast_from_complex =
        from_sol_obj->dft_bufs->bluestein->cast_from_complex;
    to_sol_obj->dft_bufs->buffered->aux_buffer_1 =
        from_sol_obj->dft_bufs->buffered->aux_buffer_1;
    to_sol_obj->dft_bufs->buffered->aux_buffer_2 =
        from_sol_obj->dft_bufs->buffered->aux_buffer_2;
    to_sol_obj->dft_bufs->buffered->aux_buf_size_per_thread =
        from_sol_obj->dft_bufs->buffered->aux_buf_size_per_thread;
    // Borrower after copy; only the node that malloc'd the pool keeps ownership.
    to_sol_obj->dft_bufs->buffered->is_aux_buffer_allocated = 0;
    to_sol_obj->dft_bufs->ct_buffer =
        from_sol_obj->dft_bufs->ct_buffer;
    to_sol_obj->decomp_scheme->thread_info->active_threads =
        from_sol_obj->decomp_scheme->thread_info->active_threads;
    to_sol_obj->dft_bufs->ct_buf_real =
        from_sol_obj->dft_bufs->ct_buf_real;
    to_sol_obj->dft_bufs->ct_buf_imag =
        from_sol_obj->dft_bufs->ct_buf_imag;
    to_sol_obj->dft_bufs->ct_buf_size = from_sol_obj->dft_bufs->ct_buf_size;
    if (from_sol_obj->dft_bufs->transpose &&
        to_sol_obj->dft_bufs->transpose)
    {
        to_sol_obj->dft_bufs->transpose->row_info =
            from_sol_obj->dft_bufs->transpose->row_info;
        to_sol_obj->dft_bufs->transpose->col_info =
            from_sol_obj->dft_bufs->transpose->col_info;
        to_sol_obj->dft_bufs->transpose->kernel =
            from_sol_obj->dft_bufs->transpose->kernel;
        if (from_sol_obj->dft_bufs->transpose->aux_mem &&
            from_sol_obj->dft_bufs->transpose->aux_mem->data &&
            from_sol_obj->dft_bufs->transpose->aux_mem->size > 0)
        {
            if (!to_sol_obj->dft_bufs->transpose->aux_mem->data)
            {
                ALLOC_ALIGN_INIT(
                    to_sol_obj->dft_bufs->transpose->aux_mem->data, FFTZ_UINT8,
                    from_sol_obj->dft_bufs->transpose->aux_mem->size);
            }
            else
            {
                FREE_ALIGN_ALLOCATED_MEM(
                    to_sol_obj->dft_bufs->transpose->aux_mem->data);
                ALLOC_ALIGN_INIT(
                    to_sol_obj->dft_bufs->transpose->aux_mem->data, FFTZ_UINT8,
                    from_sol_obj->dft_bufs->transpose->aux_mem->size);
            }
            if (to_sol_obj->dft_bufs->transpose->aux_mem->data == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->dft_bufs->transpose->aux_mem->data,
                    from_sol_obj->dft_bufs->transpose->aux_mem->data,
                    from_sol_obj->dft_bufs->transpose->aux_mem->size);
        }
        to_sol_obj->dft_bufs->transpose->aux_mem->size =
            from_sol_obj->dft_bufs->transpose->aux_mem->size;
    }
    to_sol_obj->next_sol = from_sol_obj->next_sol;
    return AOCLFFTZ_SUCCESS;
}

// maps both in & out pointers to out pointer
// incase of out-of-place problems, except the first DFT, other DFTs happen
// in-place ie., in the output buffer.
FFTZ_INT32 copy_solution_obj_out_p( aoclfftz_solution_t *to_sol_obj,
                               aoclfftz_solution_t *from_sol_obj )
{
    FFTZ_INT32 ret = copy_solution_obj(to_sol_obj, from_sol_obj);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    FFTZ_INT32 cnt;
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->dim_rank; cnt++)
    {
        to_sol_obj->decomp_scheme->dims[cnt].n =
            from_sol_obj->decomp_scheme->dims[cnt].n;
        to_sol_obj->decomp_scheme->dims[cnt].in_stride =
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;
        to_sol_obj->decomp_scheme->dims[cnt].out_stride =
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;
    }
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->vec_rank; cnt++)
    {
        to_sol_obj->decomp_scheme->vecs[cnt].n =
            from_sol_obj->decomp_scheme->vecs[cnt].n;
        to_sol_obj->decomp_scheme->vecs[cnt].in_stride =
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;
        to_sol_obj->decomp_scheme->vecs[cnt].out_stride =
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;
    }
    if (from_sol_obj->decomp_scheme->batched_vecs != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->decomp_scheme->batched_vecs);
        ALLOC_ALIGN_UNINIT(to_sol_obj->decomp_scheme->batched_vecs,
                           aoclfftz_dim_t_64_, sizeof(aoclfftz_dim_t_64_));
        if (to_sol_obj->decomp_scheme->batched_vecs == NULL)
        {
            AOCLFFTZ_ERROR("Memory allocation failed.");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        memcpy(to_sol_obj->decomp_scheme->batched_vecs,
                from_sol_obj->decomp_scheme->batched_vecs,
                sizeof(aoclfftz_dim_t_64_));
    }
    to_sol_obj->decomp_scheme->in_real =
        from_sol_obj->decomp_scheme->out_real;
    to_sol_obj->decomp_scheme->in_imag =
        from_sol_obj->decomp_scheme->out_imag;
    to_sol_obj->decomp_scheme->out_real =
        from_sol_obj->decomp_scheme->out_real;
    to_sol_obj->decomp_scheme->out_imag =
        from_sol_obj->decomp_scheme->out_imag;
    setup_rdft_dc_nyquist_offsets_ds(to_sol_obj->decomp_scheme);
    return AOCLFFTZ_SUCCESS;
}

// Copy strides from one solution to another
// except for the BATCHED_CT_L1_DIRECT solver
FFTZ_INT32 copy_strides( aoclfftz_solution_t *to_sol_obj,
                    aoclfftz_solution_t *from_sol_obj )
{
    if (from_sol_obj->strides_grp->strides->in_strides != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(
            to_sol_obj->strides_grp->strides->in_strides);
        ALLOC_ALIGN_UNINIT(
            to_sol_obj->strides_grp->strides->in_strides, FFTZ_INTP,
            from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        if (to_sol_obj->strides_grp->strides->in_strides == NULL)
        {
            AOCLFFTZ_ERROR("Memory allocation failed.");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        memcpy(to_sol_obj->strides_grp->strides->in_strides,
                from_sol_obj->strides_grp->strides->in_strides,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
    }
    if (from_sol_obj->strides_grp->strides->out_strides != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(
            to_sol_obj->strides_grp->strides->out_strides);
        ALLOC_ALIGN_UNINIT(
            to_sol_obj->strides_grp->strides->out_strides, FFTZ_INTP,
            from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        if (to_sol_obj->strides_grp->strides->out_strides == NULL)
        {
            AOCLFFTZ_ERROR("Memory allocation failed.");
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        memcpy(to_sol_obj->strides_grp->strides->out_strides,
                from_sol_obj->strides_grp->strides->out_strides,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
    }
    to_sol_obj->strides_grp->strides->v_in_stride =
        from_sol_obj->strides_grp->strides->v_in_stride;
    to_sol_obj->strides_grp->strides->v_out_stride =
        from_sol_obj->strides_grp->strides->v_out_stride;
    to_sol_obj->strides_grp->strides->v_in_h2_stride =
        from_sol_obj->strides_grp->strides->v_in_h2_stride;
    to_sol_obj->strides_grp->strides->v_out_h2_stride =
        from_sol_obj->strides_grp->strides->v_out_h2_stride;

    if (from_sol_obj->solver->kernel_c2c->count != 0)
    {
        if (from_sol_obj->strides_grp->strides_c2c->in_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_c2c->in_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_c2c->in_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_c2c->in_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_c2c->in_strides,
                    from_sol_obj->strides_grp->strides_c2c->in_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        }
        if (from_sol_obj->strides_grp->strides_c2c->out_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_c2c->out_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_c2c->out_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_c2c->out_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_c2c->out_strides,
                    from_sol_obj->strides_grp->strides_c2c->out_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        }
        to_sol_obj->strides_grp->strides_c2c->v_in_stride =
            from_sol_obj->strides_grp->strides_c2c->v_in_stride;
        to_sol_obj->strides_grp->strides_c2c->v_out_stride =
            from_sol_obj->strides_grp->strides_c2c->v_out_stride;
        to_sol_obj->strides_grp->strides_c2c->v_in_h2_stride =
            from_sol_obj->strides_grp->strides_c2c->v_in_h2_stride;
        to_sol_obj->strides_grp->strides_c2c->v_out_h2_stride =
            from_sol_obj->strides_grp->strides_c2c->v_out_h2_stride;
    }

    if (from_sol_obj->solver->kernel_r2hc->count != 0)
    {
        if (from_sol_obj->strides_grp->strides_r2hc->in_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_r2hc->in_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_r2hc->in_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_r2hc->in_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_r2hc->in_strides,
                    from_sol_obj->strides_grp->strides_r2hc->in_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        }
        if (from_sol_obj->strides_grp->strides_r2hc->out_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_r2hc->out_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_r2hc->out_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_r2hc->out_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_r2hc->out_strides,
                    from_sol_obj->strides_grp->strides_r2hc->out_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * sizeof(FFTZ_INTP));
        }
        to_sol_obj->strides_grp->strides_r2hc->v_in_stride =
            from_sol_obj->strides_grp->strides_r2hc->v_in_stride;
        to_sol_obj->strides_grp->strides_r2hc->v_out_stride =
            from_sol_obj->strides_grp->strides_r2hc->v_out_stride;
        to_sol_obj->strides_grp->strides_r2hc->v_in_h2_stride =
            from_sol_obj->strides_grp->strides_r2hc->v_in_h2_stride;
        to_sol_obj->strides_grp->strides_r2hc->v_out_h2_stride =
            from_sol_obj->strides_grp->strides_r2hc->v_out_h2_stride;
    }

    if (from_sol_obj->solver->kernel_r2hcf->count != 0)
    {
        if (from_sol_obj->strides_grp->strides_r2hcf->in_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_r2hcf->in_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_r2hcf->in_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * 2 *
                    sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_r2hcf->in_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_r2hcf->in_strides,
                    from_sol_obj->strides_grp->strides_r2hcf->in_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * 2 *
                        sizeof(FFTZ_INTP));
        }
        if (from_sol_obj->strides_grp->strides_r2hcf->out_strides != NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(
                to_sol_obj->strides_grp->strides_r2hcf->out_strides);
            ALLOC_ALIGN_UNINIT(
                to_sol_obj->strides_grp->strides_r2hcf->out_strides, FFTZ_INTP,
                from_sol_obj->decomp_scheme->dims[0].n * 2 *
                    sizeof(FFTZ_INTP));
            if (to_sol_obj->strides_grp->strides_r2hcf->out_strides == NULL)
            {
                AOCLFFTZ_ERROR("Memory allocation failed.");
                return AOCLFFTZ_MEMORY_FAILURE;
            }
            memcpy(to_sol_obj->strides_grp->strides_r2hcf->out_strides,
                    from_sol_obj->strides_grp->strides_r2hcf->out_strides,
                    from_sol_obj->decomp_scheme->dims[0].n * 2 *
                        sizeof(FFTZ_INTP));
        }
        to_sol_obj->strides_grp->strides_r2hcf->v_in_stride =
            from_sol_obj->strides_grp->strides_r2hcf->v_in_stride;
        to_sol_obj->strides_grp->strides_r2hcf->v_out_stride =
            from_sol_obj->strides_grp->strides_r2hcf->v_out_stride;
        to_sol_obj->strides_grp->strides_r2hcf->v_in_h2_stride =
            from_sol_obj->strides_grp->strides_r2hcf->v_in_h2_stride;
        to_sol_obj->strides_grp->strides_r2hcf->v_out_h2_stride =
            from_sol_obj->strides_grp->strides_r2hcf->v_out_h2_stride;
    }

    return AOCLFFTZ_SUCCESS;
}

// Copy strides from one solution to another for the BATCHED_CT_L1_DIRECT solver
FFTZ_INT32 copy_strides_batched_ct_l1_direct( aoclfftz_solution_t *to_sol_obj,
                                         aoclfftz_solution_t *from_sol_obj )
{
    /* strides (radix_m kernel): radix_m entries, count = kernel_c2c_r */
    FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_grp->strides->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_grp->strides->out_strides);
    ALLOC_ALIGN_UNINIT(to_sol_obj->strides_grp->strides->in_strides,
                       FFTZ_INTP,
                       from_sol_obj->solver->kernel_c2c_r->count *
                           sizeof(FFTZ_INTP));
    if (to_sol_obj->strides_grp->strides->in_strides == NULL)
    {
        AOCLFFTZ_ERROR("Memory allocation failed.");
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_UNINIT(to_sol_obj->strides_grp->strides->out_strides,
                       FFTZ_INTP,
                       from_sol_obj->solver->kernel_c2c_r->count *
                           sizeof(FFTZ_INTP));
    if (to_sol_obj->strides_grp->strides->out_strides == NULL)
    {
        AOCLFFTZ_ERROR("Memory allocation failed.");
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    memcpy(to_sol_obj->strides_grp->strides->in_strides,
            from_sol_obj->strides_grp->strides->in_strides,
            from_sol_obj->solver->kernel_c2c_r->count * sizeof(FFTZ_INTP));
    memcpy(to_sol_obj->strides_grp->strides->out_strides,
            from_sol_obj->strides_grp->strides->out_strides,
            from_sol_obj->solver->kernel_c2c_r->count * sizeof(FFTZ_INTP));
    /* strides_c2c (radix_r kernel): radix_r entries, count = kernel_c2c */
    FREE_ALIGN_ALLOCATED_MEM(
        to_sol_obj->strides_grp->strides_c2c->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(
        to_sol_obj->strides_grp->strides_c2c->out_strides);
    ALLOC_ALIGN_UNINIT(
        to_sol_obj->strides_grp->strides_c2c->in_strides, FFTZ_INTP,
        from_sol_obj->solver->kernel_c2c->count * sizeof(FFTZ_INTP));
    if (to_sol_obj->strides_grp->strides_c2c->in_strides == NULL)
    {
        AOCLFFTZ_ERROR("Memory allocation failed.");
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_UNINIT(
        to_sol_obj->strides_grp->strides_c2c->out_strides, FFTZ_INTP,
        from_sol_obj->solver->kernel_c2c->count * sizeof(FFTZ_INTP));
    if (to_sol_obj->strides_grp->strides_c2c->out_strides == NULL)
    {
        AOCLFFTZ_ERROR("Memory allocation failed.");
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    memcpy(to_sol_obj->strides_grp->strides_c2c->in_strides,
            from_sol_obj->strides_grp->strides_c2c->in_strides,
            from_sol_obj->solver->kernel_c2c->count * sizeof(FFTZ_INTP));
    memcpy(to_sol_obj->strides_grp->strides_c2c->out_strides,
            from_sol_obj->strides_grp->strides_c2c->out_strides,
            from_sol_obj->solver->kernel_c2c->count * sizeof(FFTZ_INTP));
    to_sol_obj->strides_grp->strides->v_in_stride =
        from_sol_obj->strides_grp->strides->v_in_stride;
    to_sol_obj->strides_grp->strides->v_out_stride =
        from_sol_obj->strides_grp->strides->v_out_stride;
    to_sol_obj->strides_grp->strides->v_in_h2_stride =
        from_sol_obj->strides_grp->strides->v_in_h2_stride;
    to_sol_obj->strides_grp->strides->v_out_h2_stride =
        from_sol_obj->strides_grp->strides->v_out_h2_stride;
    to_sol_obj->strides_grp->strides_c2c->v_in_stride =
        from_sol_obj->strides_grp->strides_c2c->v_in_stride;
    to_sol_obj->strides_grp->strides_c2c->v_out_stride =
        from_sol_obj->strides_grp->strides_c2c->v_out_stride;
    to_sol_obj->strides_grp->strides_c2c->v_in_h2_stride =
        from_sol_obj->strides_grp->strides_c2c->v_in_h2_stride;
    to_sol_obj->strides_grp->strides_c2c->v_out_h2_stride =
        from_sol_obj->strides_grp->strides_c2c->v_out_h2_stride;
    return AOCLFFTZ_SUCCESS;
}

// copy all contents except dims & vecs
// necessary in ND setup where dim_rank & vec_rank will differ for the
// sub-problem
FFTZ_VOID copy_solution_obj_wo_dims( aoclfftz_solution_t *to_sol_obj,
                                aoclfftz_solution_t *from_sol_obj )
{
    // solver
    to_sol_obj->solver->solver_type = from_sol_obj->solver->solver_type;
    to_sol_obj->solver->execute_solver =
        from_sol_obj->solver->execute_solver;
    to_sol_obj->solver->destroy_solver =
        from_sol_obj->solver->destroy_solver;
    to_sol_obj->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c->sets =
        from_sol_obj->solver->kernel_c2c->sets;
    to_sol_obj->solver->kernel_c2c->count =
        from_sol_obj->solver->kernel_c2c->count;
    to_sol_obj->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_c2c_r->sets =
        from_sol_obj->solver->kernel_c2c_r->sets;
    to_sol_obj->solver->kernel_c2c_r->count =
        from_sol_obj->solver->kernel_c2c_r->count;
    to_sol_obj->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hc->sets =
        from_sol_obj->solver->kernel_r2hc->sets;
    to_sol_obj->solver->kernel_r2hc->count =
        from_sol_obj->solver->kernel_r2hc->count;
    to_sol_obj->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hcf->kfft[BACKWARD_FFT_DIR] =
        from_sol_obj->solver->kernel_r2hcf->kfft[BACKWARD_FFT_DIR];
    to_sol_obj->solver->kernel_r2hcf->sets =
        from_sol_obj->solver->kernel_r2hcf->sets;
    to_sol_obj->solver->kernel_r2hcf->count =
        from_sol_obj->solver->kernel_r2hcf->count;

    // decomp_scheme
    to_sol_obj->decomp_scheme->in_real =
        from_sol_obj->decomp_scheme->in_real;
    to_sol_obj->decomp_scheme->in_imag =
        from_sol_obj->decomp_scheme->in_imag;
    to_sol_obj->decomp_scheme->out_real =
        from_sol_obj->decomp_scheme->out_real;
    to_sol_obj->decomp_scheme->out_imag =
        from_sol_obj->decomp_scheme->out_imag;
    to_sol_obj->decomp_scheme->cntrl_params->opt_level =
        from_sol_obj->decomp_scheme->cntrl_params->opt_level;
    to_sol_obj->decomp_scheme->cntrl_params->opt_off =
        from_sol_obj->decomp_scheme->cntrl_params->opt_off;
    to_sol_obj->decomp_scheme->cntrl_params->logger_mode =
        from_sol_obj->decomp_scheme->cntrl_params->logger_mode;
    to_sol_obj->decomp_scheme->cntrl_params->measure_stats =
        from_sol_obj->decomp_scheme->cntrl_params->measure_stats;
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads =
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads;
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model =
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model;
    to_sol_obj->decomp_scheme->thread_info->avl_threads =
        from_sol_obj->decomp_scheme->thread_info->avl_threads;
    to_sol_obj->decomp_scheme->thread_info->n_threads =
        from_sol_obj->decomp_scheme->thread_info->n_threads;
    to_sol_obj->decomp_scheme->flags = from_sol_obj->decomp_scheme->flags;
    to_sol_obj->decomp_scheme->real_in_role =
        from_sol_obj->decomp_scheme->real_in_role;
    to_sol_obj->decomp_scheme->real_out_role =
        from_sol_obj->decomp_scheme->real_out_role;

    // twiddle
    to_sol_obj->twiddle->TW = from_sol_obj->twiddle->TW;
    to_sol_obj->twiddle->load_multi_cols =
        from_sol_obj->twiddle->load_multi_cols;
    to_sol_obj->twiddle->twiddle_buf_ptr =
        from_sol_obj->twiddle->twiddle_buf_ptr;

    // dft_bufs
    to_sol_obj->dft_bufs->bluestein->B = from_sol_obj->dft_bufs->bluestein->B;
    to_sol_obj->dft_bufs->bluestein->B_out =
        from_sol_obj->dft_bufs->bluestein->B_out;
    to_sol_obj->dft_bufs->bluestein->bs_buf_size =
        from_sol_obj->dft_bufs->bluestein->bs_buf_size;
    to_sol_obj->dft_bufs->bluestein->bs_dim_offset =
        from_sol_obj->dft_bufs->bluestein->bs_dim_offset;
    to_sol_obj->dft_bufs->bluestein->pre_mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->pre_mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->pre_mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->pre_mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->post_mul[FORWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->post_mul[FORWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->post_mul[BACKWARD_FFT_DIR] =
        from_sol_obj->dft_bufs->bluestein->post_mul[BACKWARD_FFT_DIR];
    to_sol_obj->dft_bufs->bluestein->cast_to_complex =
        from_sol_obj->dft_bufs->bluestein->cast_to_complex;
    to_sol_obj->dft_bufs->bluestein->cast_from_complex =
        from_sol_obj->dft_bufs->bluestein->cast_from_complex;
    to_sol_obj->dft_bufs->buffered->aux_buffer_1 =
        from_sol_obj->dft_bufs->buffered->aux_buffer_1;
    to_sol_obj->dft_bufs->buffered->aux_buffer_2 =
        from_sol_obj->dft_bufs->buffered->aux_buffer_2;
    to_sol_obj->dft_bufs->buffered->aux_buf_size_per_thread =
        from_sol_obj->dft_bufs->buffered->aux_buf_size_per_thread;
    to_sol_obj->dft_bufs->ct_buffer =
        from_sol_obj->dft_bufs->ct_buffer;
    to_sol_obj->dft_bufs->ct_buf_real =
        from_sol_obj->dft_bufs->ct_buf_real;
    to_sol_obj->dft_bufs->ct_buf_imag =
        from_sol_obj->dft_bufs->ct_buf_imag;
    to_sol_obj->dft_bufs->ct_buf_size = from_sol_obj->dft_bufs->ct_buf_size;
    to_sol_obj->decomp_scheme->thread_info->active_threads =
        from_sol_obj->decomp_scheme->thread_info->active_threads;
    to_sol_obj->next_sol = from_sol_obj->next_sol;
}

FFTZ_VOID swap_real_ct_solutions(aoclfftz_selector_t *sel)
{
    aoclfftz_solution_t *curr = sel->solution;
    aoclfftz_solution_t *prev = NULL;
    aoclfftz_solution_t *next = NULL;
    if (sel->solution->next_sol != NULL)
    {
        /* swap first CT node */
        if (sel->solution->solver->solver_type == SOLVER_REAL_CT &&
            (is_solver_real_direct_family(
                 sel->solution->next_sol->solver->solver_type)))
        {
            sel->solution = curr->next_sol;
            curr->next_sol = sel->solution->next_sol;
            sel->solution->next_sol = curr;
        }
        /* swap remaining CT nodes */
        prev = curr;
        curr = curr->next_sol;
        while (curr && curr->next_sol)
        {
            next = curr->next_sol;
            if (curr->solver->solver_type == SOLVER_REAL_CT &&
                is_solver_real_direct_family(next->solver->solver_type))
            {
                prev->next_sol = next;
                curr->next_sol = next->next_sol;
                next->next_sol = curr;
            }
            prev = curr;
            curr = curr->next_sol;
        }
    }
}
