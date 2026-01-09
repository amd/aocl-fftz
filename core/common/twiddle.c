// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twiddle.c
 *
 *  @brief Computes and applies Twiddle factor.
 *
 *  This file contains the functions related to computing and multiplying
 *  twiddle factors to the values as needed between FFT stages
 *
 *  @author S. Biplab Raut
 *  @author Prasandh Sankarankutty
 *  @author Ashwin K. Godbole
 */

#include <math.h>
#include "core/common/twiddle.h"
#include "api/aoclfftz_internal.h"
#include "core/kernels/non_dft/transpose/transpose_utils.h"


INT32 twiddle_multiplier_for_real_float(aoclfftz_solution_t *sol,
                                        INTP freq_factor)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
    INTP base_stride = 1;

    INTP *stride_arr = NULL;
    INTP vec_stride = 1;
    if (is_backward)
    {
        stride_arr = strides->out_strides;
        vec_stride = strides->v_out_stride;
    }
    else
    {
        stride_arr = strides->in_strides;
        vec_stride = strides->v_in_stride;
    }

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    FLOAT sign = is_backward ? 1.0 : -1.0;
    FLOAT *data_r = is_backward ? (FLOAT *)out : (FLOAT *)in;
    FLOAT *data_i = is_backward ? (FLOAT *)out + 1 : (FLOAT *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
        FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP i = 1; i < radix; i++)
        {
            for (INTP j = 0; j < num_c2c_per_group; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;
                // every `num_groups` columns are duplicates
                // so skip those columns to avoid redundant loads
                INTP tw_in_index =
                    LINEAR_IDX_2D(i, j, DATA_STRIDE,
                                  num_c2c_per_group * DATA_STRIDE);
                FLOAT TW_real = twiddle_buffer_real[tw_in_index];
                FLOAT TW_imag = twiddle_buffer_imag[tw_in_index];

                for (INTP k = 0; k < num_groups; k++)
                {
                    INTP data_index = stride_arr[i] + (k * vec_stride) +
                                      (j * base_stride * 2);
                    FLOAT a = cur_data_r[data_index];
                    FLOAT b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;

                FLOAT x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
                {
                    FLOAT a = cur_data_r[stride];
                    FLOAT b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_for_real_double(aoclfftz_solution_t *sol,
                                         INTP freq_factor)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
    INTP base_stride = 1;

    INTP *stride_arr = NULL;
    INTP vec_stride = 1;
    if (is_backward)
    {
        stride_arr = strides->out_strides;
        vec_stride = strides->v_out_stride;
    }
    else
    {
        stride_arr = strides->in_strides;
        vec_stride = strides->v_in_stride;
    }

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    DOUBLE sign = is_backward ? 1.0 : -1.0;
    DOUBLE *data_r = is_backward ? (DOUBLE *)out : (DOUBLE *)in;
    DOUBLE *data_i = is_backward ? (DOUBLE *)out + 1 : (DOUBLE *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
        DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP i = 1; i < radix; i++)
        {
            for (INTP j = 0; j < num_c2c_per_group; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;
                // every `num_groups` columns are duplicates
                // so skip those columns to avoid redundant loads
                INTP tw_in_index =
                    LINEAR_IDX_2D(i, j, DATA_STRIDE,
                                  num_c2c_per_group * DATA_STRIDE);
                DOUBLE TW_real = twiddle_buffer_real[tw_in_index];
                DOUBLE TW_imag = twiddle_buffer_imag[tw_in_index];

                for (INTP k = 0; k < num_groups; k++)
                {
                    INTP data_index = stride_arr[i] + (k * vec_stride) +
                                      (j * base_stride * 2);
                    DOUBLE a = cur_data_r[data_index];
                    DOUBLE b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;

                DOUBLE x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);
                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
                {
                    DOUBLE a = cur_data_r[stride];
                    DOUBLE b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }
    return TW_SUCCESS;
}

// TODO: Support twiddle kernels for C2R problems
INT32 twiddle_multiplier_for_real(aoclfftz_solution_t *sol, INTP freq_factor)
{
    UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (dt_prec == DT_FLOAT)
    {
        twiddle_multiplier_for_real_float(sol, freq_factor);
    }
    else
    {
        twiddle_multiplier_for_real_double(sol, freq_factor);
    }
    return TW_SUCCESS;
}

#ifdef MULTI_THREADING
INT32 twiddle_multiplier_mt_for_real_float(aoclfftz_solution_t *sol,
                                           INTP freq_factor,
                                           INT32 n_threads_c2c_outer,
                                           INT32 n_threads_c2c_inner)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
    INTP base_stride = 1;

    INTP *stride_arr = NULL;
    INTP vec_stride = 1;
    if (is_backward)
    {
        stride_arr = strides->out_strides;
        vec_stride = strides->v_out_stride;
    }
    else
    {
        stride_arr = strides->in_strides;
        vec_stride = strides->v_in_stride;
    }

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    FLOAT sign = is_backward ? 1.0 : -1.0;
    FLOAT *data_r = is_backward ? (FLOAT *)out : (FLOAT *)in;
    FLOAT *data_i = is_backward ? (FLOAT *)out + 1 : (FLOAT *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
        FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP j = 0; j < num_groups; j++)
            {
                for (INTP k = 1; k < radix; k++)
                {
                    FLOAT *cur_data_r = data_r;
                    FLOAT *cur_data_i = data_i;
                    // every `num_groups` columns are duplicates
                    // so skip those columns to avoid redundant loads
                    INTP tw_in_index = LINEAR_IDX_2D(
                        k, i, DATA_STRIDE, num_c2c_per_group * DATA_STRIDE);
                    FLOAT TW_real = twiddle_buffer_real[tw_in_index];
                    FLOAT TW_imag = twiddle_buffer_imag[tw_in_index];

                    INTP data_index = stride_arr[k] + (j * vec_stride) +
                                      (i * base_stride * 2);
                    FLOAT a = cur_data_r[data_index];
                    FLOAT b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        // TODO: Parallelize these loops
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;

                FLOAT x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
                {
                    FLOAT a = cur_data_r[stride];
                    FLOAT b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_mt_for_real_double(aoclfftz_solution_t *sol,
                                            INTP freq_factor,
                                            INT32 n_threads_c2c_outer,
                                            INT32 n_threads_c2c_inner)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
    INTP base_stride = 1;

    INTP *stride_arr = NULL;
    INTP vec_stride = 1;
    if (is_backward)
    {
        stride_arr = strides->out_strides;
        vec_stride = strides->v_out_stride;
    }
    else
    {
        stride_arr = strides->in_strides;
        vec_stride = strides->v_in_stride;
    }

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    DOUBLE sign = is_backward ? 1.0 : -1.0;
    DOUBLE *data_r = is_backward ? (DOUBLE *)out : (DOUBLE *)in;
    DOUBLE *data_i = is_backward ? (DOUBLE *)out + 1 : (DOUBLE *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
        DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP j = 0; j < num_groups; j++)
            {
                for (INTP k = 1; k < radix; k++)
                {
                    DOUBLE *cur_data_r = data_r;
                    DOUBLE *cur_data_i = data_i;
                    // every `num_groups` columns are duplicates
                    // so skip those columns to avoid redundant loads
                    INTP tw_in_index = LINEAR_IDX_2D(
                        k, i, DATA_STRIDE, num_c2c_per_group * DATA_STRIDE);
                    DOUBLE TW_real = twiddle_buffer_real[tw_in_index];
                    DOUBLE TW_imag = twiddle_buffer_imag[tw_in_index];

                    INTP data_index = stride_arr[k] + (j * vec_stride) +
                                      (i * base_stride * 2);
                    DOUBLE a = cur_data_r[data_index];
                    DOUBLE b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        // TODO: Parallelize these loops
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;

                DOUBLE x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);
                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
                {
                    DOUBLE a = cur_data_r[stride];
                    DOUBLE b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_mt_for_real(aoclfftz_solution_t *sol, INTP freq_factor,
                                     INT32 n_threads_c2c_outer,
                                     INT32 n_threads_c2c_inner)
{
    UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (dt_prec == DT_FLOAT)
    {
        twiddle_multiplier_mt_for_real_float(
            sol, freq_factor, n_threads_c2c_outer, n_threads_c2c_inner);
    }
    else
    {
        twiddle_multiplier_mt_for_real_double(
            sol, freq_factor, n_threads_c2c_outer, n_threads_c2c_inner);
    }
    return TW_SUCCESS;
}
#endif

#if IN_MEMORY_TWIDDLE_FACTORS == 1
// Setup a twiddle buffer of size `radix x n_tw_batches`
VOID compute_twiddle_buffer_float(VOID *twiddle_buffer, INTP radix,
                                  INTP n_tw_batches)
{
    FLOAT *twiddle_buffer_real = (FLOAT *)twiddle_buffer;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FLOAT angle_base = -AOCLFFTZ_2_PIf / (FLOAT)(radix * n_tw_batches);
    INTP c_stride = 1 * DATA_STRIDE;
    INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0f;
    }
    for (INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0f;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 1; j < n_tw_batches; ++j)
        {
            FLOAT angle = angle_base * i * j;
            FLOAT sin_val = sinf(angle);
            FLOAT cos_val = cosf(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

// Setup a twiddle buffer of size `radix x n_tw_batches`
VOID compute_twiddle_buffer_double(VOID *twiddle_buffer, INTP radix,
                                   INTP n_tw_batches)
{
    DOUBLE *twiddle_buffer_real = (DOUBLE *)twiddle_buffer;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    DOUBLE angle_base = -AOCLFFTZ_2_PI / (DOUBLE)(radix * n_tw_batches);
    INTP c_stride = 1 * DATA_STRIDE;
    INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0;
    }
    for (INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 1; j < n_tw_batches; ++j)
        {
            DOUBLE angle = angle_base * i * j;
            DOUBLE sin_val = sin(angle);
            DOUBLE cos_val = cos(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

VOID compute_twiddle_buffer(VOID *twiddle_buffer, INTP radix, INTP n_tw_batches,
                            UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_twiddle_buffer_float(twiddle_buffer, radix, n_tw_batches);
    }
    else
    {
        compute_twiddle_buffer_double(twiddle_buffer, radix, n_tw_batches);
    }
}
#endif

#if IN_MEMORY_TWIDDLE_FACTORS == 1
VOID compute_twiddle_buffer_real_float(VOID *twiddle_buffer, INTP radix,
                                       INTP num_c2c_per_group, INTP num_groups,
                                       INTP freq_factor, UINT8 dir)
{
    FLOAT *twiddle_buffer_real = (FLOAT *)twiddle_buffer;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FLOAT sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FLOAT angle_base = (sign * AOCLFFTZ_2_PI) / (FLOAT)freq_factor;
    INTP c_stride = DATA_STRIDE;
    INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (INTP j = 0; j < num_c2c_per_group; ++j)
    {
        INTP buffer_index =
            LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 0; j < num_c2c_per_group; ++j)
        {
           // twiddle matrix size `r x num_c2c_per_group`
            FLOAT angle = angle_base * i * (j + 1);
            FLOAT sin_val = sinf(angle);
            FLOAT cos_val = cosf(angle);
            INTP buffer_index =
                LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

VOID compute_twiddle_buffer_real_double(VOID *twiddle_buffer, INTP radix,
                                        INTP num_c2c_per_group, INTP num_groups,
                                        INTP freq_factor, UINT8 dir)
{
    DOUBLE *twiddle_buffer_real = (DOUBLE *)twiddle_buffer;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    DOUBLE sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    DOUBLE angle_base = (sign * AOCLFFTZ_2_PI) / (DOUBLE)freq_factor;
    INTP c_stride = DATA_STRIDE;
    INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (INTP j = 0; j < num_c2c_per_group; ++j)
    {
        INTP buffer_index =
            LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 0; j < num_c2c_per_group; ++j)
        {
            DOUBLE angle = angle_base * i * (j + 1);
            DOUBLE sin_val = sin(angle);
            DOUBLE cos_val = cos(angle);
            // twiddle matrix size `r x num_c2c_per_group`
            INTP buffer_index =
                LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

VOID compute_twiddle_buffer_real(VOID *twiddle_buffer, INTP radix,
                                 INTP num_c2c_per_group, INTP num_groups,
                                 INTP freq_factor, UINT8 dir, UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_twiddle_buffer_real_float(twiddle_buffer, radix,
                                          num_c2c_per_group, num_groups,
                                          freq_factor, dir);
    }
    else
    {
        compute_twiddle_buffer_real_double(twiddle_buffer, radix,
                                           num_c2c_per_group, num_groups,
                                           freq_factor, dir);
    }
}

static VOID compute_sr_twiddle_buffer_float(VOID *twiddle_buffer, INTP n)
{
    INTP n4 = n / 4;
    aoclfftz_complex_f_t *tw = (aoclfftz_complex_f_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (INTP k = 0; k < n4; k++)
    {
        FLOAT angle1 = -AOCLFFTZ_2_PIf * k / n;  /* W_n^k */
        FLOAT angle3 = -3.0f * AOCLFFTZ_2_PIf * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cosf(angle1);
        tw[k * 2].imag = sinf(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cosf(angle3);
        tw[k * 2 + 1].imag = sinf(angle3);
    }
}

static VOID compute_sr_twiddle_buffer_double(VOID *twiddle_buffer, INTP n)
{
    INTP n4 = n / 4;
    aoclfftz_complex_d_t *tw = (aoclfftz_complex_d_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (INTP k = 0; k < n4; k++)
    {
        DOUBLE angle1 = -AOCLFFTZ_2_PI * k / n;  /* W_n^k */
        DOUBLE angle3 = -3.0 * AOCLFFTZ_2_PI * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cos(angle1);
        tw[k * 2].imag = sin(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cos(angle3);
        tw[k * 2 + 1].imag = sin(angle3);
    }
}

VOID compute_sr_twiddle_buffer(VOID *twiddle_buffer, INTP n, UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_sr_twiddle_buffer_float(twiddle_buffer, n);
    }
    else
    {
        compute_sr_twiddle_buffer_double(twiddle_buffer, n);
    }
}
#endif
