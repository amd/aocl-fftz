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
        INTP buffer_index = LINEAR_IDX_2D(0, j, c_stride, r_stride);
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
            INTP buffer_index = LINEAR_IDX_2D(i, j, c_stride, r_stride);
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
        INTP buffer_index = LINEAR_IDX_2D(0, j, c_stride, r_stride);
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
            INTP buffer_index = LINEAR_IDX_2D(i, j, c_stride, r_stride);
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
