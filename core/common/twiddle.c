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
FFTZ_VOID compute_twiddle_buffer_float(FFTZ_VOID *twiddle_buffer,
                                       FFTZ_INTP radix, FFTZ_INTP n_tw_batches)
{
    FFTZ_FLOAT *twiddle_buffer_real = (FFTZ_FLOAT *)twiddle_buffer;
    FFTZ_FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FFTZ_FLOAT angle_base =
        -AOCLFFTZ_2_PIf / (FFTZ_FLOAT)(radix * n_tw_batches);
    FFTZ_INTP c_stride = 1 * DATA_STRIDE;
    FFTZ_INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0f;
    }
    for (FFTZ_INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0f;
    }

    for (FFTZ_INTP i = 1; i < radix; ++i)
    {
        for (FFTZ_INTP j = 1; j < n_tw_batches; ++j)
        {
            FFTZ_FLOAT angle = angle_base * i * j;
            FFTZ_FLOAT sin_val = sinf(angle);
            FFTZ_FLOAT cos_val = cosf(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

// Setup a twiddle buffer of size `radix x n_tw_batches`
FFTZ_VOID compute_twiddle_buffer_double(FFTZ_VOID *twiddle_buffer,
                                        FFTZ_INTP radix, FFTZ_INTP n_tw_batches)
{
    FFTZ_DOUBLE *twiddle_buffer_real = (FFTZ_DOUBLE *)twiddle_buffer;
    FFTZ_DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FFTZ_DOUBLE angle_base =
        -AOCLFFTZ_2_PI / (FFTZ_DOUBLE)(radix * n_tw_batches);
    FFTZ_INTP c_stride = 1 * DATA_STRIDE;
    FFTZ_INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0;
    }
    for (FFTZ_INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0;
    }

    for (FFTZ_INTP i = 1; i < radix; ++i)
    {
        for (FFTZ_INTP j = 1; j < n_tw_batches; ++j)
        {
            FFTZ_DOUBLE angle = angle_base * i * j;
            FFTZ_DOUBLE sin_val = sin(angle);
            FFTZ_DOUBLE cos_val = cos(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

FFTZ_VOID compute_twiddle_buffer(FFTZ_VOID *twiddle_buffer, FFTZ_INTP radix,
                                 FFTZ_INTP n_tw_batches, FFTZ_UINT32 dt_prec)
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
FFTZ_VOID compute_twiddle_buffer_real_float(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP radix, FFTZ_INTP num_c2c_per_group,
    FFTZ_INTP num_groups, FFTZ_INTP freq_factor, FFTZ_UINT8 dir)
{
    FFTZ_FLOAT *twiddle_buffer_real = (FFTZ_FLOAT *)twiddle_buffer;
    FFTZ_FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FFTZ_FLOAT sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FFTZ_FLOAT angle_base = (sign * AOCLFFTZ_2_PI) / (FFTZ_FLOAT)freq_factor;
    FFTZ_INTP c_stride = DATA_STRIDE;
    FFTZ_INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
    {
        FFTZ_INTP buffer_index = LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (FFTZ_INTP i = 1; i < radix; ++i)
    {
        for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
        {
           // twiddle matrix size `r x num_c2c_per_group`
            FFTZ_FLOAT angle = angle_base * i * (j + 1);
            FFTZ_FLOAT sin_val = sinf(angle);
            FFTZ_FLOAT cos_val = cosf(angle);
            FFTZ_INTP buffer_index = LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

FFTZ_VOID compute_twiddle_buffer_real_double(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP radix, FFTZ_INTP num_c2c_per_group,
    FFTZ_INTP num_groups, FFTZ_INTP freq_factor, FFTZ_UINT8 dir)
{
    FFTZ_DOUBLE *twiddle_buffer_real = (FFTZ_DOUBLE *)twiddle_buffer;
    FFTZ_DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FFTZ_DOUBLE sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FFTZ_DOUBLE angle_base = (sign * AOCLFFTZ_2_PI) / (FFTZ_DOUBLE)freq_factor;
    FFTZ_INTP c_stride = DATA_STRIDE;
    FFTZ_INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
    {
        FFTZ_INTP buffer_index = LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (FFTZ_INTP i = 1; i < radix; ++i)
    {
        for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
        {
            FFTZ_DOUBLE angle = angle_base * i * (j + 1);
            FFTZ_DOUBLE sin_val = sin(angle);
            FFTZ_DOUBLE cos_val = cos(angle);
            // twiddle matrix size `r x num_c2c_per_group`
            FFTZ_INTP buffer_index = LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

FFTZ_VOID compute_twiddle_buffer_real(FFTZ_VOID *twiddle_buffer,
                                      FFTZ_INTP radix,
                                      FFTZ_INTP num_c2c_per_group,
                                      FFTZ_INTP num_groups,
                                      FFTZ_INTP freq_factor, FFTZ_UINT8 dir,
                                      FFTZ_UINT32 dt_prec)
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

static FFTZ_VOID compute_sr_twiddle_buffer_float(FFTZ_VOID *twiddle_buffer,
                                                 FFTZ_INTP n)
{
    FFTZ_INTP n4 = n / 4;
    aoclfftz_complex_f_t *tw = (aoclfftz_complex_f_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (FFTZ_INTP k = 0; k < n4; k++)
    {
        FFTZ_FLOAT angle1 = -AOCLFFTZ_2_PIf * k / n;  /* W_n^k */
        FFTZ_FLOAT angle3 = -3.0f * AOCLFFTZ_2_PIf * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cosf(angle1);
        tw[k * 2].imag = sinf(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cosf(angle3);
        tw[k * 2 + 1].imag = sinf(angle3);
    }
}

static FFTZ_VOID compute_sr_twiddle_buffer_double(FFTZ_VOID *twiddle_buffer,
                                                  FFTZ_INTP n)
{
    FFTZ_INTP n4 = n / 4;
    aoclfftz_complex_d_t *tw = (aoclfftz_complex_d_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (FFTZ_INTP k = 0; k < n4; k++)
    {
        FFTZ_DOUBLE angle1 = -AOCLFFTZ_2_PI * k / n;  /* W_n^k */
        FFTZ_DOUBLE angle3 = -3.0 * AOCLFFTZ_2_PI * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cos(angle1);
        tw[k * 2].imag = sin(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cos(angle3);
        tw[k * 2 + 1].imag = sin(angle3);
    }
}

FFTZ_VOID compute_sr_twiddle_buffer(FFTZ_VOID *twiddle_buffer, FFTZ_INTP n,
                                    FFTZ_UINT32 dt_prec)
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
