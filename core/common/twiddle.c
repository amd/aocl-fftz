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
#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/transpose/transpose_utils.h"

#if IN_MEMORY_TWIDDLE_FACTORS == 1

static inline
FFTZ_INTP populate_twiddle_per_simd_butterfly_float(FFTZ_FLOAT *tw,
                                                    FFTZ_INTP idx,
                                                    FFTZ_INTP radix,
                                                    FFTZ_FLOAT angle_base,
                                                    FFTZ_INTP base_col,
                                                    FFTZ_INTP tile_w)
{
    for (FFTZ_INTP k = 1; k < radix; ++k)
    {
        for (FFTZ_INTP j = 0; j < tile_w; ++j)
        {
            FFTZ_INTP col = base_col + j;
            FFTZ_FLOAT angle = angle_base * (FFTZ_FLOAT)k * (FFTZ_FLOAT)col;
            tw[idx++] = cosf(angle);
            tw[idx++] = sinf(angle);
        }
    }
    return idx;
}

static inline
FFTZ_VOID compute_twiddle_buffer_linear_float(FFTZ_FLOAT *tw,
                                              FFTZ_INTP radix, FFTZ_INTP m,
                                              FFTZ_INTP register_width,
                                              FFTZ_INTP load_multi_cols)
{
    FFTZ_FLOAT angle_base = -AOCLFFTZ_2_PIf / (FFTZ_FLOAT)(radix * m);

    if (load_multi_cols == 0)
    {
        FFTZ_INTP idx = 0;
        for (FFTZ_INTP j = 0; j < m; ++j)
        {
            for (FFTZ_INTP k = 1; k < radix; ++k)
            {
                FFTZ_FLOAT angle = angle_base * (FFTZ_FLOAT)k * (FFTZ_FLOAT)j;
                tw[idx++] = cosf(angle);
                tw[idx++] = sinf(angle);
            }
        }
        return;
    }

    FFTZ_INTP idx = 0;
    FFTZ_INTP base_col = 0;
    FFTZ_INTP rem = m;

    while (rem >= register_width)
    {
        idx = populate_twiddle_per_simd_butterfly_float(
              tw, idx, radix, angle_base, base_col, register_width);
        base_col += register_width;
        rem -= register_width;
    }
    if (register_width > NUM_SETS_256_S && rem >= NUM_SETS_256_S)
    {
        idx = populate_twiddle_per_simd_butterfly_float(
              tw, idx, radix, angle_base, base_col, NUM_SETS_256_S);
        base_col += NUM_SETS_256_S;
        rem -= NUM_SETS_256_S;
    }
    if (register_width > NUM_SETS_128_S && rem >= NUM_SETS_128_S)
    {
        idx = populate_twiddle_per_simd_butterfly_float(
              tw, idx, radix, angle_base, base_col, NUM_SETS_128_S);
        base_col += NUM_SETS_128_S;
        rem -= NUM_SETS_128_S;
    }
    if (rem > 0)
    {
        idx = populate_twiddle_per_simd_butterfly_float(
              tw, idx, radix, angle_base, base_col, rem);
    }
}

// Populate the twiddles for one SIMD butterfly.
// Returns the advanced index.
static inline
FFTZ_INTP populate_twiddle_per_simd_butterfly_double(FFTZ_DOUBLE *tw,
                                                     FFTZ_INTP idx,
                                                     FFTZ_INTP radix,
                                                     FFTZ_DOUBLE angle_base,
                                                     FFTZ_INTP base_col,
                                                     FFTZ_INTP tile_w)
{
    for (FFTZ_INTP k = 1; k < radix; ++k)
    {
        for (FFTZ_INTP j = 0; j < tile_w; ++j)
        {
            FFTZ_INTP col = base_col + j;
            FFTZ_DOUBLE angle = angle_base * (FFTZ_DOUBLE)k * (FFTZ_DOUBLE)col;
            tw[idx++] = cos(angle);
            tw[idx++] = sin(angle);
        }
    }
    return idx;
}

static inline
FFTZ_VOID compute_twiddle_buffer_linear_double(FFTZ_DOUBLE *tw,
                                               FFTZ_INTP radix,
                                               FFTZ_INTP m,
                                               FFTZ_INTP register_width,
                                               FFTZ_INTP load_multi_cols)
{
    FFTZ_DOUBLE angle_base = -AOCLFFTZ_2_PI / (FFTZ_DOUBLE)(radix * m);

    if (load_multi_cols == 0)
    {
        FFTZ_INTP idx = 0;
        for (FFTZ_INTP j = 0; j < m; ++j)
        {
            for (FFTZ_INTP k = 1; k < radix; ++k)
            {
                FFTZ_DOUBLE angle =
                    angle_base * (FFTZ_DOUBLE)k * (FFTZ_DOUBLE)j;
                tw[idx++] = cos(angle);
                tw[idx++] = sin(angle);
            }
        }
        return;
    }

    FFTZ_INTP idx = 0;
    FFTZ_INTP base_col = 0;
    FFTZ_INTP rem = m;

    while (rem >= register_width)
    {
        idx = populate_twiddle_per_simd_butterfly_double(
              tw, idx, radix, angle_base, base_col, register_width);
        base_col += register_width;
        rem -= register_width;
    }
    if (register_width > NUM_SETS_256_D && rem >= NUM_SETS_256_D)
    {
        idx = populate_twiddle_per_simd_butterfly_double(
              tw, idx, radix, angle_base, base_col, NUM_SETS_256_D);
        base_col += NUM_SETS_256_D;
        rem -= NUM_SETS_256_D;
    }
    if (register_width > NUM_SETS_128_D && (rem & 1))
    {
        idx = populate_twiddle_per_simd_butterfly_double(
              tw, idx, radix, angle_base, base_col, NUM_SETS_128_D);
    }
}

FFTZ_VOID compute_twiddle_buffer(FFTZ_VOID *twiddle_buffer, FFTZ_INTP radix,
                                 FFTZ_INTP m, FFTZ_INTP register_width,
                                 FFTZ_INTP load_multi_cols, FFTZ_UINT32 dt_prec)
{
    if (twiddle_buffer == NULL || radix < 2 || m < 1)
    {
        return;
    }

    if (dt_prec == DT_DOUBLE)
    {
        compute_twiddle_buffer_linear_double((FFTZ_DOUBLE *)twiddle_buffer,
                                             radix, m, register_width,
                                             load_multi_cols);
    }
    else
    {
        compute_twiddle_buffer_linear_float((FFTZ_FLOAT *)twiddle_buffer, radix,
                                            m, register_width, load_multi_cols);
    }
}
#endif

#if IN_MEMORY_TWIDDLE_FACTORS == 1
// Populate the twiddles for one SIMD butterfly of real-FFT c2c twiddles.
// Returns the advanced index.
static inline
FFTZ_INTP populate_twiddle_per_simd_butterfly_real_float(FFTZ_FLOAT *tw,
                                                         FFTZ_INTP idx,
                                                         FFTZ_INTP radix,
                                                         FFTZ_FLOAT angle_base,
                                                         FFTZ_INTP base_col,
                                                         FFTZ_INTP tile_w)
{
    for (FFTZ_INTP k = 1; k < radix; ++k)
    {
        for (FFTZ_INTP j = 0; j < tile_w; ++j)
        {
            FFTZ_INTP col = base_col + j;
            FFTZ_FLOAT angle =
                angle_base * (FFTZ_FLOAT)k * (FFTZ_FLOAT)(col + 1);
            tw[idx++] = cosf(angle);
            tw[idx++] = sinf(angle);
        }
    }
    return idx;
}

static inline FFTZ_INTP
populate_twiddle_per_simd_butterfly_real_double(FFTZ_DOUBLE *tw,
                                                FFTZ_INTP idx, FFTZ_INTP radix,
                                                FFTZ_DOUBLE angle_base,
                                                FFTZ_INTP base_col,
                                                FFTZ_INTP tile_w)
{
    for (FFTZ_INTP k = 1; k < radix; ++k)
    {
        for (FFTZ_INTP j = 0; j < tile_w; ++j)
        {
            FFTZ_INTP col = base_col + j;
            FFTZ_DOUBLE angle =
                angle_base * (FFTZ_DOUBLE)k * (FFTZ_DOUBLE)(col + 1);
            tw[idx++] = cos(angle);
            tw[idx++] = sin(angle);
        }
    }
    return idx;
}

FFTZ_VOID compute_twiddle_buffer_real_float(FFTZ_VOID *twiddle_buffer,
                                            FFTZ_INTP radix,
                                            FFTZ_INTP num_c2c_per_group,
                                            FFTZ_INTP freq_factor,
                                            FFTZ_UINT8 dir,
                                            FFTZ_INTP register_width,
                                            FFTZ_INTP load_multi_cols)
{
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)twiddle_buffer;

    FFTZ_FLOAT sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FFTZ_FLOAT angle_base = (sign * AOCLFFTZ_2_PI) / (FFTZ_FLOAT)freq_factor;

    if (load_multi_cols == 0)
    {
        FFTZ_INTP idx = 0;
        for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
        {
            for (FFTZ_INTP k = 1; k < radix; ++k)
            {
                FFTZ_FLOAT angle =
                    angle_base * (FFTZ_FLOAT)k * (FFTZ_FLOAT)(j + 1);
                tw[idx++] = cosf(angle);
                tw[idx++] = sinf(angle);
            }
        }
        return;
    }

    FFTZ_INTP idx = 0;
    FFTZ_INTP base_col = 0;
    FFTZ_INTP rem = num_c2c_per_group;

    while (rem >= register_width)
    {
        idx = populate_twiddle_per_simd_butterfly_real_float(
              tw, idx, radix, angle_base, base_col, register_width);
        base_col += register_width;
        rem -= register_width;
    }
    if (register_width > NUM_SETS_256_S && rem >= NUM_SETS_256_S)
    {
        idx = populate_twiddle_per_simd_butterfly_real_float(
              tw, idx, radix, angle_base, base_col, NUM_SETS_256_S);
        base_col += NUM_SETS_256_S;
        rem -= NUM_SETS_256_S;
    }
    if (register_width > NUM_SETS_128_S && rem >= NUM_SETS_128_S)
    {
        idx = populate_twiddle_per_simd_butterfly_real_float(
              tw, idx, radix, angle_base, base_col, NUM_SETS_128_S);
        base_col += NUM_SETS_128_S;
        rem -= NUM_SETS_128_S;
    }
    if (rem > 0)
    {
        idx = populate_twiddle_per_simd_butterfly_real_float(
              tw, idx, radix, angle_base, base_col, rem);
    }
}

FFTZ_VOID compute_twiddle_buffer_real_double(FFTZ_VOID *twiddle_buffer,
                                             FFTZ_INTP radix,
                                             FFTZ_INTP num_c2c_per_group,
                                             FFTZ_INTP freq_factor,
                                             FFTZ_UINT8 dir,
                                             FFTZ_INTP register_width,
                                             FFTZ_INTP load_multi_cols)
{
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)twiddle_buffer;

    FFTZ_DOUBLE sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FFTZ_DOUBLE angle_base = (sign * AOCLFFTZ_2_PI) / (FFTZ_DOUBLE)freq_factor;

    if (load_multi_cols == 0)
    {
        FFTZ_INTP idx = 0;
        for (FFTZ_INTP j = 0; j < num_c2c_per_group; ++j)
        {
            for (FFTZ_INTP k = 1; k < radix; ++k)
            {
                FFTZ_DOUBLE angle =
                    angle_base * (FFTZ_DOUBLE)k * (FFTZ_DOUBLE)(j + 1);
                tw[idx++] = cos(angle);
                tw[idx++] = sin(angle);
            }
        }
        return;
    }

    FFTZ_INTP idx = 0;
    FFTZ_INTP base_col = 0;
    FFTZ_INTP rem = num_c2c_per_group;

    while (rem >= register_width)
    {
        idx = populate_twiddle_per_simd_butterfly_real_double(
              tw, idx, radix, angle_base, base_col, register_width);
        base_col += register_width;
        rem -= register_width;
    }
    if (register_width > NUM_SETS_256_D && rem >= NUM_SETS_256_D)
    {
        idx = populate_twiddle_per_simd_butterfly_real_double(
              tw, idx, radix, angle_base, base_col, NUM_SETS_256_D);
        base_col += NUM_SETS_256_D;
        rem -= NUM_SETS_256_D;
    }
    if (register_width > NUM_SETS_128_D && (rem & 1))
    {
        idx = populate_twiddle_per_simd_butterfly_real_double(
              tw, idx, radix, angle_base, base_col, NUM_SETS_128_D);
    }
}

FFTZ_VOID
compute_twiddle_buffer_real(FFTZ_VOID *twiddle_buffer, FFTZ_INTP radix,
                            FFTZ_INTP num_c2c_per_group, FFTZ_INTP freq_factor,
                            FFTZ_UINT8 dir, FFTZ_INTP register_width,
                            FFTZ_INTP load_multi_cols, FFTZ_UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_twiddle_buffer_real_float(twiddle_buffer, radix,
                                          num_c2c_per_group, freq_factor, dir,
                                          register_width, load_multi_cols);
    }
    else
    {
        compute_twiddle_buffer_real_double(twiddle_buffer, radix,
                                           num_c2c_per_group, freq_factor, dir,
                                           register_width, load_multi_cols);
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
        FFTZ_FLOAT angle1 = -AOCLFFTZ_2_PIf * k / n;        /* W_n^k */
        FFTZ_FLOAT angle3 = -3.0f * AOCLFFTZ_2_PIf * k / n; /* W_n^(3k) */

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
        FFTZ_DOUBLE angle1 = -AOCLFFTZ_2_PI * k / n;       /* W_n^k */
        FFTZ_DOUBLE angle3 = -3.0 * AOCLFFTZ_2_PI * k / n; /* W_n^(3k) */

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
