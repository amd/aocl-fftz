// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bluestein_utils.c
 *
 * @brief Utility functions for the Bluestein FFT algorithm.
 *
 * This file provides core utility functions for the Bluestein (chirp-z)
 * algorithm including:
 * - Extended length computation for smooth FFT sizes
 * - Chirp sequence generation
 *
 * @author Srirammaswamy Srinivasan
 */

#include <math.h>
#include <string.h> /* for memset */
#include "core/common/bluestein_utils.h"

/**
 * Supported prime factors for efficient FFT computation.
 * These correspond to available prime radix kernels in the library.
 */
static const UINT32 AOCLFFTZ_SUPPORTED_PRIMES[] = {2, 3, 5, 7, 11, 13};
static const UINT32 AOCLFFTZ_NUM_SUPPORTED_PRIMES =
    sizeof(AOCLFFTZ_SUPPORTED_PRIMES) / sizeof(AOCLFFTZ_SUPPORTED_PRIMES[0]);

/**
 * @brief Computes the extended length for Bluestein algorithm.
 *
 * Finds the smallest value m >= 2n-1 such that m can be completely factored
 * using only the supported prime factors.
 *
 * @param[in] n Original input length
 * @return INTP Extended length m suitable for FFT convolution
 */
INTP get_extended_length(INTP n)
{
    INTP min_length;

    for (min_length = (2 * n) - 1;; min_length++)
    {
        INTP quo = min_length;

        for (UINT32 i = 0; i < AOCLFFTZ_NUM_SUPPORTED_PRIMES; i++)
        {
            UINT32 prime = AOCLFFTZ_SUPPORTED_PRIMES[i];
            while (quo % prime == 0)
            {
                quo /= prime;
            }
            if (quo == 1)
            {
                break;
            }
        }

        if (quo == 1)
        {
            break;
        }
    }

    return min_length;
}

/**
 * @brief Computes the chirp sequence (twiddle factors) for Bluestein algorithm.
 *
 * Generates the complex exponential sequence values, pads with zeros (to meet
 * FFT size requirement for convolution), and mirrors the sequence for
 * convolution. The sequence is stored in complex interleaved format in
 * sol->dft_bufs->bluestein->B.
 *
 * Output layout:
 *                  Bluestein sequence B of length m
 *        <------ (n) -----><-- (m-2n+1) --><----- (n-1) ----->
 *        |     values      |     zeros     | reversed values |
 *
 *   where values are complex exponentials exp(j*pi*k^2/n) for k=0..n-1
 *
 * @param[in,out] sol Solution descriptor containing output buffer and metadata
 * @param[in]     m   Extended length (>= 2n-1) for FFT convolution
 * @return INT32 BLUESTEIN_SUCCESS on success
 */
INT32 compute_chirp_sequence(aoclfftz_solution_t *sol, INTP m)
{
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INTP n = sol->decomp_scheme->dims[0].n;
    INTP n2 = n * 2;

    if (precision == DT_FLOAT)
    {
        FLOAT *B = (FLOAT *)sol->dft_bufs->bluestein->B;

        for (INTP i = 0; i < n; i++)
        {
            INTP sq_idx_mod = (i * i) % n2;
            FLOAT angle = (AOCLFFTZ_2_PIf * sq_idx_mod) / n2;
            FLOAT re = cosf(angle);
            FLOAT im = sinf(angle);

            INTP idx        = i * DATA_STRIDE;
            INTP mirror_idx = (m - i) * DATA_STRIDE;

            B[idx]     = re;
            B[idx + 1] = im;

            if (i > 0)
            {
                B[mirror_idx]     = re;
                B[mirror_idx + 1] = im;
            }
        }

        INTP pad_count = m - n2 + 1;
        if (pad_count > 0)
        {
            memset(B + n * DATA_STRIDE, 0,
                   pad_count * DATA_STRIDE * sizeof(FLOAT));
        }
    }
    else
    {
        DOUBLE *B = (DOUBLE *)sol->dft_bufs->bluestein->B;

        for (INTP i = 0; i < n; i++)
        {
            INTP sq_idx_mod = (i * i) % n2;
            DOUBLE angle = (AOCLFFTZ_2_PI * sq_idx_mod) / n2;
            DOUBLE re = cos(angle);
            DOUBLE im = sin(angle);

            INTP idx        = i * DATA_STRIDE;
            INTP mirror_idx = (m - i) * DATA_STRIDE;

            B[idx]     = re;
            B[idx + 1] = im;

            if (i > 0)
            {
                B[mirror_idx]     = re;
                B[mirror_idx + 1] = im;
            }
        }

        INTP pad_count = m - n2 + 1;
        if (pad_count > 0)
        {
            memset(B + n * DATA_STRIDE, 0,
                   pad_count * DATA_STRIDE * sizeof(DOUBLE));
        }
    }

    return BLUESTEIN_SUCCESS;
}

