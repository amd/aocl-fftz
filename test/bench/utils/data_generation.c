// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file data_generation.c
 *
 *  @brief Input data generation utility functions.
 *
 *  This file contains the utility functions for input data generation
 *  for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include <stdlib.h>
#include <string.h>
#include "test/bench/utils/data_generation.h"
#include "test/bench/aoclfftz_bench.h"
#include "test/bench/utils/bench_utils.h"

/**
 * @brief Prepare FFTZ_FLOAT input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return FFTZ_VOID
 */
FFTZ_VOID prepare_input_data_f(FFTZ_VOID *input, FFTZ_INTP n,
                               FFTZ_INTP *idx_map, FFTZ_INT32 input_type,
                               FFTZ_INT32 data_stride)
{
    FFTZ_FLOAT *input_f = (FFTZ_FLOAT *)input;
    FFTZ_INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n * data_stride; ++idx)
            {
                input_f[idx] = (20.0f / (FFTZ_FLOAT)RAND_MAX) * rand() - 10.0f;
            }
        }
        else
        {
            for (idx = 0; idx < n * data_stride; ++idx)
            {
                input_f[idx_map[idx / data_stride] * data_stride +
                        (idx % data_stride)] =
                    (20.0f / (FFTZ_FLOAT)RAND_MAX) * rand() - 10.0f;
            }
        }
    }
    // impulse input
    // range: [-10.0, 10.0)
    else if (input_type == IMPULSE_INPUT)
    {
        if (idx_map == NULL)
        {
            idx = rand() % n;
        }
        else
        {
            idx = idx_map[rand() % n];
        }
        memset(input_f, 0, n * data_stride);
        for (int i = 0; i < data_stride; i++)
        {
            input_f[(idx * data_stride) + i] =
                (20.0f / (FFTZ_FLOAT)RAND_MAX) * rand() - 10.0f;
        }
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        FFTZ_INTP half_size = n >= 2 ? (n / 2) : 1;
        FFTZ_INTP cycles = (rand() % half_size) + 2;
        FFTZ_FLOAT size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        FFTZ_INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        FFTZ_FLOAT scale = ((FFTZ_FLOAT)rand() / (FFTZ_FLOAT)RAND_MAX) * 5.0f;
        memset(input_f, 0, n * data_stride);
        if (idx_map == NULL)
        {
            for (FFTZ_INTP i = 0; i < n; i++)
            {
                input_f[((i + shift) % n) * data_stride] =
                    sinf((i * size) / n) * scale;
            }
        }
        else
        {
            for (FFTZ_INTP i = 0; i < n; i++)
            {
                input_f[idx_map[(i + shift) % n] * data_stride] =
                    sinf((i * size) / n) * scale;
            }
        }
    }
}

/**
 * @brief Prepare FFTZ_DOUBLE input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return FFTZ_VOID
 */
FFTZ_VOID prepare_input_data_d(FFTZ_VOID *input, FFTZ_INTP n,
                               FFTZ_INTP *idx_map, FFTZ_INT32 input_type,
                               FFTZ_INT32 data_stride)
{
    FFTZ_DOUBLE *input_d = (FFTZ_DOUBLE *)input;
    FFTZ_INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n * data_stride; ++idx)
            {
                input_d[idx] = (20.0 / RAND_MAX) * rand() - 10.0;
            }
        }
        else
        {
            for (idx = 0; idx < n * data_stride; ++idx)
            {
                input_d[idx_map[idx / data_stride] * data_stride +
                        (idx % data_stride)] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
            }
        }
    }
    // impulse input
    // range: [-10.0, 10.0)
    else if (input_type == IMPULSE_INPUT)
    {
        if (idx_map == NULL)
        {
            idx = rand() % n;
        }
        else
        {
            idx = idx_map[rand() % n];
        }
        memset(input_d, 0, n * data_stride);
        for (int i = 0; i < data_stride; i++)
        {
            input_d[(idx * data_stride) + i] =
                (20.0 / RAND_MAX) * rand() - 10.0;
        }
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        FFTZ_INTP half_size = n >= 2 ? (n / 2) : 1;
        FFTZ_INTP cycles = (rand() % half_size) + 2;
        FFTZ_DOUBLE size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        FFTZ_INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        FFTZ_DOUBLE scale = ((FFTZ_DOUBLE)rand() / RAND_MAX) * 5.0;
        memset(input_d, 0, n * data_stride);
        if (idx_map == NULL)
        {
            for (FFTZ_INTP i = 0; i < n; i++)
            {
                input_d[((i + shift) % n) * data_stride] =
                    cos((FFTZ_DOUBLE)(i * size) / n) * scale;
            }
        }
        else
        {
            for (FFTZ_INTP i = 0; i < n; i++)
            {
                input_d[idx_map[(i + shift) % n] * data_stride] =
                    cos((FFTZ_DOUBLE)(i * size) / n) * scale;
            }
        }
    }
}
