/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file data_generation.c
 *
 *  @brief Input data generation utility functions.
 *
 *  This file contains the utility functions for input data generation
 *  for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include <stdlib.h>
#include <string.h>
#include "test/utils/data_generation.h"
#include "test/aoclfftz_corebench.h"

/**
 * @brief Prepare FLOAT input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_f(VOID *input, INTP n, INTP *idx_map, INT32 input_type)
{
    FLOAT *input_f = (FLOAT *)input;
    INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_f[idx * T_DATA_STRIDE] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
                input_f[idx * T_DATA_STRIDE + 1] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
            }
        }
        else
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_f[idx_map[idx] * T_DATA_STRIDE] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
                input_f[idx_map[idx] * T_DATA_STRIDE + 1] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
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
        memset(input_f, 0, n * T_DATA_STRIDE);
        input_f[idx * T_DATA_STRIDE] =
            (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
        input_f[idx * T_DATA_STRIDE + 1] =
            (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        INTP half_size = n >=2 ? (n / 2) : 1;
        INTP cycles = (rand() % half_size) + 2;
        FLOAT size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        FLOAT scale = ((FLOAT)rand() / (FLOAT)RAND_MAX) * 5.0f;
        if (idx_map == NULL)
        {
            for (INTP i = 0; i < n; i++)
            {
                input_f[((i + shift) % n) * T_DATA_STRIDE] =
                    cosf((FLOAT)(i * size) / n) * scale;
                input_f[((i + shift) % n) * T_DATA_STRIDE + 1] =
                    sinf((FLOAT)(i * size) / n) * scale;
            }
        }
        else
        {
            for (INTP i = 0; i < n; i++)
            {
                input_f[idx_map[(i + shift) % n] * T_DATA_STRIDE] =
                    cosf((FLOAT)(i * size) / n) * scale;
                input_f[idx_map[(i + shift) % n] * T_DATA_STRIDE + 1] =
                    sinf((FLOAT)(i * size) / n) * scale;;
            }
        }
    }
}

/**
 * @brief Prepare DOUBLE input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_d(VOID *input, INTP n, INTP *idx_map, INT32 input_type)
{
    DOUBLE *input_d = (DOUBLE *)input;
    INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_d[idx * T_DATA_STRIDE] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
                input_d[idx * T_DATA_STRIDE + 1] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
            }
        }
        else
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_d[idx_map[idx] * T_DATA_STRIDE] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
                input_d[idx_map[idx] * T_DATA_STRIDE + 1] =
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
        memset(input_d, 0, n * T_DATA_STRIDE);
        input_d[idx * T_DATA_STRIDE] = (20.0 / RAND_MAX) * rand() - 10.0;
        input_d[idx * T_DATA_STRIDE + 1] = (20.0 / RAND_MAX) * rand() - 10.0;
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        INTP half_size = n >=2 ? (n / 2) : 1;
        INTP cycles = (rand() % half_size) + 2;
        DOUBLE size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        DOUBLE scale = ((DOUBLE)rand() / RAND_MAX) * 5.0;
        if (idx_map == NULL)
        {
            for (INTP i = 0; i < n; i++)
            {
                input_d[((i + shift) % n) * T_DATA_STRIDE] =
                    cos((DOUBLE)(i * size) / n) * scale;
                input_d[((i + shift) % n) * T_DATA_STRIDE + 1] =
                    sin((DOUBLE)(i * size) / n) * scale;;
            }
        }
        else
        {
            for (INTP i = 0; i < n; i++)
            {
                input_d[idx_map[(i + shift) % n] * T_DATA_STRIDE] =
                    cos((DOUBLE)(i * size) / n) * scale;
                input_d[idx_map[(i + shift) % n] * T_DATA_STRIDE + 1] =
                    sin((DOUBLE)(i * size) / n) * scale;;
            }
        }
    }
}
