/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file kernel.c
 *
 *  @brief Provides common functionality for a Kernel
 *
 *  This file implements common kernel functions including kernels registration
 *  related function.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#ifdef ENABLE_AVX128
#define ACCESS_AVX128 1
#else
#define ACCESS_AVX128 0
#endif

#ifdef ENABLE_AVX256
#define ACCESS_AVX256 1
#else
#define ACCESS_AVX256 0
#endif

#ifdef ENABLE_AVX512
#define ACCESS_AVX512 1
#else
#define ACCESS_AVX512 0
#endif

static INTP real_variant_limits[NUM_KERNEL_CATEGORIES] =
{
    NUM_KERNELS_IN_EACH_DFT_VARIANT,
    2 * NUM_KERNELS_IN_EACH_DFT_VARIANT,
    3 * NUM_KERNELS_IN_EACH_DFT_VARIANT,
};

static INTP limits[NUM_KERNEL_CATEGORIES] =
{
    NUM_KERNELS_IN_EACH_CATEGORY,
    2 * NUM_KERNELS_IN_EACH_CATEGORY,
    3 * NUM_KERNELS_IN_EACH_CATEGORY,
    4 * NUM_KERNELS_IN_EACH_CATEGORY
};

static INTP sets_s[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_C_S,
    NUM_SETS_128_S,
    NUM_SETS_256_S,
    NUM_SETS_512_S
};

static INTP sets_d[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_C_D,
    NUM_SETS_128_D,
    NUM_SETS_256_D,
    NUM_SETS_512_D
};

INT32 register_kernels_real(
    kernel_t kertab[NUM_KERNELS_IN_TABLE_REAL],
    kernel_fp_list_t static_kernel_table[NUM_REAL_KERNELS_VARIANTS]
                                        [NUM_KERNELS_IN_EACH_CATEGORY]
                                        [NUM_KERNEL_CATEGORIES],
    INT32 dt, INT32 dir, INT32 cpu_flags)
{
    INTP kcat_register_available[NUM_KERNEL_CATEGORIES] =
    {
        1, // C kernels are always registered
        ACCESS_AVX128 && (cpu_flags > 0),
        ACCESS_AVX256 && (cpu_flags > 1),
        ACCESS_AVX512 && (cpu_flags > 2),
    };

    INTP row_offset = 0;

    for (INTP rkvar = 0; rkvar < NUM_REAL_KERNELS_VARIANTS; rkvar++)
    {
        INTP offset = row_offset;
        for (INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
        {
            if (kcat_register_available[kcat])
            {
                for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY;
                     i++, offset++)
                {
                    if (static_kernel_table[rkvar][i][kcat].k_register_kernel !=
                        NULL)
                    {
                        kertab[offset].radix =
                            static_kernel_table[rkvar][i][kcat].radix;
                        kertab[offset].kfft =
                            static_kernel_table[rkvar][i][kcat]
                                .k_register_kernel(dt, dir);
                        kertab[offset].k_ops_cnt =
                            static_kernel_table[rkvar][i][kcat].k_ops_cnt;
                        kertab[offset].sets[DT_FLOAT - 2] = sets_s[kcat];
                        kertab[offset].sets[DT_DOUBLE - 2] = sets_d[kcat];
                    }
                }
                offset = row_offset + limits[kcat];
            }
        }
        row_offset = real_variant_limits[rkvar];
    }

    return KERNEL_SUCCESS;
}

INT32 register_kernels_complex(
    kernel_t kertab[NUM_KERNELS_IN_TABLE_COMPLEX],
    kernel_fp_list_t static_kernel_table[NUM_KERNELS_IN_EACH_CATEGORY]
                                        [NUM_KERNEL_CATEGORIES],
    INT32 dt, INT32 dir, INT32 cpu_flags)
{
    INTP kcat_register_available[NUM_KERNEL_CATEGORIES] =
    {
        1, // C kernels are always registered
        ACCESS_AVX128 && (cpu_flags > 0),
        ACCESS_AVX256 && (cpu_flags > 1),
        ACCESS_AVX512 && (cpu_flags > 2),
    };

    INTP offset = 0;

    for (INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
    {
        if (kcat_register_available[kcat])
        {
            for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++, offset++)
            {
                if (static_kernel_table[i][kcat].k_register_kernel != NULL)
                {
                    kertab[offset].radix = static_kernel_table[i][kcat].radix;
                    kertab[offset].kfft =
                        static_kernel_table[i][kcat].k_register_kernel(dt, dir);
                    kertab[offset].k_ops_cnt =
                        static_kernel_table[i][kcat].k_ops_cnt;
                    kertab[offset].sets[DT_FLOAT - 2] = sets_s[kcat];
                    kertab[offset].sets[DT_DOUBLE - 2] = sets_d[kcat];
                }
            }
            offset = limits[kcat];
        }
    }

    return KERNEL_SUCCESS;
}

#undef ACCESS_AVX128
#undef ACCESS_AVX256
#undef ACCESS_AVX512

