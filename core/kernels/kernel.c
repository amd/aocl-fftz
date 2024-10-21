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
 */

#include "core/kernels/kernel_list.h"

// Register all applicable solvers and kernels into the respective tables
// based on the input problem and cpu arch flags
INT32 register_kernels(kernel_t kertab[NUM_KERNELS_IN_TABLE],
                       INT32 dt, INT32 cpu_flags)
{
    UINT32 num_kernels = 0;
    UINT32 num_list_kernels;

    { // Only non-AVX ISA is supported, optimized C kernels applicable
        for (num_list_kernels = 0;
             num_list_kernels < NUM_KERNELS_IN_EACH_CATEGORY;
             num_list_kernels++)
        {
            if (kernels_c[num_list_kernels].k_register_kernel != NULL)
            {
                kertab[num_kernels].kfft =
                    kernels_c[num_list_kernels].k_register_kernel(dt);
                kertab[num_kernels].k_ops_cnt =
                    kernels_c[num_list_kernels].k_ops_cnt;
                kertab[num_kernels].radix = kernels_c[num_list_kernels].radix;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_C_D;
                num_kernels++;
            }
        }
    }

#ifdef ENABLE_AVX128
    if (cpu_flags >= 2) // AVX ISA is supported, 128-bit SIMD kernels applicable
    {
        for (num_list_kernels = 0;
             num_list_kernels < NUM_KERNELS_IN_EACH_CATEGORY;
             num_list_kernels++)
        {
            if (kernels_avx128[num_list_kernels].k_register_kernel != NULL)
            {
                kertab[num_kernels].kfft =
                    kernels_avx128[num_list_kernels].k_register_kernel(dt);
                kertab[num_kernels].k_ops_cnt =
                    kernels_avx128[num_list_kernels].k_ops_cnt;
                kertab[num_kernels].radix =
                    kernels_avx128[num_list_kernels].radix;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_128_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_128_D;
                num_kernels++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX256
    if (cpu_flags >= 3) // AVX2 ISA supported; 256-bit SIMD kernels applicable
    {
        for (num_list_kernels = 0;
             num_list_kernels < NUM_KERNELS_IN_EACH_CATEGORY;
             num_list_kernels++)
        {
            if (kernels_avx256[num_list_kernels].k_register_kernel != NULL)
            {
                kertab[num_kernels].kfft =
                    kernels_avx256[num_list_kernels].k_register_kernel(dt);
                kertab[num_kernels].k_ops_cnt =
                    kernels_avx256[num_list_kernels].k_ops_cnt;
                kertab[num_kernels].radix =
                    kernels_avx256[num_list_kernels].radix;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_256_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_256_D;
                num_kernels++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX512
    if (cpu_flags >= 4) // AVX512 ISA supported, 512-bit SIMD kernels applicable
    {
        for (num_list_kernels = 0;
             num_list_kernels < NUM_KERNELS_IN_EACH_CATEGORY;
             num_list_kernels++)
        {
            if (kernels_avx512[num_list_kernels].k_register_kernel != NULL)
            {
                kertab[num_kernels].kfft =
                    kernels_avx512[num_list_kernels].k_register_kernel(dt);
                kertab[num_kernels].k_ops_cnt =
                    kernels_avx512[num_list_kernels].k_ops_cnt;
                kertab[num_kernels].radix =
                    kernels_avx512[num_list_kernels].radix;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_512_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_512_D;
                num_kernels++;
            }
        }
    }
#endif

    return KERNEL_SUCCESS;
}
