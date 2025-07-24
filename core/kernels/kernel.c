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
 */

#include "core/kernels/kernel_list.h"

/**
 * @brief Register all applicable solvers and kernels into the respective dft
 * and rdft tables based on the input problem and cpu arch flags
 *
 * This function populates the kernel table with appropriate FFT kernels based
 * on:
 * - Data type (float/double)
 * - Transform direction (forward/backward)
 * - Transform type (Complex or Real transforms)
 * - Available CPU instruction set extensions (C, AVX128, AVX256, AVX512)
 *
 * The function registers three types of kernels when applicable:
 * - C2C (C2C) kernels: Always registered
 * - R2HC (R2HC) kernels: Only for real transforms
 * - R2HCF (R2HCF) kernels: Only for real transforms
 *
 * @param kertab Array to store registered kernel information
 * @param dt Data type identifier (DT_FLOAT or DT_DOUBLE)
 * @param dir Transform direction (forward or backward)
 * @param is_real Flag indicating if this is a real transform
 * @param cpu_flags CPU capability flags (0=C, 1=AVX128, 2=AVX256, 3=AVX512)
 *
 * @return KERNEL_SUCCESS on successful registration
 */
INT32 register_kernels(kernel_t kertab[NUM_KERNELS_IN_TABLE], INT32 dt,
                       INT32 dir, INT32 is_real, INT32 cpu_flags)
{
    UINT32 num_kernels = 0;
    UINT32 c2c_kernel_idx, r2hc_kernel_idx, r2hcf_kernel_idx;
    kernel_fp_list_t kernel_fp;

    // Register C-based (non-SIMD) kernels - always available
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_c_kernel:
            // Register C2C kernels
            kernel_fp = kernels_c2c_c[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain radix kernels for real transforms
                if (is_real && (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                                kernel_fp.radix == 13))
                {
                    c2c_kernel_idx++;
                    goto register_c2c_c_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_C_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
                // Register R2HC kernels
                kernel_fp = kernels_r2hc_c[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_C_D;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register R2HCF kernels
                kernel_fp = kernels_r2hcf_c[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_C_D;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }

#ifdef ENABLE_AVX128
    if (cpu_flags >= 1) // AVX128 ISA supported, 128-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx128_kernel:
            // Register AVX128 C2C kernels
            kernel_fp = kernels_c2c_avx128[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx128_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_128_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_128_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx128_kernel:
                // Register AVX128 R2HC kernels
                kernel_fp = kernels_r2hc_avx128[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx128_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_128_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_128_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX128 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx128[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_128_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_128_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX256
    if (cpu_flags >= 2) // AVX256 ISA supported; 256-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx256_kernel:
            // Register AVX256 C2C kernels
            kernel_fp = kernels_c2c_avx256[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx256_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_256_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_256_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx256_kernel:
                // Register AVX256 R2HC kernels
                kernel_fp = kernels_r2hc_avx256[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx256_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_256_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_256_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX256 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx256[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_256_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_256_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX512
    if (cpu_flags >= 3) // AVX512 ISA supported, 512-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx512_kernel:
            // Register AVX512 C2C kernels
            kernel_fp = kernels_c2c_avx512[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx512_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_512_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_512_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx512_kernel:
                // Register AVX512 R2HC kernels
                kernel_fp = kernels_r2hc_avx512[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx512_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_512_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_512_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX512 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx512[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_512_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_512_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

    return KERNEL_SUCCESS;
}

/**
 * @brief Register all applicable solvers and kernels into the fused twiddle+dft
 * and rdft tables based on the input problem and cpu arch flags
 *
 * This function populates the kernel table with appropriate FFT kernels based
 * on:
 * - Data type (float/double)
 * - Transform direction (forward/backward)
 * - Transform type (Complex or Real transforms)
 * - Available CPU instruction set extensions (C, AVX128, AVX256, AVX512)
 *
 * The function registers three types of kernels when applicable:
 * - C2C (TWIDDLE_C2C) kernels: Always registered
 * - R2HC (R2HC) kernels: Only for real transforms
 * - R2HCF (R2HCF) kernels: Only for real transforms
 *
 * @param kertab Array to store registered kernel information
 * @param dt Data type identifier (DT_FLOAT or DT_DOUBLE)
 * @param dir Transform direction (forward or backward)
 * @param is_real Flag indicating if this is a real transform
 * @param cpu_flags CPU capability flags (0=C, 1=AVX128, 2=AVX256, 3=AVX512)
 *
 * @return KERNEL_SUCCESS on successful registration
 */
INT32 register_twid_kernels(kernel_t kertab[NUM_KERNELS_IN_TABLE], INT32 dt,
                            INT32 dir, INT32 is_real, INT32 cpu_flags)
{
    UINT32 num_kernels = 0;
    UINT32 c2c_kernel_idx, r2hc_kernel_idx, r2hcf_kernel_idx;
    kernel_fp_list_t kernel_fp;

    // Register C-based (non-SIMD) kernels - always available
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_c_twid_kernel:
            // Register C2C kernels
            kernel_fp = kernels_twid_c2c_c[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain radix kernels for real transforms
                if (is_real && (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                                kernel_fp.radix == 13))
                {
                    c2c_kernel_idx++;
                    goto register_c2c_c_twid_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_C_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
                // Register R2HC kernels
                kernel_fp = kernels_r2hc_c[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_C_D;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register R2HCF kernels
                kernel_fp = kernels_r2hcf_c[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_C_S;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_C_D;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }

#ifdef ENABLE_AVX128
    if (cpu_flags >= 1) // AVX128 ISA supported, 128-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx128_twid_kernel:
            // Register AVX128 C2C kernels
            kernel_fp = kernels_twid_c2c_avx128[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx128_twid_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_128_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_128_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx128_twid_kernel:
                // Register AVX128 R2HC kernels
                kernel_fp = kernels_r2hc_avx128[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx128_twid_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_128_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_128_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX128 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx128[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_128_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_128_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX256
    if (cpu_flags >= 2) // AVX256 ISA supported; 256-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx256_twid_kernel:
            // Register AVX256 C2C kernels
            kernel_fp = kernels_twid_c2c_avx256[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx256_twid_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_256_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_256_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx256_twid_kernel:
                // Register AVX256 R2HC kernels
                kernel_fp = kernels_r2hc_avx256[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx256_twid_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_256_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_256_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX256 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx256[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_256_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_256_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

#ifdef ENABLE_AVX512
    if (cpu_flags >= 3) // AVX512 ISA supported, 512-bit SIMD kernels applicable
    {
        for (c2c_kernel_idx = 0, r2hc_kernel_idx = 0, r2hcf_kernel_idx = 0;
             c2c_kernel_idx < NUM_KERNELS_IN_EACH_CATEGORY;)
        {
register_c2c_avx512_twid_kernel:
            // Register AVX512 C2C kernels
            kernel_fp = kernels_twid_c2c_avx512[c2c_kernel_idx];
            if (kernel_fp.k_register_kernel != NULL)
            {
                // Skip certain unsupported radix kernels
                if (kernel_fp.radix == 9 || kernel_fp.radix == 11 ||
                    kernel_fp.radix == 12 || kernel_fp.radix == 13)
                {
                    c2c_kernel_idx++;
                    goto register_c2c_avx512_twid_kernel;
                }
                kertab[num_kernels].kfft =
                    kernel_fp.k_register_kernel(dt, dir);
                kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                kertab[num_kernels].radix = kernel_fp.radix;
                kertab[num_kernels].kernel_type = C2C_KERNEL;
                kertab[num_kernels].sets[DT_FLOAT - 2] = NUM_SETS_512_S;
                kertab[num_kernels].sets[DT_DOUBLE - 2] = NUM_SETS_512_D;
                num_kernels++;
            }
            c2c_kernel_idx++;

            if (is_real)
            {
register_r2hc_avx512_twid_kernel:
                // Register AVX512 R2HC kernels
                kernel_fp = kernels_r2hc_avx512[r2hc_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    // Skip radix-12 kernel (TODO: Enable radix-12 kernel)
                    if (kernel_fp.radix == 12)
                    {
                        r2hc_kernel_idx++;
                        goto register_r2hc_avx512_twid_kernel;
                    }
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HC_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_512_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_512_D * 2;
                    num_kernels++;
                }
                r2hc_kernel_idx++;

                // Register AVX512 R2HCF kernels
                kernel_fp = kernels_r2hcf_avx512[r2hcf_kernel_idx];
                if (kernel_fp.k_register_kernel != NULL)
                {
                    kertab[num_kernels].kfft =
                        kernel_fp.k_register_kernel(dt, dir);
                    kertab[num_kernels].k_ops_cnt = kernel_fp.k_ops_cnt;
                    kertab[num_kernels].radix = kernel_fp.radix;
                    kertab[num_kernels].kernel_type = R2HCF_KERNEL;
                    kertab[num_kernels].sets[DT_FLOAT - 2] =
                        NUM_SETS_512_S * 2;
                    kertab[num_kernels].sets[DT_DOUBLE - 2] =
                        NUM_SETS_512_D * 2;
                    num_kernels++;
                }
                r2hcf_kernel_idx++;
            }
        }
    }
#endif

    return KERNEL_SUCCESS;
}
