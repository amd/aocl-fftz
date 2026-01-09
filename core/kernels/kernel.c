// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#include "utils/dispatcher.h"

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

static INTP sets_complex_s[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_C_S,
    NUM_SETS_128_S,
    NUM_SETS_256_S,
    NUM_SETS_512_S
};

static INTP sets_complex_d[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_C_D,
    NUM_SETS_128_D,
    NUM_SETS_256_D,
    NUM_SETS_512_D
};

static INTP sets_real_s[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_REAL_C_S,
    NUM_SETS_REAL_128_S,
    NUM_SETS_REAL_256_S,
    NUM_SETS_REAL_512_S
};

static INTP sets_real_d[NUM_KERNEL_CATEGORIES] =
{
    NUM_SETS_REAL_C_D,
    NUM_SETS_REAL_128_D,
    NUM_SETS_REAL_256_D,
    NUM_SETS_REAL_512_D
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
                        if (rkvar == C2C_KERNEL) // c2c variant
                        {
                            kertab[offset].sets[DT_FLOAT - 2] =
                                sets_complex_s[kcat];
                            kertab[offset].sets[DT_DOUBLE - 2] =
                                sets_complex_d[kcat];
                        }
                        else // r2hc and r2hcf variants
                        {
                            kertab[offset].sets[DT_FLOAT - 2] =
                                sets_real_s[kcat];
                            kertab[offset].sets[DT_DOUBLE - 2] =
                                sets_real_d[kcat];
                        }
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
                    kertab[offset].sets[DT_FLOAT - 2] = sets_complex_s[kcat];

                    // Since the radix 48 AVX512 kernel processes only 1 set at
                    // a time, (unlike other AVX512 kernels which process
                    // multiple sets), we need to set the sets_d value
                    // accordingly.
                    if (kcat == 3 /* AVX512 */ && kertab[offset].radix == 48)
                    {
                        kertab[offset].sets[DT_DOUBLE - 2] = 1;
                    }
                    else
                    {
                        kertab[offset].sets[DT_DOUBLE - 2] = sets_complex_d[kcat];
                    }
                }
            }
            offset = limits[kcat];
        }
    }

    return KERNEL_SUCCESS;
}

/**
 * @brief Registers the appropriate elementwise multiplication kernel.
 *
 * Selects the best available SIMD implementation based on the optimization
 * level and data type:
 * - optlevel_avx512 (3): AVX512 implementation
 * - optlevel_avx256 (2): AVX256 implementation
 * - optlevel_avx128 (1): AVX128 implementation
 * - optlevel_scalar (0): Scalar C implementation
 *
 * @param[in] cpu_flags Optimization level (optimization_level_t) obtained
 *                      from get_max_build_isa_level()
 * @param[in] dt        Data type (DT_FLOAT or DT_DOUBLE)
 * @param[in] direction FORWARD_FFT_DIR  -> forward  (a .* conj(b)),
 *                      BACKWARD_FFT_DIR -> backward (a .* b)
 * @return Function pointer to the selected elementwise multiplication kernel
 */
elementwise_mul_ register_elementwise_mul_kernel(INT32 cpu_flags, INT32 dt,
                                                 UINT8 direction)
{
#ifdef ENABLE_AVX512
    if (cpu_flags >= optlevel_avx512)
    {
        return register_elementwise_mul_avx512(dt, direction);
    }
#endif

#ifdef ENABLE_AVX256
    if (cpu_flags >= optlevel_avx256)
    {
        return register_elementwise_mul_avx256(dt, direction);
    }
#endif

#ifdef ENABLE_AVX128
    if (cpu_flags >= optlevel_avx128)
    {
        return register_elementwise_mul_avx128(dt, direction);
    }
#endif

    /* Default to C implementation */
    return register_elementwise_mul_c(dt, direction);
}

/**
 * @brief Registers the appropriate normalization kernel.
 *
 * Selects the best available SIMD implementation based on the optimization
 * level and data type:
 * - optlevel_avx512 (3): AVX512 implementation
 * - optlevel_avx256 (2): AVX256 implementation
 * - optlevel_avx128 (1): AVX128 implementation
 * - optlevel_scalar (0): Scalar C implementation
 *
 * @param[in] cpu_flags Optimization level (optimization_level_t) obtained
 *                      from get_max_build_isa_level()
 * @param[in] dt        Data type (DT_FLOAT or DT_DOUBLE)
 * @return Function pointer to the selected normalization kernel
 */
normalize_ register_normalize_kernel(INT32 cpu_flags, INT32 dt)
{
#ifdef ENABLE_AVX512
    if (cpu_flags >= optlevel_avx512)
    {
        return register_normalize_avx512(dt);
    }
#endif

#ifdef ENABLE_AVX256
    if (cpu_flags >= optlevel_avx256)
    {
        return register_normalize_avx256(dt);
    }
#endif

#ifdef ENABLE_AVX128
    if (cpu_flags >= optlevel_avx128)
    {
        return register_normalize_avx128(dt);
    }
#endif

    /* Default to C implementation */
    return register_normalize_c(dt);
}

#undef ACCESS_AVX128
#undef ACCESS_AVX256
#undef ACCESS_AVX512
