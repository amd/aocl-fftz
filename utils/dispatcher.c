// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file dispatcher.c
 *
 *  @brief Implementation of Hardware capability dispatcher
 *
 *  Builds a single feature bitmask from the cpu_features module (with
 *  ENABLE_AVX* compile-time macros). The maximum SIMD level is derived
 *  from this bitmask.
 *
 *  @author Prasandh Sankarankutty
 *  @author Partiksha
 */

#include "utils/dispatcher.h"
#include "utils/cpu_features.h"

/**
 * Build the cached feature bitmask from CPU/OS detection.
 */
static FFTZ_VOID detect_capabilities(FFTZ_UINT64 *cpu_capabilities,
                                     FFTZ_UINT32 max_cpuid_leaf,
                                     cpuid_result_t cpuid_leaf1,
                                     cpuid_result_t cpuid_leaf7)
{
    if (has_avx(cpuid_leaf1) && has_avx_support(cpuid_leaf1))
    {
        *cpu_capabilities |= FEATURE_AVX;
        if (has_fma(cpuid_leaf1))
        {
            *cpu_capabilities |= FEATURE_FMA;
        }
        if (has_avx2(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX2;
        }
    }

    if (has_avx512_support(cpuid_leaf1))
    {
        if (has_avx512f(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX512F;
        }
        if (has_avx512dq(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX512DQ;
        }
        if (has_avx512vl(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX512VL;
        }
        if (has_avx512cd(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX512CD;
        }
        if (has_avx512bw(max_cpuid_leaf, cpuid_leaf7))
        {
            *cpu_capabilities |= FEATURE_AVX512BW;
        }
    }
}

/**
 * Initialize the dynamic dispatcher and detect capabilities.
 */
FFTZ_VOID init_dynamic_dispatcher(FFTZ_UINT64 *cpu_capabilities)
{
    FFTZ_UINT32 max_cpuid_leaf;
    cpuid_result_t cpuid_leaf1, cpuid_leaf7;

    initialize_cpuid_cache(&max_cpuid_leaf, &cpuid_leaf1, &cpuid_leaf7);
    detect_capabilities(cpu_capabilities, max_cpuid_leaf, cpuid_leaf1,
                        cpuid_leaf7);
}

/* Compute build max ISA dispatch level from the feature bitmask */
FFTZ_INT32 get_max_build_isa_level(FFTZ_UINT64 cpu_capabilities)
{
    FFTZ_INT32 level = optlevel_scalar;

#ifdef ENABLE_AVX128
    if ((cpu_capabilities & FEATURE_LEVEL_AVX128) == FEATURE_LEVEL_AVX128)
    {
        level = optlevel_avx128;
    }
#endif

#ifdef ENABLE_AVX256
    if ((cpu_capabilities & FEATURE_LEVEL_AVX256) == FEATURE_LEVEL_AVX256)
    {
        level = optlevel_avx256;
    }
#endif

#ifdef ENABLE_AVX512
    if ((cpu_capabilities & FEATURE_LEVEL_AVX512) == FEATURE_LEVEL_AVX512)
    {
        level = optlevel_avx512;
    }
#endif

    return level;
}
