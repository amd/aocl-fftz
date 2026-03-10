/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
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
static VOID detect_capabilities(UINT64 *cpu_capabilities, UINT32 max_cpuid_leaf,
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
VOID init_dynamic_dispatcher(UINT64 *cpu_capabilities)
{
    UINT32 max_cpuid_leaf;
    cpuid_result_t cpuid_leaf1, cpuid_leaf7;

    initialize_cpuid_cache(&max_cpuid_leaf, &cpuid_leaf1, &cpuid_leaf7);
    detect_capabilities(cpu_capabilities, max_cpuid_leaf, cpuid_leaf1,
                        cpuid_leaf7);
}

/* Compute build max ISA dispatch level from the feature bitmask */
INT32 get_max_build_isa_level(UINT64 cpu_capabilities)
{
    INT32 level = optlevel_scalar;

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
