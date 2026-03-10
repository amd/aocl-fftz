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

/** @file cpu_features.h
 *
 *  @brief Low-level CPU feature detection utilities.
 *
 *  Provides CPUID/XGETBV based detection and OS support checks
 *  for SIMD extensions such as AVX, AVX-512, and FMA.
 *
 *  @author Prasandh Sankarankutty
 *  @author Partiksha
 */

#ifndef AOCLFFTZ_CPU_FEATURES_H
#define AOCLFFTZ_CPU_FEATURES_H

#include "api/types.h"
#include <stdint.h>

typedef struct cpuid_result
{
    UINT32 eax;
    UINT32 ebx;
    UINT32 ecx;
    UINT32 edx;
} cpuid_result_t;


// Fill CPUID cache (max leaf, leaf 1, leaf 7); call once then pass to feature APIs
VOID initialize_cpuid_cache(UINT32 *max_cpuid_leaf, cpuid_result_t *cpuid_leaf1,
                            cpuid_result_t *cpuid_leaf7);

// CPUID helpers
cpuid_result_t cpuid(UINT32 leaf, UINT32 subleaf);

// XGETBV / XCR0 helpers
UINT64 get_xcr0(VOID);
INT32 is_xgetbv_supported(cpuid_result_t cpuid_leaf1);

// CPU Feature detection (pass cached cpuid_leaf1/cpuid_leaf7 from initialize_cpuid_cache)
INT32 has_avx(cpuid_result_t cpuid_leaf1);
INT32 has_avx2(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_avx512f(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_avx512dq(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_avx512cd(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_avx512bw(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_avx512vl(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
INT32 has_fma(cpuid_result_t cpuid_leaf1);

// OS support checks
INT32 has_avx_support(cpuid_result_t cpuid_leaf1);
INT32 has_avx512_support(cpuid_result_t cpuid_leaf1);

#endif /* AOCLFFTZ_CPU_FEATURES_H */
