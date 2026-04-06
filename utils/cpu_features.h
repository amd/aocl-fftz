// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
