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
    FFTZ_UINT32 eax;
    FFTZ_UINT32 ebx;
    FFTZ_UINT32 ecx;
    FFTZ_UINT32 edx;
} cpuid_result_t;

// Fill CPUID cache (max leaf, leaf 1, leaf 7); call once then pass to feature
// APIs
FFTZ_VOID initialize_cpuid_cache(FFTZ_UINT32 *max_cpuid_leaf,
                                 cpuid_result_t *cpuid_leaf1,
                                 cpuid_result_t *cpuid_leaf7);

// CPUID helpers
cpuid_result_t cpuid(FFTZ_UINT32 leaf, FFTZ_UINT32 subleaf);

// Total size in bytes of the cache at the given level (1=L1d, 2=L2, 3=L3).
// Returns 0 when the level is unavailable or CPUID cannot report it.
FFTZ_UINTP cpuid_cache_size(FFTZ_UINT32 cache_level);

// XGETBV / XCR0 helpers
FFTZ_UINT64 get_xcr0(FFTZ_VOID);
FFTZ_INT32 is_xgetbv_supported(cpuid_result_t cpuid_leaf1);

// CPU Feature detection (pass cached cpuid_leaf1/cpuid_leaf7 from
// initialize_cpuid_cache)
FFTZ_INT32 has_avx(cpuid_result_t cpuid_leaf1);
FFTZ_INT32 has_avx2(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_avx512f(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_avx512dq(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_avx512cd(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_avx512bw(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_avx512vl(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7);
FFTZ_INT32 has_fma(cpuid_result_t cpuid_leaf1);

// OS support checks
FFTZ_INT32 has_avx_support(cpuid_result_t cpuid_leaf1);
FFTZ_INT32 has_avx512_support(cpuid_result_t cpuid_leaf1);

#endif /* AOCLFFTZ_CPU_FEATURES_H */
