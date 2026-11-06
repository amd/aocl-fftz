// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file dispatcher.h
 *
 *  @brief Hardware capability dispatcher (feature-based).
 *
 *  Uses a single FFTZ_UINT64 feature bitmask to determine the SIMD optimization
 * level supported by the current CPU + OS, taking into account compiled-in ISA
 * support.
 *
 *  @author Prasandh Sankarankutty
 *  @author Partiksha
 */

#ifndef AOCLFFTZ_DISPATCHER_H
#define AOCLFFTZ_DISPATCHER_H

#include "api/types.h"

/* ================================
 * Optimization level (SIMD tier)
 * ================================ */

/**
 * Optimization Level
 *
 * Hierarchical: SCALAR < AVX128 < AVX256 < AVX512
 * Each level represents a tier of available SIMD functionality.
 * Higher levels are strict supersets of lower levels.
 */
typedef enum
{
    // No SIMD, portable baseline
    optlevel_scalar = 0,
    // 128-bit SIMD (AVX + OS support, FMA when built with ENABLE_FMA)
    optlevel_avx128 = 1,
    // 256-bit SIMD (AVX + FMA + OS support); no FMA → max AVX128
    optlevel_avx256 = 2,
    // 512-bit SIMD (AVX-512F + AVX-512DQ + OS support)
    optlevel_avx512 = 3
} optimization_level_t;

/* ============================================================================
 * Feature bitmask (single source of truth for "what the CPU can use")
 * ============================================================================
 */

/**
 * CPU feature bitmask: FFTZ_UINT64, one bit per feature (see FEATURE_* below).
 * Each bit represents availability of a specific CPU extension (with OS
 * support where applicable).
 */

/* Feature bit definitions - match capabilities used by FFTZ kernels */
#define FEATURE_AVX ((FFTZ_UINT64)1ULL << 0)  /* AVX + OS AVX support */
#define FEATURE_FMA             ((FFTZ_UINT64)1ULL << 1)  /* FMA */
#define FEATURE_AVX2            ((FFTZ_UINT64)1ULL << 2)  /* AVX2 */
#define FEATURE_AVX512F ((FFTZ_UINT64)1ULL << 3) /* AVX-512 Foundation + OS */
#define FEATURE_AVX512DQ        ((FFTZ_UINT64)1ULL << 4)  /* AVX-512 DQ + OS */
#define FEATURE_AVX512VL        ((FFTZ_UINT64)1ULL << 5)  /* AVX-512 VL + OS */
#define FEATURE_AVX512CD        ((FFTZ_UINT64)1ULL << 6)  /* AVX-512 CD + OS */
#define FEATURE_AVX512BW        ((FFTZ_UINT64)1ULL << 7)  /* AVX-512 BW + OS */

/* Composite: minimum features required for each dispatch level (for checks) */
// An ENABLE_FMA build compiles the 128-bit sources with -mfma, so the tier needs
// FMA at run time too; without it those kernels would fault on an AVX-only CPU.
// The condition must stay in step with the one guarding FMA emission in
// core/kernels/simd_includes/simd_common.h: Windows builds AVX128 under
// /arch:AVX and emits no FMA there, so requiring it would only keep AVX-only
// CPUs off this tier for nothing.
#if defined(ENABLE_FMA) && !defined(_WIN32)
#define FEATURE_LEVEL_AVX128    (FEATURE_AVX | FEATURE_FMA)
#else
#define FEATURE_LEVEL_AVX128    (FEATURE_AVX)
#endif
#define FEATURE_LEVEL_AVX256    (FEATURE_AVX | FEATURE_FMA | FEATURE_AVX2)
#define FEATURE_LEVEL_AVX512    (FEATURE_AVX512F | FEATURE_AVX512DQ)

/* ============================================================================
 * Dispatcher API
 * ============================================================================
 */

// Initialize dispatcher and detect hardware capabilities (writes feature
// bitmask to *cpu_capabilities).
FFTZ_VOID init_dynamic_dispatcher(FFTZ_UINT64 *cpu_capabilities);

// Compute build max ISA dispatch level from the feature bitmask
FFTZ_INT32 get_max_build_isa_level(FFTZ_UINT64 cpu_capabilities);

#endif /* AOCLFFTZ_DISPATCHER_H */
