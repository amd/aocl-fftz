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

/** @file dispatcher.h
 *
 *  @brief Hardware capability dispatcher (feature-based).
 *
 *  Uses a single UINT64 feature bitmask to determine the SIMD optimization level
 *  supported by the current CPU + OS, taking into account compiled-in ISA support.
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
    optlevel_scalar     = 0,   /* No SIMD, portable baseline */
    optlevel_avx128     = 1,   /* 128-bit SIMD (AVX + OS support) */
    optlevel_avx256     = 2,   /* 256-bit SIMD (AVX + FMA + OS support); no FMA → max AVX128 */
    optlevel_avx512     = 3    /* 512-bit SIMD (AVX-512F + AVX-512DQ + OS support) */
} optimization_level_t;

/* ============================================================================
 * Feature bitmask (single source of truth for "what the CPU can use")
 * ============================================================================ */

/**
 * CPU feature bitmask: UINT64, one bit per feature (see FEATURE_* below).
 * Each bit represents availability of a specific CPU extension (with OS
 * support where applicable).
 */

/* Feature bit definitions - match capabilities used by FFTZ kernels */
#define FEATURE_AVX             ((UINT64)1ULL << 0)  /* AVX + OS AVX support */
#define FEATURE_FMA             ((UINT64)1ULL << 1)  /* FMA */
#define FEATURE_AVX2            ((UINT64)1ULL << 2)  /* AVX2 */
#define FEATURE_AVX512F         ((UINT64)1ULL << 3)  /* AVX-512 Foundation + OS */
#define FEATURE_AVX512DQ        ((UINT64)1ULL << 4)  /* AVX-512 DQ + OS */
#define FEATURE_AVX512VL        ((UINT64)1ULL << 5)  /* AVX-512 VL + OS */
#define FEATURE_AVX512CD        ((UINT64)1ULL << 6)  /* AVX-512 CD + OS */
#define FEATURE_AVX512BW        ((UINT64)1ULL << 7)  /* AVX-512 BW + OS */

/* Composite: minimum features required for each dispatch level (for checks) */
#define FEATURE_LEVEL_AVX128    (FEATURE_AVX)
#define FEATURE_LEVEL_AVX256    (FEATURE_AVX | FEATURE_FMA | FEATURE_AVX2)
#define FEATURE_LEVEL_AVX512    (FEATURE_AVX512F | FEATURE_AVX512DQ)

/* ============================================================================
 * Dispatcher API
 * ============================================================================ */

// Initialize dispatcher and detect hardware capabilities (writes feature bitmask to *cpu_capabilities).
VOID init_dynamic_dispatcher(UINT64 *cpu_capabilities);

// Compute build max ISA dispatch level from the feature bitmask
INT32 get_max_build_isa_level(UINT64 cpu_capabilities);

#endif /* AOCLFFTZ_DISPATCHER_H */
