// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose.h
 *
 *  @brief Shared tiling geometry for the fused four-step twiddle + transpose
 *         kernels.
 *
 *  The fused kernel walks the n1 x n2 source in CACHE_BLOCK blocks of
 *  MICRO_TILE micro-tiles. The planner stores the inter-step twiddle table in
 *  that same blocked order so the kernel reads it sequentially (a single
 *  advancing pointer over MICRO_TILE x MICRO_TILE contiguous complex per tile),
 *  avoiding the strided tw[k1*n2 + j] access that wrecks prefetch/TLB at
 *  large N. The layout is only valid when n1 and n2 are multiples of MICRO_TILE
 *  (so every micro-tile is full), which the planner gates on.
 *
 *  CACHE_BLOCK / MICRO_TILE are square-tile edge lengths in complex elements
 *  and MUST match the per-ISA kernels; both they and the planner include this
 *  header so the two cannot drift.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef FUSED_TWIDDLE_TRANSPOSE_H
#define FUSED_TWIDDLE_TRANSPOSE_H

// Every ISA variant uses the cache-block edges below; they were measured on
// AVX-512. A block reads CACHE_BLOCK complex per source row before jumping a
// row stride, so wider blocks amortise the strided walk over more contiguous
// bytes, but CACHE_BLOCK also sets the count of open destination write streams.
// Past these values the write streams cost more than the source side recovers.

// fp64 (complex double): 4x4 register micro-tile inside a 16x16 cache block
#define FUSED_TWIDDLE_TRANSPOSE_FP64_CACHE_BLOCK 16
#define FUSED_TWIDDLE_TRANSPOSE_FP64_MICRO_TILE 4

// fp32 (complex float): 8x8 register micro-tile inside a 64x64 cache block
#define FUSED_TWIDDLE_TRANSPOSE_FP32_CACHE_BLOCK 64
#define FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE 8

#endif // FUSED_TWIDDLE_TRANSPOSE_H

