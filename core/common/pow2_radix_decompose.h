// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file pow2_radix_decompose.h
 *
 *  @brief Shared radix decomposition and kernel selection for power-of-two c2c
 *         solvers.
 *
 *  Every pow2 solver factors N = 2^k into the same radix stages from {2,4,8,16}
 *  and binds a kernel to each; only the layout the stages run over differs.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef POW2_RADIX_DECOMPOSE_H
#define POW2_RADIX_DECOMPOSE_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

// floor(log2(v)). MSVC has no __builtin_clzll, so fall back to a shift loop.
#if defined(__GNUC__) || defined(__clang__)
    #define LOG_BASE_2(n) (63 - __builtin_clzll((FFTZ_UINT64)(n)))
#else
    static inline FFTZ_INT32 fftz_log2_floor(FFTZ_UINT64 v)
    {
        FFTZ_INT32 r = 0;
        while (v >>= 1)
        {
            r++;
        }
        return r;
    }
    #define LOG_BASE_2(n) fftz_log2_floor((FFTZ_UINT64)(n))
#endif

// Balanced split of n = 2^k into near-square factors *out_n1 >= *out_n2.
// Returns 0, outputs untouched, if n is not a power of two or a factor is < 2.
FFTZ_INT32 pow2_balanced_split(FFTZ_INTP n, FFTZ_INTP *out_n1,
                               FFTZ_INTP *out_n2);

// Lowest-cost kernel entry for `radix`, ranked by find_best_kernel() at `batch`
// calls for `direction`. Both directions come from that one entry, so a node can
// be replayed either way (as Bluestein does) against a single twiddle layout.
//
// Returns 1 and fills the non-NULL outputs, or 0 if `radix` has no such entry:
// - out_kfft: the forward and backward kernels of the chosen entry;
// - out_kernel_cost: its estimated cost at `batch`;
// - out_kernel_sets: its register width, which the twiddle repack must match.
FFTZ_INT32 pow2_radix_pick_kernel(kernel_t *kertab, FFTZ_INTP radix,
                                  FFTZ_UINT8 precision, FFTZ_UINT8 direction,
                                  FFTZ_INTP batch, kfft_ *out_kfft,
                                  FFTZ_INT64 *out_kernel_cost,
                                  FFTZ_INTP *out_kernel_sets);

// Factor n = 2^k into at most POW2_MAX_DECOMP_STAGES radix stages (stage 0 from
// `kt_dft`, others `kt_twid`). Returns the stage count, largest-first, or 0.
FFTZ_INT32 pow2_radix_decompose(FFTZ_INTP n, kernel_t *kt_dft,
                                kernel_t *kt_twid, FFTZ_INTP *radixes);

#endif // POW2_RADIX_DECOMPOSE_H

