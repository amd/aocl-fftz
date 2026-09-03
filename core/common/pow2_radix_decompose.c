// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file pow2_radix_decompose.c
 *
 *  @brief Shared radix decomposition and kernel selection for power-of-two c2c
 *         solvers (see pow2_radix_decompose.h).
 *
 *  @author Ashwin K. Godbole
 */

#include "core/common/pow2_radix_decompose.h"
#include "core/solvers/solver.h"

FFTZ_INT32 pow2_balanced_split(FFTZ_INTP n, FFTZ_INTP *out_n1,
                               FFTZ_INTP *out_n2)
{
    if (n < 2 || !IS_POW2(n))
    {
        return 0;
    }

    // n = 2^k splits into 2^ceil(k/2) * 2^floor(k/2), the larger factor first.
    FFTZ_INT32 k = LOG_BASE_2(n);
    FFTZ_INTP n2 = (FFTZ_INTP)1 << (k / 2);
    FFTZ_INTP n1 = n / n2;
    if (n1 < 2 || n2 < 2)
    {
        return 0;
    }

    *out_n1 = n1;
    *out_n2 = n2;
    return 1;
}

FFTZ_INT32 pow2_radix_pick_kernel(kernel_t *kertab, FFTZ_INTP radix,
                                  FFTZ_UINT8 precision, FFTZ_UINT8 direction,
                                  FFTZ_INTP batch, kfft_ *out_kfft,
                                  FFTZ_INT64 *out_kernel_cost,
                                  FFTZ_INTP *out_kernel_sets)
{
    if (out_kernel_cost != NULL)
    {
        *out_kernel_cost = 0;
    }
    if (out_kernel_sets != NULL)
    {
        *out_kernel_sets = 1;
    }

    FFTZ_INTP base_idx = find_radix_base_idx(kertab, radix);
    if (base_idx < 0)
    {
        return 0;
    }

    kernel_choice_t best = find_best_kernel(kertab, base_idx, precision,
                                            direction, batch);
    if (best.idx < 0)
    {
        return 0;
    }

    if (kertab[best.idx].kfft[FORWARD_FFT_DIR] == NULL
        || kertab[best.idx].kfft[BACKWARD_FFT_DIR] == NULL)
    {
        return 0;
    }

    if (out_kfft != NULL)
    {
        out_kfft[FORWARD_FFT_DIR] = kertab[best.idx].kfft[FORWARD_FFT_DIR];
        out_kfft[BACKWARD_FFT_DIR] = kertab[best.idx].kfft[BACKWARD_FFT_DIR];
    }
    if (out_kernel_cost != NULL)
    {
        *out_kernel_cost = best.cost;
    }
    if (out_kernel_sets != NULL)
    {
        *out_kernel_sets = (FFTZ_INTP)kertab[best.idx].sets[precision - 2];
    }
    return 1;
}

// Whether any ISA category has a radix-`radix` entry usable in both directions.
// Defers to the picker; precision and batch only affect the discarded cost.
static FFTZ_INT32 pow2_radix_supported(kernel_t *kertab, FFTZ_INTP radix)
{
    return pow2_radix_pick_kernel(kertab, radix, DT_DOUBLE, FORWARD_FFT_DIR, 1,
                                  NULL, NULL, NULL);
}

// Greedy fallback: largest radix dividing `remaining` that has a supported
// kernel, or 0 if none.
static FFTZ_INTP pow2_radix_pick_largest(kernel_t *kertab, FFTZ_INTP remaining)
{
    FFTZ_INTP best_radix = 0;
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_INTP radix = (FFTZ_INTP)kertab[i].radix;
        if (radix == 0) // end of the kernel list
        {
            break;
        }
        if (radix <= best_radix || radix > remaining
            || (remaining % radix) != 0)
        {
            continue;
        }
        if (pow2_radix_supported(kertab, radix))
        {
            best_radix = radix;
        }
    }
    return best_radix;
}

// Factor n = 2^k into the fewest radix stages from {2,4,8,16}, spread evenly
// (512 -> 8x8x8). Emits largest-first; returns the stage count, or 0.
static FFTZ_INT32 pow2_radix_balanced(FFTZ_INTP n, FFTZ_INTP *radixes)
{
    if (n < 2 || !IS_POW2(n))
    {
        return 0; // must be a power of two >= 2
    }
    FFTZ_INT32 log2n = LOG_BASE_2(n);

    // ceil(log2n / 4): fewest stages whose per-stage radix stays <= 16.
    FFTZ_INT32 num_stages = (log2n + 3) / 4;
    if (num_stages < 1)
    {
        num_stages = 1;
    }
    if (num_stages > POW2_MAX_DECOMP_STAGES)
    {
        return 0;
    }

    FFTZ_INT32 base_exponent = log2n / num_stages;
    // The first `extra_stages` stages get one more exponent unit each.
    FFTZ_INT32 extra_stages = log2n % num_stages;
    for (FFTZ_INT32 stage = 0; stage < num_stages; stage++)
    {
        FFTZ_INT32 exponent = base_exponent + ((stage < extra_stages) ? 1 : 0);
        radixes[stage] = (FFTZ_INTP)1 << exponent;
    }
    return num_stages;
}

FFTZ_INT32 pow2_radix_decompose(FFTZ_INTP n, kernel_t *kt_dft,
                                kernel_t *kt_twid, FFTZ_INTP *radixes)
{
    FFTZ_INTP balanced_radixes[POW2_MAX_DECOMP_STAGES];
    FFTZ_INT32 balanced_stages = pow2_radix_balanced(n, balanced_radixes);
    if (balanced_stages > 0)
    {
        FFTZ_INT32 all_supported = 1;
        for (FFTZ_INT32 stage = 0; stage < balanced_stages; stage++)
        {
            kernel_t *kt = (stage == 0) ? kt_dft : kt_twid;
            radixes[stage] = balanced_radixes[stage];
            if (!pow2_radix_supported(kt, balanced_radixes[stage]))
            {
                all_supported = 0;
                break;
            }
        }
        if (all_supported)
        {
            return balanced_stages;
        }
    }

    FFTZ_INTP remaining = n;
    FFTZ_INTP leaf_radix = pow2_radix_pick_largest(kt_dft, remaining);
    if (leaf_radix == 0)
    {
        return 0;
    }
    radixes[0] = leaf_radix;
    remaining /= leaf_radix;
    FFTZ_INT32 num_stages = 1;

    while (remaining > 1 && num_stages < POW2_MAX_DECOMP_STAGES)
    {
        FFTZ_INTP radix = pow2_radix_pick_largest(kt_twid, remaining);
        if (radix == 0)
        {
            return 0;
        }
        radixes[num_stages] = radix;
        remaining /= radix;
        num_stages++;
    }
    return (remaining == 1) ? num_stages : 0;
}

