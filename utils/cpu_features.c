// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file cpu_features.c
 *
 *  @brief Implementation of low-level CPU feature detection utilities.
 *
 *  @author Prasandh Sankarankutty
 *  @author Partiksha
 *  @author Ashwin K. Godbole
 */

#include "utils/cpu_features.h"

/*
 * Platform-specific headers and mechanisms for CPUID and XGETBV.
 * - Windows: __cpuidex, _xgetbv from intrin.h
 * - Linux: inline asm("cpuid") and inline asm for xgetbv; cpuid.h included
 *   for consistency but CPUID is implemented via asm.
 */
#if defined(_WINDOWS)
 #include <intrin.h>
#else
 #include <cpuid.h>
#endif

cpuid_result_t cpuid(FFTZ_UINT32 leaf, FFTZ_UINT32 subleaf)
{
    cpuid_result_t r;

#if defined(_WINDOWS)
    FFTZ_INT32 regs[4];
    __cpuidex(regs, (FFTZ_INT32)leaf, (FFTZ_INT32)subleaf);
    r.eax = (FFTZ_UINT32)regs[0];
    r.ebx = (FFTZ_UINT32)regs[1];
    r.ecx = (FFTZ_UINT32)regs[2];
    r.edx = (FFTZ_UINT32)regs[3];
#else
    FFTZ_UINT32 eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(leaf), "c"(subleaf));
    r.eax = eax;
    r.ebx = ebx;
    r.ecx = ecx;
    r.edx = edx;
#endif
    return r;
}

FFTZ_VOID initialize_cpuid_cache(FFTZ_UINT32 *max_cpuid_leaf,
                            cpuid_result_t *cpuid_leaf1,
                            cpuid_result_t *cpuid_leaf7)
{
    cpuid_result_t leaf0 = cpuid(0u, 0u);
    *max_cpuid_leaf = leaf0.eax;

    if (*max_cpuid_leaf >= 1u)
    {
        *cpuid_leaf1 = cpuid(1u, 0u);
    }
    else
    {
        cpuid_leaf1->eax = cpuid_leaf1->ebx =
        cpuid_leaf1->ecx = cpuid_leaf1->edx = 0;
    }

    if (*max_cpuid_leaf >= 7u)
    {
        *cpuid_leaf7 = cpuid(7u, 0u);
    }
    else
    {
        cpuid_leaf7->eax = cpuid_leaf7->ebx =
        cpuid_leaf7->ecx = cpuid_leaf7->edx = 0;
    }
}

// XGETBV helpers

FFTZ_UINT64 get_xcr0(FFTZ_VOID)
{
#if defined(_WINDOWS)
    return _xgetbv(0);
#else
    FFTZ_UINT32 eax, edx;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
                         : "=a"(eax), "=d"(edx)
                         : "c"(0));
    return ((FFTZ_UINT64)edx << 32) | (FFTZ_UINT64)eax;
#endif
}

FFTZ_INT32 is_xgetbv_supported(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.OSXSAVE[bit 27]
    return (cpuid_leaf1.ecx & (1u << 27)) != 0;
}

/* ------------------------------------------------------------------------- */
/* CPU Feature Detection                                                     */
/* ------------------------------------------------------------------------- */

FFTZ_INT32 has_avx(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.AVX[bit 28]
    return (cpuid_leaf1.ecx & (1u << 28)) != 0;
}

FFTZ_INT32 has_avx2(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX2[bit 5]
    return (cpuid_leaf7.ebx & (1u << 5)) != 0;
}

FFTZ_INT32 has_avx512f(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512F[bit 16]
    return (cpuid_leaf7.ebx & (1u << 16)) != 0;
}

FFTZ_INT32 has_avx512dq(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512DQ[bit 17]
    return (cpuid_leaf7.ebx & (1u << 17)) != 0;
}

FFTZ_INT32 has_avx512cd(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512CD[bit 28]
    return (cpuid_leaf7.ebx & (1u << 28)) != 0;
}

FFTZ_INT32 has_avx512bw(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512BW[bit 30]
    return (cpuid_leaf7.ebx & (1u << 30)) != 0;
}

FFTZ_INT32 has_avx512vl(FFTZ_UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512VL[bit 31]
    return (cpuid_leaf7.ebx & (1u << 31)) != 0;
}

FFTZ_INT32 has_fma(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.FMA[bit 12]
    return (cpuid_leaf1.ecx & (1u << 12)) != 0;
}

/* ------------------------------------------------------------------------- */
/* OS support checks                                                         */
/* ------------------------------------------------------------------------- */

FFTZ_INT32 has_avx_support(cpuid_result_t cpuid_leaf1)
{
    if (!is_xgetbv_supported(cpuid_leaf1))
    {
        return 0;
    }

    // Check XCR0 bits: bit 1: XMM state; bit 2: YMM state
    FFTZ_UINT64 xcr0 = get_xcr0();
    const FFTZ_UINT64 mask = (1uLL << 1) | (1uLL << 2);

    return (xcr0 & mask) == mask;
}

FFTZ_INT32 has_avx512_support(cpuid_result_t cpuid_leaf1)
{
    if (!is_xgetbv_supported(cpuid_leaf1))
    {
        return 0;
    }

    /* For AVX-512, OS must support:
     * - XMM (bit 1)
     * - YMM (bit 2)
     * - Opmask (bit 5)
     * - ZMM_Hi256 (bit 6)
     * - Hi16_ZMM (bit 7)
     */
    FFTZ_UINT64 xcr0 = get_xcr0();
    const FFTZ_UINT64 mask = (1uLL << 1) | (1uLL << 2) |
                        (1uLL << 5) | (1uLL << 6) | (1uLL << 7);

    return (xcr0 & mask) == mask;
}

static FFTZ_INT32 is_vendor_amd(FFTZ_VOID)
{
    /* CPUID.0:EBX/EDX/ECX spell "AuthenticAMD". */
    cpuid_result_t v = cpuid(0u, 0u);
    return v.ebx == 0x68747541u  /* "Auth" */
        && v.edx == 0x69746e65u  /* "enti" */
        && v.ecx == 0x444d4163u; /* "cAMD" */
}

FFTZ_UINTP cpuid_cache_size(FFTZ_UINT32 cache_level)
{
    if (cache_level == 0u)
    {
        return 0u;
    }

    // Select the "deterministic cache parameters" leaf.
    // - Intel exposes it at leaf 0x4
    // - AMD (with topology extensions) mirrors the same encoding at leaf 0x8000001D
    // Both are sub-leaf indexed and share the EAX/EBX/ECX field layout.
    FFTZ_UINT32 leaf;
    if (is_vendor_amd())
    {
        FFTZ_UINT32 max_ext = cpuid(0x80000000u, 0u).eax;
        if (max_ext < 0x8000001Du)
        {
            return 0u;
        }
        leaf = 0x8000001Du;
    }
    else
    {
        FFTZ_UINT32 max_std = cpuid(0u, 0u).eax;
        if (max_std < 4u)
        {
            return 0u;
        }
        leaf = 4u;
    }

    // Walk the cache hierarchy sub-leaves until the terminator (type 0).
    // The 32 bound is only a guard against a malformed leaf.
    for (FFTZ_UINT32 sub = 0u; sub < 32u; sub++)
    {
        cpuid_result_t r = cpuid(leaf, sub);

        FFTZ_UINT32 cache_type = r.eax & 0x1Fu; // 0=none 1=data 2=inst 3=unified
        if (cache_type == 0u)
        {
            break;
        }
        FFTZ_UINT32 level = (r.eax >> 5) & 0x7u;

        // Match the requested level and the expected cache type for it:
        // - For L1 we want the data cache (type 1);
        // - For L2/L3 the unified cache (type 3).
        FFTZ_UINT32 want_type = (cache_level == 1u) ? 1u : 3u;
        if (level != cache_level || cache_type != want_type)
        {
            continue;
        }

        FFTZ_UINT32 line_size  = (r.ebx & 0xFFFu) + 1u;          /* bits 11:0   */
        FFTZ_UINT32 partitions = ((r.ebx >> 12) & 0x3FFu) + 1u;  /* bits 21:12  */
        FFTZ_UINT32 ways       = ((r.ebx >> 22) & 0x3FFu) + 1u;  /* bits 31:22  */
        FFTZ_UINT32 sets       = r.ecx + 1u;

        return (FFTZ_UINTP)line_size * (FFTZ_UINTP)partitions *
               (FFTZ_UINTP)ways * (FFTZ_UINTP)sets;
    }

    return 0u;
}
