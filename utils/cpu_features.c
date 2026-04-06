// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file cpu_features.c
 *
 *  @brief Implementation of low-level CPU feature detection utilities.
 *
 *  @author Prasandh Sankarankutty
 *  @author Partiksha
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

cpuid_result_t cpuid(UINT32 leaf, UINT32 subleaf)
{
    cpuid_result_t r;

#if defined(_WINDOWS)
    INT32 regs[4];
    __cpuidex(regs, (INT32)leaf, (INT32)subleaf);
    r.eax = (UINT32)regs[0];
    r.ebx = (UINT32)regs[1];
    r.ecx = (UINT32)regs[2];
    r.edx = (UINT32)regs[3];
#else
    UINT32 eax, ebx, ecx, edx;
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

VOID initialize_cpuid_cache(UINT32 *max_cpuid_leaf,
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

UINT64 get_xcr0(VOID)
{
#if defined(_WINDOWS)
    return _xgetbv(0);
#else
    UINT32 eax, edx;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
                         : "=a"(eax), "=d"(edx)
                         : "c"(0));
    return ((UINT64)edx << 32) | (UINT64)eax;
#endif
}

INT32 is_xgetbv_supported(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.OSXSAVE[bit 27]
    return (cpuid_leaf1.ecx & (1u << 27)) != 0;
}

/* ------------------------------------------------------------------------- */
/* CPU Feature Detection                                                     */
/* ------------------------------------------------------------------------- */

INT32 has_avx(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.AVX[bit 28]
    return (cpuid_leaf1.ecx & (1u << 28)) != 0;
}

INT32 has_avx2(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX2[bit 5]
    return (cpuid_leaf7.ebx & (1u << 5)) != 0;
}

INT32 has_avx512f(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512F[bit 16]
    return (cpuid_leaf7.ebx & (1u << 16)) != 0;
}

INT32 has_avx512dq(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512DQ[bit 17]
    return (cpuid_leaf7.ebx & (1u << 17)) != 0;
}

INT32 has_avx512cd(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512CD[bit 28]
    return (cpuid_leaf7.ebx & (1u << 28)) != 0;
}

INT32 has_avx512bw(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512BW[bit 30]
    return (cpuid_leaf7.ebx & (1u << 30)) != 0;
}

INT32 has_avx512vl(UINT32 max_cpuid_leaf, cpuid_result_t cpuid_leaf7)
{
    if (max_cpuid_leaf < 7u)
    {
        return 0;
    }
    // CPUID.(EAX=7,ECX=0):EBX.AVX-512VL[bit 31]
    return (cpuid_leaf7.ebx & (1u << 31)) != 0;
}

INT32 has_fma(cpuid_result_t cpuid_leaf1)
{
    // CPUID.(EAX=1):ECX.FMA[bit 12]
    return (cpuid_leaf1.ecx & (1u << 12)) != 0;
}

/* ------------------------------------------------------------------------- */
/* OS support checks                                                         */
/* ------------------------------------------------------------------------- */

INT32 has_avx_support(cpuid_result_t cpuid_leaf1)
{
    if (!is_xgetbv_supported(cpuid_leaf1))
    {
        return 0;
    }

    // Check XCR0 bits: bit 1: XMM state; bit 2: YMM state
    UINT64 xcr0 = get_xcr0();
    const UINT64 mask = (1uLL << 1) | (1uLL << 2);

    return (xcr0 & mask) == mask;
}

INT32 has_avx512_support(cpuid_result_t cpuid_leaf1)
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
    UINT64 xcr0 = get_xcr0();
    const UINT64 mask = (1uLL << 1) | (1uLL << 2) |
                        (1uLL << 5) | (1uLL << 6) | (1uLL << 7);

    return (xcr0 & mask) == mask;
}
