/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file utils.c
 *
 *  @brief Utility functions that are used by library framework and methods.
 *
 *  This file contains the utility functions to provide functionalities like
 *  CPU feature detection, logger, timer and others.
 *
 *  @author S. Biplab Raut
 */

#include "utils/utils.h"

#define AVX512_SUPPORTED 3
#define AVX256_SUPPORTED 2
#define AVX128_SUPPORTED 1
#define C_SUPPORTED 0
#define UNSUPPORTED -1

INTP is_AVX_supported(INT32 logger_mode)
{
    INTP ret;
    INTP eax, ebx, ecx, edx;
    cpu_features_detection(0x00000001, 0, &eax, &ebx, &ecx, &edx);
    ret = ((ecx & 0x18000000) == 0x18000000);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "AVX SIMD %s supported",
                           (ret ? "is" : "is not"));
#endif
    return ret;
}

INTP is_AVX2_supported(INT32 logger_mode)
{
    INTP ret;
    INTP eax, ebx, ecx, edx;
    cpu_features_detection(0x00000007, 0, &eax, &ebx, &ecx, &edx);
    ret = ((ebx & (1 << 5)) != 0);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "AVX2 SIMD %s supported",
                           (ret ? "is" : "is not"));
#endif
    return ret;
}

static inline INTP xgetbv(INTP opt)
{
    INT32 eax, edx;
    __asm__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(opt));
    return eax;
}

INTP is_AVX512_supported(INT32 logger_mode)
{
    INTP ret = 0;
    INTP eax, ebx, ecx, edx;
    // Below is the set of checks for AVX512 detection
    // 1. Check CPU support for ZMM state management using OSXSAVE
    // Its support also implies that XGETBV is enabled for application use
    cpu_features_detection(0x1, 0, &eax, &ebx, &ecx, &edx);
    if ((ecx & 0x08000000) == 0x08000000)
    {
        // 2. Check OS support for XGETBV instruction and ZMM register state
        INTP reg_support_bits = (7 << 5) | (1 << 2) | (1 << 1);
        if ((xgetbv(0) & reg_support_bits) == reg_support_bits)
        {
            // 3. Check CPU support for AVX-512 Foundation instructions
            cpu_features_detection(7, 0, &eax, &ebx, &ecx, &edx);
            if (ebx & (1 << 16))
            {
                ret = 1;
            }
        }
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode,
                           "AVX512 SIMD %s supported", (ret ? "is" : "is not"));
#endif
    return ret;
}

INTP is_FMA_supported(INT32 logger_mode)

{
    INTP ret;
    INTP eax, ebx, ecx, edx;
    cpu_features_detection(0x00000001, 0, &eax, &ebx, &ecx, &edx);
    ret = ((ecx & (1 << 12)) != 0);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "FMA %s supported",
                           (ret ? "is" : "is not"));
#endif
    return ret;
}

// CPU Features detection using CPUID
#ifdef AOCLFFTZ_CPUID_SIMD_DETECTION
#ifndef _WINDOWS
inline VOID cpu_features_detection(INTP fn, INTP optVal,
                                   INTP *eax, INTP *ebx,
                                   INTP *ecx, INTP *edx)
{
    *eax = fn;
    *ecx = optVal;
    *ebx = 0;
    *edx = 0;
    __asm__("cpuid            \n\t"
            : "+a"(*eax), "+b"(*ebx), "+c"(*ecx), "+d"(*edx));
}
#else
#include <intrin.h>
inline VOID cpu_features_detection(INTP fn, INTP optVal,
                                   INTP *eax, INTP *ebx,
                                   INTP *ecx, INTP *edx)
{
    INT32 CPUInfo[4];

    __cpuid(CPUInfo, fn);

    *eax = CPUInfo[0];
    *ebx = CPUInfo[1];
    *ecx = CPUInfo[2];
    *edx = CPUInfo[3];
}
#endif
#endif

EXPORT_SYM_DYN INT32 setup_dynamic_dispatcher(INT32 opt_off, INT32 opt_level,
                               INT32 logger_mode)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    if (opt_off)
    {
        return UNSUPPORTED;
    }

    if (opt_level == 0)
    {
        return C_SUPPORTED;
    }
#ifdef ENABLE_AVX512
    if (opt_level > 2 && is_AVX512_supported(logger_mode)) // opt_level == 3
    {
        return AVX512_SUPPORTED;
    }
#endif
#ifdef ENABLE_AVX256
    if (opt_level > 1 && is_AVX_supported(logger_mode) &&
        is_FMA_supported(logger_mode) == 0) // opt_level == 2
    {
        return AVX256_SUPPORTED;
    }
#endif
#ifdef ENABLE_AVX128
    if (opt_level > 0 && is_AVX_supported(logger_mode)) // opt_level == 1
    {
        return AVX128_SUPPORTED;
    }
#endif
    return C_SUPPORTED;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
}
