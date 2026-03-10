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
#include "utils/dispatcher.h"

// Note: This is not thread safe, nor is it multiple instance safe.
//       When called through a multiple instance interface, the logger mode
//       provided in the last setup call will be used.
INT32 global_logger_mode = 0;

/**
 * Resolve effective dispatch level (clamped between user opt_level and HW+Build ISA level):
 * 1. init_dynamic_dispatcher() runs capability detection.
 * 2. effective_level = min(opt_level, get_max_build_isa_level()) so it stays between
 *    requested opt_level and hardware-supported level.
 *
 * @return: INT32 effective optimization level (optlevel_*).
 *         if opt_off or if opt_level <= 0, optlevel_scalar else effective_level.
 */
INT32 setup_dynamic_dispatcher(INT32 opt_off, INT32 opt_level)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (opt_off)
    {
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
        return optlevel_scalar;
    }
    if (opt_level <= 0)
    {
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
        return optlevel_scalar;
    }
    /* CPU feature bitmask (UINT64). */
    UINT64 cpu_capabilities = (UINT64)0;
    init_dynamic_dispatcher(&cpu_capabilities);
    INT32 hw_level = get_max_build_isa_level(cpu_capabilities); // Build max ISA dispatch level

    // Effective dispatch level is the minimum of the requested opt_level and the build max ISA dispatch level
    INT32 cpu_flags = (opt_level > hw_level) ? hw_level : opt_level;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return cpu_flags;
}

const CHAR* get_status_string(aoclfftz_error_type status)
{
    switch (status)
    {
        case AOCLFFTZ_TIME_OUT:
            return "AOCLFFTZ_TIME_OUT";
        case AOCLFFTZ_MEMORY_FAILURE:
            return "AOCLFFTZ_MEMORY_FAILURE";
        case AOCLFFTZ_INVALID_INPUT:
            return "AOCLFFTZ_INVALID_INPUT";
        case AOCLFFTZ_SETUP_FAILURE:
            return "AOCLFFTZ_SETUP_FAILURE";
        case AOCLFFTZ_EXECUTION_FAILURE:
            return "AOCLFFTZ_EXECUTION_FAILURE";
        case AOCLFFTZ_SUCCESS:
            return "AOCLFFTZ_SUCCESS";
        default:
            return "Unrecognized error code";
    }
}
