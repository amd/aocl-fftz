// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
