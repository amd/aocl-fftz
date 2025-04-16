/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file aoclfftz_core_wrapper.h
 *
 *  @brief Contains wrapper function declarations for core funtions
 *  with dllexport attribute.
 *
 *  This file contains the wrapper function declarations for core functions
 *  with `__declspec(dllexport)` attribute for Windows compatibility.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_CORE_WRAPPER_H
#define AOCLFFTZ_CORE_WRAPPER_H

#include "core/common/memory_manager.h"
#include "core/common/strides.h"
#include "core/kernels/kernel.h"
#include "core/solvers/solver.h"
#include "selector/selector.h"
#include "core/kernels/transpose/transpose_utils.h"
#include "core/kernels/transpose/transpose_kernels.h"

// Re-delcaring this struct to avoid using core/kernels/kernel_list.h file
typedef struct wrapper_kernel_fp_list
{
    k_register_kernel_ k_register_kernel;
    k_ops_cnt_ k_ops_cnt;
    UINT32 radix;
} wrapper_kernel_fp_list_t;

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

// C2C Kernels
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft2c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft3c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft4c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft5c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft6c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft7c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft8c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft9c_wrapper(UINT8 precision,
                                                      UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft10c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft11c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft12c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft13c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft14c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft15c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft16c_wrapper(UINT8 precision,
                                                       UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft2avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft3avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft4avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft5avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft6avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft7avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft8avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft9avx128_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft10avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft11avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft12avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft13avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft14avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft15avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft16avx128_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft2avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft3avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft4avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft5avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft6avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft7avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft8avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft9avx256_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft10avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft11avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft12avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft13avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft14avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft15avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft16avx256_wrapper(UINT8 precision,
                                                            UINT8 direction);
// AVX512
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft2avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft3avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft4avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft5avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft6avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft7avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft8avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft9avx512_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft10avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft11avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft12avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft13avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft14avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft15avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft16avx512_wrapper(UINT8 precision,
                                                            UINT8 direction);

// R2HC Kernels
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft2c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft3c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft4c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft5c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft6c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft7c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft8c_wrapper(UINT8 precision,
                                                            UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft10c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft12c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft14c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft15c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hc_rfft16c_wrapper(UINT8 precision,
                                                             UINT8 direction);

// R2HC-Fused Kernels
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft2c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft3c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft4c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft5c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft6c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft7c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft8c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft10c_wrapper(UINT8 precision,
                                                              UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft12c_wrapper(UINT8 precision,
                                                             UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft14c_wrapper(UINT8 precision,
                                                              UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft15c_wrapper(UINT8 precision,
                                                              UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_r2hcf_rfft16c_wrapper(UINT8 precision,
                                                              UINT8 direction);

#ifdef ENABLE_AVX128
// R2HC AVX128 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft2avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft3avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft4avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft5avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft6avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft7avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft8avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft10avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft14avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft15avx128_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX128 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft2avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft7avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft10avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft14avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft15avx128_wrapper(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
// R2HC AVX256 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft2avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft3avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft4avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft5avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft6avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft7avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft8avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft10avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft14avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft15avx256_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX256 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft2avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft10avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft14avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft15avx256_wrapper(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX512
// R2HC AVX512 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft2avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft3avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft4avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft5avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft6avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft7avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft8avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft10avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft14avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hc_rfft15avx512_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX512 Kernels
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft2avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft10avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft14avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN ops_cycles_t
get_ops_cnt_r2hcf_rfft15avx512_wrapper(UINT8 precision, UINT8 direction);
#endif
/* ---------------- kernels : register_kernel_fft* ---------------- */

// C2C Kernels
EXPORT_SYM_DYN kfft_ register_kernel_fft2c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft3c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft4c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft5c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft6c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft7c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft8c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft9c_wrapper(UINT8 precision,
                                                   UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft10c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft11c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft12c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft13c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft14c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft15c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft16c_wrapper(UINT8 precision,
                                                    UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft2avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft3avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft4avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft5avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft6avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft7avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft8avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft9avx128_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft10avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft11avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft12avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft13avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft14avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft15avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft16avx128_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft2avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft3avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft4avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft5avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft6avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft7avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft8avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft9avx256_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft10avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft11avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft12avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft13avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft14avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft15avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft16avx256_wrapper(UINT8 precision,
                                                         UINT8 direction);
// R2HC Kernels
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft2c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft3c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft4c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft5c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft6c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft7c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft8c_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft10c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft12c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft14c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft15c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hc_rfft16c_wrapper(UINT8 precision,
                                                          UINT8 direction);
// R2HC-Fused Kernels
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft2c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft3c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft4c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft5c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft6c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft7c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft8c_wrapper(UINT8 precision,
                                                          UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft10c_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft12c_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft14c_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft15c_wrapper(UINT8 precision,
                                                           UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_r2hcf_rfft16c_wrapper(UINT8 precision,
                                                           UINT8 direction);
// AVX512
EXPORT_SYM_DYN kfft_ register_kernel_fft2avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft3avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft4avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft5avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft6avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft7avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft8avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft9avx512_wrapper(UINT8 precision,
                                                        UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft10avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft11avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft12avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft13avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft14avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft15avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);
EXPORT_SYM_DYN kfft_ register_kernel_fft16avx512_wrapper(UINT8 precision,
                                                         UINT8 direction);

#ifdef ENABLE_AVX128
// R2HC AVX128 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft2avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft3avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft4avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft5avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft6avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft7avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft8avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft10avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft14avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft15avx128_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX128 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft2avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft7avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft10avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft14avx128_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft15avx128_wrapper(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
// R2HC AVX256 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft2avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft3avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft4avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft5avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft6avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft7avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft8avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft10avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft14avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft15avx256_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX256 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft2avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft10avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft14avx256_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft15avx256_wrapper(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX512
// R2HC AVX512 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft2avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft3avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft4avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft5avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft6avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft7avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft8avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft10avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft14avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hc_rfft15avx512_wrapper(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX512 Kernels
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft2avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft10avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft14avx512_wrapper(UINT8 precision, UINT8 direction);
EXPORT_SYM_DYN kfft_
register_kernel_r2hcf_rfft15avx512_wrapper(UINT8 precision, UINT8 direction);
#endif
/* ---------------- kernels : permuted_copy_* ---------------- */

EXPORT_SYM_DYN VOID permuted_copy_c_fp32_wrapper(VOID *in, VOID *out, INTP n,
                                                 INTP radix,
                                                 aoclfftz_strides_t *strides,
                                                 UINT8 data_stride);
EXPORT_SYM_DYN VOID permuted_copy_c_fp64_wrapper(VOID *in, VOID *out, INTP n,
                                                 INTP radix,
                                                 aoclfftz_strides_t *strides,
                                                 UINT8 data_stride);

/* ---------------- memory allocators/destroys ---------------- */

EXPORT_SYM_DYN
aoclfftz_decomp_scheme_t *alloc_decomp_scheme_wrapper(INT32 vec_rank,
                                                      INT32 dim_rank);
EXPORT_SYM_DYN aoclfftz_solution_t *alloc_solution_wrapper(INT32 vec_rank,
                                                           INT32 dim_rank);
EXPORT_SYM_DYN aoclfftz_selector_t *alloc_selector_wrapper(INT32 vec_rank,
                                                           INT32 dim_rank,
                                                           VOID *scratch_space);
EXPORT_SYM_DYN VOID *alloc_twiddle_for_solution_wrapper(UINT8 rad_size,
                                                        UINT8 dt_prec);
EXPORT_SYM_DYN VOID destroy_selector_wrapper(aoclfftz_selector_t *sel);
EXPORT_SYM_DYN VOID destroy_solution_wrapper(aoclfftz_solution_t *sol);
EXPORT_SYM_DYN
VOID destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme);
EXPORT_SYM_DYN VOID destroy_handle_wrapper(VOID *handle);

/* ---------------- fuse vector wrapper ---------------- */
EXPORT_SYM_DYN VOID fuse_vecs_wrapper(aoclfftz_solution_t *sol);

/* ---------------- strides wrapper ---------------- */
EXPORT_SYM_DYN VOID populate_stride_array_wrapper(INTP *strides,
                      INTP stride_val,
                      INTP n,
                      UINT8 compute_half_complex,
                      UINT8 adjust_to_full_complex);

/* ---------------- fused strides wrapper ---------------- */
EXPORT_SYM_DYN VOID prepare_fused_kernel_strides_wrapper(INTP *strides,
                                                         INTP radix,
                                                         INTP offset);

/* ---------------- wrapper kernel tables ---------------- */

// C2C Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_c[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2c_wrapper, get_ops_cnt_fft2c_wrapper, 2},
    {register_kernel_fft3c_wrapper, get_ops_cnt_fft3c_wrapper, 3},
    {register_kernel_fft4c_wrapper, get_ops_cnt_fft4c_wrapper, 4},
    {register_kernel_fft5c_wrapper, get_ops_cnt_fft5c_wrapper, 5},
    {register_kernel_fft6c_wrapper, get_ops_cnt_fft6c_wrapper, 6},
    {register_kernel_fft7c_wrapper, get_ops_cnt_fft7c_wrapper, 7},
    {register_kernel_fft8c_wrapper, get_ops_cnt_fft8c_wrapper, 8},
    {register_kernel_fft9c_wrapper, get_ops_cnt_fft9c_wrapper, 9},
    {register_kernel_fft10c_wrapper, get_ops_cnt_fft10c_wrapper, 10},
    {register_kernel_fft11c_wrapper, get_ops_cnt_fft11c_wrapper, 11},
    {register_kernel_fft12c_wrapper, get_ops_cnt_fft12c_wrapper, 12},
    {register_kernel_fft13c_wrapper, get_ops_cnt_fft13c_wrapper, 13},
    {register_kernel_fft14c_wrapper, get_ops_cnt_fft14c_wrapper, 14},
    {register_kernel_fft15c_wrapper, get_ops_cnt_fft15c_wrapper, 15},
    {register_kernel_fft16c_wrapper, get_ops_cnt_fft16c_wrapper, 16},
    {NULL, NULL, 20},
    {NULL, NULL, 25},
    {NULL, NULL, 32},
    {NULL, NULL, 64}
};
#ifdef ENABLE_AVX128
static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx128[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2avx128_wrapper, get_ops_cnt_fft2avx128_wrapper, 2},
    {register_kernel_fft3avx128_wrapper, get_ops_cnt_fft3avx128_wrapper, 3},
    {register_kernel_fft4avx128_wrapper, get_ops_cnt_fft4avx128_wrapper, 4},
    {register_kernel_fft5avx128_wrapper, get_ops_cnt_fft5avx128_wrapper, 5},
    {register_kernel_fft6avx128_wrapper, get_ops_cnt_fft6avx128_wrapper, 6},
    {register_kernel_fft7avx128_wrapper, get_ops_cnt_fft7avx128_wrapper, 7},
    {register_kernel_fft8avx128_wrapper, get_ops_cnt_fft8avx128_wrapper, 8},
    {register_kernel_fft9avx128_wrapper, get_ops_cnt_fft9avx128_wrapper, 9},
    {register_kernel_fft10avx128_wrapper, get_ops_cnt_fft10avx128_wrapper, 10},
    {register_kernel_fft11avx128_wrapper, get_ops_cnt_fft11avx128_wrapper, 11},
    {register_kernel_fft12avx128_wrapper, get_ops_cnt_fft12avx128_wrapper, 12},
    {register_kernel_fft13avx128_wrapper, get_ops_cnt_fft13avx128_wrapper, 13},
    {register_kernel_fft14avx128_wrapper, get_ops_cnt_fft14avx128_wrapper, 14},
    {register_kernel_fft15avx128_wrapper, get_ops_cnt_fft15avx128_wrapper, 15},
    {register_kernel_fft16avx128_wrapper, get_ops_cnt_fft16avx128_wrapper, 16},
    {NULL, NULL, 20},
    {NULL, NULL, 25},
    {NULL, NULL, 32},
    {NULL, NULL, 64}
};
#endif
#ifdef ENABLE_AVX256
static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx256[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2avx256_wrapper, get_ops_cnt_fft2avx256_wrapper, 2},
    {register_kernel_fft3avx256_wrapper, get_ops_cnt_fft3avx256_wrapper, 3},
    {register_kernel_fft4avx256_wrapper, get_ops_cnt_fft4avx256_wrapper, 4},
    {register_kernel_fft5avx256_wrapper, get_ops_cnt_fft5avx256_wrapper, 5},
    {register_kernel_fft6avx256_wrapper, get_ops_cnt_fft6avx256_wrapper, 6},
    {register_kernel_fft7avx256_wrapper, get_ops_cnt_fft7avx256_wrapper, 7},
    {register_kernel_fft8avx256_wrapper, get_ops_cnt_fft8avx256_wrapper, 8},
    {register_kernel_fft9avx256_wrapper, get_ops_cnt_fft9avx256_wrapper, 9},
    {register_kernel_fft10avx256_wrapper, get_ops_cnt_fft10avx256_wrapper, 10},
    {register_kernel_fft11avx256_wrapper, get_ops_cnt_fft11avx256_wrapper, 11},
    {register_kernel_fft12avx256_wrapper, get_ops_cnt_fft12avx256_wrapper, 12},
    {register_kernel_fft13avx256_wrapper, get_ops_cnt_fft13avx256_wrapper, 13},
    {register_kernel_fft14avx256_wrapper, get_ops_cnt_fft14avx256_wrapper, 14},
    {register_kernel_fft15avx256_wrapper, get_ops_cnt_fft15avx256_wrapper, 15},
    {register_kernel_fft16avx256_wrapper, get_ops_cnt_fft16avx256_wrapper, 16},
    {NULL, NULL, 13},
    {NULL, NULL, 20},
    {NULL, NULL, 25},
    {NULL, NULL, 32},
    {NULL, NULL, 64}
};
#endif
#ifdef ENABLE_AVX512
static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx512[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2avx512_wrapper, get_ops_cnt_fft2avx512_wrapper, 2},
    {register_kernel_fft3avx512_wrapper, get_ops_cnt_fft3avx512_wrapper, 3},
    {register_kernel_fft4avx512_wrapper, get_ops_cnt_fft4avx512_wrapper, 4},
    {register_kernel_fft5avx512_wrapper, get_ops_cnt_fft5avx512_wrapper, 5},
    {register_kernel_fft6avx512_wrapper, get_ops_cnt_fft6avx512_wrapper, 6},
    {register_kernel_fft7avx512_wrapper, get_ops_cnt_fft7avx512_wrapper, 7},
    {register_kernel_fft8avx512_wrapper, get_ops_cnt_fft8avx512_wrapper, 8},
    {register_kernel_fft9avx512_wrapper, get_ops_cnt_fft9avx512_wrapper, 9},
    {register_kernel_fft10avx512_wrapper, get_ops_cnt_fft10avx512_wrapper, 10},
    {register_kernel_fft11avx512_wrapper, get_ops_cnt_fft11avx512_wrapper, 11},
    {register_kernel_fft12avx512_wrapper, get_ops_cnt_fft12avx512_wrapper, 12},
    {register_kernel_fft13avx512_wrapper, get_ops_cnt_fft13avx512_wrapper, 13},
    {register_kernel_fft14avx512_wrapper, get_ops_cnt_fft14avx512_wrapper, 14},
    {register_kernel_fft15avx512_wrapper, get_ops_cnt_fft15avx512_wrapper, 15},
    {register_kernel_fft16avx512_wrapper, get_ops_cnt_fft16avx512_wrapper, 16},
    {NULL, NULL, 16},
    {NULL, NULL, 20},
    {NULL, NULL, 25},
    {NULL, NULL, 32},
    {NULL, NULL, 64}
};
#endif

// R2HC Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_c[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hc_rfft2c_wrapper, get_ops_cnt_r2hc_rfft2c_wrapper, 2},
    {register_kernel_r2hc_rfft3c_wrapper, get_ops_cnt_r2hc_rfft3c_wrapper, 3},
    {register_kernel_r2hc_rfft4c_wrapper, get_ops_cnt_r2hc_rfft4c_wrapper, 4},
    {register_kernel_r2hc_rfft5c_wrapper, get_ops_cnt_r2hc_rfft5c_wrapper, 5},
    {register_kernel_r2hc_rfft6c_wrapper, get_ops_cnt_r2hc_rfft6c_wrapper, 6},
    {register_kernel_r2hc_rfft7c_wrapper, get_ops_cnt_r2hc_rfft7c_wrapper, 7},
    {register_kernel_r2hc_rfft8c_wrapper, get_ops_cnt_r2hc_rfft8c_wrapper, 8},
    {register_kernel_r2hc_rfft10c_wrapper,
     get_ops_cnt_r2hc_rfft10c_wrapper, 10},
    {register_kernel_r2hc_rfft12c_wrapper,
     get_ops_cnt_r2hc_rfft12c_wrapper, 12},
    {register_kernel_r2hc_rfft14c_wrapper,
     get_ops_cnt_r2hc_rfft14c_wrapper, 14},
    {register_kernel_r2hc_rfft15c_wrapper,
     get_ops_cnt_r2hc_rfft15c_wrapper, 15},
    {register_kernel_r2hc_rfft16c_wrapper,
     get_ops_cnt_r2hc_rfft16c_wrapper, 16}
};

// R2HC-Fused Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_c[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hcf_rfft2c_wrapper, get_ops_cnt_r2hcf_rfft2c_wrapper, 2},
    {register_kernel_r2hcf_rfft3c_wrapper, get_ops_cnt_r2hcf_rfft3c_wrapper, 3},
    {register_kernel_r2hcf_rfft4c_wrapper, get_ops_cnt_r2hcf_rfft4c_wrapper, 4},
    {register_kernel_r2hcf_rfft5c_wrapper, get_ops_cnt_r2hcf_rfft5c_wrapper, 5},
    {register_kernel_r2hcf_rfft6c_wrapper, get_ops_cnt_r2hcf_rfft6c_wrapper, 6},
    {register_kernel_r2hcf_rfft7c_wrapper, get_ops_cnt_r2hcf_rfft7c_wrapper, 7},
    {register_kernel_r2hcf_rfft8c_wrapper, get_ops_cnt_r2hcf_rfft8c_wrapper, 8},
    {register_kernel_r2hcf_rfft10c_wrapper,
     get_ops_cnt_r2hcf_rfft10c_wrapper, 10},
    {register_kernel_r2hcf_rfft12c_wrapper,
     get_ops_cnt_r2hcf_rfft12c_wrapper, 12},
    {register_kernel_r2hcf_rfft14c_wrapper,
     get_ops_cnt_r2hcf_rfft14c_wrapper, 14},
    {register_kernel_r2hcf_rfft15c_wrapper,
     get_ops_cnt_r2hcf_rfft15c_wrapper, 15},
    {register_kernel_r2hcf_rfft16c_wrapper,
     get_ops_cnt_r2hcf_rfft16c_wrapper, 16}
};

#ifdef ENABLE_AVX128
// R2HC-AVX128 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx128[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hc_rfft2avx128_wrapper,
     get_ops_cnt_r2hc_rfft2avx128_wrapper, 2},
    {register_kernel_r2hc_rfft3avx128_wrapper,
     get_ops_cnt_r2hc_rfft3avx128_wrapper, 3},
    {register_kernel_r2hc_rfft4avx128_wrapper,
     get_ops_cnt_r2hc_rfft4avx128_wrapper, 4},
    {register_kernel_r2hc_rfft5avx128_wrapper,
     get_ops_cnt_r2hc_rfft5avx128_wrapper, 5},
    {register_kernel_r2hc_rfft6avx128_wrapper,
     get_ops_cnt_r2hc_rfft6avx128_wrapper, 6},
    {register_kernel_r2hc_rfft7avx128_wrapper,
     get_ops_cnt_r2hc_rfft7avx128_wrapper, 7},
    {register_kernel_r2hc_rfft8avx128_wrapper,
     get_ops_cnt_r2hc_rfft8avx128_wrapper, 8},
    {register_kernel_r2hc_rfft10avx128_wrapper,
     get_ops_cnt_r2hc_rfft10avx128_wrapper, 10},
    {register_kernel_r2hc_rfft14avx128_wrapper,
     get_ops_cnt_r2hc_rfft14avx128_wrapper, 14},
    {register_kernel_r2hc_rfft15avx128_wrapper,
     get_ops_cnt_r2hc_rfft15avx128_wrapper, 15}
};

// R2HC-Fused -AVX128 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx128[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hcf_rfft2avx128_wrapper,
     get_ops_cnt_r2hcf_rfft2avx128_wrapper, 2},
    {register_kernel_r2hcf_rfft7avx128_wrapper,
     get_ops_cnt_r2hcf_rfft7avx128_wrapper, 7},
    {register_kernel_r2hcf_rfft10avx128_wrapper,
     get_ops_cnt_r2hcf_rfft10avx128_wrapper, 10},
    {register_kernel_r2hcf_rfft14avx128_wrapper,
     get_ops_cnt_r2hcf_rfft14avx128_wrapper, 14},
    {register_kernel_r2hcf_rfft15avx128_wrapper,
     get_ops_cnt_r2hcf_rfft15avx128_wrapper, 15}
};
#endif

#ifdef ENABLE_AVX256
// R2HC-AVX256 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx256[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hc_rfft2avx256_wrapper,
     get_ops_cnt_r2hc_rfft2avx256_wrapper, 2},
    {register_kernel_r2hc_rfft3avx256_wrapper,
     get_ops_cnt_r2hc_rfft3avx256_wrapper, 3},
    {register_kernel_r2hc_rfft4avx256_wrapper,
     get_ops_cnt_r2hc_rfft4avx256_wrapper, 4},
    {register_kernel_r2hc_rfft5avx256_wrapper,
     get_ops_cnt_r2hc_rfft5avx256_wrapper, 5},
    {register_kernel_r2hc_rfft6avx256_wrapper,
     get_ops_cnt_r2hc_rfft6avx256_wrapper, 6},
    {register_kernel_r2hc_rfft7avx256_wrapper,
     get_ops_cnt_r2hc_rfft7avx256_wrapper, 7},
    {register_kernel_r2hc_rfft8avx256_wrapper,
     get_ops_cnt_r2hc_rfft8avx256_wrapper, 8},
    {register_kernel_r2hc_rfft10avx256_wrapper,
     get_ops_cnt_r2hc_rfft10avx256_wrapper, 10},
    {register_kernel_r2hc_rfft14avx256_wrapper,
     get_ops_cnt_r2hc_rfft14avx256_wrapper, 14},
    {register_kernel_r2hc_rfft15avx256_wrapper,
     get_ops_cnt_r2hc_rfft15avx256_wrapper, 15}
};

// R2HC-Fused -AVX256 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx256[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hcf_rfft2avx256_wrapper,
     get_ops_cnt_r2hcf_rfft2avx256_wrapper, 2},
    {register_kernel_r2hcf_rfft10avx256_wrapper,
     get_ops_cnt_r2hcf_rfft10avx256_wrapper, 10},
    {register_kernel_r2hcf_rfft14avx256_wrapper,
     get_ops_cnt_r2hcf_rfft14avx256_wrapper, 14},
    {register_kernel_r2hcf_rfft15avx256_wrapper,
     get_ops_cnt_r2hcf_rfft15avx256_wrapper, 15}
};
#endif

#ifdef ENABLE_AVX512
// R2HC-AVX512 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx512[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hc_rfft2avx512_wrapper,
     get_ops_cnt_r2hc_rfft2avx512_wrapper, 2},
    {register_kernel_r2hc_rfft3avx512_wrapper,
     get_ops_cnt_r2hc_rfft3avx512_wrapper, 3},
    {register_kernel_r2hc_rfft4avx512_wrapper,
     get_ops_cnt_r2hc_rfft4avx512_wrapper, 4},
    {register_kernel_r2hc_rfft5avx512_wrapper,
     get_ops_cnt_r2hc_rfft5avx512_wrapper, 5},
    {register_kernel_r2hc_rfft6avx512_wrapper,
     get_ops_cnt_r2hc_rfft6avx512_wrapper, 6},
    {register_kernel_r2hc_rfft7avx512_wrapper,
     get_ops_cnt_r2hc_rfft7avx512_wrapper, 7},
    {register_kernel_r2hc_rfft8avx512_wrapper,
     get_ops_cnt_r2hc_rfft8avx512_wrapper, 8},
    {register_kernel_r2hc_rfft10avx512_wrapper,
     get_ops_cnt_r2hc_rfft10avx512_wrapper, 10},
    {register_kernel_r2hc_rfft14avx512_wrapper,
     get_ops_cnt_r2hc_rfft14avx512_wrapper, 14},
    {register_kernel_r2hc_rfft15avx512_wrapper,
     get_ops_cnt_r2hc_rfft15avx512_wrapper, 15}
};

// R2HC-Fused -AVX512 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx512[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_r2hcf_rfft2avx512_wrapper,
     get_ops_cnt_r2hcf_rfft2avx512_wrapper, 2},
    {register_kernel_r2hcf_rfft10avx512_wrapper,
     get_ops_cnt_r2hcf_rfft10avx512_wrapper, 10},
    {register_kernel_r2hcf_rfft14avx512_wrapper,
     get_ops_cnt_r2hcf_rfft14avx512_wrapper, 14},
    {register_kernel_r2hcf_rfft15avx512_wrapper,
     get_ops_cnt_r2hcf_rfft15avx512_wrapper, 15}
};
#endif

// Transpose wrappers
#define TRANSPOSE_WRAPPER_DECL(kernel_name, TYPE, isa)                         \
    EXPORT_SYM_DYN VOID CONCAT(FUNC(kernel_name, TYPE, isa),                   \
                               _wrapper)(TRANSPOSE_KERNEL_ARGS)

#define TRANSPOSE_WRAPPER_ALL_TYPES_DECL(kernel_name, isa)                     \
    TRANSPOSE_WRAPPER_DECL(kernel_name, FLOAT, isa);                           \
    TRANSPOSE_WRAPPER_DECL(kernel_name, DOUBLE, isa);                          \
    TRANSPOSE_WRAPPER_DECL(kernel_name, aoclfftz_complex_f_t, isa);            \
    TRANSPOSE_WRAPPER_DECL(kernel_name, aoclfftz_complex_d_t, isa);

TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tiq_iterative, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tisq_iterative, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tiq_recursive_buf, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tir_cycles, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tisr_cycles, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tos_iterative, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tos_blocked, c);

#endif // AOCLFFTZ_CORE_WRAPPER_H
