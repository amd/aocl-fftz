/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file fftw_wrapper.h
 *
 *  @brief Contains constant declarations and function prototypes for
 *  FFTW wrapper.
 *
 */

#ifndef FFTW_WRAPPER_H
#define FFTW_WRAPPER_H

#include "aoclfftz.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WINDOWS
#define EXPORT_SYM_DYN __declspec(dllexport)
#else
#define EXPORT_SYM_DYN
#endif

#define DIR_FORWARD (-1)
#define DIR_BACKWARD (+1)
#define OPTION_MEASURE (0U)
#define OPTION_DESTROY_INPUT (1U << 0)
#define OPTION_UNALIGNED (1U << 1)
#define OPTION_CONSERVE_MEMORY (1U << 2)
#define OPTION_EXHAUSTIVE (1U << 3)
#define OPTION_PRESERVE_INPUT (1U << 4)
#define OPTION_PATIENT (1U << 5)
#define OPTION_ESTIMATE (1U << 6)

#define COMPLEX_TYPE(Real, Complex) typedef Real Complex[2]

#define API_NAME_CONCAT(prefix, name) prefix ## name
#define API_NAME_MANGLE_DOUBLE(name) API_NAME_CONCAT(fftw_, name)
#define API_NAME_MANGLE_FLOAT(name) API_NAME_CONCAT(fftwf_, name)

#define FFTW_WRAPPER_API(GEN, Real, Complex)                                        \
                                                                                    \
COMPLEX_TYPE(Real, Complex);                                                        \
                                                                                    \
typedef struct VOID* GEN(plan);                                                     \
                                                                                    \
typedef aoclfftz_dim_t GEN(iodim);                                                  \
typedef aoclfftz_dim_t_64_ GEN(iodim64);                                            \
                                                                                    \
typedef VOID* GEN(write_char_func);                                                 \
typedef VOID* GEN(read_char_func);                                                  \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(execute)(const GEN(plan) sol);                                              \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft)(INT32 rdims, const INT32* ndim,                                   \
    Complex* idata, Complex* odata, INT32 dir, UINT32 options);                     \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_1d)(INT32 ndim, Complex* idata, Complex* odata, INT32 dir,         \
    UINT32 options);                                                                \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_2d)(INT32 ndim0, INT32 ndim1,                                      \
    Complex* idata, Complex* odata, INT32 dir, UINT32 options);                     \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_3d)(INT32 ndim0, INT32 ndim1, INT32 ndim2,                         \
    Complex* idata, Complex* odata, INT32 dir, UINT32 options);                     \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_many_dft)(INT32 rdims, const INT32* ndim,                              \
    INT32 nproblems,                                                                \
    Complex* idata, const INT32* inoffset,                                          \
    INT32 instride, INT32 inaddr,                                                   \
    Complex* odata, const INT32* onoffset,                                          \
    INT32 outstride, INT32 outaddr,                                                 \
    INT32 dir, UINT32 options);                                                     \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru_dft)(INT32 rdims, const GEN(iodim)* dims,                         \
    INT32 num_problems,                                                             \
    const GEN(iodim)* pdims,                                                        \
    Complex* idata, Complex* odata,                                                 \
    INT32 dir, UINT32 options);                                                     \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru64_dft)(INT32 rdims,                                               \
    const GEN(iodim64)* dims,                                                       \
    INT32 num_problems,                                                             \
    const GEN(iodim64)* pdims,                                                      \
    Complex* idata, Complex* odata,                                                 \
    INT32 dir, UINT32 options);                                                     \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(execute_dft)(const GEN(plan) sol, Complex* idata, Complex* odata);          \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_many_dft_r2c)(INT32 rdims, const INT32* ndim,                          \
    INT32 nproblems,                                                                \
    Real* idata, const INT32* inoffset,                                             \
    INT32 instride, INT32 inaddr,                                                   \
    Complex* odata, const INT32* onoffset,                                          \
    INT32 outstride, INT32 outaddr,                                                 \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_r2c)(INT32 rdims, const INT32* ndim,                               \
    Real* idata, Complex* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_r2c_1d)(INT32 ndim, Real* idata, Complex* odata, UINT32 options);  \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_r2c_2d)(INT32 ndim0, INT32 ndim1,                                  \
    Real* idata, Complex* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_r2c_3d)(INT32 ndim0, INT32 ndim1,                                  \
    INT32 ndim2,                                                                    \
    Real* idata, Complex* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_many_dft_c2r)(INT32 rdims, const INT32* ndim,                          \
    INT32 nproblems,                                                                \
    Complex* idata, const INT32* inoffset,                                          \
    INT32 instride, INT32 inaddr,                                                   \
    Real* odata, const INT32* onoffset,                                             \
    INT32 outstride, INT32 outaddr,                                                 \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_c2r)(INT32 rdims, const INT32* ndim,                               \
    Complex* idata, Real* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_c2r_1d)(INT32 ndim, Complex* idata, Real* odata, UINT32 options);  \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_c2r_2d)(INT32 ndim0, INT32 ndim1,                                  \
    Complex* idata, Real* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_dft_c2r_3d)(INT32 ndim0, INT32 ndim1,                                  \
    INT32 ndim2,                                                                    \
    Complex* idata, Real* odata, UINT32 options);                                   \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru_dft_r2c)(INT32 rdims, const GEN(iodim)* dims,                     \
    INT32 num_problems,                                                             \
    const GEN(iodim)* pdims,                                                        \
    Real* idata, Complex* odata,                                                    \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru_dft_c2r)(INT32 rdims, const GEN(iodim)* dims,                     \
    INT32 num_problems,                                                             \
    const GEN(iodim)* pdims,                                                        \
    Complex* idata, Real* odata,                                                    \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru64_dft_r2c)(INT32 rdims,                                           \
    const GEN(iodim64)* dims,                                                       \
    INT32 num_problems,                                                             \
    const GEN(iodim64)* pdims,                                                      \
    Real* idata, Complex* odata,                                                    \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN GEN(plan)                                                            \
    GEN(plan_guru64_dft_c2r)(INT32 rdims,                                           \
    const GEN(iodim64)* dims,                                                       \
    INT32 num_problems,                                                             \
    const GEN(iodim64)* pdims,                                                      \
    Complex* idata, Real* odata,                                                    \
    UINT32 options);                                                                \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(execute_dft_r2c)(const GEN(plan) sol, Real* idata, Complex* odata);         \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(execute_dft_c2r)(const GEN(plan) sol, Complex* idata, Real* odata);         \
                                                                                    \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(destroy_plan)(GEN(plan) sol);                                               \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(cleanup)(VOID);                                                             \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(plan_with_nthreads)(INT32 nthreads);                                        \
                                                                                    \
EXPORT_SYM_DYN INT32                                                                \
    GEN(init_threads)(VOID);                                                        \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(cleanup_threads)(VOID);                                                     \
                                                                                    \
EXPORT_SYM_DYN VOID*                                                                \
    GEN(malloc)(size_t ndim);                                                       \
                                                                                    \
EXPORT_SYM_DYN Real*                                                                \
    GEN(alloc_real)(size_t ndim);                                                   \
EXPORT_SYM_DYN Complex*                                                             \
    GEN(alloc_complex)(size_t ndim);                                                \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(free)(VOID* sol);                                                           \
                                                                                    \
EXPORT_SYM_DYN VOID                                                                 \
    GEN(flops)(const GEN(plan) sol,                                                 \
    DOUBLE* add, DOUBLE* mul, DOUBLE* fmas);                                        \
EXPORT_SYM_DYN DOUBLE                                                               \
    GEN(estimate_cost)(const GEN(plan) sol);                                        \
                                                                                    \
EXPORT_SYM_DYN DOUBLE                                                               \
    GEN(cost)(const GEN(plan) sol);                                                 \
                                                                                    \
EXPORT_SYM_DYN INT32                                                                \
    GEN(alignment_of)(Real* sol);

FFTW_WRAPPER_API(API_NAME_MANGLE_DOUBLE, DOUBLE, fftw_complex)
FFTW_WRAPPER_API(API_NAME_MANGLE_FLOAT, FLOAT, fftwf_complex)

#define FFTW_FORWARD DIR_FORWARD
#define FFTW_BACKWARD DIR_BACKWARD
#define FFTW_MEASURE OPTION_MEASURE
#define FFTW_DESTROY_INPUT OPTION_DESTROY_INPUT
#define FFTW_UNALIGNED OPTION_UNALIGNED
#define FFTW_CONSERVE_MEMORY OPTION_CONSERVE_MEMORY
#define FFTW_EXHAUSTIVE OPTION_EXHAUSTIVE
#define FFTW_PRESERVE_INPUT OPTIONPRESERVE_INPUT
#define FFTW_PATIENT OPTION_PATIENT
#define FFTW_ESTIMATE OPTION_ESTIMATE

#ifdef __cplusplus
}
#endif

#endif /* FFTW_WRAPPER_H */
