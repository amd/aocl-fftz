// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper.h
 *
 *  @brief Contains constant declarations and function prototypes for
 *  FFTW wrapper.
 *
 */

#ifndef FFTW_WRAPPER_H
#define FFTW_WRAPPER_H

#include <stdio.h>
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

#define FFTW_WRAPPER_API(GEN, Real, Complex)                                   \
                                                                               \
    COMPLEX_TYPE(Real, Complex);                                               \
                                                                               \
    typedef FFTZ_VOID *GEN(plan);                                              \
                                                                               \
    typedef struct                                                             \
    {                                                                          \
        FFTZ_INT32 n;                                                          \
        FFTZ_INT32 is;                                                         \
        FFTZ_INT32 os;                                                         \
    } GEN(iodim);                                                              \
    typedef struct                                                             \
    {                                                                          \
        FFTZ_INTP n;                                                           \
        FFTZ_INTP is;                                                          \
        FFTZ_INTP os;                                                          \
    } GEN(iodim64);                                                            \
                                                                               \
    typedef FFTZ_VOID (*GEN(write_char_func))(FFTZ_CHAR c, FFTZ_VOID *);       \
    typedef FFTZ_INT32 (*GEN(read_char_func))(FFTZ_VOID *);                    \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(execute)(const GEN(plan) sol);                \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft)(                                    \
        FFTZ_INT32 rdims, const FFTZ_INT32 *ndim, Complex *idata,              \
        Complex *odata, FFTZ_INT32 dir, FFTZ_UINT32 options);                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan)                                                   \
        GEN(plan_dft_1d)(FFTZ_INT32 ndim, Complex * idata, Complex * odata,    \
                         FFTZ_INT32 dir, FFTZ_UINT32 options);                 \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_2d)(                                 \
        FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, Complex * idata, Complex * odata,  \
        FFTZ_INT32 dir, FFTZ_UINT32 options);                                  \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_3d)(                                 \
        FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, FFTZ_INT32 ndim2, Complex * idata, \
        Complex * odata, FFTZ_INT32 dir, FFTZ_UINT32 options);                 \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_many_dft)(                               \
        FFTZ_INT32 rdims, const FFTZ_INT32 *ndim, FFTZ_INT32 nproblems,        \
        Complex *idata, const FFTZ_INT32 *inoffset, FFTZ_INT32 instride,       \
        FFTZ_INT32 inaddr, Complex *odata, const FFTZ_INT32 *onoffset,         \
        FFTZ_INT32 outstride, FFTZ_INT32 outaddr, FFTZ_INT32 dir,              \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru_dft)(                               \
        FFTZ_INT32 rdims, const GEN(iodim) * dims, FFTZ_INT32 num_problems,    \
        const GEN(iodim) * pdims, Complex * idata, Complex * odata,            \
        FFTZ_INT32 dir, FFTZ_UINT32 options);                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru64_dft)(                             \
        FFTZ_INT32 rdims, const GEN(iodim64) * dims, FFTZ_INT32 num_problems,  \
        const GEN(iodim64) * pdims, Complex * idata, Complex * odata,          \
        FFTZ_INT32 dir, FFTZ_UINT32 options);                                  \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(execute_dft)(const GEN(plan) sol,             \
                                              Complex *idata, Complex *odata); \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_many_dft_r2c)(                           \
        FFTZ_INT32 rdims, const FFTZ_INT32 *ndim, FFTZ_INT32 nproblems,        \
        Real *idata, const FFTZ_INT32 *inoffset, FFTZ_INT32 instride,          \
        FFTZ_INT32 inaddr, Complex *odata, const FFTZ_INT32 *onoffset,         \
        FFTZ_INT32 outstride, FFTZ_INT32 outaddr, FFTZ_UINT32 options);        \
                                                                               \
    EXPORT_SYM_DYN GEN(plan)                                                   \
        GEN(plan_dft_r2c)(FFTZ_INT32 rdims, const FFTZ_INT32 *ndim,            \
                          Real *idata, Complex *odata, FFTZ_UINT32 options);   \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_r2c_1d)(                             \
        FFTZ_INT32 ndim, Real * idata, Complex * odata, FFTZ_UINT32 options);  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan)                                                   \
        GEN(plan_dft_r2c_2d)(FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, Real * idata, \
                             Complex * odata, FFTZ_UINT32 options);            \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_r2c_3d)(                             \
        FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, FFTZ_INT32 ndim2, Real * idata,    \
        Complex * odata, FFTZ_UINT32 options);                                 \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_many_dft_c2r)(                           \
        FFTZ_INT32 rdims, const FFTZ_INT32 *ndim, FFTZ_INT32 nproblems,        \
        Complex *idata, const FFTZ_INT32 *inoffset, FFTZ_INT32 instride,       \
        FFTZ_INT32 inaddr, Real *odata, const FFTZ_INT32 *onoffset,            \
        FFTZ_INT32 outstride, FFTZ_INT32 outaddr, FFTZ_UINT32 options);        \
                                                                               \
    EXPORT_SYM_DYN GEN(plan)                                                   \
        GEN(plan_dft_c2r)(FFTZ_INT32 rdims, const FFTZ_INT32 *ndim,            \
                          Complex *idata, Real *odata, FFTZ_UINT32 options);   \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_c2r_1d)(                             \
        FFTZ_INT32 ndim, Complex * idata, Real * odata, FFTZ_UINT32 options);  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_c2r_2d)(                             \
        FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, Complex * idata, Real * odata,     \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_dft_c2r_3d)(                             \
        FFTZ_INT32 ndim0, FFTZ_INT32 ndim1, FFTZ_INT32 ndim2, Complex * idata, \
        Real * odata, FFTZ_UINT32 options);                                    \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru_dft_r2c)(                           \
        FFTZ_INT32 rdims, const GEN(iodim) * dims, FFTZ_INT32 num_problems,    \
        const GEN(iodim) * pdims, Real * idata, Complex * odata,               \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru_dft_c2r)(                           \
        FFTZ_INT32 rdims, const GEN(iodim) * dims, FFTZ_INT32 num_problems,    \
        const GEN(iodim) * pdims, Complex * idata, Real * odata,               \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru64_dft_r2c)(                         \
        FFTZ_INT32 rdims, const GEN(iodim64) * dims, FFTZ_INT32 num_problems,  \
        const GEN(iodim64) * pdims, Real * idata, Complex * odata,             \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN GEN(plan) GEN(plan_guru64_dft_c2r)(                         \
        FFTZ_INT32 rdims, const GEN(iodim64) * dims, FFTZ_INT32 num_problems,  \
        const GEN(iodim64) * pdims, Complex * idata, Real * odata,             \
        FFTZ_UINT32 options);                                                  \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(execute_dft_r2c)(                             \
        const GEN(plan) sol, Real *idata, Complex *odata);                     \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(execute_dft_c2r)(                             \
        const GEN(plan) sol, Complex *idata, Real *odata);                     \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(destroy_plan)(GEN(plan) sol);                 \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(cleanup)(FFTZ_VOID);                          \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(set_timelimit)(FFTZ_DOUBLE t);                \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(plan_with_nthreads)(FFTZ_INT32 nthreads);     \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(planner_nthreads)(FFTZ_VOID);                \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(init_threads)(FFTZ_VOID);                    \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(cleanup_threads)(FFTZ_VOID);                  \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(threads_set_callback)(                        \
        FFTZ_VOID(*parallel_loop)(FFTZ_VOID * (*work)(FFTZ_CHAR *),            \
                                  FFTZ_CHAR * jobdata, size_t elsize,          \
                                  FFTZ_INT32 njobs, FFTZ_VOID * data),         \
        FFTZ_VOID * data);                                                     \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID *GEN(malloc)(size_t ndim);                        \
                                                                               \
    EXPORT_SYM_DYN Real *GEN(alloc_real)(size_t ndim);                         \
    EXPORT_SYM_DYN Complex *GEN(alloc_complex)(size_t ndim);                   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(free)(FFTZ_VOID * sol);                       \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(flops)(const GEN(plan) sol, FFTZ_DOUBLE *add, \
                                        FFTZ_DOUBLE *mul, FFTZ_DOUBLE *fmas);  \
    EXPORT_SYM_DYN FFTZ_DOUBLE GEN(estimate_cost)(const GEN(plan) sol);        \
                                                                               \
    EXPORT_SYM_DYN FFTZ_DOUBLE GEN(cost)(const GEN(plan) sol);                 \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(alignment_of)(Real * sol);                   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(fprint_plan)(const GEN(plan) sol, FILE *f);   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(print_plan)(const GEN(plan) sol);             \
                                                                               \
    EXPORT_SYM_DYN FFTZ_CHAR *GEN(sprint_plan)(const GEN(plan) sol);           \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(make_planner_thread_safe)(FFTZ_VOID);         \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(forget_wisdom)(FFTZ_VOID);                    \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(export_wisdom_to_filename)(                  \
        const FFTZ_CHAR *f);                                                   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(export_wisdom_to_file)(FILE * f);             \
                                                                               \
    EXPORT_SYM_DYN FFTZ_CHAR *GEN(export_wisdom_to_string)(FFTZ_VOID);         \
                                                                               \
    EXPORT_SYM_DYN FFTZ_VOID GEN(export_wisdom)(                               \
        GEN(write_char_func) write_char, FFTZ_VOID * data);                    \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(import_system_wisdom)(FFTZ_VOID);            \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(import_wisdom_from_filename)(                \
        const FFTZ_CHAR *f);                                                   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(import_wisdom_from_file)(FILE * f);          \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(import_wisdom_from_string)(                  \
        const FFTZ_CHAR *s);                                                   \
                                                                               \
    EXPORT_SYM_DYN FFTZ_INT32 GEN(import_wisdom)(                              \
        GEN(read_char_func) read_char, FFTZ_VOID * data);                      \
                                                                               \
    EXPORT_SYM_DYN extern const FFTZ_CHAR GEN(version)[];

    FFTW_WRAPPER_API(API_NAME_MANGLE_DOUBLE, FFTZ_DOUBLE, fftw_complex)
    FFTW_WRAPPER_API(API_NAME_MANGLE_FLOAT, FFTZ_FLOAT, fftwf_complex)

#define FFTW_FORWARD DIR_FORWARD
#define FFTW_BACKWARD DIR_BACKWARD
#define FFTW_MEASURE OPTION_MEASURE
#define FFTW_DESTROY_INPUT OPTION_DESTROY_INPUT
#define FFTW_UNALIGNED OPTION_UNALIGNED
#define FFTW_CONSERVE_MEMORY OPTION_CONSERVE_MEMORY
#define FFTW_EXHAUSTIVE OPTION_EXHAUSTIVE
#define FFTW_PRESERVE_INPUT OPTION_PRESERVE_INPUT
#define FFTW_PATIENT OPTION_PATIENT
#define FFTW_ESTIMATE OPTION_ESTIMATE

#ifdef __cplusplus
}
#endif

#endif /* FFTW_WRAPPER_H */
