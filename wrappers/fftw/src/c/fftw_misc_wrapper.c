// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_misc_wrapper.c
 *
 *  @brief Contains wrapper implementations of FFTW miscellaneous APIs.
 *
 *  This file contains implementations for the miscellaneous APIs provided
 *  by FFTW.
 */

#include "src/translator/fftz_translator.h"

FFTZ_INT32 thread_num = 1;

const FFTZ_CHAR fftw_version[128]  =
    AOCLFFTZ_LIBRARY_VERSION " (FFTW compatible)";
const FFTZ_CHAR fftwf_version[128] =
    AOCLFFTZ_LIBRARY_VERSION " (FFTW compatible)";

/* Single empty string returned by export_wisdom_to_string. No allocation, so no
 * leak. If the caller passes this pointer to fftw_free, we skip the free (safe
 * no-op). */
static FFTZ_CHAR fftw_export_wisdom_empty_string[] = "";

// Allocate aligned memory for given number of bytes
FFTZ_VOID *fftw_malloc(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, n);
    return ptr;
}

// Allocate aligned memory for given number of bytes
FFTZ_VOID *fftwf_malloc(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, n);
    return ptr;
}

// Allocate aligned memory for complex double datatype for given size
fftw_complex *fftw_alloc_complex(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, sizeof(fftw_complex) * n);
    return ptr;
}

// Allocate aligned memory for complex float datatype for given size
fftwf_complex *fftwf_alloc_complex(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, sizeof(fftwf_complex) * n);
    return ptr;
}

double *fftw_alloc_real(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, sizeof(double) * n);
    return ptr;
}

float *fftwf_alloc_real(size_t n)
{
    FFTZ_VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, FFTZ_VOID, sizeof(float) * n);
    return ptr;
}

FFTZ_VOID fftw_free(FFTZ_VOID *mem_ptr)
{
    if (mem_ptr == (FFTZ_VOID *)fftw_export_wisdom_empty_string)
    {
        return;
    }
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

FFTZ_VOID fftwf_free(FFTZ_VOID *mem_ptr)
{
    if (mem_ptr == (FFTZ_VOID *)fftw_export_wisdom_empty_string)
    {
        return;
    }
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

// FFTW planner stores some persistant data other than plan, which can be
// destroyed using cleanup() but this has no requirement in AOCL-FFTZ. Hence
// having it as an empty function.
FFTZ_VOID fftw_cleanup(FFTZ_VOID)
{
}

FFTZ_VOID fftwf_cleanup(FFTZ_VOID)
{
}

FFTZ_VOID fftw_plan_with_nthreads(FFTZ_INT32 nthreads)
{
    thread_num = nthreads;
}

FFTZ_VOID fftwf_plan_with_nthreads(FFTZ_INT32 nthreads)
{
    thread_num = nthreads;
}

FFTZ_INT32 fftw_planner_nthreads(FFTZ_VOID)
{
    return thread_num;
}

FFTZ_INT32 fftwf_planner_nthreads(FFTZ_VOID)
{
    return thread_num;
}

FFTZ_INT32 fftw_init_threads(FFTZ_VOID)
{
    thread_num = 1;
    return 1;
}

FFTZ_INT32 fftwf_init_threads(FFTZ_VOID)
{
    thread_num = 1;
    return 1;
}

FFTZ_VOID fftw_cleanup_threads(FFTZ_VOID)
{
   thread_num = 1; // reset to default
}

FFTZ_VOID fftwf_cleanup_threads(FFTZ_VOID)
{
    thread_num = 1; // reset to default
}

FFTZ_VOID fftw_set_timelimit(FFTZ_DOUBLE t)
{
    (void)t;
}

FFTZ_VOID fftwf_set_timelimit(FFTZ_DOUBLE t)
{
    (void)t;
}

FFTZ_VOID fftw_threads_set_callback(
    FFTZ_VOID (*parallel_loop)(FFTZ_VOID *(*work)(FFTZ_CHAR *),
    FFTZ_CHAR *jobdata, size_t elsize, FFTZ_INT32 njobs, FFTZ_VOID *data),
    FFTZ_VOID *data)
{
    (void)parallel_loop;
    (void)data;
}

FFTZ_VOID fftwf_threads_set_callback(
    FFTZ_VOID (*parallel_loop)(FFTZ_VOID *(*work)(FFTZ_CHAR *),
    FFTZ_CHAR *jobdata, size_t elsize, FFTZ_INT32 njobs, FFTZ_VOID *data),
    FFTZ_VOID *data)
{
    (void)parallel_loop;
    (void)data;
}

FFTZ_VOID fftw_fprint_plan(const fftw_plan p, FILE *f)
{
    (void)p;
    (void)f;
}

FFTZ_VOID fftwf_fprint_plan(const fftwf_plan p, FILE *f)
{
    (void)p;
    (void)f;
}

FFTZ_VOID fftw_print_plan(const fftw_plan p)
{
    (void)p;
}

FFTZ_VOID fftwf_print_plan(const fftwf_plan p)
{
    (void)p;
}

FFTZ_CHAR *fftw_sprint_plan(const fftw_plan p)
{
    (void)p;
    return (FFTZ_CHAR *)fftw_export_wisdom_empty_string;
}

FFTZ_CHAR *fftwf_sprint_plan(const fftwf_plan p)
{
    (void)p;
    return (FFTZ_CHAR *)fftw_export_wisdom_empty_string;
}

FFTZ_VOID fftw_flops(const fftw_plan p, double *add, double *mul, double *fmas)
{
    (void)p;
    (void)add;
    (void)mul;
    (void)fmas;
}

FFTZ_VOID fftwf_flops(const fftwf_plan p, double *add, double *mul,
                      double *fmas)
{
    (void)p;
    (void)add;
    (void)mul;
    (void)fmas;
}

double fftw_estimate_cost(const fftw_plan p)
{
    (void)p;
    return 0;
}

double fftwf_estimate_cost(const fftwf_plan p)
{
    (void)p;
    return 0;
}

double fftw_cost(const fftw_plan p)
{
    (void)p;
    return 0;
}

double fftwf_cost(const fftwf_plan p)
{
    (void)p;
    return 0;
}

FFTZ_INT32 fftw_alignment_of(double *p)
{
    return (int)(((uintptr_t)p) % 16);
}

FFTZ_INT32 fftwf_alignment_of(float *p)
{
    return (int)(((uintptr_t)p) % 16);
}

/*
 * FFTW3 wisdom API stubs. We have no wisdom to save or load, but apps expect
 * these functions. Export functions return success (1) since there is nothing
 * to export and the no-op is harmless. Import functions return 0 (failure)
 * since there is no wisdom to load. export_to_string returns a fixed empty
 * string (no malloc). If the app calls fftw_free on that pointer, we do
 * nothing—no crash, no leak. We never return NULL.
 */
FFTZ_VOID fftw_forget_wisdom(FFTZ_VOID)
{
}

FFTZ_VOID fftw_make_planner_thread_safe(FFTZ_VOID)
{
}

FFTZ_INT32 fftw_export_wisdom_to_filename(const FFTZ_CHAR *filename)
{
    (void)filename;
    return 1;
}

FFTZ_VOID fftw_export_wisdom_to_file(FILE *f)
{
    (void)f;
}

FFTZ_CHAR *fftw_export_wisdom_to_string(FFTZ_VOID)
{
    return (FFTZ_CHAR *)fftw_export_wisdom_empty_string;
}

FFTZ_VOID fftw_export_wisdom(fftw_write_char_func write_char, FFTZ_VOID *data)
{
    (void)write_char;
    (void)data;
}

FFTZ_INT32 fftw_import_system_wisdom(FFTZ_VOID)
{
    return 0;
}

FFTZ_INT32 fftw_import_wisdom_from_filename(const FFTZ_CHAR *filename)
{
    (void)filename;
    return 0;
}

FFTZ_INT32 fftw_import_wisdom_from_file(FILE *f)
{
    (void)f;
    return 0;
}

FFTZ_INT32 fftw_import_wisdom_from_string(const FFTZ_CHAR *s)
{
    (void)s;
    return 0;
}

FFTZ_INT32 fftw_import_wisdom(fftw_read_char_func read_char, FFTZ_VOID *data)
{
    (void)read_char;
    (void)data;
    return 0;
}

FFTZ_VOID fftwf_forget_wisdom(FFTZ_VOID)
{
}

FFTZ_VOID fftwf_make_planner_thread_safe(FFTZ_VOID)
{
}

FFTZ_INT32 fftwf_export_wisdom_to_filename(const FFTZ_CHAR *filename)
{
    (void)filename;
    return 1;
}

FFTZ_VOID fftwf_export_wisdom_to_file(FILE *f)
{
    (void)f;
}

FFTZ_CHAR *fftwf_export_wisdom_to_string(FFTZ_VOID)
{
    return (FFTZ_CHAR *)fftw_export_wisdom_empty_string;
}

FFTZ_VOID fftwf_export_wisdom(fftwf_write_char_func write_char, FFTZ_VOID *data)
{
    (void)write_char;
    (void)data;
}

FFTZ_INT32 fftwf_import_system_wisdom(FFTZ_VOID)
{
    return 0;
}

FFTZ_INT32 fftwf_import_wisdom_from_filename(const FFTZ_CHAR *filename)
{
    (void)filename;
    return 0;
}

FFTZ_INT32 fftwf_import_wisdom_from_file(FILE *f)
{
    (void)f;
    return 0;
}

FFTZ_INT32 fftwf_import_wisdom_from_string(const FFTZ_CHAR *s)
{
    (void)s;
    return 0;
}

FFTZ_INT32 fftwf_import_wisdom(fftwf_read_char_func read_char, FFTZ_VOID *data)
{
    (void)read_char;
    (void)data;
    return 0;
}
