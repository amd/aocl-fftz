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

INT32 thread_num = 1;

const CHAR fftw_version[128]  = AOCLFFTZ_LIBRARY_VERSION " (FFTW compatible)";
const CHAR fftwf_version[128] = AOCLFFTZ_LIBRARY_VERSION " (FFTW compatible)";

/* Single empty string returned by export_wisdom_to_string. No allocation, so no leak.
 * If the caller passes this pointer to fftw_free, we skip the free (safe no-op). */
static CHAR fftw_export_wisdom_empty_string[] = "";

// Allocate aligned memory for given number of bytes
VOID *fftw_malloc(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, n);
    return ptr;
}

// Allocate aligned memory for given number of bytes
VOID *fftwf_malloc(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, n);
    return ptr;
}

// Allocate aligned memory for complex double datatype for given size
fftw_complex *fftw_alloc_complex(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(fftw_complex) * n);
    return ptr;
}

// Allocate aligned memory for complex float datatype for given size
fftwf_complex *fftwf_alloc_complex(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(fftwf_complex) * n);
    return ptr;
}

double *fftw_alloc_real(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(double) * n);
    return ptr;
}

float *fftwf_alloc_real(size_t n)
{
    VOID *ptr = NULL;
    ALLOC_ALIGN_UNINIT(ptr, VOID, sizeof(float) * n);
    return ptr;
}

VOID fftw_free(VOID *mem_ptr)
{
    if (mem_ptr == (VOID *)fftw_export_wisdom_empty_string)
    {
        return;
    }
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

VOID fftwf_free(VOID *mem_ptr)
{
    if (mem_ptr == (VOID *)fftw_export_wisdom_empty_string)
    {
        return;
    }
    FREE_ALIGN_ALLOCATED_MEM(mem_ptr);
}

// FFTW planner stores some persistant data other than plan, which can be
// destroyed using cleanup() but this has no requirement in AOCL-FFTZ. Hence having
// it as an empty function.
VOID fftw_cleanup(VOID)
{
}

VOID fftwf_cleanup(VOID)
{
}

VOID fftw_plan_with_nthreads(INT32 nthreads)
{
    thread_num = nthreads;
}

VOID fftwf_plan_with_nthreads(INT32 nthreads)
{
    thread_num = nthreads;
}

INT32 fftw_planner_nthreads(VOID)
{
    return thread_num;
}

INT32 fftwf_planner_nthreads(VOID)
{
    return thread_num;
}

INT32 fftw_init_threads(VOID)
{
    thread_num = 1;
    return 1;
}

INT32 fftwf_init_threads(VOID)
{
    thread_num = 1;
    return 1;
}

VOID fftw_cleanup_threads(VOID)
{
   thread_num = 1; // reset to default
}

VOID fftwf_cleanup_threads(VOID)
{
    thread_num = 1; // reset to default
}

VOID fftw_set_timelimit(DOUBLE t)
{
    (void)t;
}

VOID fftwf_set_timelimit(DOUBLE t)
{
    (void)t;
}

VOID fftw_threads_set_callback(
    VOID (*parallel_loop)(VOID *(*work)(CHAR *),
    CHAR *jobdata, size_t elsize, INT32 njobs, VOID *data), VOID *data)
{
    (void)parallel_loop;
    (void)data;
}

VOID fftwf_threads_set_callback(
    VOID (*parallel_loop)(VOID *(*work)(CHAR *),
    CHAR *jobdata, size_t elsize, INT32 njobs, VOID *data), VOID *data)
{
    (void)parallel_loop;
    (void)data;
}

VOID fftw_fprint_plan(const fftw_plan p, FILE *f)
{
    (void)p;
    (void)f;
}

VOID fftwf_fprint_plan(const fftwf_plan p, FILE *f)
{
    (void)p;
    (void)f;
}

VOID fftw_print_plan(const fftw_plan p)
{
    (void)p;
}

VOID fftwf_print_plan(const fftwf_plan p)
{
    (void)p;
}

CHAR *fftw_sprint_plan(const fftw_plan p)
{
    (void)p;
    return (CHAR *)fftw_export_wisdom_empty_string;
}

CHAR *fftwf_sprint_plan(const fftwf_plan p)
{
    (void)p;
    return (CHAR *)fftw_export_wisdom_empty_string;
}

VOID fftw_flops(const fftw_plan p, double *add, double *mul, double *fmas)
{
    (void)p;
    (void)add;
    (void)mul;
    (void)fmas;
}

VOID fftwf_flops(const fftwf_plan p, double *add, double *mul, double *fmas)
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

INT32 fftw_alignment_of(double *p)
{
    return (int)(((uintptr_t)p) % 16);
}

INT32 fftwf_alignment_of(float *p)
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
VOID fftw_forget_wisdom(VOID)
{
}

VOID fftw_make_planner_thread_safe(VOID)
{
}

INT32 fftw_export_wisdom_to_filename(const CHAR *filename)
{
    (void)filename;
    return 1;
}

VOID fftw_export_wisdom_to_file(FILE *f)
{
    (void)f;
}

CHAR *fftw_export_wisdom_to_string(VOID)
{
    return (CHAR *)fftw_export_wisdom_empty_string;
}

VOID fftw_export_wisdom(fftw_write_char_func write_char, VOID *data)
{
    (void)write_char;
    (void)data;
}

INT32 fftw_import_system_wisdom(VOID)
{
    return 0;
}

INT32 fftw_import_wisdom_from_filename(const CHAR *filename)
{
    (void)filename;
    return 0;
}

INT32 fftw_import_wisdom_from_file(FILE *f)
{
    (void)f;
    return 0;
}

INT32 fftw_import_wisdom_from_string(const CHAR *s)
{
    (void)s;
    return 0;
}

INT32 fftw_import_wisdom(fftw_read_char_func read_char, VOID *data)
{
    (void)read_char;
    (void)data;
    return 0;
}

VOID fftwf_forget_wisdom(VOID)
{
}

VOID fftwf_make_planner_thread_safe(VOID)
{
}

INT32 fftwf_export_wisdom_to_filename(const CHAR *filename)
{
    (void)filename;
    return 1;
}

VOID fftwf_export_wisdom_to_file(FILE *f)
{
    (void)f;
}

CHAR *fftwf_export_wisdom_to_string(VOID)
{
    return (CHAR *)fftw_export_wisdom_empty_string;
}

VOID fftwf_export_wisdom(fftwf_write_char_func write_char, VOID *data)
{
    (void)write_char;
    (void)data;
}

INT32 fftwf_import_system_wisdom(VOID)
{
    return 0;
}

INT32 fftwf_import_wisdom_from_filename(const CHAR *filename)
{
    (void)filename;
    return 0;
}

INT32 fftwf_import_wisdom_from_file(FILE *f)
{
    (void)f;
    return 0;
}

INT32 fftwf_import_wisdom_from_string(const CHAR *s)
{
    (void)s;
    return 0;
}

INT32 fftwf_import_wisdom(fftwf_read_char_func read_char, VOID *data)
{
    (void)read_char;
    (void)data;
    return 0;
}
