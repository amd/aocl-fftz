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

/** @file fftz_translator.h
 *
 *  @brief Contains utility functions required for translating FFTW APIs by
 *  FFTW wrapper.
 *
 *  This file provides macros and enums for FFTW wrapper.
 */

#include "../../utils/allocator.h"
#include "aoclfftz.h"
#include "api/fftw_wrapper.h"

extern INT32 thread_num;

#define INIT_PTHR_FFT(pthr_fft)                                                \
    {                                                                          \
        pthr_fft.num_threads = thread_num;                                     \
        pthr_fft.dynamic_load_model = 0;                                       \
    }

#define INIT_CNTRL_PARAMS(cntrl_params)                                        \
    {                                                                          \
        cntrl_params.opt_level = 4;                                            \
        cntrl_params.opt_off = 0;                                              \
        cntrl_params.logger_mode = AOCLFFTZ_LOG_NONE;                          \
        cntrl_params.measure_stats = 0;                                        \
    }

#define INIT_PD(problem, dv_desc, sign, in, out, ffttype)                      \
    {                                                                          \
        problem->flags = init_flag(sign, in, out, ffttype);                    \
        problem->vec_rank = dv_desc->vec_rank;                                 \
        problem->dim_rank = dv_desc->dim_rank;                                 \
        problem->vecs = dv_desc->vecs;                                         \
        problem->dims = dv_desc->dims;                                         \
        INIT_PTHR_FFT(problem->pthr_fft);                                      \
        INIT_CNTRL_PARAMS(problem->cntrl_params);                              \
    }

#define DESTROY_DESC(p_desc, dv_desc)                                          \
    {                                                                          \
        if (p_desc != NULL)                                                    \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(p_desc->dims);                            \
            FREE_ALIGN_ALLOCATED_MEM(p_desc->vecs);                            \
        }                                                                      \
        FREE_ALIGN_ALLOCATED_MEM(p_desc);                                      \
        FREE_ALIGN_ALLOCATED_MEM(dv_desc);                                     \
    }

typedef enum
{
    FORWARD = 0,
    BACKWARD = 1
} direction_t;

typedef enum
{
    COMPLEX = 0,
    REAL = 1
} fft_type_t;

typedef struct
{
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t *dims;
    aoclfftz_dim_t *vecs;
} dv_desc;

typedef struct
{
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
} dv_desc_64_;

aoclfftz_flags_t init_flag(INT32 sign, VOID *in, VOID *out, fft_type_t ffttype);

fftw_plan get_handle_d(dv_desc *dv_desc, INT32 sign, VOID *in, VOID *out,
                       fft_type_t ffttype);

fftwf_plan get_handle_f(dv_desc *dv_desc, INT32 sign, VOID *in, VOID *out,
                        fft_type_t ffttype);

fftw_plan get_handle_d_64_(dv_desc_64_ *dv_desc, INT32 sign, VOID *in,
                           VOID *out, fft_type_t ffttype);

fftwf_plan get_handle_f_64_(dv_desc_64_ *dv_desc, INT32 sign, VOID *in,
                            VOID *out, fft_type_t ffttype);

dv_desc *get_dv_desc(INT32 rank, const INT32 *n);

dv_desc *get_r2c_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace);

dv_desc *get_c2r_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace);

dv_desc *get_many_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                          const INT32 *inembed, INT32 istride, INT32 idist,
                          const INT32 *onembed, INT32 ostride, INT32 odist);

dv_desc *get_many_r2c_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                              const INT32 *inembed, INT32 istride, INT32 idist,
                              const INT32 *onembed, INT32 ostride, INT32 odist,
                              INT32 is_inplace);

dv_desc *get_many_c2r_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                              const INT32 *inembed, INT32 istride, INT32 idist,
                              const INT32 *onembed, INT32 ostride, INT32 odist,
                              INT32 is_inplace);

dv_desc *get_guru_dv_desc(INT32 rank, const fftw_iodim *dims,
                          INT32 howmany_rank, const fftw_iodim *howmany_dims);

dv_desc_64_ *get_guru_64_dv_desc(INT32 rank, const fftw_iodim64 *dims,
                                 INT32 howmany_rank,
                                 const fftw_iodim64 *howmany_dims);

VOID reverse_array(const INT32 *p_array, INT32 *p_reversed_array, INT32 size);
dv_desc *get_fortran_guru_dv_desc(INT32 rank, const INT32 *n, const INT32 *is,
                                  const INT32 *os, INT32 howmany_rank,
                                  const INT32 *h_n, const INT32 *h_is,
                                  const INT32 *h_os);
#define MIN_ALIGNMENT 16

#ifdef _WIN32

#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
    {                                                                          \
        ptr = (type *)_aligned_malloc(num_bytes, MIN_ALIGNMENT);               \
    }

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
    {                                                                          \
        if (mem_ptr)                                                           \
        {                                                                      \
            _aligned_free(mem_ptr);                                            \
        }                                                                      \
        mem_ptr = NULL;                                                        \
    }

#else

#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
    {                                                                          \
        if (posix_memalign((VOID **)(&ptr), MIN_ALIGNMENT, num_bytes))         \
        {                                                                      \
            ptr = NULL;                                                        \
        }                                                                      \
    }

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
    {                                                                          \
        if (mem_ptr)                                                           \
        {                                                                      \
            free(mem_ptr);                                                     \
        }                                                                      \
        mem_ptr = NULL;                                                        \
    }

#endif
