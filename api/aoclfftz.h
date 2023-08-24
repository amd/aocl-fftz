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
 
 /** @file aoclfftz.h
 *  
 *  @brief Interface APIs and data structures of AOCL FFTZ library.
 *
 *  This file contains the APIs and associated data structures that
 *  are responsible for setting up and executing the single-threaded,
 *  and multi-threaded FFT operations.
 *
 *  @note Different variants of APIs and data structures are exposed to
 *  support float and double precision types in LP64 and ILP64 data models.  
 *
 *  @author S. Biplab Raut
 */
 
#ifndef AOCLFFTZ_H
#define AOCLFFTZ_H

#include <stddef.h>
#include "types.h"

#ifdef __cplusplus
extern "C" {

#endif

#ifdef _WINDOWS
#define EXPORT_SYM_DYN __declspec(dllexport)
#else
#define EXPORT_SYM_DYN
#endif

#define AOCLFFTZ_LIBRARY_VERSION "AOCL-FFTZ 0.0.1"

//Error return codes of aocl-fftz library
//Add new error codes at the code to retain the existing error code values
typedef enum
{
    AOCLFFTZ_TIME_OUT = -6,     //Operation took long time than expected
    AOCLFFTZ_MPI_COMM_FAILURE,  //Error encourered in MPI comminucation
    AOCLFFTZ_MEMORY_FAILURE,    //Error related to Memory access or operation
    AOCLFFTZ_INVALID_INPUT,     //Invalid size, format, type or precision of input
    AOCLFFTZ_SETUP_FAILURE,     //Error in setup of the library
    AOCLFFTZ_EXECUTION_FAILURE, //Error in execution of the library
    AOCLFFTZ_SUCCESS            //Successful operation
} aoclfftz_error_type;

//tensor dimension for LP64
typedef struct
{
    INT32 n;
    INT32 in_stride;
    INT32 out_stride;
} aoclfftz_dim_t;

//tensor dimension for ILP64
typedef struct
{
    ptrdiff_t n;
    ptrdiff_t in_stride;
    ptrdiff_t out_stride;
} aoclfftz_dim_t_64_;

//params for parallel SMP fft computation
typedef struct
{
    //Number of max threads to granted for use
    INT32 num_threads;
    //Allow the library to determine how many threads to be used
    INT32 dynamic_load_model;
} aoclfftz_smp_pfft;

//control params for optimizations, logs, stats and others
typedef struct
{
    //Levels: 0 - non-SIMD algorithmic optimizations, 1 - SSE2 optimizations,
    //2 - AVX optimizations, 3 - AVX2 optimizations, 4 - AVX512 optimizations
    INT32 opt_level;
    INT32 opt_off; //Turn off all optimizations
    INT32 logger_mode;
    INT32 measure_stats;     
} aoclfftz_cntrl_params;

//float LP64
typedef struct
{
    FLOAT*in;
    FLOAT*out;
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t *dims;
    aoclfftz_dim_t *vecs;
    //Bits-> 0 : in-place(0) or out-of-place(1),
    //       1 : in-order(0) or out-of-order(1),
    //       2 : forward(0) or backward(1),
    //       3 : complex(0) or real
    UINT32 flags;
    aoclfftz_smp_pfft pthr_fft;
    aoclfftz_cntrl_params cntrl_params;
} aoclfftz_prob_desc_f;

//double LP64
typedef struct
{
    DOUBLE *in;
    DOUBLE *out;
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t *dims;
    aoclfftz_dim_t *vecs;
    //Bits-> 0 : in-place(0) or out-of-place(1),
    //       1 : in-order(0) or out-of-order(1),
    //       2 : forward(0) or backward(1),
    //       3 : complex(0) or real
    UINT32 flags;
    aoclfftz_smp_pfft pthr_fft;
    aoclfftz_cntrl_params cntrl_params;
} aoclfftz_prob_desc_d;

//float ILP64
typedef struct
{
    FLOAT*in;
    FLOAT*out;
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    //Bits-> 0 : in-place(0) or out-of-place(1),
    //       1 : in-order(0) or out-of-order(1),
    //       2 : forward(0) or backward(1),
    //       3 : complex(0) or real
    UINT32 flags;
    aoclfftz_smp_pfft pthr_fft;
    aoclfftz_cntrl_params cntrl_params;
} aoclfftz_prob_desc_f_64_;

//double LP64
typedef struct
{
    DOUBLE *in;
    DOUBLE*out;
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    //Bits-> 0 : in-place(0) or out-of-place(1),
    //       1 : in-order(0) or out-of-order(1),
    //       2 : forward(0) or backward(1),
    //       3 : complex(0) or real
    UINT32 flags;
    aoclfftz_smp_pfft pthr_fft;
    aoclfftz_cntrl_params cntrl_params;
} aoclfftz_prob_desc_d_64_;

/* Single-threaded and multi-threaded FFT unified APIs */
//float LP64
EXPORT_SYM_DYN VOID *aoclfftz_setup_f(aoclfftz_prob_desc_f *problem);
EXPORT_SYM_DYN INT32 aoclfftz_execute_f(VOID *handle);
EXPORT_SYM_DYN VOID aoclfftz_destroy_f(VOID *handle);

//double LP64
EXPORT_SYM_DYN VOID *aoclfftz_setup_d(aoclfftz_prob_desc_d *problem);
EXPORT_SYM_DYN INT32 aoclfftz_execute_d(VOID *handle);
EXPORT_SYM_DYN VOID aoclfftz_destroy_d(VOID *handle);

//float ILP64
EXPORT_SYM_DYN VOID *aoclfftz_setup_f_64_(aoclfftz_prob_desc_f_64_ *problem);
EXPORT_SYM_DYN INT32 aoclfftz_execute_f_64_(VOID *handle);
EXPORT_SYM_DYN VOID aoclfftz_destroy_f_64_(VOID *handle);

//double ILP64
EXPORT_SYM_DYN VOID *aoclfftz_setup_d_64_(aoclfftz_prob_desc_d_64_ *problem);
EXPORT_SYM_DYN INT32 aoclfftz_execute_d_64_(VOID *handle);
EXPORT_SYM_DYN VOID aoclfftz_destroy_d_64_(VOID *handle);

//Interface API to get aocl-fftz library version string.
EXPORT_SYM_DYN const CHAR *aoclfftz_version(VOID);

#ifdef __cplusplus
}
#endif

#endif //AOCLFFTZ_H
