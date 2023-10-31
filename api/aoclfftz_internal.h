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

 /** @file aoclfftz_internal.h
 *
 *  @brief Top-level data structures used across different modules that are not
 *  publicly exposed but are internal to the AOCL FFTZ library.
 *
 *  This file contains the internal library-wide data structures including
 *  top-level Handle and DFT module structures.
 *
 *  @note Different variants of data structures are defined to
 *  support float and double precision types in LP64 and ILP64 data models.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_INTERNAL_H
#define AOCLFFTZ_INTERNAL_H

#include "types.h"
#include "aoclfftz.h"

#define AOCLFFTZ_INTERNAL_LIBRARY_VERSION "AOCL-FFTZ Internal 1.0"


#define NUM_PRECISIONS 2 //Float, Double : Can be increased to add FP16 or FP8
// 0, 1 reserved for FP8 & FP16
#define DT_FLOAT 2
#define DT_DOUBLE 3
#define MAX_GUARANTEED_CACHEABLE_SIZE (2097152) //2MB

//Set and Get Flags bits
#define IS_OUT_OF_PLACE(flags) (flags & 0x1)
#define IS_OUT_OF_ORDER(flags) (flags & 0x2)
#define FFT_DIR(flags) (flags & 0x4)
#define IS_REAL(flags) (flags & 0x8)
#define SET_PRECISION(flags, val) (flags |= (val << 30))
#define DT_PRECISION_FLAG(flags) (flags >> 30)
#define DT_PRECISION_BYTES(dt_prec) \
    UINT32 _dt = dt_prec;           \
    dt_bytes = 1;                   \
    while (_dt > 0)                 \
    {                               \
        dt_bytes *= 2;              \
        _dt--;                      \
    }
#define SET_SELECTOR_MODE(flags, val) \
    if (val == 0)                     \
    {                                 \
        flags &= (~(1 << 16));        \
    }                                 \
    else                              \
    {                                 \
        flags |= (1 << 16);           \
    }
#define GET_SELECTOR_MODE(flags) ((flags << 15) >> 31)
//Move the base address of void pointer by adding offset
#define MOVE_ADDR(base_addr, offset) (VOID *)((CHAR *)base_addr + offset)

#define NUM_FFT_DIRS 2
#define FORWARD_FFT_DIR 0
#define BACKWARD_FFT_DIR 1
#define NUM_SOLVERS 16
#define NUM_KERNELS_IN_TABLE 256
#define NUM_KERNEL_CATEGORIES 4
#define NUM_KERNELS_IN_EACH_CATEGORY 64
#define DATA_STRIDE 2 //Offset to next data, 2 for complex number

//AMD ZEN CPU Instruction approximated latency cycles
#define AMD_ZEN_FP_FMA_CYCLES 4
#define AMD_ZEN_FP_MUL_CYCLES 3
#define AMD_ZEN_FP_ADD_CYCLES 1
#define AMD_ZEN_FP_MOVE_CYCLES 1//Need to fix this after more experiments
#define AMD_ZEN_FP_PERM_CYCLES 1
#define AMD_ZEN_FP_OTHER_CYCLES 1

//Forward declarations
typedef struct aoclfftz_solution aoclfftz_solution_t;
typedef struct aoclfftz_generic_solver aoclfftz_generic_solver_t;
typedef struct aoclfftz_strides aoclfftz_strides_t;
typedef struct aoclfftz_twiddle aoclfftz_twiddle_t;
typedef struct aoclfftz_bluestein aoclfftz_bluestein_t;

//Computational cost analysis of solution of an executed problem/sub-problem
typedef struct cost_analysis
{
    INT64 ops;
    INT64 time;
} cost_analysis_t;

//Kernel template function pointer for performing FFT
typedef VOID (*kfft_) (VOID *in_real, VOID *in_imag,
                       VOID *out_real, VOID *out_imag,
                       ptrdiff_t n,
                       aoclfftz_strides_t *strides);

//Solver execute template function pointer
typedef INT32 (*dft_solver_) (aoclfftz_solution_t* solution);

//Base data structure acting as an abstract class that is derived by the
//top-level DFT data structure and implemented by all the solvers
struct aoclfftz_generic_solver
{
    INT32 solver_type;
    dft_solver_ execute_solver;
    VOID (*destroy_solver) (aoclfftz_solution_t *solution);
    kfft_ kernel_r;
    kfft_ kernel_m;
};

//Holds info on the main problem or decomposed sub-problem in current dimension
typedef struct {
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    //VOID *in;
    VOID *in_real;
    VOID *in_imag;
    //VOID *out;
    VOID *out_real;
    VOID *out_imag;
    aoclfftz_cntrl_params *cntrl_params;
    aoclfftz_smp_pfft *pthr_fft;
    //Application side flag bits =>
    //  in/out-of place:0-bit, in/out-of order:1-bit, dir:2-bit, real/comp:3-bit..
    //Library side internal flag bits =>
    //  datatype:30-31 bits for precision
    //  2-bits: 64-bit(11), 32-bit(10), 16-bit(01), 8-bit(00)
    //  selector mode: 16th-bit
    UINT32 flags;
} aoclfftz_decomp_scheme_t;

//Holds element-wise and radix-wise strides of the sub-problem decomposition
//that is acted upon by a specific kernel
typedef struct aoclfftz_strides
{
    ptrdiff_t in_stride;
    ptrdiff_t out_stride;
    ptrdiff_t v_in_stride;
    ptrdiff_t v_out_stride;
} aoclfftz_strides_t;

//Holds twiddle factors used by a specific kernel for the given problem
typedef struct aoclfftz_twiddle
{
    VOID *TW;
} aoclfftz_twiddle_t;

//Holds bluestein sequence B used by the bluestein solver
//When FFT is computed for B, it will be stored in B_out and
//is_B_out_valid will be set to 1.
//Also holds the internal input and output buffers.
typedef struct aoclfftz_bluestein
{
    VOID *B;
    VOID *B_out;
    VOID *in;
    VOID *out;
    UINT8 is_B_out_valid;
} aoclfftz_bluestein_t;

//Solution data structure that is returned as a handle by the setup API and
//used by the execute API.
typedef struct aoclfftz_solution
{
    aoclfftz_generic_solver_t *solver;
    aoclfftz_decomp_scheme_t *decomp_scheme;
    aoclfftz_strides_t *strides;
    aoclfftz_twiddle_t *twiddle;
    aoclfftz_bluestein_t *bluestein;
    aoclfftz_solution_t *next_sol;
} aoclfftz_solution_t;

//float LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_f *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_f;

//double LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_d *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_d;

//float ILP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_f_64_ *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_f_64_;

//double LP64
//DFT data structure that holds all other module objects and is the top-level
//data structure of the library.
typedef struct
{
    aoclfftz_prob_desc_d_64_ *prob_desc;
    aoclfftz_solution_t *sol_handle;
} aoclfftz_dft_d_64_;

#endif //AOCLFFTZ_INTERNAL_H
