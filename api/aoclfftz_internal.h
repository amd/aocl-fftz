// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#ifdef MULTI_THREADING
#include <omp.h>
#endif
#include "types.h"
#include "aoclfftz.h"

#define AOCLFFTZ_2_PI 6.2831853071795864769252867665590057683943388
#define AOCLFFTZ_2_PIf 6.2831853071795864769252867665590057683943388f

#define NUM_PRECISIONS 2 // Float, Double : Can be increased to add FP16 or FP8
// 0, 1 reserved for FP8 & FP16
#define DT_FLOAT 2
#define DT_DOUBLE 3
// Set and Get Flags bits
#define BIT_FLAG32_ON(flags, nbit) ((flags) |= (0x1 << (nbit)))
#define BIT_FLAG32_OFF(flags, nbit) ((flags) &= ~(0x1 << (nbit)))
#define GET_BIT_FLAG32(flags, nbit) (((flags) >> (nbit)) & 0x1)

#define SET_BIT_FLAG32(flags, nbit, value)                                     \
    do                                                                         \
    {                                                                          \
        if (value)                                                             \
        {                                                                      \
            BIT_FLAG32_ON(flags, nbit);                                        \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            BIT_FLAG32_OFF(flags, nbit);                                       \
        }                                                                      \
    } while (0)

// Get Flags
#define IS_OUT_OF_PLACE(flags) GET_BIT_FLAG32(flags, 0)
#define IS_OUT_OF_ORDER(flags) GET_BIT_FLAG32(flags, 1)
#define FFT_DIR(flags) GET_BIT_FLAG32(flags, 2)
#define IS_REAL(flags) GET_BIT_FLAG32(flags, 3)
#define DT_PRECISION_FLAG(flags) (flags >> 30)

// Set Flags
#define SET_PRECISION(flags, val) (flags = (flags & 0x3FFFFFFF) | (val << 30))
#define SET_INPLACE(flags) SET_BIT_FLAG32(flags, 0, 0)
#define SET_OUTOFPLACE(flags) SET_BIT_FLAG32(flags, 0, 1)

#define SET_BIT_REPRODUCIBLE(flags, val) SET_BIT_FLAG32(flags, 4, val)
#define GET_BIT_REPRODUCIBLE(flags) GET_BIT_FLAG32(flags, 4)

#define SET_NOT_INNERMOST_DIM(flags) SET_BIT_FLAG32(flags, 10, 1)
#define IS_NOT_INNERMOST_DIM(flags) GET_BIT_FLAG32(flags, 10)

#define SET_BUFFERED(flags) SET_BIT_FLAG32(flags, 11, 1)
#define UNSET_BUFFERED(flags) SET_BIT_FLAG32(flags, 11, 0)
#define IS_BUFFERED(flags) GET_BIT_FLAG32(flags, 11)

// Get size of datatype based on the precision
#define DT_PRECISION_BYTES(dt_prec) (1 << dt_prec)

/*
 * @brief Get size of datatype from solution, in bytes
 * */
#define SOL_DT_SIZE(sol)                                                       \
    DT_PRECISION_BYTES(DT_PRECISION_FLAG(sol->decomp_scheme->flags))

#define SET_SELECTOR_MODE(flags, value) SET_BIT_FLAG32(flags, 16, value)
#define GET_SELECTOR_MODE(flags) GET_BIT_FLAG32(flags, 16)

// +-------------------------------+-------------------------------+
// |         SET_TRANSPOSE         |    SET_STANDALONE_TRANSPOSE   |
// |-------------------------------+-------------------------------+
// | * used at solver (CT) level   | * used at selector level      |
// | * transpose + dft operations  | * transpose operation, no dft |
// +-------------------------------+-------------------------------+

#define SET_TRANSPOSE(flags, val) SET_BIT_FLAG32(flags, 9, val)
#define GET_TRANSPOSE(flags) GET_BIT_FLAG32(flags, 9)

#define SET_STANDALONE_TRANSPOSE(flags, val) SET_BIT_FLAG32(flags, 8, val)
#define GET_STANDALONE_TRANSPOSE(flags) GET_BIT_FLAG32(flags, 8)

// Move the base address of void pointer by adding offset
#define MOVE_ADDR(base_addr, offset) (VOID *)((CHAR *)base_addr + offset)

#define IS_POW2(x) (((x) & ((x) - 1)) == 0)

#define NUM_FFT_DIRS 2
#define FORWARD_FFT_DIR 0
#define BACKWARD_FFT_DIR 1
#define DATA_STRIDE 2 // Offset to next data, 2 for complex number

#define NUM_REAL_KERNELS_VARIANTS 3
#define NUM_KERNEL_CATEGORIES 4

// Number of standard kernels (radix 2 to 16)
#define NUM_STANDARD_KERNELS 15
// Number of higher radix kernels (radix > 16, e.g., radix 48)
#define NUMBER_OF_HIGHER_RADIX_KERNELS 2
// Total number of kernels in each category
#define NUM_KERNELS_IN_EACH_CATEGORY (NUM_STANDARD_KERNELS + NUMBER_OF_HIGHER_RADIX_KERNELS)

#define NUM_KERNELS_IN_EACH_DFT_VARIANT                                        \
    (NUM_KERNELS_IN_EACH_CATEGORY * NUM_KERNEL_CATEGORIES)

#define NUM_KERNELS_IN_TABLE_COMPLEX NUM_KERNELS_IN_EACH_DFT_VARIANT

#define NUM_KERNELS_IN_TABLE_REAL                                              \
    (NUM_KERNELS_IN_EACH_DFT_VARIANT * NUM_REAL_KERNELS_VARIANTS)

#define MAX_NUM_KERNELS_IN_TABLE NUM_KERNELS_IN_TABLE_REAL // max of real and complex

// AMD ZEN CPU Instruction approximated latency cycles
#define AMD_ZEN_FP_FMA_CYCLES 4
#define AMD_ZEN_FP_MUL_CYCLES 3
#define AMD_ZEN_FP_ADD_CYCLES 1
#define AMD_ZEN_FP_MOVE_CYCLES 1 // Need to fix this after more experiments
#define AMD_ZEN_FP_PERM_CYCLES 1
#define AMD_ZEN_FP_OTHER_CYCLES 1

/**
 * A "group" is a repeating pattern of DFT kernels at any RealFFT stage.
 * Each group
 *   starts with one R2HC kernel,
 *   followed by a set of C2C kernels(optional) and
 *   finally another R2HC kernel(optional).
 * If an R2HC kernel is present at end, we merge it with first to form one R2HCF
 * kernel. Hence, a RDFT group will either have one R2HC or one R2HCF.
 *
 * Therefore,
 *     Number of groups = Num_R2HCF > 0 ? Num_R2HCF : Num_R2HC. \
 * OR: Number of groups = max(Num_R2HCF, Num_R2HC) \
 * OR: Number of groups = Num_R2HCF + Num_R2HC \
 * OR: Number of groups = (Total points in problem) / (product of radices till
 *       current stage) = product of radices after current stage
 */
#define NUM_RFFT_GROUPS(solver)                                                 \
    (solver)->kernel_r2hcf->count + (solver)->kernel_r2hc->count

// Forward declarations
typedef struct aoclfftz_solution aoclfftz_solution_t;
typedef struct aoclfftz_generic_solver aoclfftz_generic_solver_t;
typedef struct aoclfftz_strides aoclfftz_strides_t;
typedef struct aoclfftz_twiddle aoclfftz_twiddle_t;
typedef struct aoclfftz_bluestein aoclfftz_bluestein_t;
typedef struct aoclfftz_buffered aoclfftz_buffered_t;
typedef struct aoclfftz_sr aoclfftz_sr_t;
typedef struct aoclfftz_executor aoclfftz_executor_t;
typedef struct aoclfftz_realhelper aoclfftz_realhelper_t;

// Computational cost analysis of solution of an executed problem/sub-problem
typedef struct cost_analysis
{
    INT64 ops;
    INT64 time;
} cost_analysis_t;

// Kernel template function pointer for performing FFT
typedef VOID (*kfft_) (VOID *in_real, VOID *in_imag,
                       VOID *out_real, VOID *out_imag,
                       INTP n,
                       aoclfftz_strides_t *strides,
                       VOID *twd, UINT8 flag);

// Kernel information data structure holds the kernel function pointer and the
// number of sets it can process in parallel based on the kernel type(C/SIMD).
// This set information will be used by MT direct solver to find the kernel type
// (C/AVX128/AVX256/AVX512) at runtime, adjust the number of batch iteration and
// assign number of threads accordingly.
typedef struct kernel_info
{
    kfft_ kfft;
    UINTP count; // used for Real FFT solvers: at any time, r2hc->count + 2 * r2hcf->count + 2 * c2c->count = vecs->n
    UINT8 sets;
} kernel_info_t;

// Thread information structure holds the threading related information for the
// solution of given problem
typedef struct thread_info
{
    aoclfftz_smp_pfft_t *pthr_fft; // Thread information from problem descriptor
    INT32 avl_threads;  // Available number of threads at any point of execution
    INT32 n_threads;    // Number of threads assigned to a particular solver
} thread_info_t;

// Solver execute template function pointer
typedef INT32 (*dft_solver_)(aoclfftz_solution_t *solution);

// Executor function pointer
typedef INT32 (*execute_)(aoclfftz_executor_t *executor_obj);

// Base data structure acting as an abstract class that is derived by the
// top-level DFT data structure and implemented by all the solvers
typedef struct aoclfftz_generic_solver
{
    INT32 solver_type;
    dft_solver_ execute_solver;
    VOID (*destroy_solver)(aoclfftz_solution_t *solution);
    kernel_info_t *kernel_c2c;
    kernel_info_t *kernel_c2c_r; // Used by batched_ct_l1_direct solver only
    kernel_info_t *kernel_r2hc;
    kernel_info_t *kernel_r2hcf;
} aoclfftz_generic_solver_t;

// Holds info on the main problem or decomposed sub-problem in current dimension
typedef struct aoclfftz_decomp_scheme
{
    INT32 vec_rank;
    INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    aoclfftz_dim_t_64_ *batched_vecs; // used in batched-direct solver, otherwise NULL
    // VOID *in;
    VOID *in_real;
    VOID *in_imag;
    // VOID *out;
    VOID *out_real;
    VOID *out_imag;
    aoclfftz_cntrl_params_t *cntrl_params;
    thread_info_t *thread_info;
    UINT32 decomp_level;
    // Application side flag bits
    //   bit 0: (0) in-place / (1) out-of-place
    //   bit 1: (0) in-order / (1) out-of-order
    //   bit 2: (0) forward  / (1) backward
    //   bit 3: (0) complex  / (1) real
    //   bit 4: (bit reproducibility) (0) disabled / (1) enabled
    // Library side internal flag bits
    //  transpose (standalone) (no DFT): 8th-bit
    //  transpose (alongside DFT): 9th-bit
    //   bit 8     : (0) no-transpose / (1) transpose
    //   bit 9     : (0) (transpose+fft) / (1) fft (no transpose)
    //   bit 10    : (0) innermost dimension / (1) not innermost dimension (of ND-dim problem)
    //   bit 11    : (0) not buffered / (1) buffered
    //   bit 16    : (0) fixed selector mode / (1) auto tuner selector mode
    //   bit 30-31 : floating point datatype precision
    //               (00) 8-bit / (01) 16-bit / (10) 32-bit / (11) 64-bit
    UINT32 flags;
} aoclfftz_decomp_scheme_t;

// TW Holds twiddle factors used by a specific kernel for the given problem
// twiddle_buf_ptr holds the pointer to the twiddle buffer (TW), that is used to
// free the memory allocated for TW in a post processed solution, where the same
// memory pointer is used across array of next solutions.
typedef struct aoclfftz_twiddle
{
    VOID *twiddle_buf_ptr; /*< pointer to owned twiddle buffer. It has to be allocated/freed with current struct. */
    VOID *TW; /*< pointer to shared twiddle buffer. It must not be freed/allocated with the struct. */
    UINTP cols; /*< number of columns/batches/sets that are used to set up the twiddle buffer in a CT problem */
    UINTP load_multi_cols; /*< determines whether multiple columns are to be loaded from the twiddle buffer per iteration in the twiddle kernels */
} aoclfftz_twiddle_t;

// Holds bluestein sequence B used by the bluestein solver
// When FFT is computed for B, it will be stored in B_out and
// is_B_out_valid will be set to 1.
// Also holds the internal input and output buffers.
typedef struct aoclfftz_bluestein
{
    VOID *B;
    VOID *B_out;
    VOID *in;
    VOID *out;
    UINT8 is_B_out_valid;
} aoclfftz_bluestein_t;

typedef struct aoclfftz_buffered
{
    VOID *aux_buffer_1;
    VOID *aux_buffer_2;
    // this is used to store the address of last direct solution's output buffer
    // NOTE: This is required since we cannot immediately get the address of the
    //       last node from one of the starting nodes.
    //       It can be avoided if we introduce support circular doubly
    //       linked-list or an additional field to point dependend non-next
    //       nodes from the current solution.
    VOID **out_ptr;
} aoclfftz_buffered_t;

// Holds split-radix solver specific sub-solutions and buffers.
// The SR algorithm decomposes an N-point FFT into:
//   - Even: N/2-point sub-problem (stored via next_sol[0])
//   - Odd1: N/4-point sub-problem (indices 1, 5, 9, ...)
//   - Odd3: N/4-point sub-problem (indices 3, 7, 11, ...)
// The input_copy buffer is used for in-place transforms to preserve input data
// before the sub-problems overwrite the output buffer.
typedef struct aoclfftz_sr
{
    aoclfftz_solution_t *odd1_sol;  // N/4-point sub-solution for odd-1 indices
    aoclfftz_solution_t *odd3_sol;  // N/4-point sub-solution for odd-3 indices
    VOID *input_copy;               // Pre-allocated buffer for in-place input safety copy
    INTP  input_copy_size;          // Size of input_copy buffer in bytes
} aoclfftz_sr_t;

// Internal types to denote complex numbers in fftz's transpose routines
typedef struct aoclfftz_complex_f
{
    FLOAT real, imag;
} aoclfftz_complex_f_t;

typedef struct aoclfftz_complex_d
{
    DOUBLE real, imag;
} aoclfftz_complex_d_t;

typedef enum aoclfftz_transpose_dtype
{
    // enum value : [3 bit value] = (datatype flag value)(is real)
    TYPE_FLOAT = (2 << 1) | 1,         // 101
    TYPE_FLOATCOMPLEX = (2 << 1) | 0,  // 100
    TYPE_DOUBLE = (3 << 1) | 1,        // 111
    TYPE_DOUBLECOMPLEX = (3 << 1) | 0, // 110
} aoclfftz_transpose_dtype;

// A data structure to track the visited locations in a matrix
typedef struct aoclfftz_transpose_aux_mem
{
    UINT8 *data;
    INTP size;
} aoclfftz_transpose_aux_mem_t;

// function pointer compatible with all transpose kernel function signatures
typedef void (*aoclfftz_transpose_kernel)(VOID *, VOID *, aoclfftz_dim_t_64_,
                                          aoclfftz_dim_t_64_,
                                          aoclfftz_transpose_aux_mem_t *);

typedef struct aoclfftz_transpose
{
    aoclfftz_dim_t_64_ row_info;
    aoclfftz_dim_t_64_ col_info;
    aoclfftz_transpose_kernel kernel;
    aoclfftz_transpose_aux_mem_t *aux_mem;
} aoclfftz_transpose_t;

/////////////////////////// STRIDE RELATED : START ////////////////////////////

// Holds element-wise and radix-wise strides of the sub-problem decomposition
// that is acted upon by a specific kernel
typedef struct aoclfftz_strides
{
    INTP *in_strides;
    INTP *out_strides;
    INTP v_in_stride;
    INTP v_out_stride;
} aoclfftz_strides_t;

typedef struct aoclfftz_strides_grp
{
    aoclfftz_strides_t* strides;        // for complex Kernels
    aoclfftz_strides_t* strides_c2c;    // for real C2C Kernels
    aoclfftz_strides_t* strides_r2hc;   // for real R2HC Kernels
    aoclfftz_strides_t* strides_r2hcf;  // for real R2HC-Fused Kernels
    aoclfftz_strides_t* strides_c2r_ct_op; /* used for real C2R out-of-place CT problems;
                                              to avoid modifying the input buffer of the first CT stage,
                                              conjugated values are stored in the auxiliary buffer,
                                              `strides_c2r_ct_op` will hold input strides as same as strides_c2c
                                              and output strides as unit-strides to align with auxiliary buffer.
                                            */
} aoclfftz_strides_grp_t;

/////////////////////////// STRIDE RELATED : END //////////////////////////////

/////////////////////////// BUFS RELATED : START //////////////////////////////
typedef struct aoclfftz_dft_bufs
{
    aoclfftz_bluestein_t* bluestein;
    aoclfftz_buffered_t* buffered;
    aoclfftz_transpose_t* transpose;
    aoclfftz_solution_t* nd_sol; // may hold one of the solutions of ND
    aoclfftz_sr_t* sr; // split-radix solver specific data (sub-solutions + buffers)
    VOID *ct_buffer; // auxiliary buffer for CT problems
    VOID *ct_buf_real; // real part of ct_buffer
    VOID *ct_buf_imag; // imaginary part of ct_buffer
    VOID *ct_buf_real_in; /* update `ct_buf_real_in` pointers used to store the modified input in the
                             first CT stage for C2R out-of-place CT problems.
                             use the auxiliary buffer which is not used for computation in that stage
                             to store the modified input.
                           */
    INTP ct_buf_size; // size of ct_buffer per thread
    INT32 num_ct_buf; // number of ct_buffer allocated in total. It should be
                      // equal to the number of threads assigned to the first CT
                      // stage in the solution.
    UINT32 ct_buf_allocated; // to know that the solution originally allocated
                             // the buffer and is responsible for freeing it in
                             // the end.
} aoclfftz_dft_bufs_t;
/////////////////////////// BUFS RELATED : END ////////////////////////////////

// Solution data structure that is returned as a handle by the setup API and
// used by the execute API.
// Recommended memory layout recommendations: (use jemalloc)
// (aoclfftz_solution_t *sol) => nodes chained by next_sol links should come
// from a contiguous memory pool
// Elements within a node => solver->decomp_scheme->strides_grpdft_bufs shall
// come from contiguous memory region. twiddle (one-time separate alloc region)
typedef struct aoclfftz_solution
{
    aoclfftz_generic_solver_t *solver;
    aoclfftz_decomp_scheme_t *decomp_scheme;
    aoclfftz_strides_grp_t *strides_grp;
    aoclfftz_dft_bufs_t *dft_bufs;
    aoclfftz_twiddle_t *twiddle;
    aoclfftz_solution_t **next_sol;
} aoclfftz_solution_t;

// Helper data structure to store setup-time information related to real solvers
// and selectors.
typedef struct aoclfftz_realhelper
{
    INTP problem_size;
    INTP freq_factor; /*< frequency factor: For FWD (time->frequency conversion), it starts from 1 to problem_size. Reverse for BWD. */
    UINT32 stage;
    UINT8 is_last_stage;
    UINT8 is_CT;
    UINT8 is_buffered_invoked;
    INT32 num_aux_buf;
} aoclfftz_realhelper_t;

execute_ register_execute_dft(VOID);

#endif // AOCLFFTZ_INTERNAL_H
