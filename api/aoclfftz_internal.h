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

/*
 * Real FFT Cooley-Tukey execution order (single source of truth). Selected by the
 * SELECT_REAL_FFT_EXECUTION_ORDER CMake option, which maps to REAL_FFT_EXECUTION_ORDER:
 *   ITERATIVE         (0): legacy iterative mode; Direct-first traversal with SWAP reordering.
 *   PARTIAL_RECURSION (1): recursive CT-first tree; Direct nodes tail-chain via HAS_NEXT.
 *   TRUE_RECURSION    (2): CT solver orchestrates recurse-then-combine (Direct = pure leaf),
 *                          mirroring the Complex FFT CT solver traversal.
 * All three modes are numerically identical; they differ only in execution traversal.
 */
#define REAL_FFT_ORDER_ITERATIVE 0
#define REAL_FFT_ORDER_PARTIAL_RECURSION 1
#define REAL_FFT_ORDER_TRUE_RECURSION 2

#ifndef REAL_FFT_EXECUTION_ORDER
#define REAL_FFT_EXECUTION_ORDER REAL_FFT_ORDER_TRUE_RECURSION
#endif

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

#define SET_FFT_DIR(flags, val) SET_BIT_FLAG32(flags, 2, val)

#define SET_COMPLEX(flags) SET_BIT_FLAG32(flags, 3, 0)

#define SET_BIT_REPRODUCIBLE(flags, val) SET_BIT_FLAG32(flags, 4, val)
#define GET_BIT_REPRODUCIBLE(flags) GET_BIT_FLAG32(flags, 4)

#define SET_NOT_INNERMOST_DIM(flags) SET_BIT_FLAG32(flags, 10, 1)
#define IS_NOT_INNERMOST_DIM(flags) GET_BIT_FLAG32(flags, 10)

#define SET_BUFFERED(flags) SET_BIT_FLAG32(flags, 11, 1)
#define UNSET_BUFFERED(flags) SET_BIT_FLAG32(flags, 11, 0)
#define IS_BUFFERED(flags) GET_BIT_FLAG32(flags, 11)

// Per-node I/O role for the REAL (R2C/C2R) execute path. Storing the role
// (fixed at setup) instead of an absolute pointer keeps the tree read-only, so
// each caller resolves its own buffer via aoclfftz_resolve_real_io().
#define REAL_USE_IO_BUF       0 // read from / write to the handle's own input / output
                                // buffer, carried in ctx->in_real / ctx->out_real
#define REAL_USE_AUX_AND_SWAP 1 // read from / write to the aux buffers, carried in
                                // ctx->aux_pool_base_1 / ctx->aux_pool_base_2

// Get size of datatype based on the precision
#define DT_PRECISION_BYTES(dt_prec) (1 << dt_prec)

#define DT_SIZE(flags) DT_PRECISION_BYTES(DT_PRECISION_FLAG(flags))

/*
 * @brief Get size of datatype from solution, in bytes
 * */
#define SOL_DT_SIZE(sol) DT_SIZE(sol->decomp_scheme->flags)

// Get size of datatype from execution context, in bytes
#define CTX_DT_SIZE(ctx) DT_SIZE((ctx)->flags)

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
#define MOVE_ADDR(base_addr, offset)                                           \
    (FFTZ_VOID *)((FFTZ_CHAR *)base_addr + offset)

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
// Largest radix in the real kernel tables.
#define MAX_REAL_KERNEL_RADIX 16
// Total number of kernels in each category
#define NUM_KERNELS_IN_EACH_CATEGORY                                           \
  (NUM_STANDARD_KERNELS + NUMBER_OF_HIGHER_RADIX_KERNELS)

#define NUM_KERNELS_IN_EACH_DFT_VARIANT                                        \
    (NUM_KERNELS_IN_EACH_CATEGORY * NUM_KERNEL_CATEGORIES)

#define NUM_KERNELS_IN_TABLE_COMPLEX NUM_KERNELS_IN_EACH_DFT_VARIANT

#define NUM_KERNELS_IN_TABLE_REAL                                              \
    (NUM_KERNELS_IN_EACH_DFT_VARIANT * NUM_REAL_KERNELS_VARIANTS)

#define MAX_NUM_KERNELS_IN_TABLE                                               \
    NUM_KERNELS_IN_TABLE_REAL // max of real and complex

// Pow2 solver tunables:
// - below MIN_N the split yields <= 2 stages, which direct/CT already handle;
// - MAX_DECOMP_STAGES caps the radix stages any pow2 solver may plan.
#define POW2_ITERATIVE_MIN_N 512
#define POW2_MAX_DECOMP_STAGES 16

// Assumed cache sizes when CPUID does not report one.
#define DEFAULT_L1D_BYTES (32 * 1024)
#define DEFAULT_L2_BYTES (512 * 1024)

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
#define NUM_RFFT_GROUPS(solver)                                                \
    (solver)->kernel_r2hcf->count + (solver)->kernel_r2hc->count

// Compiler-portable atomics. Extend the branches below for new toolchains.
#if defined(__GNUC__) || defined(__clang__)

#define AOCLFFTZ_ATOMIC_CMP_XCHG(ptr, expected, desired)                       \
    __atomic_compare_exchange_n((ptr), (expected), (desired), 0,               \
                                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)

#define AOCLFFTZ_ATOMIC_STORE(ptr, value)                                      \
    __atomic_store_n((ptr), (value), __ATOMIC_RELEASE)

#else
#error "AOCL-FFTZ: atomics not supported on this compiler/platform."
#endif

// Forward declarations
typedef struct aoclfftz_solution aoclfftz_solution_t;
typedef struct aoclfftz_generic_solver aoclfftz_generic_solver_t;
typedef struct aoclfftz_strides aoclfftz_strides_t;
typedef struct aoclfftz_twiddle aoclfftz_twiddle_t;
typedef struct aoclfftz_bluestein aoclfftz_bluestein_t;
typedef struct aoclfftz_buffered aoclfftz_buffered_t;
typedef struct aoclfftz_sr aoclfftz_sr_t;
typedef struct aoclfftz_pow2_iterative aoclfftz_pow2_iterative_t;
typedef struct aoclfftz_pow2_fourstep aoclfftz_pow2_fourstep_t;
typedef struct aoclfftz_executor aoclfftz_executor_t;
typedef struct aoclfftz_realhelper aoclfftz_realhelper_t;

// Stack-local execution context passed through the solver tree.
// Holds per-call mutable state so the solution tree remains read-only.
typedef struct aoclfftz_mutable_ctx
{
    FFTZ_VOID *in_real;              // Input buffer real part
    FFTZ_VOID *in_imag;              // Input buffer imag part
    FFTZ_VOID *out_real;             // Output buffer real part
    FFTZ_VOID *out_imag;             // Output buffer imag part
    FFTZ_VOID *ct_buf_base;          // ct_buffer allocated by
                                     // BUFFERED/NDIM/CTL1D solvers
    FFTZ_VOID *bs_in_base;           // Bluestein per-call input scratch
    FFTZ_VOID *bs_out_base;          // Bluestein per-call output scratch
    FFTZ_VOID *sr_input_copy_base;   // Split-radix per-call input copy scratch
    FFTZ_VOID *pow2_buf_base;        // Pow2-iterative per-call ping-pong pool
    FFTZ_VOID *aux_pool_base_1;      // REAL_BUFFERED aux ping-pong pool 1; the
                                     // buffered node re-points it to this
                                     // thread's slice before descending
    FFTZ_VOID *aux_pool_base_2;      // REAL_BUFFERED aux ping-pong pool 2 (same
                                     // per-thread slicing as pool 1)
    FFTZ_VOID *aux_pool_base_ndim;   // REAL_NDIM C2R aux pool base
    FFTZ_VOID *c2c_strides_base;     // C2C stride scratch pool for the
                                     // single-threaded real Direct CT nodes; one
                                     // MAX_REAL_KERNEL_RADIX-entry slot per thread
                                     // that may run their C2C kernels
    FFTZ_VOID *transpose_aux_base;   // Standalone-transpose visited-cell bitmap
    FFTZ_INTP ct_offset;             // Byte offset into the ct_buffer,
                                     // accumulated per-thread by mt_batched
    FFTZ_UINT32 flags;               // Plan flags (direction, precision, etc.)
    FFTZ_INT32 slot_idx;             // Slices bs_[in/out]_base for Bluestein/MT_Bluestein,
                                     // the aux pools for REAL_BUFFERED/REAL_NDIM
                                     // and c2c_strides_base for real Direct CT.
} aoclfftz_mutable_ctx_t;

// Per-handle scratch byte sizes & the immutable execution context recorded at
// setup time. The scratch sizes are used by aoclfftz_execute_io to allocate a
// fresh per-call scratch slab so that concurrent application threads can share
// a single handle without trampling on each other's internal scratch.
//
// All sizes are in bytes. A zero value means the corresponding scratch
// region is not needed by this plan.
typedef struct aoclfftz_immutable_metadata
{
    FFTZ_UINTP bs_buffer_size;          // Total Bluestein pool size, summed
                                        // over all Bluestein nodes
    FFTZ_UINTP sr_input_copy_size;      // Split-radix in-place input copy
    FFTZ_UINTP pow2_buf_size;           // Pow2 scratch pool, shared: max over
                                        // the plan's iterative/four-step nodes
    FFTZ_INT32 pow2_buf_needs_zero;     // 1 when a four-step node shares the
                                        // pool and needs its pad lanes zeroed
    FFTZ_UINTP ct_buffer_total_size;    // CT scratch pool size for the owners
                                        // -> NDIM, BUFFERED, CTL1D
    FFTZ_UINTP aux_buffered_pool_size;  // REAL_BUFFERED aux ping-pong pool size,
                                        // two pools of this size are allocated
    FFTZ_UINTP aux_ndim_pool_size;      // REAL_NDIM C2R aux pool size
    FFTZ_UINTP c2c_strides_pool_size;   // Real direct solvers' C2C stride
                                        // scratch pool
    FFTZ_UINTP transpose_aux_size;      // Standalone-transpose bitmap size
    aoclfftz_mutable_ctx_t base_ctx;    // execution context built at setup time
    FFTZ_INT32 setup_buffers_acquired;  // 0 = setup-time buffers free; whoever
                                        // grabs them flips to 1, so others
                                        // allocate their own scratch
} aoclfftz_immutable_metadata_t;

// Computational cost analysis of solution of an executed problem/sub-problem
typedef struct cost_analysis
{
    FFTZ_INT64 ops;
    FFTZ_INT64 time;
} cost_analysis_t;

// Kernel template function pointer for performing FFT
typedef FFTZ_VOID (*kfft_)(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                      FFTZ_INTP n,
                      aoclfftz_strides_t *strides,
                      FFTZ_VOID *twd, FFTZ_UINT8 flag);

// Kernel information data structure holds the kernel function pointer and the
// number of sets it can process in parallel based on the kernel type(C/SIMD).
// This set information will be used by MT direct solver to find the kernel type
// (C/AVX128/AVX256/AVX512) at runtime, adjust the number of batch iteration and
// assign number of threads accordingly.
typedef struct kernel_info
{
    kfft_ kfft[NUM_FFT_DIRS]; // contains kernel function pointers for forward
                              // and backward directions
    FFTZ_UINTP count; // used for Real FFT solvers: at any time, r2hc->count + 2
                      // * r2hcf->count + 2 * c2c->count = vecs->n
    FFTZ_UINT8 sets;  // number of sets processable in parallel by kernel type
                      // (C/AVX-variants)
} kernel_info_t;

// Thread information structure holds the threading related information for the
// solution of given problem
typedef struct thread_info
{
    aoclfftz_smp_pfft_t *pthr_fft;  // Thread information from problem descriptor
    FFTZ_INT32 avl_threads;         // Available number of threads at any point of execution
    FFTZ_INT32 active_threads;      // number of threads active at this node (product of the
                                    // threads spawned by each MT_BATCHED level above it)
    FFTZ_INT32 n_threads;           // Number of threads assigned to a particular solver
    FFTZ_INT32 ndim_concurrency;    // Number of innermost NDIM instances
                                    // running concurrently in this subtree
} thread_info_t;

// Solver execute template function pointer
typedef FFTZ_INT32 (*dft_solver_)(aoclfftz_solution_t *solution,
                                  aoclfftz_mutable_ctx_t *ctx);

// Executor function pointer
typedef FFTZ_INT32 (*execute_)(aoclfftz_executor_t *executor_obj,
                               aoclfftz_mutable_ctx_t *ctx);

// Base data structure acting as an abstract class that is derived by the
// top-level DFT data structure and implemented by all the solvers
typedef struct aoclfftz_generic_solver
{
    FFTZ_INT32 solver_type;
    dft_solver_ execute_solver;
    FFTZ_VOID (*destroy_solver)(aoclfftz_solution_t *solution);
    kernel_info_t *kernel_c2c;
    kernel_info_t *kernel_c2c_r; // Used by batched_ct_l1_direct solver only
    kernel_info_t *kernel_r2hc;
    kernel_info_t *kernel_r2hcf;
} aoclfftz_generic_solver_t;

// Holds info on the main problem or decomposed sub-problem in current dimension
typedef struct aoclfftz_decomp_scheme
{
    FFTZ_INT32 vec_rank;
    FFTZ_INT32 dim_rank;
    aoclfftz_dim_t_64_ *dims;
    aoclfftz_dim_t_64_ *vecs;
    // used in batched-direct solver, otherwise NULL
    aoclfftz_dim_t_64_ *batched_vecs;
    // FFTZ_VOID *in;
    FFTZ_VOID *in_real;
    FFTZ_VOID *in_imag;
    // FFTZ_VOID *out;
    FFTZ_VOID *out_real;
    FFTZ_VOID *out_imag;
    aoclfftz_cntrl_params_t *cntrl_params;
    thread_info_t *thread_info;
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
    //   bit 10    : (0) innermost dimension / (1) not innermost dimension (of
    //   ND-dim problem) bit 11    : (0) not buffered / (1) buffered bit 16    :
    //   (0) fixed selector mode / (1) auto tuner selector mode bit 30-31 :
    //   floating point datatype precision
    //               (00) 8-bit / (01) 16-bit / (10) 32-bit / (11) 64-bit
    FFTZ_UINT32 flags;
    // Real forward R2C post-process: precomputed DC/Nyquist imag slot indices.
    FFTZ_INTP nyquist_im_offset_direct; // per transform; direct R2C zero
    FFTZ_INTP nyquist_im_offset_ct;     // full output span; CT R2C last-stage
                                        // zero.
    // Per-node I/O roles for the REAL execute path (setup-time constants).
    // REAL_USE_IO_BUF: read from / write to the handle's own input / output buffers
    // REAL_USE_AUX_AND_SWAP: read from / write to the aux buffers
    FFTZ_UINT8 real_in_role;
    FFTZ_UINT8 real_out_role;
} aoclfftz_decomp_scheme_t;

// TW Holds twiddle factors used by a specific kernel for the given problem
// twiddle_buf_ptr holds the pointer to the twiddle buffer (TW), that is used to
// free the memory allocated for TW in a post processed solution, where the same
// memory pointer is used across array of next solutions.
typedef struct aoclfftz_twiddle
{
    FFTZ_VOID *twiddle_buf_ptr; /*< pointer to owned twiddle buffer. It has to
                                   be allocated/freed with current struct. */
    FFTZ_VOID *TW;   /*< pointer to shared twiddle buffer. It must not be
                        freed/allocated with the struct. */
    FFTZ_UINTP load_multi_cols; /*< determines whether multiple columns are to
                                   be loaded from the twiddle buffer per
                                   iteration in the twiddle kernels */
} aoclfftz_twiddle_t;

// Function pointer for elementwise multiplication kernels.
// Two direction-specialized variants are stored in mul[NUM_FFT_DIRS]:
// mul[FORWARD_FFT_DIR] computes a .* conj(b), mul[BACKWARD_FFT_DIR]
// computes a .* b. start_idx and stride are used by strided variants.
typedef FFTZ_VOID (*elementwise_mul_)(FFTZ_VOID *out, FFTZ_VOID *a, 
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_INTP start_idx, FFTZ_INTP stride);

// out[i*out_stride] = factor * (a[i] (*) B[i]) where factor = 1/m (used for
// normalisation)
typedef FFTZ_VOID (*elementwise_mul_fused_norm_)(FFTZ_VOID *out, FFTZ_VOID *a,
                                                 FFTZ_VOID *b, FFTZ_INTP n,
                                                 FFTZ_DOUBLE factor,
                                                 FFTZ_INTP out_stride);

// Conversion kernels for the real Bluestein solver, which operates on complex
// data and therefore converts its input on entry and its result on exit.
// Selected per plan from cpu_flags and precision.
//
// Each name reads source2destination, where r denotes reals, c denotes
// complex (all n points) and hc denotes half complex (the n/2+1 points that
// represent a real spectrum). All share the signature (dst, src, n, stride):
//   r2c  : n reals -> n complex, imaginary parts zeroed
//   c2hc : retains the first n/2+1 points, discards the remainder
//   hc2c : n/2+1 points -> all n, via X[n-k] = conj(X[k])
//   c2r  : n complex -> the real part of each
typedef FFTZ_VOID (*type_convert_)(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                                   FFTZ_INTP stride);

// Fused four-step inter-step twiddle + transpose in a single pass over the
// n1 x n2 source: every element is read once and written once, and the
// pre-blocked twiddle table is walked by one forward-moving pointer (no
// non-temporal stores are involved). Row strides are in complex elements,
// twiddle row stride is n2:
//   out[j*out_row_stride + k1] = in[k1*in_row_stride + j] * twiddles[k1*n2 + j]
typedef FFTZ_VOID (*fused_twiddle_transpose_)(FFTZ_VOID *in, FFTZ_VOID *out,
                                        FFTZ_VOID *twiddles, FFTZ_INTP n1,
                                        FFTZ_INTP n2, FFTZ_INTP in_row_stride,
                                        FFTZ_INTP out_row_stride);

// Holds the Bluestein chirp sequence B and its FFT B_out (computed once
// during plan setup), plus elementwise-multiply/fused_norm_multiply kernels
// bound at plan setup.
//
// B/B_out are allocated (non-NULL) only on the owning node and are read at
// execution time.
typedef struct aoclfftz_bluestein
{
    FFTZ_VOID *B;
    FFTZ_VOID *B_out;
    // Step 1: input × chirp (B). Unit or strided-in ele_mul, bound at setup.
    elementwise_mul_ pre_mul[NUM_FFT_DIRS];
    // Step 2b: FFT result × chirp FFT (B_out). Unit-stride ele_mul.
    elementwise_mul_ mul[NUM_FFT_DIRS];
    // Step 3: (1/m) * output * chirp (post_mul[] on B);
    elementwise_mul_fused_norm_ post_mul[NUM_FFT_DIRS];
    // Conversion kernels for a real Bluestein node; NULL on complex nodes. A
    // plan executes a single direction, so the selector binds only the pair
    // that direction requires.
    // problem layout -> n complex, pre-process
    type_convert_ cast_to_complex;
    // n complex -> problem layout, post-process
    type_convert_ cast_from_complex;
    FFTZ_INTP bs_buf_size;   // bytes per per-call bs_in/out scratch slot
    FFTZ_INTP bs_dim_offset; // byte offset of this dim in bs scratch pool
} aoclfftz_bluestein_t;

typedef struct aoclfftz_buffered
{
    FFTZ_VOID *aux_buffer_1;
    FFTZ_VOID *aux_buffer_2;
    // 1: this node allocated aux_buffer_1/2 (must free);
    // 0: offset / alias into shared pool.
    FFTZ_UINT8 is_aux_buffer_allocated;
    // Padded aux_buffer size (REAL_NDIM / REAL_BUFFERED) per thread; 0 if
    // unused.
    FFTZ_INTP aux_buf_size_per_thread;
} aoclfftz_buffered_t;

// Holds split-radix solver specific sub-solutions and buffers.
// The SR algorithm decomposes an N-point FFT into:
//   - Even: N/2-point sub-problem (stored via next_sol)
//   - Odd1: N/4-point sub-problem (indices 1, 5, 9, ...)
//   - Odd3: N/4-point sub-problem (indices 3, 7, 11, ...)
// The input_copy buffer is used for in-place transforms to preserve input data
// before the sub-problems overwrite the output buffer.
typedef struct aoclfftz_sr
{
    aoclfftz_solution_t *odd1_sol;  // N/4-point sub-solution for odd-1 indices
    aoclfftz_solution_t *odd3_sol;  // N/4-point sub-solution for odd-3 indices
    // Pre-allocated buffer for in-place input safety copy
    FFTZ_VOID *input_copy;
    FFTZ_INTP  input_copy_size;          // Size of input_copy buffer in bytes
} aoclfftz_sr_t;

// Internal types to denote complex numbers in fftz's transpose routines
typedef struct aoclfftz_complex_f
{
    FFTZ_FLOAT real, imag;
} aoclfftz_complex_f_t;

typedef struct aoclfftz_complex_d
{
    FFTZ_DOUBLE real, imag;
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
    FFTZ_UINT8 *data;
    FFTZ_INTP size;
} aoclfftz_transpose_aux_mem_t;

// function pointer compatible with all transpose kernel function signatures
typedef void (*aoclfftz_transpose_kernel)(FFTZ_VOID *, FFTZ_VOID *,
                                          aoclfftz_dim_t_64_,
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
    FFTZ_INTP *in_strides;
    FFTZ_INTP *out_strides;
    FFTZ_INTP v_in_stride;
    FFTZ_INTP v_out_stride;
    FFTZ_INTP v_in_h2_stride;
    FFTZ_INTP v_out_h2_stride;
} aoclfftz_strides_t;

typedef struct aoclfftz_strides_grp
{
    aoclfftz_strides_t* strides;        // for complex Kernels
    aoclfftz_strides_t* strides_c2c;    // for real C2C Kernels
    aoclfftz_strides_t* strides_r2hc;   // for real R2HC Kernels
    aoclfftz_strides_t* strides_r2hcf;  // for real R2HC-Fused Kernels
} aoclfftz_strides_grp_t;

/////////////////////////// STRIDE RELATED : END //////////////////////////////

// What every pow2 radix stage needs: the kernel and how it walks memory. The
// four-step solver plans nothing beyond this, so it uses the struct as is; the
// iterative solver embeds it and adds its group walk.
typedef struct aoclfftz_pow2_stage
{
    FFTZ_INTP  radix;     // stage radix
    kfft_ kfft[NUM_FFT_DIRS]; // stage kernel, per FFT direction: a node can
                              // be executed either way (e.g. Bluestein's
                              // inner FFT)
    FFTZ_VOID *twiddle;   // stage twiddle table (NULL for the leaf stage)
    aoclfftz_strides_t strides; // radix + element strides, precomputed at setup
                                // (in/out arrays heap-owned per stage) so the
                                // executor only replays them
} aoclfftz_pow2_stage_t;

// One radix stage of the power-of-2 iterative solver (SOLVER_POW2_ITERATIVE):
// the shared stage plus the group walk this solver drives it with.
typedef struct aoclfftz_pow2_iterative_stage
{
    aoclfftz_pow2_stage_t stage_info;
    FFTZ_INTP  sets;      // kernel register width (sets processed in parallel)
    FFTZ_INTP  count;          // element count argument passed to the kernel
    FFTZ_INTP  num_groups;     // number of kernel invocations for this stage
    FFTZ_INTP  src_grp_stride; // byte offset between consecutive source groups
    FFTZ_INTP  dst_grp_stride; // byte offset between consecutive destination
                               // groups
} aoclfftz_pow2_iterative_stage_t;

// Fused leaf solver: `stages` is a flat array, not solution-tree nodes, so the
// tree walkers cannot see it. `aoclfftz_pow2_iterative_stage` lists the
// per-stage state that this solver builds itself.
typedef struct aoclfftz_pow2_iterative
{
    aoclfftz_pow2_iterative_stage_t *stages;  // [num_stages], leaf first
    FFTZ_VOID *pingpong_buf; // solver-owned ping-pong pool: one slot of two
                             // buffers (A and B) per concurrent thread
    FFTZ_INTP buf_bytes;     // bytes in one buffer (A or B)
    FFTZ_UINTP pool_bytes;   // bytes in the whole pool (slots * 2 * buf_bytes)
    FFTZ_INT32 num_stages;
} aoclfftz_pow2_iterative_t;

// Power-of-2 four-step solver (SOLVER_POW2_FOURSTEP). Like the iterative
// solver it is a fused leaf: `stages` below are flat arrays, not tree nodes.
// - split   : N = n1 * n2, balanced (n1 >= n2); input read as an n1 x n2 matrix
// - sub1    : n1-point FFT down each of the n2 columns, written transposed
// - fused   : inter-step twiddle multiply + transpose, in a single pass
// - sub2    : n2-point FFT down each of the n1 columns, output natural order
// - selected when N spills the iterative solver's 2x L1D budget but n1 still
//   fits L2 (see is_pow2_solvable)
// - declined at setup, falling through to CT, when no fused kernel exists or
//   n1/n2 are not micro-tile multiples
typedef struct aoclfftz_pow2_fourstep_subfft
{
    FFTZ_INTP  fft_len;    // sub-FFT length (n1 for sub1, n2 for sub2)
    FFTZ_INTP  num_cols;   // contiguous columns batched (n2 for sub1, n1 for
                           // sub2)
    FFTZ_INTP  row_stride; // padded distance (complex elements) to the next
                           // row; >= num_cols
    FFTZ_INT32 num_stages;
    aoclfftz_pow2_stage_t *stages; // [num_stages], leaf first
} aoclfftz_pow2_fourstep_subfft_t;

typedef struct aoclfftz_pow2_fourstep
{
    FFTZ_INTP n1; // outer factor (sub1 length)
    FFTZ_INTP n2; // inner factor (sub2 length)

    // sub1: n1-point FFT down each of n2 columns.
    aoclfftz_pow2_fourstep_subfft_t sub1;
    // sub2: n2-point FFT down each of n1 columns.
    aoclfftz_pow2_fourstep_subfft_t sub2;

    FFTZ_VOID *step_twiddles; // inter-step twiddle factor table (n1 x n2)
    // Fused step2 (twiddle) + step3 (transpose), per FFT direction.
    fused_twiddle_transpose_ fused_twiddle_transpose[NUM_FFT_DIRS];

    FFTZ_VOID *scratch;    // solver-owned ping-pong pool, recycled by the whole
                           // pipeline: one slot of two padded buffers (A and B)
                           // per active thread, each thread using the slot at
                           // its ctx->slot_idx. The gate admits single-threaded
                           // plans only, so today that is one slot; concurrent
                           // execute_io calls get their own copy of the pool
                           // instead of a slot.
    FFTZ_INTP buf_bytes;   // bytes in one padded buffer (A or B)
    FFTZ_UINTP pool_bytes; // bytes in the whole pool
                           // (active_threads * 2 * buf_bytes)
} aoclfftz_pow2_fourstep_t;

/////////////////////////// BUFS RELATED : START //////////////////////////////
typedef struct aoclfftz_dft_bufs
{
    aoclfftz_bluestein_t* bluestein;
    aoclfftz_buffered_t* buffered;
    aoclfftz_transpose_t* transpose;
    aoclfftz_solution_t* nd_sol; // may hold one of the solutions of ND
    aoclfftz_sr_t *sr;
    aoclfftz_pow2_iterative_t* pow2_iterative;
    // four-step pow2 solver specific data (sub-FFT setups + buffers);
    // heap-allocated on demand (NULL otherwise)
    aoclfftz_pow2_fourstep_t* pow2_fourstep;
    FFTZ_VOID *ct_buffer; // auxiliary buffer for CT problems
    FFTZ_VOID *ct_buf_real; // real part of ct_buffer
    FFTZ_VOID *ct_buf_imag; // imaginary part of ct_buffer
    FFTZ_INTP ct_buf_size; // 64-byte aligned size per ct_buf / thread slot
    FFTZ_UINT32 ct_buf_allocated; // to know that the solution originally
                                  // allocated the buffer and is responsible for
                                  // freeing it in the end.
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
    aoclfftz_solution_t *next_sol;
} aoclfftz_solution_t;

// Helper data structure to store setup-time information related to real solvers
// and selectors.
typedef struct aoclfftz_realhelper
{
    FFTZ_INTP problem_size;
    /** frequency factor: For FWD (time->frequency conversion), it starts from 1
     * to problem_size. Reverse for BWD. */
    FFTZ_INTP freq_factor;
    FFTZ_UINT32 stage;
    FFTZ_UINT8 is_last_stage;
    FFTZ_UINT8 is_CT;
    FFTZ_UINT8 is_buffered_invoked;
} aoclfftz_realhelper_t;

execute_ register_execute_dft(FFTZ_VOID);

#endif // AOCLFFTZ_INTERNAL_H
