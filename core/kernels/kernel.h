// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file kernel.h
 *
 *  @brief Kernel template and related routines for computing DFT computations.
 *
 *  This file defines the kernel template that is used to statically derive the
 *  kernels of different precisions (float and double) related to all the
 *  compute types (C, AVX128, AVX256, AVX512).
 *
 *  @note Different variants of data structures are defined to
 *  support float and double precision types by default in ILP64 data model.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 *  @author Ashwin K. Godbole
 */

#ifndef AOCLFFTZ_KERNEL_H
#define AOCLFFTZ_KERNEL_H

#include "api/aoclfftz_internal.h"
#include "utils/utils.h"

// macro to make in/out stride array volatile
// hack to prevent compiler optimization to prevent register to stack movements
// on the variable
#define VOLATILE_STRIDE_ARRAY

// Implies the number of sets that can be processed in parallel.
// Computed using - (register_width / (2 * 8 * sizeof(floating point)))
#define NUM_SETS_C_S 1
#define NUM_SETS_C_D 1
#define NUM_SETS_128_S 2
#define NUM_SETS_128_D 1
#define NUM_SETS_256_S 4
#define NUM_SETS_256_D 2
#define NUM_SETS_512_S 8
#define NUM_SETS_512_D 4
#define NUM_SETS_REAL_C_S 1
#define NUM_SETS_REAL_C_D 1
#define NUM_SETS_REAL_128_S 4
#define NUM_SETS_REAL_128_D 2
#define NUM_SETS_REAL_256_S 8
#define NUM_SETS_REAL_256_D 4
#define NUM_SETS_REAL_512_S 16
#define NUM_SETS_REAL_512_D 8

// Error return codes related to Kernel
// Add more codes at the top
typedef enum
{
    KERNEL_FAILURE = -1,
    KERNEL_SUCCESS // Successful operation
} aoclfftz_kernel_status;

// Kernel types
typedef enum
{
    R2HC_KERNEL = 0,
    R2HCF_KERNEL,
    C2C_KERNEL,
} aoclfftz_kernel_type;

// Holds the kernel level operational complexity in terms approximate cycles
typedef struct ops_cycles
{
    FFTZ_USHORT fma;
    FFTZ_USHORT mul;
    FFTZ_USHORT add;
    FFTZ_USHORT move;
    FFTZ_USHORT perm;
    FFTZ_USHORT other;
} ops_cycles_t;

// Function pointer to get kernel compute operations in terms of approx cycles
typedef ops_cycles_t (*k_ops_cnt_)(FFTZ_UINT8, FFTZ_UINT8);
typedef kfft_ (*k_register_kernel_)(FFTZ_UINT8, FFTZ_UINT8);

// Kernel data structure that holds kernel function pointers and other
// associated parameters related to radix and compute operations
typedef struct kernel
{
    // Contains kernel function pointers for forward and backward directions.
    kfft_ kfft[NUM_FFT_DIRS];
    k_ops_cnt_ k_ops_cnt;
    FFTZ_UINT32 radix;
    aoclfftz_kernel_type kernel_type;
    FFTZ_UINT8 sets[NUM_PRECISIONS];
} kernel_t;

// Data structure containing kernel function pointers corresponding to the
// registration, and operation count of the kernel
typedef struct kernel_fp_list
{
    k_register_kernel_ k_register_kernel;
    k_ops_cnt_ k_ops_cnt;
    FFTZ_UINT32 radix;
} kernel_fp_list_t;

// Group of kernel tables holding different variants
// ele_mul and normalize hold the elementwise multiplication and normalization
// kernels selected for the plan based on cpu_flags, registered once by
// register_solvers_kernels.
typedef struct kernel_tables
{
    kernel_t *kt_dft;
    kernel_t *kt_twid_dft;
    kernel_t *kt_rdft;
    elementwise_mul_ ele_mul[NUM_FFT_DIRS];
    normalize_ normalize;
} kernel_tables_t;

// Function declarations for the common routines
FFTZ_INT32 register_kernels_real(
    kernel_t kertab[NUM_KERNELS_IN_TABLE_REAL],
    kernel_fp_list_t static_kernel_table[NUM_REAL_KERNELS_VARIANTS]
                                        [NUM_KERNELS_IN_EACH_CATEGORY]
                                        [NUM_KERNEL_CATEGORIES],
    FFTZ_INT32 dt, FFTZ_INT32 dir, FFTZ_INT32 cpu_flags);

// Register complex kernels into `kertab`. `static_kernel_table` populates
// bidirectional `kfft` or `kfft[FORWARD_FFT_DIR]`.
// The optional `static_kernel_table_bwd` populates `kfft[BACKWARD_FFT_DIR]`
// when present; otherwise the backward slot aliases the forward slot so both
// array entries are always non-NULL.
FFTZ_INT32 register_kernels_complex(
    kernel_t kertab[NUM_KERNELS_IN_TABLE_COMPLEX],
    kernel_fp_list_t static_kernel_table[NUM_KERNELS_IN_EACH_CATEGORY]
                                        [NUM_KERNEL_CATEGORIES],
    kernel_fp_list_t static_kernel_table_bwd[NUM_KERNELS_IN_EACH_CATEGORY]
                                            [NUM_KERNEL_CATEGORIES],
    FFTZ_INT32 dt, FFTZ_INT32 dir, FFTZ_INT32 cpu_flags);

// Selects the elementwise multiplication kernel variant for the given
// cpu_flags, data type and direction. direction == FORWARD_FFT_DIR returns
// the forward (a .* conj(b)) kernel; direction == BACKWARD_FFT_DIR returns
// the backward (a .* b) kernel. Called twice per plan from
// register_solvers_kernels (once for fwd, once for bwd).
elementwise_mul_ register_elementwise_mul_kernel(FFTZ_INT32 cpu_flags,
                                                 FFTZ_INT32 dt,
                                                 FFTZ_UINT8 direction);

// Selects the normalization kernel variant for the given cpu_flags and data
// type. Called once per plan from register_solvers_kernels.
normalize_ register_normalize_kernel(FFTZ_INT32 cpu_flags, FFTZ_INT32 dt);

// Kernel function declarations for different floating point precision types
// supported in scalar and vector compute variants

// Get Operations Count Functions

// C2C Kernels
ops_cycles_t get_ops_cnt_fft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft5c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft6c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft7c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft8c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft9c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft11c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft12c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft13c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft14c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft15c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft20c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft48c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

// Direction-specific C2C Twiddle Kernels
ops_cycles_t get_ops_cnt_twid_fwd_fft2c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft3c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft4c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft5c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft7c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft8c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft9c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft10c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft11c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft12c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft13c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft14c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft15c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft16c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft2c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft3c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft4c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft5c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft7c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft8c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft9c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft10c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft11c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft12c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft13c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft14c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft15c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft16c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft2c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft3c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft4c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft5c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft7c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft8c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft9c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft10c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft11c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft12c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft13c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft14c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft15c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft16c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft2c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft3c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft4c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft5c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft7c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft8c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft9c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft10c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft11c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft12c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft13c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft14c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft15c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft16c(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);

// R2HC Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft13c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);

// R2HC-Fused Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft9c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft11c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft12c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft13c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);

// Register kernels

// C2C Kernels
kfft_ register_kernel_fft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft5c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft6c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft7c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft8c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft9c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft11c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft12c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft13c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft14c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft15c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft20c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft48c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

kfft_ register_kernel_twid_fwd_fft2c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft3c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft4c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft5c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft13c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft2c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft3c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft4c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft5c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft13c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft2c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft3c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft4c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft5c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft13c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft2c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft3c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft4c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft5c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft13c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction);

elementwise_mul_ register_elementwise_mul_c(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
normalize_ register_normalize_c(FFTZ_UINT8 precision);

// R2HC Kernels
kfft_ register_kernel_r2hc_rfft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft5c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft6c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft7c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft8c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft9c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft11c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft12c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft13c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft14c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft15c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

// R2HC-Fused Kernels
kfft_ register_kernel_r2hcf_rfft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft9c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft11c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft12c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft13c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

// AVX Kernels
#ifdef ENABLE_AVX128
ops_cycles_t get_ops_cnt_fft2avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft20avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft48avx128(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_fwd_fft2avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft3avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft4avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft5avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft6avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft7avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft8avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft9avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft10avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft11avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft12avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft13avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft14avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft15avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft16avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_fwd_fft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft4avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft5avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft6avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft7avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft8avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft9avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft10avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft11avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft12avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft13avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft14avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft15avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft16avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_bwd_fft2avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft3avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft4avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft5avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft6avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft7avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft8avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft9avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft10avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft11avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft12avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft13avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft14avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft15avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft16avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_bwd_fft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft4avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft5avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft6avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft7avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft8avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft9avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft10avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft11avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft12avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft13avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft14avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft15avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft16avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft2avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft3avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft4avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft5avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft6avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft7avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft8avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft9avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft10avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft11avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft12avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft13avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft14avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft15avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft16avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft4avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft5avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft6avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft7avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft8avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft9avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft10avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft11avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft12avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft13avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft14avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft15avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft16avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft2avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft3avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft4avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft5avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft6avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft7avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft8avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft9avx128(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft10avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft11avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft12avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft13avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft14avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft15avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft16avx128(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft4avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft5avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft6avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft7avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft8avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft9avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft10avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft11avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft12avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft13avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft14avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft15avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft16avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC AVX128 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft9avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft11avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft12avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft13avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft16avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC-Fused AVX128 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft9avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft11avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft13avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx128(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);

kfft_ register_kernel_fft2avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft3avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft4avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft5avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft6avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft7avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft8avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft9avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft10avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft11avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft12avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft13avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft14avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft15avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft16avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft20avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft48avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

elementwise_mul_ register_elementwise_mul_avx128(FFTZ_UINT8 precision,
                                                 FFTZ_UINT8 direction);
normalize_ register_normalize_avx128(FFTZ_UINT8 precision);

// R2HC AVX128 Kernels
kfft_ register_kernel_r2hc_rfft2avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft9avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft11avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft12avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft13avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft16avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);

// R2HC-Fused AVX128 Kernels
kfft_ register_kernel_r2hcf_rfft2avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft9avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft11avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft12avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft13avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);

#endif

#ifdef ENABLE_AVX256
ops_cycles_t get_ops_cnt_fft2avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft20avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft48avx256(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_fwd_fft2avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft3avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft4avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft5avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft6avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft7avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft8avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft9avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft10avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft11avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft12avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft13avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft14avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft15avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft16avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_fwd_fft2avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft3avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft4avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft5avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft6avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft7avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft8avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft9avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft10avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft11avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft12avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft13avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft14avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft15avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft16avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_bwd_fft2avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft3avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft4avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft5avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft6avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft7avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft8avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft9avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft10avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft11avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft12avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft13avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft14avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft15avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft16avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_bwd_fft2avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft3avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft4avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft5avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft6avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft7avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft8avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft9avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft10avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft11avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft12avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft13avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft14avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft15avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft16avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft2avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft3avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft4avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft5avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft6avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft7avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft8avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft9avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft10avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft11avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft12avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft13avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft14avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft15avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft16avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft2avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft3avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft4avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft5avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft6avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft7avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft8avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft9avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft10avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft11avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft12avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft13avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft14avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft15avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft16avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft2avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft3avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft4avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft5avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft6avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft7avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft8avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft9avx256(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft10avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft11avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft12avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft13avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft14avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft15avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft16avx256(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft2avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft3avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft4avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft5avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft6avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft7avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft8avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft9avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft10avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft11avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft12avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft13avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft14avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft15avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft16avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC AVX256 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft9avx256(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft11avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft12avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft13avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft16avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC-Fused AVX256 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft9avx256(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft11avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft13avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx256(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);

kfft_ register_kernel_fft2avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft3avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft4avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft5avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft6avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft7avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft8avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft9avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft10avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft11avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft12avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft13avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft14avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft15avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft16avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft20avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft48avx256(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

elementwise_mul_ register_elementwise_mul_avx256(FFTZ_UINT8 precision,
                                                 FFTZ_UINT8 direction);
normalize_ register_normalize_avx256(FFTZ_UINT8 precision);

// R2HC AVX256 Kernels
kfft_ register_kernel_r2hc_rfft2avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft9avx256(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft11avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft12avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft13avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft16avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);

// R2HC-Fused AVX256 Kernels
kfft_ register_kernel_r2hcf_rfft2avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft9avx256(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft11avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft12avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft13avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx256(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);

#endif

#ifdef ENABLE_AVX512
ops_cycles_t get_ops_cnt_fft2avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft20avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_fft48avx512(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_fwd_fft2avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft3avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft4avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft5avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft6avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft7avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft8avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft9avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft10avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft11avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft12avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft13avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft14avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft15avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_fwd_fft16avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_fwd_fft2avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft3avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft4avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft5avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft6avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft7avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft8avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft9avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft10avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft11avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft12avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft13avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft14avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft15avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_fwd_fft16avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

ops_cycles_t get_ops_cnt_twid_bwd_fft2avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft3avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft4avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft5avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft6avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft7avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft8avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft9avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft10avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft11avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft12avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft13avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft14avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft15avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_bwd_fft16avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);

kfft_ register_kernel_twid_bwd_fft2avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft3avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft4avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft5avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft6avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft7avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft8avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft9avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft10avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft11avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft12avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft13avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft14avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft15avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_bwd_fft16avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft2avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft3avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft4avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft5avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft6avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft7avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft8avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft9avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft10avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft11avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft12avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft13avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft14avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft15avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_r2c_fft16avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft2avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft3avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft4avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft5avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft6avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft7avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft8avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft9avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft10avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft11avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft12avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft13avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft14avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft15avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_r2c_fft16avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft2avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft3avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft4avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft5avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft6avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft7avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft8avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft9avx512(FFTZ_UINT8 precision,
                                             FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft10avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft11avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft12avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft13avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft14avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft15avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_twid_c2r_fft16avx512(FFTZ_UINT8 precision,
                                              FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft2avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft3avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft4avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft5avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft6avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft7avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft8avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft9avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft10avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft11avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft12avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft13avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft14avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft15avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
kfft_ register_kernel_twid_c2r_fft16avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC AVX512 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft9avx512(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft11avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft12avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft13avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft16avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);

// R2HC-Fused AVX512 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft9avx512(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft11avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft13avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx512(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction);

kfft_ register_kernel_fft2avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft3avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft4avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft5avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft6avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft7avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft8avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft9avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft10avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft11avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft12avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft13avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft14avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft15avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft16avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft20avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);
kfft_ register_kernel_fft48avx512(FFTZ_UINT8 precision, FFTZ_UINT8 direction);

elementwise_mul_ register_elementwise_mul_avx512(FFTZ_UINT8 precision,
                                                 FFTZ_UINT8 direction);
normalize_ register_normalize_avx512(FFTZ_UINT8 precision);

// R2HC AVX512 Kernels
kfft_ register_kernel_r2hc_rfft2avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft9avx512(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft11avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft12avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft13avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hc_rfft16avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);

// R2HC-Fused AVX512 Kernels
kfft_ register_kernel_r2hcf_rfft2avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft9avx512(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft11avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft12avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft13avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction);

#endif

// Permuted Copy Kernels
FFTZ_VOID permuted_copy_c_fp32(FFTZ_VOID *in, FFTZ_VOID *out, FFTZ_INTP n,
                               FFTZ_INTP size, FFTZ_INTP in_stride,
                               FFTZ_INTP out_stride, FFTZ_INTP v_in_stride,
                               FFTZ_INTP v_out_stride);
FFTZ_VOID permuted_copy_c_fp64(FFTZ_VOID *in, FFTZ_VOID *out, FFTZ_INTP n,
                               FFTZ_INTP size, FFTZ_INTP in_stride,
                               FFTZ_INTP out_stride, FFTZ_INTP v_in_stride,
                               FFTZ_INTP v_out_stride);

#endif // AOCLFFTZ_KERNEL_H
