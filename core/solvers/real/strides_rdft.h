// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file strides_rdft.h
 *
 *  @brief Declarations for the real FFT stride utility functions.
 *
 *  This file contains the function declarations that compute and manipulate
 *  stride array values for real, complex and half-complex data, and that set
 *  up the strides a real solution hands to its R2HC, R2HCF and C2C kernels,
 *  for the user buffer as well as the regrouped aux buffer layout.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Partiksha
 */

#ifndef STRIDES_RDFT_H
#define STRIDES_RDFT_H

#include "api/aoclfftz_internal.h"

/**
 * @brief Paired input and output stride values used for element-level access,
 * vector/batch traversal, and C2C kernel batch stepping in both direct
 * and CT decomposition scenarios.
 * Note: Stride arrays cannot be stored here.
 */
typedef struct base_strides
{
    FFTZ_INTP in_stride;  /**< Stride for input buffer access */
    FFTZ_INTP out_stride; /**< Stride for output buffer access */
} base_strides_t;

/**
 * @brief In RFFT, output can be the real problem input buffer or a temp buffer.
 * This function checks if the output is the real problem output buffer.
 */
static inline FFTZ_UINT8 is_output_prob_buffer(aoclfftz_solution_t *sol)
{
    FFTZ_UINT32 is_fwd = FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;
    FFTZ_UINT32 is_last_stage = sol->next_sol == NULL;
    return (is_fwd && is_last_stage);
}

/**
 * @brief In RFFT, input can be the real problem input buffer or a temp buffer.
 * This function checks if the input is the real problem input buffer.
 */
static inline FFTZ_UINT8 is_input_prob_buffer(aoclfftz_solution_t *sol)
{
    return (FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR &&
            NUM_RFFT_GROUPS(sol->solver) == 1);
}

FFTZ_VOID populate_stride_array(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                FFTZ_INTP n, FFTZ_UINT8 compute_half_complex,
                                FFTZ_UINT8 adjust_to_full_complex);
FFTZ_VOID populate_stride_array_r2hc(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                     FFTZ_INTP n,
                                     FFTZ_UINT8 compute_half_complex,
                                     FFTZ_UINT8 adjust_to_full_complex,
                                     FFTZ_INT8 is_user_buffer,
                                     FFTZ_INTP num_groups);
FFTZ_VOID populate_stride_array_r2hcf(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                      FFTZ_INTP n,
                                      FFTZ_UINT8 compute_half_complex,
                                      FFTZ_UINT8 adjust_to_full_complex,
                                      FFTZ_INT8 is_user_buffer,
                                      FFTZ_INTP num_groups);
FFTZ_VOID prepare_real_c2c_kernel_strides(FFTZ_INTP *in, FFTZ_INTP *out,
                                          FFTZ_INTP radix, FFTZ_INTP n,
                                          FFTZ_INTP stride);

/**
 * @brief Compute the element, vector and C2C base strides of a solution, for
 * a CT stage and for a direct problem size respectively.
 */
FFTZ_VOID set_ct_base_strides(aoclfftz_solution_t *sol,
                              aoclfftz_realhelper_t realhelper,
                              base_strides_t *element_strides,
                              base_strides_t *vector_strides,
                              base_strides_t *c2c_stride);
FFTZ_VOID set_base_strides(aoclfftz_solution_t *sol,
                           base_strides_t *element_strides,
                           base_strides_t *vector_strides);

/**
 * @brief Spread the base strides over the per-kernel vector strides (c2c,
 * r2hc, r2hcf), including the asymmetric endpoint/interior band steps used by
 * the regrouped aux buffer.
 */
FFTZ_VOID set_vector_strides_for_kernels(aoclfftz_solution_t *sol,
                                         base_strides_t vector_strides,
                                         base_strides_t c2c_strides,
                                         FFTZ_UINT8 use_asymmetric_kernel,
                                         aoclfftz_realhelper_t realhelper,
                                         FFTZ_INTP with_r2hcf);

/**
 * @brief Fill the stride arrays a solution hands to its R2HC, R2HCF and C2C
 * kernels, for the user buffer as well as the regrouped aux buffer layout.
 */
FFTZ_INT32 setup_r2hc_stride_arrays(aoclfftz_solution_t *sol,
                                    aoclfftz_realhelper_t realhelper,
                                    base_strides_t element_strides,
                                    FFTZ_INTP num_groups,
                                    FFTZ_INTP num_c2c_per_group);
FFTZ_INT32 setup_r2hcf_stride_arrays(aoclfftz_solution_t *sol,
                                     aoclfftz_realhelper_t realhelper,
                                     base_strides_t element_strides);
FFTZ_INT32 setup_c2c_stride_arrays(aoclfftz_solution_t *sol,
                                   aoclfftz_realhelper_t realhelper,
                                   base_strides_t element_strides,
                                   base_strides_t c2c_stride,
                                   FFTZ_INTP num_groups,
                                   FFTZ_INTP with_r2hcf,
                                   FFTZ_INTP num_c2c_per_group);
FFTZ_VOID prepare_fused_c2c_asymmetric_strides(aoclfftz_solution_t *sol);

#endif // STRIDES_RDFT_H
