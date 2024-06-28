/*
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
 *  @brief AOCL-FFTZ Library - Interface APIs and data structures.
 *
 *  This file contains the APIs and associated data structures that
 *  are responsible for setting up and executing the single-threaded,
 *  and multi-threaded FFT operations.
 *
 *  @author S. Biplab Raut
 */

/**
 *  @defgroup group_api AOCL-FFTZ - API
 *  @brief Interface APIs of AOCL-FFTZ library.
 *
 *  APIs for setting up and executing the single-threaded, and multi-threaded
 *  FFT operations.
 *
 *  @note Different variants of APIs are exposed to support float and double
 *  precision types in LP64 and ILP64 data models.
 *
 */

/**
 *  @defgroup group_types AOCL-FFTZ - Types
 *  @brief Data structures and typedef declarations of AOCL-FFTZ library.
 *
 *  Data strutures and typedef declarations used for AOCL-FFTZ APIs.
 *
 *  @note Different variants of data structures are available to support float
 *  and double precision types in LP64 and ILP64 data models.
 */

#ifndef AOCLFFTZ_H
#define AOCLFFTZ_H

#include <stddef.h>
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#ifdef _WINDOWS
#define EXPORT_SYM_DYN __declspec(dllexport)
#else
#define EXPORT_SYM_DYN
#endif

#define AOCLFFTZ_LIBRARY_VERSION "AOCL-FFTZ 0.2.1"
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**  @ingroup group_types
     * aoclfftz_error_type
     * Error return codes of aocl-fftz library.
*/
// Add new error codes at the code to retain the existing error code values
typedef enum
{
    AOCLFFTZ_TIME_OUT = -6,     /**< Operation took long time than expected */
    AOCLFFTZ_MPI_COMM_FAILURE,  /**< Error encountered in MPI communication */
    AOCLFFTZ_MEMORY_FAILURE,    /**< Error related to Memory access or operation */
    AOCLFFTZ_INVALID_INPUT,     /**< Invalid size, format, type or precision of input */
    AOCLFFTZ_SETUP_FAILURE,     /**< Error in setup of the library */
    AOCLFFTZ_EXECUTION_FAILURE, /**< Error in execution of the library */
    AOCLFFTZ_SUCCESS            /**< Successful operation */
} aoclfftz_error_type;

/**  @ingroup group_types
     * aoclfftz_dim_t
     * Tensor dimension for LP64.
*/
typedef struct aoclfftz_dim
{
    INT32 n;            /**< Problem dimension size */
    INT32 in_stride;    /**< Stride for input buffer */
    INT32 out_stride;   /**< Stride for output buffer */
} aoclfftz_dim_t;

/**  @ingroup group_types
     * aoclfftz_dim_t_64_
     * Tensor dimension for ILP64.
*/
typedef struct aoclfftz_dim_64_
{
    INTP n;            /**< Problem dimension size */
    INTP in_stride;    /**< Stride for input buffer */
    INTP out_stride;   /**< Stride for output buffer */
} aoclfftz_dim_t_64_;

/**  @ingroup group_types
     * aoclfftz_smp_pfft_t
     * params for parallel SMP fft computation.
*/
typedef struct aoclfftz_smp_pfft
{
    INT32 num_threads;        /**< Number of max threads to granted for use */
    INT32 dynamic_load_model; /**< Allow the library to determine how many threads to be used */
} aoclfftz_smp_pfft_t;

/**  @ingroup group_types
     * aoclfftz_cntrl_params_t
     * control params for optimizations, logs, stats and others.
*/
typedef struct aoclfftz_cntrl_params
{
    INT32 opt_level;        /**< Set Optimization level with following values\n
     Levels:\n 0 - non-SIMD algorithmic optimizations\n
     1 - AVX128 optimizations\n
     2 - AVX256 optimizations\n
     3 - AVX512 optimizations */
    INT32 opt_off;          /**< Turn off all optimizations */
    INT32 logger_mode;      /**< Set Logger mode with following values
    0 - no logging\n 1 - error\n 2 - info\n 3 - debug\n 4 - trace */
    INT32 measure_stats;    /**< Enable/Disable measure stats */
} aoclfftz_cntrl_params_t;

/**  @ingroup group_types
     * aoclfftz_prob_desc_f
     * problem descriptor for float LP64
*/
typedef struct
{
    FLOAT *in;              /**< Input buffer */
    FLOAT *out;             /**< Output buffer */
    INT32 vec_rank;         /**< Vector rank */
    INT32 dim_rank;         /**< Dimension rank */
    aoclfftz_dim_t *dims;   /**< Multi-dimensional tensor dimensions for LP64 */
    aoclfftz_dim_t *vecs;   /**< Vector tensor dimensions for LP64 */
    UINT32 flags;           /**< Set flags
     where each bit represents the following\n
     Bit 0 : in-place(0) or out-of-place(1),\n
     Bit 1 : in-order(0) or out-of-order(1),\n
     Bit 2 : forward(0) or backward(1),\n
     Bit 3 : complex(0) or real(1),\n
     Bit 8 : fft(0) or standalone transpose(1) */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP fft computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_f;

/**  @ingroup group_types
     * aoclfftz_prob_desc_d
     * problem descriptor for double LP64
*/
typedef struct
{
    DOUBLE *in;             /**< Input buffer */
    DOUBLE *out;            /**< Output buffer */
    INT32 vec_rank;         /**< Vector rank */
    INT32 dim_rank;         /**< Dimension rank */
    aoclfftz_dim_t *dims;   /**< Multi-dimensional tensor dimensions for LP64 */
    aoclfftz_dim_t *vecs;   /**< Vector tensor dimensions for LP64 */
    UINT32 flags;   /**< Set flags
     where each bit represents the following\n
     Bit 0 : in-place(0) or out-of-place(1),\n
     Bit 1 : in-order(0) or out-of-order(1),\n
     Bit 2 : forward(0) or backward(1),\n
     Bit 3 : complex(0) or real(1),\n
     Bit 8 : fft(0) or standalone transpose(1) */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP fft computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_d;

// float ILP64
/**  @ingroup group_types
     * aoclfftz_prob_desc_f_64_
     * problem descriptor for float ILP64
*/
typedef struct
{
    FLOAT *in;                  /**< Input buffer */
    FLOAT *out;                 /**< Output buffer */
    INT32 vec_rank;             /**< Vector rank */
    INT32 dim_rank;             /**< Dimension rank */
    aoclfftz_dim_t_64_ *dims;   /**< Multi-dimensional tensor dimensions for ILP64 */
    aoclfftz_dim_t_64_ *vecs;   /**< Vector tensor dimensions for ILP64 */
    UINT32 flags;               /**< Set flags
     where each bit represents the following\n
     Bit 0 : in-place(0) or out-of-place(1),\n
     Bit 1 : in-order(0) or out-of-order(1),\n
     Bit 2 : forward(0) or backward(1),\n
     Bit 3 : complex(0) or real(1),\n
     Bit 8 : fft(0) or standalone transpose(1) */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP fft computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_f_64_;

/**  @ingroup group_types
     * aoclfftz_prob_desc_d_64_
     * problem descriptor for double ILP64
*/
typedef struct
{
    DOUBLE *in;                 /**< Input buffer */
    DOUBLE *out;                /**< Output buffer */
    INT32 vec_rank;             /**< Vector rank */
    INT32 dim_rank;             /**< Dimension rank */
    aoclfftz_dim_t_64_ *dims;   /**< Multi-dimensional tensor dimensions for ILP64 */
    aoclfftz_dim_t_64_ *vecs;   /**< Vector tensor dimensions for ILP64 */
    UINT32 flags;               /**< Set flags
     where each bit represents the following\n
     Bit 0 : in-place(0) or out-of-place(1),\n
     Bit 1 : in-order(0) or out-of-order(1),\n
     Bit 2 : forward(0) or backward(1),\n
     Bit 3 : complex(0) or real(1),\n
     Bit 8 : fft(0) or standalone transpose(1) */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP fft computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_d_64_;

/* Single-threaded and multi-threaded FFT unified APIs */
// float LP64
/**  @ingroup group_api
     * @brief Generates a solution handle for the given input problem.\n
     * This generated handle is passed to the execute/destroy APIs.
     *
     * @param problem FLOAT LP64 problem descriptor object
     * @return solution handle as an opaque pointer
     */
EXPORT_SYM_DYN VOID *aoclfftz_setup_f(aoclfftz_prob_desc_f *problem);

// double LP64
/**  @ingroup group_api
     * @brief Generates a solution handle for the given input problem.\n
     * This generated handle is passed to the execute/destroy APIs.
     *
     * @param problem DOUBLE LP64 problem descriptor object
     * @return solution handle as an opaque pointer
     */
EXPORT_SYM_DYN VOID *aoclfftz_setup_d(aoclfftz_prob_desc_d *problem);

// float ILP64
/**  @ingroup group_api
     * @brief Generates a solution handle for the given input problem.\n
     * This generated handle is passed to the execute/destroy APIs.
     *
     * @param problem FLOAT ILP64 problem descriptor object
     * @return solution handle as an opaque pointer
     */
EXPORT_SYM_DYN VOID *aoclfftz_setup_f_64_(aoclfftz_prob_desc_f_64_ *problem);

// double ILP64
/**  @ingroup group_api
     * @brief Generates a solution handle for the given input problem.\n
     * This generated handle is passed to the execute/destroy APIs.
     *
     * @param problem DOUBLE ILP64 problem descriptor object
     * @return solution handle as an opaque pointer
     */
EXPORT_SYM_DYN VOID *aoclfftz_setup_d_64_(aoclfftz_prob_desc_d_64_ *problem);

/**  @ingroup group_api
     * @brief Receives a solution handle and performs FFT computation for that solution.\n
     * Returns the status of execution as an INT32 value.
     *
     * @param handle solution handle (generated by setup api)
     * @return
     * | Result     | Description |
     * |:-----------|:------------|
     * | Success    |`AOCLFFTZ_SUCCESS`                |
     * | Fail       |`AOCLFFTZ_EXECUTION_FAILURE`      |
     */
EXPORT_SYM_DYN INT32 aoclfftz_execute(VOID *handle);
/**  @ingroup group_api
     * @brief Destroys the solution handle created by aoclfftz_setup_* APIs
     *
     * @param handle solution handle (generated by setup api)
     *
     */
EXPORT_SYM_DYN VOID aoclfftz_destroy(VOID *handle);

/**  @ingroup group_api
     * @brief Interface API to get aocl-fftz library version string.
     *
     * @return aocl-fftz library version string
     */
EXPORT_SYM_DYN const CHAR *aoclfftz_version(VOID);

#ifdef __cplusplus
}
#endif

#endif // AOCLFFTZ_H
