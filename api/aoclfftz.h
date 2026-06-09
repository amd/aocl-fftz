// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
 *  @defgroup APIs AOCL-FFTZ - APIs
 *  @brief Interface APIs of AOCL-FFTZ library.
 *
 *  This section contains APIs for setting up, executing and destroying the
 *  single-threaded, and multi-threaded FFT operations.
 *
 *  @note Different variants of setup APIs are exposed to support float and
 *  double precision types in LP64 and ILP64 data models.
 *
 */

/**
 * @defgroup Data_Structures AOCL-FFTZ - Data Structures and Type Definitions
 * @brief This section contains typedef definitions used by the AOCL-FFTZ
 * library's interface APIs.
 *
 * @note Different variants of data structures are available to support float
 * and double precision types in LP64 and ILP64 data models.
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

#define AOCLFFTZ_LIBRARY_VERSION "AOCL-FFTZ 5.2.2"
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * @ingroup Data_Structures
 * @brief Error return codes of `AOCL-FFTZ` library.
 */
// Add new error codes at the end to retain the existing error code values
typedef enum
{
    AOCLFFTZ_TIME_OUT = -5,     /**< Operation took long time than expected */
    AOCLFFTZ_MEMORY_FAILURE,    /**< Error related to Memory access or operation */
    AOCLFFTZ_INVALID_INPUT,     /**< Invalid size, format, type or precision of input */
    AOCLFFTZ_SETUP_FAILURE,     /**< Error in setup of the library */
    AOCLFFTZ_EXECUTION_FAILURE, /**< Error in execution of the library */
    AOCLFFTZ_SUCCESS            /**< Successful operation */
} aoclfftz_error_type;

/**
 * @ingroup Data_Structures
 * @brief Logging modes to control verbosity of `AOCL-FFTZ` library output.
 */
typedef enum
{
    AOCLFFTZ_LOG_NONE = 0,     /**< `0` : Disable all logging */
    AOCLFFTZ_LOG_INFO = 1,     /**< `1` : Detailed debugging logs */
    AOCLFFTZ_LOG_TRACE = 2,    /**< `2` : Logging with complete execution trace */
    AOCLFFTZ_LOG_DEBUG = 3,    /**< `3` : Most detailed logging, with debug logs */
} aoclfftz_logger_mode;

/**
 * @ingroup Data_Structures
 * @brief Configuration flags to control critical FFT execution parameters.
*/
typedef struct aoclfftz_flags
{
    UINT8 fft_type;             /**< Complex(0) or Real(1) >*/
    UINT8 fft_direction;        /**< Forward(0) or Backward(1) >*/
    UINT8 storage_order;        /**< In-order(0) or Out-of-order(1)>*/
    UINT8 fft_placement;        /**< In-place(0) or Out-of-place(1) */
    UINT8 transpose_mode;       /**< fft(0) or standalone transpose(1) (Not supported, must be default value `0`)*/
    UINT8 bit_reproducibility;  /**< Disable(0) or Enable(1) the bit reproducibility mode*/
} aoclfftz_flags_t;

/**
 * @ingroup Data_Structures
 * @brief Tensor dimension for LP64.
 */
typedef struct aoclfftz_dim
{
    INT32 n;            /**< Dimension length */
    INT32 in_stride;    /**< Stride for input buffer */
    INT32 out_stride;   /**< Stride for output buffer */
} aoclfftz_dim_t;

/**
 * @ingroup Data_Structures
 * @brief Tensor dimension for ILP64.
 */
typedef struct aoclfftz_dim_64_
{
    INTP n;            /**< Dimension length */
    INTP in_stride;    /**< Stride for input buffer */
    INTP out_stride;   /**< Stride for output buffer */
} aoclfftz_dim_t_64_;

/**
 * @ingroup Data_Structures
 * @brief Params for parallel SMP FFT computation.
 */
typedef struct aoclfftz_smp_pfft
{
    INT32 num_threads;         /**< Number of max threads granted for use.\n
    * `num_threads` must be greater than 0. */
    UINT32 dynamic_load_model; /**< Specifies the model for determining the number of threads.
     * - 0: Use `num_threads` as the maximum number of threads.
     * - 1: Dynamically determine the number of threads (currently configured to
     *      take upto the max system threads)
     */
} aoclfftz_smp_pfft_t;

/**
 * @ingroup Data_Structures
 * @brief Control params for optimizations, logs and stats.
 */
typedef struct aoclfftz_cntrl_params
{
    INT32 opt_level;        /**< Set Optimization level with following values\n
    * Levels:
    * - 0 - non-SIMD algorithmic optimizations
    * - 1 - AVX128 optimizations
    * - 2 - AVX256 optimizations
    * - 3 - AVX512 optimizations */
    INT32 opt_off;                         /**< Turn off all optimizations */
    aoclfftz_logger_mode logger_mode;      /**< Set Logger mode with following values.\n
    * - @ref AOCLFFTZ_LOG_NONE - Disable all logging
    * - @ref AOCLFFTZ_LOG_INFO - Detailed debugging logs
    * - @ref AOCLFFTZ_LOG_TRACE - Logging with complete execution trace
    * - @ref AOCLFFTZ_LOG_DEBUG - Most detailed logging, with debug logs */
    INT32 measure_stats;    /**< Enable/Disable measure stats (Not supported, must be default value `0`) */
} aoclfftz_cntrl_params_t;

/**
 * @rst
 * .. _aoclfftz_prob_desc_f:
 * @endrst
 * @ingroup Data_Structures
 * @brief Defines problem descriptor for float data on LP64 machines.
 */
typedef struct
{
    FLOAT *in;              /**< input buffer: Points to signal points for forward FFT and frequency points for backward FFT */
    FLOAT *out;             /**< output buffer: Points frequency points for forward FFT and signal points for backward FFT */
    INT32 vec_rank;         /**< `vec_rank` is the number of batch/vector dimensions. length of `vecs` array(must be >= 1). */
    INT32 dim_rank;         /**< `dim_rank` is the number of signal/frequency dimensions. length of `dims` array(must be >= 1). */
    aoclfftz_dim_t *dims;   /**< Multi-dimensional tensor dimensions for LP64 */
    aoclfftz_dim_t *vecs;   /**< Vector tensor dimensions for LP64 */
    aoclfftz_flags_t flags; /**< Struct for configuration flags to control critical FFT execution parameters like -\n
                                 `fft_type`, `fft_direction`, `storage_order`, `fft_placement`, `transpose_mode`. */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP FFT computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_f;

/**
 * @rst
 * .. _aoclfftz_prob_desc_d:
 * @endrst
 * @ingroup Data_Structures
 * @brief Defines problem descriptor for double data on LP64 machines.
 */
typedef struct
{
    DOUBLE *in;             /**< input buffer: Points to signal points for forward FFT and frequency points for backward FFT */
    DOUBLE *out;            /**< output buffer: Points frequency points for forward FFT and signal points for backward FFT */
    INT32 vec_rank;         /**< `vec_rank` is the number of batch/vector dimensions. length of `vecs` array(must be >= 1). */
    INT32 dim_rank;         /**< `dim_rank` is the number of signal/frequency dimensions. length of `dims` array(must be >= 1). */
    aoclfftz_dim_t *dims;   /**< Multi-dimensional tensor dimensions for LP64 */
    aoclfftz_dim_t *vecs;   /**< Vector tensor dimensions for LP64 */
    aoclfftz_flags_t flags; /**< Struct for configuration flags to control critical FFT execution parameters like -\n
                                 `fft_type`, `fft_direction`, `storage_order`, `fft_placement`, `transpose_mode`. */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP FFT computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_d;

/**
 * @rst
 * .. _aoclfftz_prob_desc_f_64_:
 * @endrst
 * @ingroup Data_Structures
 * @brief Defines problem descriptor for float data on ILP64 machines.
 */
typedef struct
{
    FLOAT *in;                  /**< input buffer: Points to signal points for forward FFT and frequency points for backward FFT */
    FLOAT *out;                 /**< output buffer: Points to frequency points for forward FFT and signal points for backward FFT */
    INT32 vec_rank;             /**< `vec_rank` is the number of batch/vector dimensions. length of `vecs` array(must be >= 1). */
    INT32 dim_rank;             /**< `dim_rank` is the number of signal/frequency dimensions. length of `dims` array(must be >= 1). */
    aoclfftz_dim_t_64_ *dims;   /**< Multi-dimensional tensor dimensions for ILP64 */
    aoclfftz_dim_t_64_ *vecs;   /**< Vector tensor dimensions for ILP64 */
    aoclfftz_flags_t flags;     /**< Struct for configuration flags to control critical FFT execution parameters like -\n
                                    * `fft_type`, `fft_direction`, `storage_order`, `fft_placement`, `transpose_mode`. */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP FFT computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_f_64_;

/**
 * @rst
 * .. _aoclfftz_prob_desc_d_64_:
 * @endrst
 * @ingroup Data_Structures
 * @brief Defines problem descriptor for double data on ILP64 machines.
 */
typedef struct
{
    DOUBLE *in;                 /**< input buffer: Points to signal points for forward FFT and frequency points for backward FFT */
    DOUBLE *out;                /**< output buffer: Points to frequency points for forward FFT and signal points for backward FFT */
    INT32 vec_rank;             /**< `vec_rank` is the number of batch/vector dimensions. length of `vecs` array(must be >= 1). */
    INT32 dim_rank;             /**< `dim_rank` is the number of signal/frequency dimensions. length of `dims` array(must be >= 1). */
    aoclfftz_dim_t_64_ *dims;   /**< Multi-dimensional tensor dimensions for ILP64 */
    aoclfftz_dim_t_64_ *vecs;   /**< Vector tensor dimensions for ILP64 */
    aoclfftz_flags_t flags;     /**< Struct for configuration flags to control critical FFT execution parameters like - \n
                                    * `fft_type`, `fft_direction`, `storage_order`, `fft_placement`, `transpose_mode`. */
    aoclfftz_smp_pfft_t pthr_fft;         /**< Struct for parallel SMP FFT computation */
    aoclfftz_cntrl_params_t cntrl_params; /**< Struct for optimizations, logs, stat params */
} aoclfftz_prob_desc_d_64_;

/* Single-threaded and multi-threaded FFT unified APIs */
/**
 * @rst
 * .. _aoclfftz_setup_f:
 * @endrst
 * @ingroup APIs
 * @brief Generates a solution handle for a given problem of type FLOAT on LP64 systems.
 *
 * This API validates the problem descriptor and generates a solution handle
 * for the given input problem.
 * The problem descriptor parameters must be fully initialized prior to invocation,
 * including:
 *   - Allocation of memory for input/output buffers
 *   - Allocation and initialization of dims/vecs tensor dimensions
 *   - Configuration of all fields within the relevant problem descriptor
 *     structure @ref aoclfftz_prob_desc_f
 *
 * This generated handle is passed to the execute APIs to perform FFT
 * computation and should be destroyed by @ref aoclfftz_destroy.
 *
 * @param problem FLOAT LP64 problem descriptor object.
 * @return
 * | Result                                 | Description |
 * |:---------------------------------------|:-------------------------------------------------------------------------------------|
 * | Opaque pointer to FFT solution handle  |  Setup succeeded with a valid solution for execution of the given problem (SUCCESS)  |
 * | NULL                                   |  Setup failed (FAIL)                                                                 |
 */
EXPORT_SYM_DYN VOID *aoclfftz_setup_f(aoclfftz_prob_desc_f *problem);

/**
 * @ingroup APIs
 * @brief Generates a solution handle for a given problem of type DOUBLE on LP64 systems.
 *
 * This API validates the problem descriptor and generates a solution handle
 * for the given input problem.
 * The problem descriptor parameters must be fully initialized prior to invocation,
 * including:
 *   - Allocation of memory for input/output buffers
 *   - Allocation and initialization of dims/vecs tensor dimensions
 *   - Configuration of all fields within the relevant problem descriptor
 *     structure @ref aoclfftz_prob_desc_d
 *
 * This generated handle is passed to the execute APIs to perform FFT
 * computation and should be destroyed by @ref aoclfftz_destroy.
 *
 * @param problem DOUBLE LP64 problem descriptor object
 * @return
 * | Result                                 | Description |
 * |:---------------------------------------|:-------------------------------------------------------------------------------------|
 * | Opaque pointer to FFT solution handle  |  Setup succeeded with a valid solution for execution of the given problem (SUCCESS)  |
 * | NULL                                   |  Setup failed (FAIL)                                                                 |
 */
EXPORT_SYM_DYN VOID *aoclfftz_setup_d(aoclfftz_prob_desc_d *problem);

/**
 * @ingroup APIs
 * @brief Generates a solution handle for a given problem of type FLOAT on ILP64 systems.
 *
 * This API validates the problem descriptor and generates a solution handle
 * for the given input problem.
 * The problem descriptor parameters must be fully initialized prior to invocation,
 * including:
 *   - Allocation of memory for input/output buffers
 *   - Allocation and initialization of dims/vecs tensor dimensions
 *   - Configuration of all fields within the relevant problem descriptor
 *     structure @ref aoclfftz_prob_desc_f_64_
 *
 * This generated handle is passed to the execute APIs to perform FFT
 * computation and should be destroyed by @ref aoclfftz_destroy.
 *
 * @param problem FLOAT ILP64 problem descriptor object
 * @return
 * | Result                                 | Description |
 * |:---------------------------------------|:-------------------------------------------------------------------------------------|
 * | Opaque pointer to FFT solution handle  |  Setup succeeded with a valid solution for execution of the given problem (SUCCESS)  |
 * | NULL                                   |  Setup failed (FAIL)                                                                 |
 */
EXPORT_SYM_DYN VOID *aoclfftz_setup_f_64_(aoclfftz_prob_desc_f_64_ *problem);

/**
 * @ingroup APIs
 * @brief Generates a solution handle for a given problem of type DOUBLE on ILP64 systems.
 *
 * This API validates the problem descriptor and generates a solution handle
 * for the given input problem.
 * The problem descriptor parameters must be fully initialized prior to invocation,
 * including:
 *   - Allocation of memory for input/output buffers
 *   - Allocation and initialization of dims/vecs tensor dimensions
 *   - Configuration of all fields within the relevant problem descriptor
 *     structure @ref aoclfftz_prob_desc_d_64_
 *
 * This generated handle is passed to the execute APIs to perform FFT
 * computation and should be destroyed by @ref aoclfftz_destroy.
 *
 * @param problem DOUBLE ILP64 problem descriptor object
 * @return
 * | Result                                 | Description |
 * |:---------------------------------------|:-------------------------------------------------------------------------------------|
 * | Opaque pointer to FFT solution handle  |  Setup succeeded with a valid solution for execution of the given problem (SUCCESS)  |
 * | NULL                                   |  Setup failed (FAIL)                                                                 |
 */
EXPORT_SYM_DYN VOID *aoclfftz_setup_d_64_(aoclfftz_prob_desc_d_64_ *problem);

/**
 * @ingroup APIs
 * @brief Performs FFT computation based on the solution handle and returns
 * the status of execution as one of the aoclfftz_error_type values.
 *
 * The input and output buffers defined in the setup stage will be used for
 * computation without requiring additional buffer specification.
 * Setup API `aoclfftz_setup_*` must be invoked before calling this execute API.
 *
 * @param handle solution handle
 * @return
 * | Result                             | Description |
 * |:-----------------------------------|:------------|
 * | @ref AOCLFFTZ_SUCCESS              |  SUCCESS    |
 * | @ref AOCLFFTZ_EXECUTION_FAILURE    |  FAIL       |
 */
EXPORT_SYM_DYN aoclfftz_error_type aoclfftz_execute(VOID *handle);

/**
 * @ingroup APIs
 * @brief Performs FFT computation on the given buffers based on the
 * solution handle and returns the status of execution as one of the
 * aoclfftz_error_type values.
 *
 * This API uses an already computed solution handle but applies it to the newly
 * provided input and output buffers instead of those defined during setup.
 * To avoid memory access issues, the `in` and `out` buffers must have
 * sufficient memory allocated based on the problem size defined in the
 * solution handle.
 * Setup API `aoclfftz_setup_*` must have been invoked in the past to acquire
 * a valid solution handle before calling this execute API with any new set
 * of input and output buffers.
 *
 * @param handle solution handle
 * @param in pointer to input buffer
 * @param out pointer to output buffer
 * @return
 * | Result                             | Description |
 * |:-----------------------------------|:------------|
 * | @ref AOCLFFTZ_SUCCESS              |  SUCCESS    |
 * | @ref AOCLFFTZ_EXECUTION_FAILURE    |  FAIL       |
 */
EXPORT_SYM_DYN aoclfftz_error_type aoclfftz_execute_io(VOID *handle, VOID *in, VOID *out);

/**
 * @ingroup APIs
 * @brief Destroys the solution handle created by aoclfftz_setup_* APIs
 *
 * This API frees all memory allocated within the solution handle and
 * destroys the handle. It should be called after all FFT computations are
 * completed to prevent memory leaks.
 * After calling this API, the handle becomes invalid and should not be used
 * again.
 *
 * @param handle solution handle
 *
 */
EXPORT_SYM_DYN VOID aoclfftz_destroy(VOID *handle);

/**
 * @ingroup APIs
 * @brief Interface API to get `AOCL-FFTZ` library version string.
 *
 * @return `AOCL-FFTZ` library version string
 */
EXPORT_SYM_DYN const CHAR *aoclfftz_version(VOID);

#ifdef __cplusplus
}
#endif

#endif // AOCLFFTZ_H
