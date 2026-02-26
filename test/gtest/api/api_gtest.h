/*
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
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

#ifndef AOCLFFTZ_API_GTEST_H
#define AOCLFFTZ_API_GTEST_H

#include <string>
#include <limits>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <climits>
#include <iostream> // Add this include for debug printing
#include <omp.h>
#include "gtest/gtest.h"
#include "test/gtest/common_gtest_utils.h"

extern "C"
{
#include "utils/utils.h"
#include "api/aoclfftz.h"
#include "api/types.h"
#include "api/aoclfftz_internal.h"
}

// Enum to define valid and invalid test cases
typedef enum
{
    INVALID = -1,
    VALID
} InvalidCase;

// Default configuration parameters
#define DEFAULT_OPT_OFF 1
#define DEFAULT_OPT_LEVEL 2
#define DEFAULT_FLAGS 0b0001
#define DEFAULT_NUM_THREADS 1
#define DEFAULT_DYNAMIC_LOAD_MODEL 1
#define COMPLEX_DATA_STRIDE 2

// Unsupported flags and ranks for testing
const std::vector<aoclfftz_flags_t> unsupported_flags = {
    {1, 0, 1, 0, 0}, // real, forward, out_of_order, inplace, FFT
    {0, 1, 1, 1, 1}, // complex, backward, in-order, out-of-place, transpose
};

const std::vector<INT32> unsupported_rank = { INT32_MIN, -1 };

// Random number generator setup for generating invalid test values
#define INIT_RANDOM_NUM_GEN() \
    std::random_device rd; \
    std::mt19937 prng(rd()); \
    std::uniform_int_distribution<> dist_invalid(INT32_MIN, 0); \
    prng.seed(42);

// Function to check if the handle is destroyed
static bool is_handle_null(VOID *handle)
{
    return (handle == NULL);
}

// Template class for testing the AOCL-FFTZ API
template<typename ProblemType>
class AoclfftzAPITest : public ::testing::Test
{
public:
    INT32 optOff;
    INT32 optLevel;
    INT32 num_threads;
    UINT32 dynamic_load_model;
    ProblemType *problem;
    VOID *handle;
    UINTP input_size;
    UINTP output_size;
    void SetUp() override
    {
        problem = NULL;
        handle = NULL;

        problem = new ProblemType();
        if (problem == NULL)
        {
            throw std::runtime_error(
                "Memory allocation failed for the problem.");
        }
        // Set default flags to out-of-place, in-order, forward, complex, FFT
        problem->flags = {0};
        problem->flags.fft_placement = 1;

        create_default_pdesc(); // Create a sample problem for testing
    }

    void TearDown() override
    {
        if (problem != NULL)
        {
            cleanup_problem();
            delete problem;
            problem = NULL;
        }
    }

    // Function to clean up the problem descriptor
    VOID cleanup_problem()
    {
        if (problem == NULL)
        {
            return;
        }
        /* Default value of is_inplace is 0,
         * If problem->flags are invalid, free the memory buffer based on default flags */
        bool is_inplace = isValidFlags(problem->flags) ?
                                    !problem->flags.fft_placement : 0;
        if (problem->in)
        {
            delete[] problem->in;
            problem->in = NULL;
        }
        if (problem->out && !is_inplace)
        {
            delete[] problem->out;
        }
        problem->out = NULL;
        if (problem->dims)
        {
            delete[] problem->dims;
            problem->dims = NULL;
        }
        if (problem->vecs)
        {
            delete[] problem->vecs;
            problem->vecs = NULL;
        }
    }

    bool isValidFlags(aoclfftz_flags_t flags)
    {
        if (flags.storage_order || flags.transpose_mode || flags.fft_type)
        {
            return 0;
        }
        return 1;
    }

    VOID get_inout_size(UINTP *in_size, UINTP *out_size)
    {
        in_size[0] = input_size;
        out_size[0] = output_size;
    }

    // Function to create a sample problem for testing
    template<typename DataType, typename DimT>
    VOID create_pdesc(bool is_forward = true, bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if (problem == NULL)
        {
            return;
        }
        INT32 in_size = 0, out_size = 0;
        problem->dim_rank = 3;
        problem->vec_rank = 1;
        problem->dims = new DimT[problem->dim_rank];
        problem->vecs = new DimT[problem->vec_rank];
        if (problem->dims == NULL || problem->vecs == NULL)
        {
            cleanup_problem();
            throw std::runtime_error("Memory allocation failed "
                                                        "for dims or vecs!");
        }
        // Update flags based on transform direction and placement
        // Only update fft_direction and fft_placement, preserve other flag fields
        if (is_forward)
        {
            problem->flags.fft_direction = 0;
        }
        else
        {
            problem->flags.fft_direction = 1;
        }
        if (is_inplace)
        {
            problem->flags.fft_placement = 0;
        }
        else
        {
            problem->flags.fft_placement = 1;
        }
        // Set dims values
        problem->dims[0].n = 10;
        problem->dims[0].in_stride = 1;
        problem->dims[0].out_stride = 1;
        problem->dims[1].n = 30;
        problem->dims[1].in_stride = problem->dims[0].n *
                                     problem->dims[0].in_stride;
        problem->dims[1].out_stride = problem->dims[0].n *
                                      problem->dims[0].out_stride;
        problem->dims[2].n = 30;
        problem->dims[2].in_stride = problem->dims[1].n *
                                     problem->dims[1].in_stride;
        problem->dims[2].out_stride = problem->dims[1].n *
                                      problem->dims[1].out_stride;
        // Set vecs values
        problem->vecs[0].n = 1;
        problem->vecs[0].in_stride = problem->dims[problem->dim_rank - 1].n *
                            problem->dims[problem->dim_rank - 1].in_stride;
        problem->vecs[0].out_stride = problem->dims[problem->dim_rank - 1].n *
                            problem->dims[problem->dim_rank - 1].out_stride;

        in_size = problem->vecs[0].n * problem->vecs[0].in_stride *
                                       COMPLEX_DATA_STRIDE;
        out_size = problem->vecs[0].n * problem->vecs[0].out_stride *
                                        COMPLEX_DATA_STRIDE;

        // Allocating input buffer
        input_size  = in_size * sizeof(DataType);
        problem->in  = new DataType[in_size];
        // Allocating output buffer
        if (is_inplace)
        {
            output_size = input_size;
            problem->out = problem->in;
        }
        else
        {
            output_size = out_size * sizeof(DataType);
            problem->out = new DataType[out_size];
        }

        if (problem->in == NULL || problem->out == NULL)
        {
            cleanup_problem();
            throw std::runtime_error("Memory allocation failed for input "
                                                    "or output arrays");
        }
        for (INT32 i = 0; i < in_size; i++)
        {
            problem->in[i] =
                get_value_based_on_strategy<DataType>(value_strategy, in_size);
        }
        if (!is_inplace)
        {
            for (INT32 i = 0; i < out_size; i++)
            {
                problem->out[i] = (DataType)0.0;
            }
        }
        problem->pthr_fft.dynamic_load_model = DEFAULT_DYNAMIC_LOAD_MODEL;
        problem->pthr_fft.num_threads = DEFAULT_NUM_THREADS;
        problem->cntrl_params.logger_mode = AOCLFFTZ_LOG_NONE;
        problem->cntrl_params.measure_stats = 0;
        problem->cntrl_params.opt_level = -1;
        problem->cntrl_params.opt_off = 1;
    }

    /**
     * @brief Create a 1D problem descriptor with large size
     *
     */
    template<typename DataType, typename DimT>
    VOID create_1d_pdesc(bool is_forward = true, bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if (problem == NULL)
        {
            return;
        }
        INT32 in_size = 0, out_size = 0;
        problem->dim_rank = 1;
        problem->vec_rank = 1;
        problem->dims = new DimT[problem->dim_rank];
        problem->vecs = new DimT[problem->vec_rank];
        if (problem->dims == NULL || problem->vecs == NULL)
        {
            cleanup_problem();
            throw std::runtime_error("Memory allocation failed for dims or vecs!");
        }

        // Update flags based on transform direction and placement
        if (is_forward)
        {
            problem->flags.fft_direction = 0;
        }
        else
        {
            problem->flags.fft_direction = 1;
        }
        if (is_inplace)
        {
            problem->flags.fft_placement = 0;
        }
        else
        {
            problem->flags.fft_placement = 1;
        }

        // Set dims values - 1D with large size (16384 = 2^14, common FFT size)
        problem->dims[0].n = 16384;
        problem->dims[0].in_stride = 1;
        problem->dims[0].out_stride = 1;

        // Set vecs values
        problem->vecs[0].n = 1;
        problem->vecs[0].in_stride = problem->dims[0].n *
                                     problem->dims[0].in_stride;
        problem->vecs[0].out_stride = problem->dims[0].n *
                                      problem->dims[0].out_stride;

        in_size = problem->vecs[0].n * problem->vecs[0].in_stride *
                                       COMPLEX_DATA_STRIDE;
        out_size = problem->vecs[0].n * problem->vecs[0].out_stride *
                                        COMPLEX_DATA_STRIDE;

        // Allocating input buffer
        input_size = in_size * sizeof(DataType);
        problem->in = new DataType[in_size];

        // Allocating output buffer
        if (is_inplace)
        {
            output_size = input_size;
            problem->out = problem->in;
        }
        else
        {
            output_size = out_size * sizeof(DataType);
            problem->out = new DataType[out_size];
        }

        if (problem->in == NULL || problem->out == NULL)
        {
            cleanup_problem();
            throw std::runtime_error("Memory allocation failed for input or output arrays");
        }

        // Initialize input buffer with values based on strategy
        for (INT32 i = 0; i < in_size; i++)
        {
            problem->in[i] =
                get_value_based_on_strategy<DataType>(value_strategy, in_size);
        }

        // Initialize output buffer (for out-of-place only)
        if (!is_inplace)
        {
            for (INT32 i = 0; i < out_size; i++)
            {
                problem->out[i] = static_cast<DataType>(0.0);
            }
        }

        // Set threading and control parameters
        problem->pthr_fft.dynamic_load_model = DEFAULT_DYNAMIC_LOAD_MODEL;
        problem->pthr_fft.num_threads = DEFAULT_NUM_THREADS;
        problem->cntrl_params.logger_mode = AOCLFFTZ_LOG_NONE;
        problem->cntrl_params.measure_stats = 0;
        problem->cntrl_params.opt_level = -1;
        problem->cntrl_params.opt_off = 1;
    }

    // Calls the appropriate sample problem creation based on problem type
    VOID create_default_pdesc(bool is_forward = true, bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            create_pdesc<FLOAT, aoclfftz_dim_t>(is_forward, is_inplace,
                                                value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d>::value)
        {
            create_pdesc<DOUBLE, aoclfftz_dim_t>(is_forward, is_inplace,
                                                 value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_f_64_>::value)
        {
            create_pdesc<FLOAT, aoclfftz_dim_t_64_>(is_forward, is_inplace,
                                                    value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d_64_>::value)
        {
            create_pdesc<DOUBLE, aoclfftz_dim_t_64_>(is_forward, is_inplace,
                                                     value_strategy);
        }
        else
        {
            throw std::runtime_error("Unsupported problem type "
                                                "for create_pdesc.");
        }
    }

    /**
     * @brief Wrapper to create 1D problem descriptor based on problem type
     *
     * This function dispatches to the appropriate create_1d_pdesc template
     * instantiation based on the problem type (float/double and 32-bit/64-bit
     * indices).
     */
    VOID create_default_1d_pdesc(bool is_forward = true,
            bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            create_1d_pdesc<FLOAT, aoclfftz_dim_t>(is_forward, is_inplace,
                                                   value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_d>::value)
        {
            create_1d_pdesc<DOUBLE, aoclfftz_dim_t>(is_forward, is_inplace,
                                                    value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_f_64_>::value)
        {
            create_1d_pdesc<FLOAT, aoclfftz_dim_t_64_>(is_forward, is_inplace,
                                                       value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_d_64_>::value)
        {
            create_1d_pdesc<DOUBLE, aoclfftz_dim_t_64_>(is_forward, is_inplace,
                                                        value_strategy);
        }
        else
        {
            throw std::runtime_error("Unsupported problem type for create_1d_pdesc.");
        }
    }

    // Calls the appropriate setup API based on the specified problem type
    VOID *aoclfftz_setup(ProblemType *problem)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            return aoclfftz_setup_f(problem);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d>::value)
        {
            return aoclfftz_setup_d(problem);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_f_64_>::value)
        {
            return aoclfftz_setup_f_64_(problem);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d_64_>::value)
        {
            return aoclfftz_setup_d_64_(problem);
        }
        else
        {
            throw std::runtime_error("Unsupported problem type.");
        }
    }

    // Functions to retrieve supported option levels and flags
    std::vector<INT32> get_supported_optlevels()
    {
        return {0, 1, 2, 3};
    }

    std::vector<aoclfftz_flags_t> get_supported_flags()
    {
        std::vector<aoclfftz_flags_t> flags;
        for (UINT32 in_place : {0,1})
        {
            for (UINT32 in_order : {0})
            {
                for (UINT32 forward : {0,1})
                {
                    for (UINT32 complex : {0})
                    {
                        aoclfftz_flags flag;
                        flag.fft_placement       = in_place;
                        flag.storage_order       = in_order;
                        flag.fft_direction       = forward;
                        flag.fft_type            = complex;
                        flag.transpose_mode      = 0;
                        flag.bit_reproducibility = 0;
                        flags.push_back(flag);
                    }
                }
            }
        }
        return flags;
    }

    // Function to run the setup and validate the handle
    void run_setup_and_validate(INT32 err_no)
    {
        if (err_no == INVALID)
        {
            handle = aoclfftz_setup(problem);
            EXPECT_EQ(handle, nullptr);
        }
        else if (err_no == VALID)
        {
            handle = aoclfftz_setup(problem);
            EXPECT_NE(handle, nullptr);
            aoclfftz_destroy(handle);
        }
    }

    // Validates execute_io correctness using execute output as reference
    VOID validate_execute_io(bool is_forward)
    {
        cleanup_problem();
        create_default_pdesc(is_forward);

        handle = aoclfftz_setup(problem);

        // invoke execute API
        INT32 exe = aoclfftz_execute(handle);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

        // involve execute_io API
        UINTP input_size = 0;
        UINTP output_size = 0;
        get_inout_size(&input_size, &output_size);

        VOID *in, *out;
        in = malloc(input_size);
        out = malloc(output_size);

        VOID *temp_out = malloc(output_size);
        memcpy(in, problem->in, input_size);
        memcpy(temp_out, problem->out, output_size);
        memset(out, 0, output_size);
        exe = aoclfftz_execute_io(handle, in, out);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

        // Compare 'out' buffer against output buffer in problem desc
        INT32 ret = memcmp(out, temp_out, output_size);
        free(temp_out);
        temp_out = NULL;
        EXPECT_EQ(ret, 0); // Expect successful comparison

        aoclfftz_destroy(handle);
        free(in);
        in = NULL;
        free(out);
        out = NULL;
    }

    // Tests aoclfftz_execute_io after freeing original buffers and passing new ones.
    VOID validate_execute_io_after_buffer_free_and_alloc(bool is_forward, bool use_1d_problem)
    {
        cleanup_problem();

        if (use_1d_problem)
        {
            create_default_1d_pdesc(is_forward, false); // 1D out-of-place
        }
        else
        {
            create_default_pdesc(is_forward, false); // 3D out-of-place
        }

        handle = aoclfftz_setup(problem);
        EXPECT_NE(handle, nullptr) << "Setup failed";
        if (handle == nullptr)
        {
            return;
        }

        // First execute with original buffers to get reference output
        INT32 exe = aoclfftz_execute(handle);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS) << "First execute failed";

        UINTP in_size = 0;
        UINTP out_size = 0;
        get_inout_size(&in_size, &out_size);

        // Allocate all buffers together
        VOID *reference_output = malloc(out_size);
        VOID *new_in = malloc(in_size);
        VOID *new_out = malloc(out_size);

        // Check allocations, cleanup and return on failure
        if (reference_output == NULL || new_in == NULL || new_out == NULL)
        {
            ADD_FAILURE() << "Memory allocation failed";
            goto cleanup_and_return;
        }

        memcpy(new_in, problem->in, in_size);
        memcpy(reference_output, problem->out, out_size);
        memset(new_out, 0, out_size);

        // Free original buffers (simulating user freeing their buffers)
        delete[] problem->in;
        delete[] problem->out;
        problem->in = nullptr;
        problem->out = nullptr;

        // Call execute_io with new buffers
        exe = aoclfftz_execute_io(handle, new_in, new_out);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS) 
            << "execute_io failed after buffer realloc"
            << " [1D=" << use_1d_problem << ", forward=" << is_forward << "]";

        // Verify output matches reference (same input should produce same output)
        if(exe == AOCLFFTZ_SUCCESS)
        {
            INT32 cmp_result = memcmp(new_out, reference_output, out_size);
            EXPECT_EQ(cmp_result, 0)
                << "Output mismatch after buffer realloc"
                << " [1D=" << use_1d_problem << ", forward=" << is_forward << "]";
        }

    cleanup_and_return:
        aoclfftz_destroy(handle);
        handle = nullptr;
        free(reference_output);
        free(new_in);
        free(new_out);
    }
};
#endif // AOCLFFTZ_API_GTEST_H
