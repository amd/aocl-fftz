// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

#include "test/gtest/api/api_gtest.h"

// Define the typed test case
TYPED_TEST_SUITE_P(AoclfftzAPITest);

// Setup API test cases
TYPED_TEST_P(AoclfftzAPITest, PTEST_CNTRL_PARAMETERS)
{
    for (auto opt_off : {0,1})
    {
        // Invalid optlevel -2 to ensure setup doesn't fail on invalid inputs
        for (auto opt_level : {-2, -1, 0, 1, 2, 3})
        {
            this->problem->cntrl_params.opt_off = opt_off;
            this->problem->cntrl_params.opt_level = opt_level;
            this->run_setup_and_validate(VALID); // Run setup with valid case
        }
    }
}

TYPED_TEST_P(AoclfftzAPITest, PTEST_FLAGS)
{
    for (auto flags : this->get_supported_flags())
    {
        this->cleanup_problem();
        bool is_fwd = flags.fft_direction ? 0 : 1;
        bool is_inplace = flags.fft_placement ? 0 : 1;
        this->problem->flags.fft_placement  = flags.fft_placement;
        this->problem->flags.storage_order  = flags.storage_order;
        this->problem->flags.fft_direction  = flags.fft_direction;
        this->problem->flags.fft_type       = flags.fft_type;
        this->problem->flags.transpose_mode = flags.transpose_mode;
        /* While changing the flag to test, problem descriptor must be changed
         * as well to make sure that problem descriptor is valid and based on
         * updated flags. */
        this->create_default_pdesc(is_fwd, is_inplace);
        this->run_setup_and_validate(VALID);
    }
}

TYPED_TEST_P(AoclfftzAPITest, PTEST_THREADS)
{
    for (auto num_threads : {1})
    {
        for (auto dynamic_load_model : {1})
        {
            this->problem->pthr_fft.num_threads = num_threads;
            this->problem->pthr_fft.dynamic_load_model = dynamic_load_model;
            this->run_setup_and_validate(VALID);
        }
    }
}

// Test with all valid combinations of opt_off, opt_level, flags, pthr_fft
TYPED_TEST_P(AoclfftzAPITest, PTEST_COMBINE)
{
    for (auto opt_off : {0,1})
    {
        for (auto opt_level : this->get_supported_optlevels())
        {
            for (auto flags : this->get_supported_flags())
            {
                for (auto num_threads : {1})
                {
                    for (auto dynamic_load_model : {1})
                    {
                        bool is_fwd = flags.fft_direction ? 0 : 1;
                        bool is_inplace = flags.fft_placement ? 0 : 1;
                        /* While changing the flag to test, problem descriptor
                         * must be changed as well to make sure that problem
                         * descriptor is valid and based on updated flags. */
                        this->cleanup_problem();
                        this->problem->flags = flags;
                        this->create_default_pdesc(is_fwd, is_inplace);
                        this->problem->cntrl_params.opt_off = opt_off;
                        this->problem->cntrl_params.opt_level = opt_level;
                        this->problem->pthr_fft.num_threads = num_threads;
                        this->problem->pthr_fft.dynamic_load_model =
                                                            dynamic_load_model;
                        this->run_setup_and_validate(VALID);
                    }
                }
            }
        }
    }
}

// Setup variants negative testing
TYPED_TEST_P(AoclfftzAPITest, NTEST_PROBLEM_DESCRIPTOR)
{
    if (this->problem != NULL)
    {
        this->cleanup_problem();
        delete this->problem;
        this->problem = NULL;
    }
    this->run_setup_and_validate(INVALID); // Run setup with invalid case
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_IN_BUFFER)
{
    if (this->problem != NULL && this->problem->in != NULL)
    {
        delete[] this->problem->in;
        this->problem->in = NULL;
    }
    this->run_setup_and_validate(INVALID);
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_OUT_BUFFER)
{
    if (this->problem->out != NULL)
    {
        delete[] this->problem->out;
        this->problem->out = NULL;
    }
    this->run_setup_and_validate(INVALID);
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_CNTRL_PARAMS)
{
    INIT_RANDOM_NUM_GEN();
    for (FFTZ_INT32 i = 0; i < 100; i++) // Fuzzing range
    {
        this->problem->cntrl_params.opt_level = dist_invalid(prng);
        this->problem->cntrl_params.opt_off = dist_invalid(prng);
        this->problem->cntrl_params.logger_mode =
            (aoclfftz_logger_mode)(dist_invalid(prng));
        this->problem->cntrl_params.measure_stats = dist_invalid(prng);
        this->run_setup_and_validate(VALID);
    }
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_FLAGS)
{
    for (auto flags : unsupported_flags)
    {
        /* No need to call `create_default_pdesc()` for unsupported flags,
         * problem descriptor has default initialization. */
        this->problem->flags.fft_placement  = flags.fft_placement;
        this->problem->flags.storage_order  = flags.storage_order;
        this->problem->flags.fft_direction  = flags.fft_direction;
        this->problem->flags.fft_type       = flags.fft_type;
        this->problem->flags.transpose_mode = flags.transpose_mode;
        this->run_setup_and_validate(INVALID);
    }
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_INPLACE_IO_STRIDES)
{
    /* While changing the flag to test, problem descriptor must be changed
     * as well to make sure that problem descriptor is valid and based on
     * updated flags. */
    this->cleanup_problem();
    this->problem->flags = {0};
    bool is_fwd = this->problem->flags.fft_direction ? 0 : 1;
    bool is_inplace = this->problem->flags.fft_placement ? 0 : 1;
    this->create_default_pdesc(is_fwd, is_inplace);
    // Setting non-identical input and output stride values, expecting
    // setup to fail, as inplace problems require same input & output strides
    this->problem->dims[2].in_stride = 200;
    this->problem->dims[2].out_stride = 250;
    this->run_setup_and_validate(INVALID);
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_DIM_RANK)
{
    for (auto rank : unsupported_rank)
    {
        this->problem->dim_rank = rank;
        this->run_setup_and_validate(INVALID);
    }
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_VEC_RANK)
{
    for (auto rank : unsupported_rank)
    {
        this->problem->vec_rank = rank;
        this->run_setup_and_validate(INVALID);
    }
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_DIMS)
{
    if (this->problem->dims != NULL)
    {
        delete[] this->problem->dims;
        this->problem->dims = NULL;
    }
    this->run_setup_and_validate(INVALID);
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_VECS)
{
    if (this->problem->vecs != NULL)
    {
        delete[] this->problem->vecs;
        this->problem->vecs = NULL;
    }
    this->run_setup_and_validate(INVALID);
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_DIMS_STRIDES)
{
    INIT_RANDOM_NUM_GEN(); // Initialize random number generator
    for (FFTZ_INT32 i = 0; i < 100; i++)
    {
        if (this->problem->dims != NULL)
        {
            this->problem->dims->n = dist_invalid(prng);
            this->problem->dims->in_stride = dist_invalid(prng);
            this->problem->dims->out_stride = dist_invalid(prng);
            this->run_setup_and_validate(INVALID);
        }
    }
}

TYPED_TEST_P(AoclfftzAPITest, NTEST_VECS_STRIDES)
{
    INIT_RANDOM_NUM_GEN(); // Initialize random number generator
    for (FFTZ_INT32 i = 0; i < 100; i++)
    {
        if (this->problem->vecs != NULL)
        {
            this->problem->vecs->n = dist_invalid(prng);
            this->problem->vecs->in_stride = dist_invalid(prng);
            this->problem->vecs->out_stride = dist_invalid(prng);
            this->run_setup_and_validate(INVALID);
        }
    }
}

// Execute API test cases
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_VALIDHANDLE)
{
    this->handle = this->aoclfftz_setup(this->problem);
    FFTZ_INT32 exe = aoclfftz_execute(this->handle);
    EXPECT_EQ(exe, 0); // Expect successful execution
    aoclfftz_destroy(this->handle);
}

// Execute_dft API test cases - Forward FFT Transform
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_IO_VALIDHANDLE_FORWARD)
{
    // this test compares the outputs obtained through
    //  (a) invoke of execute API where in/out buffers are bound to problem desc
    //  (b) invoke of execute_io API where in/out buffers are passed explicitly
    // Forward FFT transform: problem->flags = 0b0001

    this->validate_execute_io(true);
}

// Execute_io API test cases - Backward FFT Transform
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_IO_VALIDHANDLE_BACKWARD)
{
    // this test compares the outputs obtained through
    //  (a) invoke of execute API where in/out buffers are bound to problem desc
    //  (b) invoke of execute_io API where in/out buffers are passed explicitly
    // Backward FFT transform: problem->flags = 0b0001

    this->validate_execute_io(false);
}

// Test execute_io with new buffers after freeing original - 1D case (ct_buffer
// == NULL)
TYPED_TEST_P(AoclfftzAPITest,
             PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_1D_FORWARD)
{
    this->validate_execute_io_after_buffer_free_and_alloc(true,
                                                          true); // 1D forward
}

TYPED_TEST_P(AoclfftzAPITest,
             PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_1D_BACKWARD)
{
    this->validate_execute_io_after_buffer_free_and_alloc(false,
                                                          true); // 1D backward
}

// Test execute_io with new buffers after freeing original - 3D case (ct_buffer
// != NULL)
TYPED_TEST_P(AoclfftzAPITest,
             PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_3D_FORWARD)
{
    this->validate_execute_io_after_buffer_free_and_alloc(true,
                                                          false); // 3D forward
}

TYPED_TEST_P(AoclfftzAPITest,
             PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_3D_BACKWARD)
{
    this->validate_execute_io_after_buffer_free_and_alloc(false,
                                                          false); // 3D backward
}

// Execute API test with near-edge values and output validation
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_WITH_NEAR_EDGE_VALUES)
{
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_pdesc(is_forward, is_inplace,
                                      InputValueStrategy::NEAR_EDGE);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr);

            // Execute the FFT
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

            // Validate that output values are reasonable (not NaN, Infinity,
            // etc.)
            FFTZ_UINTP output_size_bytes = 0;
            FFTZ_UINTP input_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            using DataType =
                std::remove_pointer_t<decltype(this->problem->out)>;
            FFTZ_UINTP num_elements = output_size_bytes / sizeof(DataType);
            output_validation_stats stats =
                            validate_output_array<DataType>(this->problem->out,
                                                            num_elements);

            FFTZ_UINTP invalid_count = stats.nan_count + stats.inf_count;
            EXPECT_EQ(invalid_count, 0U)
                << "Found " << invalid_count << " invalid values ("
                << stats.nan_count << " NaN, " << stats.inf_count
                << " Inf) in output array" << " out of " << num_elements
                << " total elements" << " [is_forward=" << is_forward
                << ", is_inplace=" << is_inplace << "]";

            aoclfftz_destroy(this->handle);
        }
    }
}

// Robustness test with special values (NaN, Inf, MAX, MIN) - validates NaN
// propagation
TYPED_TEST_P(AoclfftzAPITest, PTEST_ROBUSTNESS_WITH_NAN_VALUES)
{
    // This test ensures the API doesn't crash with extreme inputs
    // and validates that if input has NaN, output should be entirely filled
    // with NaN
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_1d_pdesc(is_forward, is_inplace,
                                          InputValueStrategy::SPECIAL_VALUES);

            FFTZ_UINTP input_size_bytes = 0;
            FFTZ_UINTP output_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            // Use decltype to automatically get the data type from problem->in
            using DataType = std::remove_pointer_t<decltype(this->problem->in)>;
            FFTZ_UINTP num_input_elements = input_size_bytes / sizeof(DataType);
            FFTZ_UINTP num_output_elements = output_size_bytes / sizeof(
                DataType);

            // Analyze input array before execution
            output_validation_stats input_stats =
                            validate_output_array<DataType>(this->problem->in,
                                                            num_input_elements);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr)
                << "Setup should not return null handle even with special "
                   "values"
                << " [is_forward=" << is_forward
                << ", is_inplace=" << is_inplace << "]";

            // Execute the FFT - it may succeed or fail, but should not crash
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS)
                << "Execute should return AOCLFFTZ_SUCCESS"
                << " [is_forward=" << is_forward
                << ", is_inplace=" << is_inplace << "]";

            // Analyze output array after execution
            output_validation_stats output_stats =
                        validate_output_array<DataType>(this->problem->out,
                                                        num_output_elements);

            // Validate NaN propagation: if input has NaN, output MUST also have
            // NaN
            if (input_stats.nan_count > 0)
            {
                EXPECT_EQ(output_stats.nan_count, num_output_elements)
                    << "NaN propagation failed: Input had "
                    << input_stats.nan_count << " NaN values, "
                    << "but only " << output_stats.nan_count
                    << " out of " << num_output_elements
                    << " output elements are NaN (expected ALL to be NaN)"
                    << " [is_forward=" << is_forward << ", is_inplace="
                    << is_inplace << "]";
            }

            aoclfftz_destroy(this->handle);
        }
    }
}

// Execute API test with special non-NaN values - output should only contain NaN
// or Inf
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_WITH_SPECIAL_EXCEPT_NAN_VALUES)
{
    // Test FFT execution with special values excluding NaN (Inf, MAX, MIN)
    // Output should only contain NaN or Infinity (no finite values including
    // zeros)
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_1d_pdesc(is_forward, is_inplace,
                                        InputValueStrategy::SPECIAL_EXCEPT_NAN);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr);

            // Execute the FFT
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

            // Validate that output contains only NaN or Inf (no finite values)
            FFTZ_UINTP output_size_bytes = 0;
            FFTZ_UINTP input_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            using DataType =
                std::remove_pointer_t<decltype(this->problem->out)>;
            FFTZ_UINTP num_elements = output_size_bytes / sizeof(DataType);
            output_validation_stats stats =
                            validate_output_array<DataType>(this->problem->out,
                                                            num_elements);

            FFTZ_UINTP special_count = stats.nan_count + stats.inf_count;
            EXPECT_EQ(special_count, num_elements)
                << "Output validation failed for special non-NaN inputs: "
                << "Expected all " << num_elements
                << " elements to be NaN or Inf, but found "
                << special_count << " special values ("
                << stats.nan_count << " NaN, " << stats.inf_count << " Inf), "
                << stats.zero_count << " zeros, " << stats.nonzero_count
                << " finite non-zeros"
                << " [is_forward=" << is_forward << ", is_inplace="
                << is_inplace << "]";

            // Additional validation: no finite values should exist
            EXPECT_EQ(stats.zero_count, 0U)
                << "Found " << stats.zero_count << " zero values (expected 0)"
                << " [is_forward=" << is_forward << ", is_inplace="
                << is_inplace << "]";

            EXPECT_EQ(stats.nonzero_count, 0U)
                << "Found " << stats.nonzero_count
                << " finite non-zero values (expected 0)"
                << " [is_forward=" << is_forward
                << ", is_inplace=" << is_inplace << "]";

            aoclfftz_destroy(this->handle);
        }
    }
}

// Execute API test with only tiny values - output should not be entirely zero
// It should not contain NaN or Inf
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_WITH_ONLY_TINY_VALUES)
{
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_pdesc(is_forward, is_inplace,
                                      InputValueStrategy::TINY_VALUES_ONLY);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr);

            // Execute the FFT
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

            // Validate that output values are not entirely zero and not NaN/Inf
            FFTZ_UINTP output_size_bytes = 0;
            FFTZ_UINTP input_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            using DataType =
                std::remove_pointer_t<decltype(this->problem->out)>;
            FFTZ_UINTP num_elements = output_size_bytes / sizeof(DataType);
            output_validation_stats stats =
                            validate_output_array<DataType>(this->problem->out,
                                                            num_elements);

            FFTZ_UINTP invalid_count = stats.nan_count + stats.inf_count;
            EXPECT_EQ(invalid_count, 0U)
                << "Output validation failed for tiny inputs: Found "
                << invalid_count << " invalid values (" << stats.nan_count
                << " NaN, " << stats.inf_count << " Inf)" << " out of "
                << num_elements << " total elements" << " [is_forward="
                << is_forward << ", is_inplace=" << is_inplace << "]";

            EXPECT_GT(stats.nonzero_count, 0U)
                << "Output validation failed for tiny inputs: All outputs are "
                   "zero"
                << " [is_forward=" << is_forward
                << ", is_inplace=" << is_inplace << "]";

            aoclfftz_destroy(this->handle);
        }
    }
}

// Execute API test with only large values - output should not be infinity or
// NaN
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_WITH_ONLY_LARGE_VALUES)
{
    // Test FFT execution with only large values (close to but not exceeding
    // MAX)
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_pdesc(is_forward, is_inplace,
                                      InputValueStrategy::LARGE_VALUES_ONLY);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr);

            // Execute the FFT
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

            // Validate that output values are not infinity or NaN
            FFTZ_UINTP output_size_bytes = 0;
            FFTZ_UINTP input_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            using DataType =
                std::remove_pointer_t<decltype(this->problem->out)>;
            FFTZ_UINTP num_elements = output_size_bytes / sizeof(DataType);
            output_validation_stats stats =
                            validate_output_array<DataType>(this->problem->out,
                                                            num_elements);

            FFTZ_UINTP invalid_count = stats.nan_count + stats.inf_count;
            EXPECT_EQ(invalid_count, 0U)
                << "Output validation failed for large inputs: Found "
                << invalid_count << " invalid values (" << stats.nan_count
                << " NaN, " << stats.inf_count << " Inf)"
                << " out of " << num_elements << " total elements"
                << " [is_forward=" << is_forward << ", is_inplace="
                << is_inplace << "]";

            aoclfftz_destroy(this->handle);
        }
    }
}

// Execute API test with full zero input - output should be all zeros
TYPED_TEST_P(AoclfftzAPITest, PTEST_EXECUTE_WITH_ONLY_ZERO_VALUES)
{
    for (auto is_forward : {true, false})
    {
        for (auto is_inplace : {true, false})
        {
            this->cleanup_problem();
            this->create_default_pdesc(is_forward, is_inplace,
                                       InputValueStrategy::FULL_ZERO);

            this->handle = this->aoclfftz_setup(this->problem);
            EXPECT_NE(this->handle, nullptr);

            // Execute the FFT
            FFTZ_INT32 exe = aoclfftz_execute(this->handle);

            EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

            // Validate that all output values are zero
            FFTZ_UINTP output_size_bytes = 0;
            FFTZ_UINTP input_size_bytes = 0;
            this->get_inout_size(&input_size_bytes, &output_size_bytes);

            using DataType =
                std::remove_pointer_t<decltype(this->problem->out)>;
            FFTZ_UINTP num_elements = output_size_bytes / sizeof(DataType);
            output_validation_stats stats =
                            validate_output_array<DataType>(this->problem->out,
                                                            num_elements);

            EXPECT_EQ(stats.zero_count, num_elements)
                << "Output validation failed for zero inputs: Expected all "
                << num_elements << " elements to be zero, but found "
                << stats.zero_count << " zeros, "
                << stats.nonzero_count << " non-zeros, "
                << stats.nan_count << " NaN, " << stats.inf_count << " Inf"
                << " [is_forward=" << is_forward << ", is_inplace="
                << is_inplace << "]";

            aoclfftz_destroy(this->handle);
        }
    }
}

// Destroy API test cases
TYPED_TEST_P(AoclfftzAPITest, PTEST_DESTROY_VALIDHANDLE_SOLUTION)
{
    this->handle = this->aoclfftz_setup(this->problem);
    aoclfftz_execute(this->handle);
    aoclfftz_destroy(this->handle);
    ASSERT_FALSE(is_handle_null(this->handle));
}

TYPED_TEST_P(AoclfftzAPITest, PTEST_DESTROY_WITHOUT_EXECUTE)
{
    this->handle = this->aoclfftz_setup(this->problem);
    aoclfftz_destroy(this->handle);
    ASSERT_FALSE(is_handle_null(this->handle));
}

// Register all test cases together
REGISTER_TYPED_TEST_SUITE_P(
    AoclfftzAPITest,
    PTEST_CNTRL_PARAMETERS,
    PTEST_FLAGS,
    PTEST_THREADS,
    PTEST_COMBINE,
    NTEST_PROBLEM_DESCRIPTOR,
    NTEST_IN_BUFFER,
    NTEST_OUT_BUFFER,
    NTEST_CNTRL_PARAMS,
    NTEST_FLAGS,
    NTEST_INPLACE_IO_STRIDES,
    NTEST_DIM_RANK,
    NTEST_VEC_RANK,
    NTEST_DIMS,
    NTEST_VECS,
    NTEST_DIMS_STRIDES,
    NTEST_VECS_STRIDES,
    PTEST_EXECUTE_VALIDHANDLE,
    PTEST_EXECUTE_IO_VALIDHANDLE_FORWARD,
    PTEST_EXECUTE_IO_VALIDHANDLE_BACKWARD,
    PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_1D_FORWARD,
    PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_1D_BACKWARD,
    PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_3D_FORWARD,
    PTEST_EXECUTE_IO_OOP_ALLOC_NEW_FREE_OLD_3D_BACKWARD,
    PTEST_EXECUTE_WITH_NEAR_EDGE_VALUES,
    PTEST_ROBUSTNESS_WITH_NAN_VALUES,
    PTEST_EXECUTE_WITH_SPECIAL_EXCEPT_NAN_VALUES,
    PTEST_EXECUTE_WITH_ONLY_TINY_VALUES,
    PTEST_EXECUTE_WITH_ONLY_LARGE_VALUES,
    PTEST_EXECUTE_WITH_ONLY_ZERO_VALUES,
    PTEST_DESTROY_VALIDHANDLE_SOLUTION,
    PTEST_DESTROY_WITHOUT_EXECUTE
);

// Using TestTypes for different problem types
using TestTypes = ::testing::Types<aoclfftz_prob_desc_f, aoclfftz_prob_desc_d,
                        aoclfftz_prob_desc_f_64_, aoclfftz_prob_desc_d_64_>;

INSTANTIATE_TYPED_TEST_SUITE_P(FFTZ_tests_setup_API,
                                AoclfftzAPITest, TestTypes);

TEST(AoclfftzAPITest, NTEST_EXECUTE_INVALIDHANDLE)
{
    FFTZ_VOID *handle = NULL;
    FFTZ_INT32 exe = aoclfftz_execute(handle);
    EXPECT_EQ(exe, AOCLFFTZ_EXECUTION_FAILURE)
                << "Execution return run_setup_and_validate failure";
}

TEST(AoclfftzAPITest, NTEST_EXECUTE_IO_INVALIDHANDLE)
{
    FFTZ_VOID* handle = NULL;
    FFTZ_VOID *in = NULL, *out = NULL;
    FFTZ_INT32 exe = aoclfftz_execute_io(handle, in, out);
    EXPECT_EQ(exe, -1) << "Execution failure for aoclfftz_execute_io API";
}

TEST(AoclfftzAPITest, NTEST_DESTROY_NULL_HANDLE)
{
    FFTZ_VOID *handle = NULL;
    aoclfftz_destroy(handle);
    EXPECT_TRUE(is_handle_null(handle));
}

#ifdef AOCLFFTZ_API_CONCURRENCY_TESTS
// ===========================================================================
// Concurrent aoclfftz_execute_io tests (C2C)
// ---------------------------------------------------------------------------
// Verifies that aoclfftz_execute_io is safe to call from multiple application
// threads simultaneously on a single shared handle. The cases below cover the
// solvers that exercise the per-call scratch surface introduced for MT-safety:
//   * Bluestein  -> bs_in_base / bs_out_base
//   * Split-radix -> sr_input_copy_base
//   * NDim/CTL1D -> ct_buffer
// ===========================================================================

namespace concurrent_exec_io
{
    // Stress one problem across both inplace & out-of-place over a range of internal
    // thread counts, with the number of application threads calling execute_io
    // concurrently such that internal_threads * app_threads ~= cores
    template<typename Fixture>
    void sweep(Fixture *f, const std::vector<FFTZ_INT32> &dims,
               FFTZ_INT32 batch, bool is_real = false)
    {
        const FFTZ_INT32 max_procs = omp_get_num_procs();
        // The app thread count below is derived from max_procs, so on a single
        // usable core every sweep reports success against a knowingly racy library,
        // so skip rather than pass vacuously.
        if (max_procs < 2)
        {
            GTEST_SKIP() << "concurrency sweep needs at least 2 usable cores, "
                            "omp_get_num_procs() reports " << max_procs;
        }
        // App threads always run concurrently; internal threads only matter in
        // a multi-threaded library, so a single-thread build sweeps just 1.
#ifdef MULTI_THREADING
        const std::vector<FFTZ_INT32> internal_thread_counts =
            {1, 3, 8, 40, max_procs};
#else
        const std::vector<FFTZ_INT32> internal_thread_counts = {1};
#endif
        // Both C2C directions share one solver tree, so sweeping forward covers
        // the scratch surface. R2C and C2R are distinct trees and need both.
        const bool directions[] = {true, false};
        const FFTZ_INT32 num_directions = is_real ? 2 : 1;

        for (FFTZ_INT32 d = 0; d < num_directions; d++)
        {
            const bool is_forward = directions[d];
            for (bool inplace : {false, true})
            {
                for (FFTZ_INT32 num_threads : internal_thread_counts)
                {
                    if (num_threads > max_procs)
                    {
                        continue;
                    }
                    const FFTZ_INT32 concurrent_api_count =
                        max_procs / num_threads;
                    f->run_concurrent_execute_io(dims, batch, inplace,
                                                 is_forward,
                                                 concurrent_api_count,
                                                 num_threads, is_real);
                    if (::testing::Test::HasFatalFailure())
                    {
                        return;
                    }
                }
            }
        }
    }
} // namespace concurrent_exec_io

template<typename ProblemType>
class AoclfftzConcurrentTest : public AoclfftzAPITest<ProblemType>
{
};

TYPED_TEST_SUITE_P(AoclfftzConcurrentTest);

// Each test runs both placements (OOP + in-place) via sweep();
TYPED_TEST_P(AoclfftzConcurrentTest, C2C_DIRECT)
{
    concurrent_exec_io::sweep(this, {15}, 1);
}

TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BATCHED_DIRECT)
{
    concurrent_exec_io::sweep(this, {16}, 8);
}

TYPED_TEST_P(AoclfftzConcurrentTest, C2C_CTL1D)
{
    concurrent_exec_io::sweep(this, {256}, 32);
}

TYPED_TEST_P(AoclfftzConcurrentTest, C2C_NDIM)
{
    concurrent_exec_io::sweep(this, {21, 25, 32}, 1);
}

TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BATCHED_NDIM)
{
    concurrent_exec_io::sweep(this, {5, 25, 32}, 7);
}

TYPED_TEST_P(AoclfftzConcurrentTest, C2C_SPLIT_RADIX)
{
    concurrent_exec_io::sweep(this, {4096}, 1);
}

// Bluestein: prime size > 16 (not kernel-supported, not CT-solvable).
TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BLUESTEIN)
{
    concurrent_exec_io::sweep(this, {199}, 1);
}

// Batched Bluestein: batched parent with Bluestein child.
TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BATCHED_BLUESTEIN)
{
    concurrent_exec_io::sweep(this, {53}, 10);
}

// Batched NDim Bluestein with (outer dim < inner dim).
TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BATCHED_NDIM_BS_1)
{
    concurrent_exec_io::sweep(this, {19, 97}, 5);
}

// Batched NDim Bluestein with (outer dim > inner dim).
TYPED_TEST_P(AoclfftzConcurrentTest, C2C_BATCHED_NDIM_BS_2)
{
    concurrent_exec_io::sweep(this, {97, 19}, 3);
}

REGISTER_TYPED_TEST_SUITE_P(
    AoclfftzConcurrentTest,
    C2C_DIRECT,
    C2C_BATCHED_DIRECT,
    C2C_CTL1D,
    C2C_NDIM,
    C2C_BATCHED_NDIM,
    C2C_SPLIT_RADIX,
    C2C_BLUESTEIN,
    C2C_BATCHED_BLUESTEIN,
    C2C_BATCHED_NDIM_BS_1,
    C2C_BATCHED_NDIM_BS_2
);

// Concurrency is the focus here, not type coverage, so run a single type
// (double) to keep the heavy stress suite fast.
using ConcurrentTestTypes = ::testing::Types<aoclfftz_prob_desc_d>;
INSTANTIATE_TYPED_TEST_SUITE_P(FFTZ_tests_concurrent_execute_io,
                                AoclfftzConcurrentTest, ConcurrentTestTypes);

// ===========================================================================
// Concurrent aoclfftz_execute_io tests (REAL: R2C / C2R)
// ---------------------------------------------------------------------------
// Same intent as the C2C suite, for the real path made concurrency-safe via
// the read-only tree + per-call ctx. Each case runs both R2C (forward) and C2R
// (backward), each in both placements, over a range of internal thread counts.
//
// The cases reach every per-call scratch resource: the aux ping-pong pair,
// the C2R-only ndim aux, the per-thread stride slots used by the real CT C2C
// kernels (all sliced by slot_idx), and the Bluestein pool.
// ===========================================================================

template<typename ProblemType>
class AoclfftzConcurrentRealTest : public AoclfftzAPITest<ProblemType>
{
};

TYPED_TEST_SUITE_P(AoclfftzConcurrentRealTest);

// 1D direct
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_DIRECT)
{
    concurrent_exec_io::sweep(this, {16}, 1, true);
}

// 1D fused batched-direct node.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_DIRECT)
{
    concurrent_exec_io::sweep(this, {16}, 8, true);
}

// CT One-level solver: checks ping-pong buffers are not shared between
// concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_CT_ONE_LEVEL)
{
    concurrent_exec_io::sweep(this, {60}, 1, true);
}

// Batched CT one-level solver: checks ping-pong buffers are partitioned within a
// single execute_io call and also are not shared between concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_CT_ONE_LEVEL)
{
    concurrent_exec_io::sweep(this, {60}, 8, true);
}

// Multi-level CT chain: checks ping-pong buffers are not shared between
// concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_CT_MULTILEVEL)
{
    concurrent_exec_io::sweep(this, {625}, 1, true);
}

// Batched multi-level CT chain: checks ping-pong buffers are partitioned within a
// single execute_io call and also are not shared between concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_CT_MULTILEVEL)
{
    concurrent_exec_io::sweep(this, {625}, 5, true);
}

// REAL_NDIM: checks aux_pool_base_ndim is not shared between concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_NDIM)
{
    concurrent_exec_io::sweep(this, {21, 22, 25}, 1, true);
}

// Batched REAL_NDIM: checks aux_pool_base_ndim is partitioned within a single
// execute_io call and also are not shared between concurrent execute_io calls.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_NDIM)
{
    concurrent_exec_io::sweep(this, {21, 22, 25}, 3, true);
}

// Outer dim is prime, inner dim is composite.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_NDIM_BS_1)
{
    concurrent_exec_io::sweep(this, {19, 21}, 5, true);
}

// Outer dim is prime, inner dim is prime: (outer dim < inner dim).
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_NDIM_BS_2)
{
    concurrent_exec_io::sweep(this, {19, 97}, 5, true);
}

// Outer dim is prime, inner dim is prime: (outer dim > inner dim).
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_NDIM_BS_3)
{
    concurrent_exec_io::sweep(this, {97, 19}, 5, true);
}

// Outer dim is composite, inner dim is prime.
TYPED_TEST_P(AoclfftzConcurrentRealTest, REAL_BATCHED_NDIM_BS_4)
{
    concurrent_exec_io::sweep(this, {18, 23}, 5, true);
}

REGISTER_TYPED_TEST_SUITE_P(
    AoclfftzConcurrentRealTest,
    REAL_DIRECT,
    REAL_BATCHED_DIRECT,
    REAL_CT_ONE_LEVEL,
    REAL_BATCHED_CT_ONE_LEVEL,
    REAL_CT_MULTILEVEL,
    REAL_BATCHED_CT_MULTILEVEL,
    REAL_NDIM,
    REAL_BATCHED_NDIM,
    REAL_BATCHED_NDIM_BS_1,
    REAL_BATCHED_NDIM_BS_2,
    REAL_BATCHED_NDIM_BS_3,
    REAL_BATCHED_NDIM_BS_4
);

using ConcurrentRealTestTypes = ::testing::Types<aoclfftz_prob_desc_d>;
INSTANTIATE_TYPED_TEST_SUITE_P(FFTZ_tests_concurrent_execute_io_real,
                               AoclfftzConcurrentRealTest,
                               ConcurrentRealTestTypes);
#endif // AOCLFFTZ_API_CONCURRENCY_TESTS
