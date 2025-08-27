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

#include "test/gtest/api/api_gtest.h"

// Define the typed test case
TYPED_TEST_SUITE_P(AoclfftzAPITest);

// Setup API test cases
TYPED_TEST_P(AoclfftzAPITest, PTEST_CNTRL_PARAMETERS)
{
    for (auto optOff : {0,1})
    {
        // Invalid optlevel -2 to ensure setup doesn't fail on invalid inputs
        for (auto optLevel : {-2, -1, 0, 1, 2, 3})
        {
            this->problem->cntrl_params.opt_off = optOff;
            this->problem->cntrl_params.opt_level = optLevel;
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
    for (auto optOff : {0,1})
    {
        for (auto optLevel : this->get_supported_optlevels())
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
                        this->problem->cntrl_params.opt_off = optOff;
                        this->problem->cntrl_params.opt_level = optLevel;
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
    for (INT32 i = 0; i < 100; i++) // Fuzzing range
    {
        this->problem->cntrl_params.opt_level = dist_invalid(prng);
        this->problem->cntrl_params.opt_off = dist_invalid(prng);
        this->problem->cntrl_params.logger_mode = dist_invalid(prng);
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
    for (INT32 i = 0; i < 100; i++)
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
    for (INT32 i = 0; i < 100; i++)
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
    INT32 exe = aoclfftz_execute(this->handle);
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

// FIXME : Revisit the logic
// Destroy API test cases
TYPED_TEST_P(AoclfftzAPITest, PTEST_DESTROY_VALIDHANDLE_SOLUTION)
{
    this->handle = this->aoclfftz_setup(this->problem);
    INT32 exe = aoclfftz_execute(this->handle);
    aoclfftz_destroy(this->handle);
    ASSERT_FALSE(this->is_handle_null(this->handle));
}

TYPED_TEST_P(AoclfftzAPITest, PTEST_DESTROY_WITHOUT_EXECUTE)
{
    this->handle = this->aoclfftz_setup(this->problem);
    aoclfftz_destroy(this->handle);
    ASSERT_FALSE(this->is_handle_null(this->handle));
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
    VOID *handle = NULL;
    INT32 exe = aoclfftz_execute(handle);
    EXPECT_EQ(exe, AOCLFFTZ_EXECUTION_FAILURE)
                << "Execution return run_setup_and_validate failure";
}

TEST(AoclfftzAPITest, NTEST_EXECUTE_IO_INVALIDHANDLE)
{
    VOID* handle = NULL;
    VOID *in = NULL, *out = NULL;
    INT32 exe = aoclfftz_execute_io(handle, in, out);
    EXPECT_EQ(exe, -1) << "Execution failure for aoclfftz_execute_io API";
}

TEST(AoclfftzAPITest, NTEST_DESTROY_NULL_HANDLE)
{
    VOID *handle = NULL;
    aoclfftz_destroy(handle);
    EXPECT_TRUE(AoclfftzAPITest<void>::is_handle_null(handle));
}
