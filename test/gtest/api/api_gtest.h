// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AOCLFFTZ_API_GTEST_H
#define AOCLFFTZ_API_GTEST_H

#include <string>
#include <limits>
#include <vector>
#include <memory>
#include <type_traits>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <climits>
#include <algorithm>
#include <iostream> // Add this include for debug printing
#ifdef AOCLFFTZ_API_CONCURRENCY_TESTS
#include <omp.h>
#endif
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
} invalid_case;

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

const std::vector<FFTZ_INT32> unsupported_rank = { INT32_MIN, -1 };

// Random number generator setup for generating invalid test values
#define INIT_RANDOM_NUM_GEN() \
    std::random_device rd; \
    std::mt19937 prng(rd()); \
    std::uniform_int_distribution<> dist_invalid(INT32_MIN, 0); \
    prng.seed(42);

// Function to check if the handle is destroyed
static bool is_handle_null(FFTZ_VOID *handle)
{
    return (handle == NULL);
}

// Template class for testing the AOCL-FFTZ API
template<typename ProblemType>
class AoclfftzAPITest : public ::testing::Test
{
public:
    FFTZ_INT32 opt_off;
    FFTZ_INT32 opt_level;
    FFTZ_INT32 num_threads;
    FFTZ_UINT32 dynamic_load_model;
    ProblemType *problem;
    FFTZ_VOID *handle;
    FFTZ_UINTP input_size;
    FFTZ_UINTP output_size;
    void SetUp() override
    {
        problem = NULL;
        handle = NULL;
        num_threads = DEFAULT_NUM_THREADS;
        dynamic_load_model = DEFAULT_DYNAMIC_LOAD_MODEL;

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
    FFTZ_VOID cleanup_problem()
    {
        if (problem == NULL)
        {
            return;
        }
        /* Default value of is_inplace is 0,
         * If problem->flags are invalid, free the memory buffer based on default flags */
        bool is_inplace = is_valid_flags(problem->flags) ?
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

    bool is_valid_flags(aoclfftz_flags_t flags)
    {
        if (flags.storage_order || flags.transpose_mode || flags.fft_type)
        {
            return 0;
        }
        return 1;
    }

    FFTZ_VOID get_inout_size(FFTZ_UINTP *in_size, FFTZ_UINTP *out_size)
    {
        in_size[0] = input_size;
        out_size[0] = output_size;
    }

    // Create a sample problem for testing: fixed 3D 10 x 30 x 30, batch 1.
    // Thin wrapper over the generalized create_custom_pdesc.
    template<typename DataType, typename DimT>
    FFTZ_VOID create_pdesc(bool is_forward = true, bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        num_threads = DEFAULT_NUM_THREADS;
        create_custom_pdesc<DataType, DimT>({10, 30, 30}, 1, is_forward,
                is_inplace, value_strategy, DEFAULT_DYNAMIC_LOAD_MODEL);
    }

    /**
     * @brief Create a 1D problem descriptor with large size (16384 = 2^14).
     *
     * Thin wrapper over the generalized create_custom_pdesc.
     */
    template<typename DataType, typename DimT>
    FFTZ_VOID create_1d_pdesc(bool is_forward = true, bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        num_threads = DEFAULT_NUM_THREADS;
        create_custom_pdesc<DataType, DimT>({16384}, 1, is_forward,
                is_inplace, value_strategy, DEFAULT_DYNAMIC_LOAD_MODEL);
    }

    /**
     * @brief Build a C2C problem from arbitrary dims and a batch count.
     *
     * Generalized builder behind all create_*_pdesc helpers: derives contiguous
     * strides, allocates buffers, and fills the input.
     */
    template<typename DataType, typename DimT>
    FFTZ_VOID create_custom_pdesc(const std::vector<FFTZ_INT32> &dim_sizes,
            FFTZ_INT32 batch,
            bool is_forward, bool is_inplace,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE,
            FFTZ_UINT32 dyn_load_model = 0,
            FFTZ_INT32 internal_threads = DEFAULT_NUM_THREADS)
    {
        if (problem == NULL)
        {
            return;
        }
        if (dim_sizes.empty())
        {
            throw std::runtime_error("create_custom_pdesc requires at least "
                                     "one dimension");
        }

        FFTZ_INT32 in_size = 0, out_size = 0;
        problem->dim_rank = (FFTZ_INT32)dim_sizes.size();
        problem->vec_rank = 1;
        problem->dims = new DimT[problem->dim_rank];
        problem->vecs = new DimT[problem->vec_rank];
        if (problem->dims == NULL || problem->vecs == NULL)
        {
            cleanup_problem();
            throw std::runtime_error(
                "Memory allocation failed for dims or vecs!");
        }

        // Preserve all other flag fields, only set direction and placement.
        problem->flags.fft_direction = is_forward ? 0 : 1;
        problem->flags.fft_placement = is_inplace ? 0 : 1;

        // Derive unit-innermost contiguous strides from the dim sizes.
        FFTZ_INTP stride = 1;
        for (FFTZ_INT32 d = 0; d < problem->dim_rank; d++)
        {
            problem->dims[d].n          = dim_sizes[d];
            problem->dims[d].in_stride  = stride;
            problem->dims[d].out_stride = stride;
            stride *= dim_sizes[d];
        }

        problem->vecs[0].n          = batch;
        problem->vecs[0].in_stride  = stride;
        problem->vecs[0].out_stride = stride;

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
            throw std::runtime_error("Memory allocation failed for input or "
                                     "output arrays");
        }

        // Initialize input buffer with values based on strategy
        for (FFTZ_INT32 i = 0; i < in_size; i++)
        {
            problem->in[i] =
                get_value_based_on_strategy<DataType>(value_strategy, in_size);
        }

        // Initialize output buffer (for out-of-place only)
        if (!is_inplace)
        {
            for (FFTZ_INT32 i = 0; i < out_size; i++)
            {
                problem->out[i] = static_cast<DataType>(0.0);
            }
        }

        // Set threading and control parameters
        problem->pthr_fft.dynamic_load_model = dyn_load_model;
        problem->pthr_fft.num_threads = internal_threads;
        problem->cntrl_params.logger_mode = AOCLFFTZ_LOG_NONE;
        problem->cntrl_params.measure_stats = 0;
        problem->cntrl_params.opt_level = -1;
        problem->cntrl_params.opt_off = 1;
    }

    /**
     * @brief Dispatch create_custom_pdesc to the right template instantiation
     * based on the problem type (float/double and 32-bit/64-bit indices).
     */
    FFTZ_VOID create_default_custom_pdesc(
            const std::vector<FFTZ_INT32> &dim_sizes,
            FFTZ_INT32 batch, bool is_forward, bool is_inplace,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE,
            FFTZ_INT32 internal_threads = DEFAULT_NUM_THREADS)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            create_custom_pdesc<FFTZ_FLOAT, aoclfftz_dim_t>(dim_sizes, batch,
                                    is_forward, is_inplace, value_strategy,
                                    /*dyn_load_model=*/0, internal_threads);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d>::value)
        {
            create_custom_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t>(dim_sizes, batch,
                                    is_forward, is_inplace, value_strategy,
                                    /*dyn_load_model=*/0, internal_threads);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_f_64_>::value)
        {
            create_custom_pdesc<FFTZ_FLOAT, aoclfftz_dim_t_64_>(dim_sizes,
                                    batch, is_forward, is_inplace,
                                    value_strategy, /*dyn_load_model=*/0,
                                    internal_threads);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d_64_>::value)
        {
            create_custom_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t_64_>(dim_sizes,
                                    batch, is_forward, is_inplace,
                                    value_strategy, /*dyn_load_model=*/0,
                                    internal_threads);
        }
        else
        {
            throw std::runtime_error("Unsupported problem type for "
                                     "create_custom_pdesc.");
        }
    }

    // Calls the appropriate sample problem creation based on problem type
    FFTZ_VOID create_default_pdesc(bool is_forward = true,
        bool is_inplace = false,
        InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            create_pdesc<FFTZ_FLOAT, aoclfftz_dim_t>(is_forward, is_inplace,
                                                value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d>::value)
        {
            create_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t>(is_forward, is_inplace,
                                                 value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_f_64_>::value)
        {
            create_pdesc<FFTZ_FLOAT, aoclfftz_dim_t_64_>(is_forward, is_inplace,
                                                    value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                                            aoclfftz_prob_desc_d_64_>::value)
        {
            create_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t_64_>(is_forward,
                                                          is_inplace,
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
    FFTZ_VOID create_default_1d_pdesc(bool is_forward = true,
            bool is_inplace = false,
            InputValueStrategy value_strategy = InputValueStrategy::MID_RANGE)
    {
        if constexpr (std::is_same<ProblemType, aoclfftz_prob_desc_f>::value)
        {
            create_1d_pdesc<FFTZ_FLOAT, aoclfftz_dim_t>(is_forward, is_inplace,
                                                   value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_d>::value)
        {
            create_1d_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t>(is_forward, is_inplace,
                                                    value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_f_64_>::value)
        {
            create_1d_pdesc<FFTZ_FLOAT, aoclfftz_dim_t_64_>(is_forward,
                                                            is_inplace,
                                                            value_strategy);
        }
        else if constexpr (std::is_same<ProblemType,
                           aoclfftz_prob_desc_d_64_>::value)
        {
            create_1d_pdesc<FFTZ_DOUBLE, aoclfftz_dim_t_64_>(is_forward,
                is_inplace, value_strategy);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported problem type for create_1d_pdesc.");
        }
    }

    // Calls the appropriate setup API based on the specified problem type
    FFTZ_VOID *aoclfftz_setup(ProblemType *problem)
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
    std::vector<FFTZ_INT32> get_supported_optlevels()
    {
        return {0, 1, 2, 3};
    }

    std::vector<aoclfftz_flags_t> get_supported_flags()
    {
        std::vector<aoclfftz_flags_t> flags;
        for (FFTZ_UINT32 in_place : {0,1})
        {
            for (FFTZ_UINT32 in_order : {0})
            {
                for (FFTZ_UINT32 forward : {0,1})
                {
                    for (FFTZ_UINT32 complex : {0})
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
    void run_setup_and_validate(FFTZ_INT32 err_no)
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
            handle = nullptr;
        }
    }

    // Validates execute_io correctness using execute output as reference
    FFTZ_VOID validate_execute_io(bool is_forward)
    {
        cleanup_problem();
        create_default_pdesc(is_forward);

        handle = aoclfftz_setup(problem);

        // invoke execute API
        FFTZ_INT32 exe = aoclfftz_execute(handle);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

        // involve execute_io API
        FFTZ_UINTP input_size = 0;
        FFTZ_UINTP output_size = 0;
        get_inout_size(&input_size, &output_size);

        FFTZ_VOID *in, *out;
        in = malloc(input_size);
        out = malloc(output_size);

        FFTZ_VOID *temp_out = malloc(output_size);
        memcpy(in, problem->in, input_size);
        memcpy(temp_out, problem->out, output_size);
        memset(out, 0, output_size);
        exe = aoclfftz_execute_io(handle, in, out);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS);

        // Compare 'out' buffer against output buffer in problem desc
        FFTZ_INT32 ret = memcmp(out, temp_out, output_size);
        free(temp_out);
        temp_out = NULL;
        EXPECT_EQ(ret, 0); // Expect successful comparison

        aoclfftz_destroy(handle);
        handle = nullptr;
        free(in);
        in = NULL;
        free(out);
        out = NULL;
    }

    // Tests aoclfftz_execute_io after freeing original buffers and passing new
    // ones.
    FFTZ_VOID
    validate_execute_io_after_buffer_free_and_alloc(bool is_forward,
                                                    bool use_1d_problem)
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
        FFTZ_INT32 exe = aoclfftz_execute(handle);
        EXPECT_EQ(exe, AOCLFFTZ_SUCCESS) << "First execute failed";

        FFTZ_UINTP in_size = 0;
        FFTZ_UINTP out_size = 0;
        get_inout_size(&in_size, &out_size);

        // Allocate all buffers together
        FFTZ_VOID *reference_output = malloc(out_size);
        FFTZ_VOID *new_in = malloc(in_size);
        FFTZ_VOID *new_out = malloc(out_size);

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

        // Verify output matches reference (same input should produce same
        // output)
        if(exe == AOCLFFTZ_SUCCESS)
        {
            FFTZ_INT32 cmp_result = memcmp(new_out, reference_output, out_size);
            EXPECT_EQ(cmp_result, 0) << "Output mismatch after buffer realloc"
                                     << " [1D=" << use_1d_problem
                                     << ", forward=" << is_forward << "]";
        }

    cleanup_and_return:
        aoclfftz_destroy(handle);
        handle = nullptr;
        free(reference_output);
        free(new_in);
        free(new_out);
    }

    // Per-application-thread arguments for the concurrent execute_io stress.
    struct concurrent_args
    {
        FFTZ_VOID  *handle;
        FFTZ_VOID  *in;
        FFTZ_VOID  *out;
        FFTZ_INT32  iters;
        FFTZ_UCHAR  inplace;
        FFTZ_INTP   buf_elems;
        FFTZ_UCHAR  result_buffer; // out: where the chained result landed (0=in, 1=out)
        FFTZ_INT32  ret;
    };

    /**
     * @brief Concurrency stress for aoclfftz_execute_io on a shared handle.
     *
     * `app_threads` threads each run `iters` chained transforms on one shared
     * handle with their own distinct-content buffers (out-of-place ping-pongs
     * in/out each iteration; in-place reuses the same buffer). Each thread's
     * final buffer is checked against a serial reference running the identical
     * chain, so cross-thread scratch contamination shows up as a divergence.
     */
    FFTZ_VOID run_concurrent_execute_io(
                                    const std::vector<FFTZ_INT32> &dim_sizes,
                                    FFTZ_INT32 batch, bool is_inplace,
                                    bool is_forward, FFTZ_INT32 app_threads,
                                    FFTZ_INT32 internal_threads)
    {
        using DataType = std::remove_pointer_t<decltype(problem->in)>;

        // Chained transforms per thread. Kept low so tests stay fast.
        const FFTZ_INT32 iters = 5;

        cleanup_problem();
        create_default_custom_pdesc(dim_sizes, batch, is_forward, is_inplace,
                                    InputValueStrategy::MID_RANGE,
                                    internal_threads);

        handle = aoclfftz_setup(problem);
        ASSERT_NE(handle, nullptr) << "setup failed";

        // Destroy the handle on every exit path (including early ASSERT returns)
        std::unique_ptr<void, void(*)(void *)> handle_guard(handle,
            [](void *h){ aoclfftz_destroy(h); });

        const FFTZ_INTP buf_elems = (FFTZ_INTP)(input_size / sizeof(DataType));

        auto build_args = [&](FFTZ_VOID *input, FFTZ_VOID *output) {
            concurrent_args cargs = {
                handle,
                input,
                output,
                iters,
                is_inplace,
                buf_elems,
                0, // result_buffer; set by execute_io_loop.
                AOCLFFTZ_EXECUTION_FAILURE
            };
            return cargs;
        };

        // Setup inputs: seed each thread's input and compute its serial reference.
        // The references run here sequentially, so they cannot trip the race.
        std::vector<std::vector<DataType>> thread_inputs(app_threads,
                                        std::vector<DataType>(buf_elems));
        std::vector<std::vector<DataType>> expected_outputs(app_threads,
                                        std::vector<DataType>(buf_elems));
        std::vector<std::vector<DataType>> concurrent_inputs(app_threads);
        std::vector<std::vector<DataType>> concurrent_outputs(app_threads,
                                        std::vector<DataType>(buf_elems));
        std::vector<concurrent_args> thread_args(app_threads);
        // Computing each thread's serial reference is costly, so only the first
        // reuse_threshold get a unique (input, expected) pair; later threads
        // reuse one round-robin.
        // All app_threads still run concurrently at the time of concurrency test.
        const FFTZ_INT32 reuse_threshold = 10;
        for (FFTZ_INT32 t = 0; t < app_threads; t++)
        {
            if (t >= reuse_threshold)
            {
                FFTZ_INT32 src = t % reuse_threshold;
                thread_inputs[t] = thread_inputs[src];
                expected_outputs[t] = expected_outputs[src];
            }
            else
            {
                fill_concurrent_input(thread_inputs[t], t + 1);

                // Serial reference: run the chain on a throwaway copy of input.
                std::vector<DataType> ref_input = thread_inputs[t];
                concurrent_args ref_args = build_args(ref_input.data(),
                        is_inplace ? ref_input.data()
                                   : expected_outputs[t].data());
                execute_io_loop<DataType>(&ref_args);
                ASSERT_EQ(ref_args.ret, AOCLFFTZ_SUCCESS)
                    << "reference chain failed for seed "
                    << (t + 1);
                // If the chain ended in ref_input (in-place, or an even
                // out-of-place ping-pong count), copy it into expected_outputs.
                if (is_inplace || ref_args.result_buffer == 0)
                {
                    std::copy(ref_input.begin(), ref_input.end(),
                              expected_outputs[t].begin());
                }
            }

            // Concurrent run: each thread owns private buffers, shares the
            // handle. Its input starts from the pristine master copy.
            concurrent_inputs[t] = thread_inputs[t];
            thread_args[t] = build_args(concurrent_inputs[t].data(),
                is_inplace ? concurrent_inputs[t].data()
                           : concurrent_outputs[t].data());
        }

        // Concurrency test: run all app_threads at once on the shared handle.
#ifdef AOCLFFTZ_API_CONCURRENCY_TESTS
        #pragma omp parallel for num_threads(app_threads)
#endif
        for (FFTZ_INT32 t = 0; t < app_threads; t++)
        {
            execute_io_loop<DataType>(&thread_args[t]);
        }

        // Validation: every thread's result must match its serial reference
        // (a mismatch signals a race).
        for (FFTZ_INT32 t = 0; t < app_threads; t++)
        {
            ASSERT_EQ(thread_args[t].ret, AOCLFFTZ_SUCCESS)
                << "execute_io failed on thread " << t;
            const DataType *concurrent_result =
                (is_inplace || thread_args[t].result_buffer == 0)
                    ? concurrent_inputs[t].data()
                    : concurrent_outputs[t].data();
            for (FFTZ_INTP i = 0; i < buf_elems; i++)
            {
                ASSERT_EQ(concurrent_result[i], expected_outputs[t][i])
                    << "thread " << t << " output[" << i << "] diverged";
            }
        }

        // handle_guard owns its own copy of the pointer and frees the handle on
        // scope exit;
        handle = nullptr;
    }

private:
    // Deterministic per-thread random values in [-1, 1];
    template<typename DataType>
    static void fill_concurrent_input(std::vector<DataType> &buf,
                                      FFTZ_INT32 seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (size_t i = 0; i < buf.size(); i++)
        {
            buf[i] = static_cast<DataType>(distribution(rng));
        }
    }

    // Run `iters` chained transforms on the shared handle, widening the window
    // for a concurrency bug. Out-of-place ping-pongs in/out; in-place reuses one
    // buffer. result_buffer: 0=in, 1=out.
    template<typename DataType>
    static void execute_io_loop(concurrent_args *args)
    {
        FFTZ_INT32 status = AOCLFFTZ_SUCCESS;
        FFTZ_VOID *current_input = args->in, *current_output = args->out;
        FFTZ_INT32 completed = 0;
        for (FFTZ_INT32 i = 0; i < args->iters; i++)
        {
            status = aoclfftz_execute_io(args->handle,
                                         current_input, current_output);
            if (status != AOCLFFTZ_SUCCESS)
            {
                break;
            }
            completed++;

            // For out-of-place transforms:
            // swap the input and output buffers for the next iteration
            if (!args->inplace)
            {
                std::swap(current_input, current_output);
            }
        }
        // Out-of-place result is in `out` after an odd number of transforms.
        args->result_buffer = (!args->inplace && (completed % 2 == 1)) ? 1 : 0;
        args->ret = status;
    }
};
#endif // AOCLFFTZ_API_GTEST_H
