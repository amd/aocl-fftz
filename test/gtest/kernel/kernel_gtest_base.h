/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file kernel_gtest_base.h
 *
 * @brief Base file for GTest kernel tests.
 *
 * This file contains the classes, functions and global kernel tables to be used
 * for running kernel unit tests using GTest.
 *
 * @author Srirammaswamy Srinivasan
 * @author Ashwin K. Godbole
 * @author Jeevanantham N
 *
 */

#ifndef AOCLFFTZ_KERNEL_GTEST_BASE_H
#define AOCLFFTZ_KERNEL_GTEST_BASE_H

#include <gtest/gtest.h>
#include <math.h>
#include <utility>
#include <chrono>

extern "C"
{
#include "core/common/strides.h"
}
#include "test/gtest/kernel/kernel_gtest_utils.h"
#include "test/gtest/gtest_types.h"
#include "utils/allocator.h"

#define CLEANUP_CODE                                    \
    do {                                                \
        FREE_ALIGN_ALLOCATED_MEM(k_in);                 \
        FREE_ALIGN_ALLOCATED_MEM(twk_in);               \
        FREE_ALIGN_ALLOCATED_MEM(k_stride.in_strides);  \
        if (is_out_of_place)                            \
        {                                               \
            FREE_ALIGN_ALLOCATED_MEM(k_out);            \
            FREE_ALIGN_ALLOCATED_MEM(twk_out);          \
            FREE_ALIGN_ALLOCATED_MEM(k_stride.out_strides); \
        }                                               \
        FREE_ALIGN_ALLOCATED_MEM(twiddle_buffer);       \
    } while(0)

/**
 * @brief Base class for the AOCLFFTZ Kernel Tests
 *
 * @tparam T type of the input / output. (Supported types: FLOAT, DOUBLE)
 */
// clang-format off
template <class T>
class AoclfftzKernelTestBase
    : public ::testing::TestWithParam<std::tuple<aoclfftz_kernel_test_params_t,
                                      std::tuple<INTP, INTP, INTP, UINT8,
                                      UINT8, UINT8>>>
// clang-format on
{
  protected:
    bool is_complex;          // true for complex-fft and false for real-fft
    UINT8 data_stride;        // 1 for real-fft and 2 for complex-fft
    bool use_special;         // whether to use special inputs or not
    UINT32 radix;             // radix of the FFT kernel
    INTP length;              // no. of points in the data or data length
    INTP input_length;        // length with input strides
    INTP output_length;       // length with output strides
    kfft_ fft_kernel;         // pointer to the kernel function
    kfft_ fft_reverse_kernel; // pointer to the kernel function in reverse
                              // direction
    INTP in_stride;           // input stride value
    INTP out_stride;          // output stride value
    UINT8 kernel_type;        // kernel type
    UINT8 is_bwd;             // direction, 1 -> BWD, 0 -> FWD
    UINT8 is_out_of_place;    // result placement, 0 -> inplace, 1 -> outplace
    INTP offset;              // no. of sets (1 offset size = radix)
    UINT32 random_seed;       // seed value of random number generator
    INTP in_stride_w_ds;      // in stride value multiplied with data_stride
    INTP out_stride_w_ds;     // out stride value multiplied with data_stride
    T tolerance; // precision tolerance for output comparison
    INT8 buf_size_multiplier; // factor for scaled input,
                              // 1 -> R2HC Kernels, 2 -> R2HCF Kernels

    VOID SetUp() override
    {
        length = 0;
        // using the microseconds of current time as seed so the different
        // iterations test iterations will have different seed value
        random_seed = (std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count()) %
                      UINT32_MAX;
        srand(random_seed);
    }

    /**
     * @brief A high level function to run the twiddle kernel tests based on the
     * given test parameter (aoclfftz_kernel_test_params_t)
     *
     * @param special if true, special values like NaN, infinity and
     * sub-normals values will be used in input data (default: false)
     *
     */
    VOID run_twiddle_kernel_test()
    {
        auto param      = std::get<0>(GetParam());
        auto io_param   = std::get<1>(GetParam());
        in_stride       = std::get<0>(io_param);
        out_stride      = std::get<1>(io_param);
        offset          = std::get<2>(io_param);
        is_bwd          = std::get<3>(io_param);
        is_out_of_place = std::get<4>(io_param);
        UINT8 load_multi_cols_param = std::get<5>(io_param);
        radix           = std::get<0>(param);
        kernel_type     = std::get<1>(param);

        // to prevent the "goto jumps over variable initialization" issue
        kfft_ tw_kernel = nullptr;
        wrapper_kernel_fp_list *table = nullptr;
        fft_kernel = nullptr;
        T *k_in = nullptr;
        T *twk_in = nullptr;
        T *k_out = nullptr;
        T *twk_out = nullptr;
        VOID *twiddle_buffer = nullptr;
        aoclfftz_strides_t k_stride;
        k_stride.in_strides = nullptr;
        k_stride.out_strides = nullptr;
        INT32 error = 0;
        T *k_in_r = nullptr;
        T *k_in_i = nullptr;
        T *k_out_r = nullptr;
        T *k_out_i = nullptr;
        T *twk_in_r = nullptr;
        T *twk_in_i = nullptr;
        T *twk_out_r = nullptr;
        T *twk_out_i = nullptr;
        aoclfftz_twiddle_t tws;

        bool use_input_params = !is_out_of_place;
        aocl_fftz_test_input input_type = aocl_fftz_test_input::RANDOM;

        if (kernel_type < aocl_fftz_kernel_type::C2C_TWID_C ||
            kernel_type > aocl_fftz_kernel_type::C2C_TWID_AVX512)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Given kernel is not a twiddle kernel.");
            return;
        }

        tw_kernel = get_twiddle_kernel<T>(radix, is_bwd, kernel_type);
        if (tw_kernel == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_(std::string("Twiddle kernel for radix " +
                                             std::to_string(radix) +
                                             " not found.")
                                    .c_str());
        }

        table = get_kernel_table(aocl_fftz_kernel_type::C2C_C);

        if (table == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Kernel table of C2C kernels is empty.");
        }

        fft_kernel = get_kernel<T>(table, is_bwd, radix);
        if (fft_kernel == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_(std::string("C2C kernel for radix " +
                                             std::to_string(radix) +
                                             " not found.")
                                    .c_str());
        }

        length = radix * offset; // offset is actually the number of sets
        input_length  = length * in_stride;
        output_length = length * out_stride;
        in_stride_w_ds  = in_stride * data_stride;
        out_stride_w_ds = out_stride * data_stride;

        k_in = prepare_input(input_type); // the input to the regular kernel
        if (k_in == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Input preparation failed");
        }

        twk_in = nullptr; // the input to the twiddle kernel
        ALLOC_ALIGN_UNINIT(twk_in, T, sizeof(T) * input_length * data_stride);
        if (twk_in == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Twiddle kernel input memory "
                                 "allocation failed");
        }

        // copy the regular kernel's input to the twiddle kernel's input
        memcpy(twk_in, k_in, input_length * data_stride * sizeof(T));

        k_out = nullptr; // the output of the regular kernel
        twk_out = nullptr; // the output of the twiddle kernel

        if (is_out_of_place)
        {
            ALLOC_ALIGN_UNINIT(k_out, T,
                               sizeof(T) * output_length * data_stride);
            ALLOC_ALIGN_UNINIT(twk_out, T,
                               sizeof(T) * output_length * data_stride);

            if (k_out == nullptr || twk_out == nullptr)
            {
                CLEANUP_CODE;
                GTEST_FATAL_FAILURE_("Output memory allocation failed");
            }
        }
        else
        {
            k_out = k_in;
            twk_out = twk_in;
        }

        // prepare the strides for the regular kernel
        ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
        if (k_stride.in_strides == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Input stride memory allocation failed");
        }
        populate_stride_array_wrapper(k_stride.in_strides, in_stride_w_ds,
                                      radix, 0, 0);

        if (is_out_of_place)
        {
            ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
                               radix * sizeof(INTP));
            if (k_stride.out_strides == nullptr)
            {
                CLEANUP_CODE;
                GTEST_FATAL_FAILURE_("Output stride memory "
                                     "allocation failed");
            }
            populate_stride_array_wrapper(k_stride.out_strides, out_stride_w_ds,
                                          radix, 0, 0);
        }
        else
        {
            k_stride.out_strides = k_stride.in_strides;
        }

        k_stride.v_in_stride  = in_stride_w_ds * radix;
        k_stride.v_out_stride = out_stride_w_ds * radix;

        k_in_r  = k_in;
        k_in_i  = k_in + 1;
        k_out_r = k_out;
        k_out_i = k_out + 1;

        twk_in_r  = twk_in;
        twk_in_i  = twk_in + 1;
        twk_out_r = twk_out;
        twk_out_i = twk_out + 1;

        // setup the twiddle buffer
        ALLOC_ALIGN_INIT(twiddle_buffer, UINT8,
                         data_stride * sizeof(T) * offset * radix);

        if (twiddle_buffer == nullptr)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_("Twiddle buffer memory allocation failed");
        }

        if (load_multi_cols_param)
        {
            // load_multi_cols = 1: compute different twiddles for each column
            compute_twiddle_buffer_wrapper<T>(twiddle_buffer, radix, offset);
        }
        else
        {
            // load_multi_cols = 0 (broadcast mode): compute one set of twiddles
            // and replicate it for all columns
            compute_twiddle_buffer_wrapper<T>(twiddle_buffer, radix, 1);
            // Replicate the first column's twiddles to all other columns
            T *tw_ptr = (T *)twiddle_buffer;
            INTP twiddle_set_size = data_stride * radix;
            for (INTP col = 1; col < offset; col++)
            {
                memcpy(tw_ptr + col * twiddle_set_size,
                       tw_ptr,
                       twiddle_set_size * sizeof(T));
            }
        }

        tws.TW = twiddle_buffer;
        tws.twiddle_buf_ptr = twiddle_buffer;
        tws.cols = offset;
        tws.load_multi_cols = load_multi_cols_param;

        // perform the twiddle multiplication on the kernel's input buffer
        // this is to simulate the condition where the m (offset) fft has been
        // performed and the twiddle multiplication followed by the kernel
        // (radix/r) fft has to be done next.
        if (is_bwd)
        {
            error = gtest_twiddle_multiplier_no_transpose(
            k_in_i, k_in_r, radix, offset, in_stride_w_ds,
            k_stride.v_in_stride, twiddle_buffer);
        }
        else
        {
            error = gtest_twiddle_multiplier_no_transpose(
            k_in_r, k_in_i, radix, offset, in_stride_w_ds,
            k_stride.v_in_stride, twiddle_buffer);
        }

        if (error == 0)
        {
            CLEANUP_CODE;
            GTEST_FATAL_FAILURE_(
                "Twiddle multiplication (after regular kernel) failed");
        }

        // execute the regular kernel
        fft_kernel(k_in_r, k_in_i, k_out_r, k_out_i, offset, &k_stride, NULL,
                   is_bwd);

        // execute the twiddle kernel
        tw_kernel(twk_in_r, twk_in_i, twk_out_r, twk_out_i, offset, &k_stride,
                  &tws, is_bwd);

        EXPECT_TRUE(is_error_safe(k_out, twk_out, output_length * data_stride,
                                  tolerance, use_input_params))
            << "Twiddle kernel failed, seed: " << random_seed << "\n";

        // Free allocated memory
        CLEANUP_CODE;
    }

    /**
     * @brief A high level function to run the kernel tests based on the given
     * test parameter (aoclfftz_kernel_test_params_t)
     *
     * @param special if true, special values like NaN, infinity and
     * sub-normals values will be used in input data (default: false)
     *
     */
    VOID run_kernel_test(bool special = false)
    {
        use_special     = special;
        auto param      = std::get<0>(GetParam());
        auto io_param   = std::get<1>(GetParam());
        in_stride       = std::get<0>(io_param);
        out_stride      = std::get<1>(io_param);
        offset          = std::get<2>(io_param);
        is_bwd          = std::get<3>(io_param);
        is_out_of_place = std::get<4>(io_param);
        // UINT8 load_multi_cols = std::get<5>(io_param);  // unused in non-twiddle tests
        radix           = std::get<0>(param);
        kernel_type     = std::get<1>(param);

        // Each set represents a data of size `radix`
        // hence, data length = radix * offset
        length = radix * offset;
        wrapper_kernel_fp_list *table = get_kernel_table(kernel_type);
        if (table == nullptr)
        {
            GTEST_FATAL_FAILURE_(
                "Kernel table is empty or invalid kernel type");
            return;
        }
        if (is_complex)
        {
            fft_kernel         = get_kernel<T>(table, is_bwd, radix);
            fft_reverse_kernel = fft_kernel;
        }
        else
        {
            fft_kernel         = get_kernel<T>(table, is_bwd, radix);
            fft_reverse_kernel = get_kernel<T>(table, !is_bwd, radix);
        }
        if (fft_kernel == nullptr)
        {
            GTEST_FATAL_FAILURE_(
                std::string("Radix" + std::to_string(radix) +
                            " kernel not found in the kernel table")
                    .c_str());
            return;
        }

        input_length  = length * in_stride  * buf_size_multiplier;
        output_length = length * out_stride * buf_size_multiplier;
        // precompute strides with data stride
        in_stride_w_ds  = in_stride * data_stride;
        out_stride_w_ds = out_stride * data_stride;

        run_dft_reference_test(use_special ?
                                aocl_fftz_test_input::RANDOM_SPECIAL :
                                aocl_fftz_test_input::RANDOM);
    } // run_test

    /**
     * @brief A high level to prepare inputs based on the given `input_type`
     *
     * @param input_type test input type
     * @return input data of size `length`
     */
    T *prepare_input(aocl_fftz_test_input input_type)
    {
        switch (input_type)
        {
        case aocl_fftz_test_input::RANDOM:
            return prepare_random_input();
        case aocl_fftz_test_input::RANDOM_SPECIAL:
            return prepare_random_input_with_special();
        default:
            GTEST_NONFATAL_FAILURE_("Invalid test input type");
            return nullptr;
        }
    } // prepare_input

    /**
     * @brief A function to generate random complex inputs
     *
     * @return input data of size `length`
     */
    T *prepare_random_input()
    {
        T *input = NULL;
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * data_stride);
        if(input == nullptr)
        {
            GTEST_NONFATAL_FAILURE_(
                "Memory allocation failed for random input preparation");
            return input;
        }
        for (INTP idx = 0; idx < input_length * data_stride; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            // generate an integer in the range [0, 19999]
            // divide it by 1000 to get in the range of [0.000, 19.999]
            // subtract by 10.0 to get in the range of [-10.000, 9.999]
            input[idx] = (((T)(rand() % 20000)) / (T)(1000)) - (T)(10);
        }
        return input;
    } // prepare_random_input

    /**
     * @brief A function to generate random complex inputs with special and
     * extreme inputs at random places which include NaN, infinity, negative
     * infinity, zero, floating point min/max/subnormal-min values in positive
     * and negative range.
     *
     * @return input data of size `length`
     */
    T *prepare_random_input_with_special()
    {
        T *input = NULL;
        ALLOC_ALIGN_UNINIT(input, T,sizeof(T) * input_length * data_stride);
        if(input == nullptr)
        {
            GTEST_NONFATAL_FAILURE_(
                "Memory allocation failed for special input preparation");
            return input;
        }
        for (INTP idx = 0; idx < input_length * data_stride; idx += in_stride)
        {
            input[idx] = get_fp_special_value<T>(rand());
        }
        return input;
    } // prepare_random_noise_input

    /**
     * @brief Get the relative error of the two complex arrays `a` and `b`
     *
     * @param a first data array to compare
     * @param b second data array to compare
     * @param use_input_params if true, input-length and input-stride will be
     * used instead of output-length and output-stride
     * @return T calculated error value
     */
    T get_error(T *a, T *b, INTP _length, bool use_input_params = false)
    {
        T max_e       = 0.0;
        T max_mag     = 0.0;
        INT32 _stride = use_input_params ? in_stride : out_stride;
        _stride *= data_stride;
        for (INTP idx = 0; idx < _length; idx += _stride)
        {
            T e;
            T mag;
            if (is_complex) // complex kernels
            {
                e   = (std::max)(std::abs(a[idx] - b[idx]),
                               std::abs(a[idx + 1] - b[idx + 1]));
                mag = (std::min)(
                    (std::max)(std::abs(a[idx]), std::abs(a[idx + 1])),
                    (std::max)(std::abs(b[idx]), std::abs(b[idx + 1])));
            }
            else // real kernels
            {
                e   = (std::abs)(a[idx] - b[idx]);
                mag = (std::min)(std::abs(a[idx]), std::abs(b[idx]));
            }
            max_e   = (std::max)(max_e, e);
            max_mag = (std::max)(max_mag, mag);
        }

        // FIXME: fix these contraints for tiny values
        // Verify output correctness when the max_e and max_mag values are too
        // small. When max_mag value is too small, dividing with very small
        // value can lead to large error value.
        if (max_e == 0.0 && max_mag == 0.0)
        {
            return 0.0;
        }
        if (max_mag == 0.0)
        {
            return max_e;
        }
        return (max_e / max_mag);
    }

    /**
    * @brief Get the relative error of the two complex arrays `a` and `b`
    *
    * @param a first data array to compare
    * @param b second data array to compare
    * @param use_input_params if true, input-length and input-stride will be
    * used instead of output-length and output-stride
    * @return T calculated error value
    */
    bool is_error_safe(T *a, T *b, INTP _length, T tolerance,
                       bool use_input_params = false)
    {
        T max_e = 0.0;
        T max_mag = 0.0;
        INT32 _stride = use_input_params ? in_stride : out_stride;
        _stride *= data_stride;

        bool passed_checks = true;
        // check if there are any NaNs or Infs in the buffers
        for (INTP idx = 0; idx < _length; idx += _stride)
        {
            if (std::isnan(a[idx]) || std::isinf(a[idx]) ||
                std::isnan(a[idx + 1]) || std::isinf(a[idx + 1]))
            {
                GTEST_LOG_(INFO) << "Buffer 'a' has NaN or Inf at idx " << idx
                                 << " or " << idx + 1 << std::endl;
                passed_checks = false;
            }
            if (std::isnan(b[idx]) || std::isinf(b[idx]) ||
                std::isnan(b[idx + 1]) || std::isinf(b[idx + 1]))
            {
                GTEST_LOG_(INFO) << "Buffer 'b' has NaN or Inf at idx " << idx
                                 << " or " << idx + 1 << std::endl;
                passed_checks = false;
            }
        }

        if (!passed_checks)
        {
            // dump the buffers one by one to stderr
            for (INTP idx = 0; idx < _length; idx += _stride)
            {
                GTEST_LOG_(INFO) << a[idx] << ", " << a[idx + 1] << " | "
                                 << b[idx] << ", " << b[idx + 1];
            }
            GTEST_LOG_(INFO) << std::endl << std::endl;
            return false;
        }

        for (INTP idx = 0; idx < _length; idx += _stride)
        {
            T e;
            T mag;
            e = (std::max)(std::abs(a[idx] - b[idx]),
                           std::abs(a[idx + 1] - b[idx + 1]));
            mag =
                (std::min)((std::max)(std::abs(a[idx]), std::abs(a[idx + 1])),
                           (std::max)(std::abs(b[idx]), std::abs(b[idx + 1])));
            max_e = (std::max)(max_e, e);
            max_mag = (std::max)(max_mag, mag);
        }

        GTEST_LOG_(INFO) << "max error: " << max_e
                         << " -- acceptable error (relative): "
                         << tolerance * max_mag;

        // FIXME: if the max_mag is zero, then we need some other metric to
        // compute the error.
        if (max_mag == 0.0)
        {
            max_mag = 1.0;
        }
        return max_e <= tolerance * max_mag;
    }

    /**
     * @brief Expand the buffer from half complex format of size n to
     * full complex format of size n * 2
     *
     * @example with n = 6
     * input  : 1r 2r 2i 3r 3i 4r
     * output : 1r 0i 2r 2i 3r 3i 4r 0i 3r -3i 2r -2i
     *
     * @param out output buffer
     * @param in input buffer
     * @param r size of a set (radix)
     * @param n no. of sets
     * @param s stride
     * @return VOID
     */
    VOID convert_halfcomplex_to_fullcomplex(T *out, T *in, INTP r, INTP n,
                                            INTP s)
    {
        for (INTP b = 0; b < n; b++)
        {
            INTP half_idx = b * r;
            INTP full_idx = half_idx * 2;
            // introduce imaginary part for the first half-complex point
            out[full_idx * s]     = in[half_idx * s];
            out[full_idx * s + 1] = 0.0;
            // copy the complex points in-order first and then fill its
            // conjugates in reverse order
            for (INTP i = (r - 1) / 2 - 1; i >= 0 ; i--)
            {
                INTP j1 = (i * 2) + 2;
                INTP j2 = ((r - i) * 2) - 2;
                out[(full_idx + j1) * s] = in[(half_idx + j1) * s - 1];
                out[(full_idx + j1) * s + 1] = in[(half_idx + j1) * s];
                out[(full_idx + j2) * s] = in[(half_idx + j1) * s - 1];
                out[(full_idx + j2) * s + 1] = -in[(half_idx + j1) * s];
            }
            // introduce imaginary part for the last half-complex point
            if (r % 2 == 0)
            {
                out[(full_idx + r) * s] = in[(half_idx + r) * s - 1];
                out[(full_idx + r) * s + 1] = 0.0;
            }
        }
    }

    /**
     * @brief Shrink the buffer from full complex format of size n * 2 to
     * half complex format of size n
     *
     * @example with n = 6
     * input  : 1r 0i 2r 2i 3r 3i 4r 0i 3r -3i 2r -2i
     * output : 1r 2r 2i 3r 3i 4r
     *
     * @param out output buffer
     * @param in input buffer
     * @param r size of a set (radix)
     * @param n no. of sets
     * @param s stride
     * @return VOID
     */
    VOID convert_fullcomplex_to_halfcomplex(T *out, T *in, INTP r, INTP n,
                                            INTP s)
    {
        for (INTP b = 0; b < n; b++)
        {
            INTP half_idx = b * r;
            INTP full_idx = half_idx * 2;
            // copy the real part of first complex point and skip the imag part
            // since it is zero
            out[half_idx * s] = in[full_idx * s];
            // copy the complex points from half-complex to full-complex buffer
            // and skip its conjugates
            for (INTP i = 0; i < (r - 1) / 2; i++)
            {
                INTP j = (i + 1) * 2;
                out[(half_idx + j) * s - 1] = in[(full_idx + j) * s];
                out[(half_idx + j) * s] = in[(full_idx + j) * s + 1];
            }
            // copy the real part of mid complex point and skip the imag part
            // since it is zero
            if (r % 2 == 0)
            {
                out[(half_idx + 1 * r) * s - 1] = in[(full_idx + r) * s];
            }
        }
    }

    /**
     * @brief A function to split standard and shifted dft inputs
     *
     * @param input_type test input type
     * @param data half complex/real input data
     * @param standard_dft_data half complex/real input of standard dft
     * @param shifted_dft_data half complex/real input of shifted dft
     * @param is_half_complex 1 -> half complex format | 0 -> real
     * @param strides stride array for standard_dft_data, shifted_dft_data,
     *                data_full
     * @param v_stride vector stride
     *
     * @example suppose given strides for fused data for radix-3, stride-1 and
     * batch-2 --> 0, 1, 2, 3, 4, 5
     * and assuming that standard and shifted DFT data is represented
     * by P1 and P2 symbols respectively.
     * Fused data in half complex form as input according to the given strides
     * is represented as -
     * P1, P2, P2, P1, P1, P2, P1, P2, P2, P1, P1, P2
     * This function splits standard and shifted DFT as shown below -
     * standard_dft_data -> P1, 0,  0,  P1, P1, 0,  P1, 0,  0,  P1, P1, 0
     * shifted_dft_data  -> 0,  P2, P2, 0,  0,  P2, 0,  P2, P2, 0,  0,  P2
     *
     * Similarly for interleaved format -
     * Given Fused data  -> P1, P2, P1, P2, P1, P2, P1, P2, P1, P2, P1, P2
     * standard_dft_data -> P1, 0,  P1, 0,  P1, 0,  P1, 0,  P1, 0,  P1, 0
     * shifted_dft_data  -> 0,  P2, 0,  P2, 0,  P2, 0,  P2, 0,  P2, 0,  P2
     */
    VOID split_r2hcf_data(aocl_fftz_test_input input_type, T *data,
                        T *standard_dft_data, T *shifted_dft_data,
                        INT8 is_half_complex, INTP *strides, INTP v_stride)
    {
        if (is_half_complex)
        {
            for (INTP i = 0; i < offset; i++)
            {
                INTP is_shifted_data = 0;
                for (INTP j = 0; j < radix * buf_size_multiplier;)
                {
                    if (!j)
                    {
                        // first element within one offset (is_shifted_data is always 0 here)
                        standard_dft_data[strides[j]] = data[strides[j]];
                        is_shifted_data = 1;
                        j++;
                    }
                    else if (j == radix * buf_size_multiplier - 1)
                    {
                        // last element within offset
                        if (is_shifted_data)
                        {
                            shifted_dft_data[strides[j]] = data[strides[j]];
                        }
                        else
                        {
                            standard_dft_data[strides[j]] = data[strides[j]];
                        }
                        j++;
                    }
                    else if (is_shifted_data)
                    {
                        shifted_dft_data[strides[j]] = data[strides[j]];
                        shifted_dft_data[strides[j+1]] = data[strides[j+1]];
                        is_shifted_data = 0;
                        j += 2;
                    }
                    else
                    {
                        standard_dft_data[strides[j]] = data[strides[j]];
                        standard_dft_data[strides[j+1]] = data[strides[j+1]];
                        is_shifted_data = 1;
                        j += 2;
                    }
                }
                data    += v_stride;
                standard_dft_data += v_stride;
                shifted_dft_data += v_stride;
            }
        }
        else
        {
            // input is real for forward FFT
            for (INTP i = 0; i < offset; i++)
            {
                for (INTP j = 0; j<radix * buf_size_multiplier;
                     j += buf_size_multiplier)
                {
                    standard_dft_data[strides[j]] = data[strides[j]];
                    shifted_dft_data[strides[j+1]] = data[strides[j+1]];
                }
                data    += v_stride;
                standard_dft_data += v_stride;
                shifted_dft_data += v_stride;
            }
        }
    }

    /**
     * @brief A function to combine the standard and shifted DFT buffers inputs
     * together and works exactly opposite to function split_r2hcf_data()
     *
     * @param input_type test input type
     * @param data half complex/real input data
     * @param standard_dft_data half complex/real input of standard dft
     * @param shifted_dft_data half complex/real input of shifted dft
     * @param is_half_complex 1 -> half complex format | 0 -> real
     * @param strides stride array for standard_dft_data, shifted_dft_data,
     *                data_full
     * @param v_stride vector stride
     */
    VOID combine_data_for_r2hcf(aocl_fftz_test_input input_type, T *data,
                        T *standard_dft_data, T *shifted_dft_data,
                        INT8 is_half_complex, INTP *strides, INTP v_stride)
    {
        if (is_half_complex)
        {
            for (INTP i = 0; i < offset; i++)
            {
                bool is_shifted_data = false;
                for (INTP j = 0; j < radix * buf_size_multiplier;)
                {
                    if (!j)
                    {
                        // first element within one offset (is_shifted_data is always false here)
                        data[strides[j]] = standard_dft_data[strides[j]];
                        is_shifted_data = true;
                        j++;
                    }
                    else if (j == radix * buf_size_multiplier - 1)
                    {
                        // last element within offset
                        if (is_shifted_data)
                        {
                            data[strides[j]] = shifted_dft_data[strides[j]];
                        }
                        else
                        {
                            data[strides[j]] = standard_dft_data[strides[j]];
                        }
                        j++;
                    }
                    else if (is_shifted_data)
                    {
                        data[strides[j]] = shifted_dft_data[strides[j]];
                        data[strides[j+1]] = shifted_dft_data[strides[j+1]];
                        is_shifted_data = false;
                        j += 2;
                    }
                    else
                    {
                        data[strides[j]] = standard_dft_data[strides[j]];
                        data[strides[j+1]] = standard_dft_data[strides[j+1]];
                        is_shifted_data = true;
                        j += 2;
                    }
                }
                data += v_stride;
                standard_dft_data += v_stride;
                shifted_dft_data  += v_stride;
            }
        }
        else
        {
            // input is real for forward FFT
            for (INTP i = 0; i < offset; i++)
            {
                for (INTP j = 0; j < radix * buf_size_multiplier;
                     j += buf_size_multiplier)
                {
                    data[strides[j]] = standard_dft_data[strides[j]];
                    data[strides[j + 1]] = shifted_dft_data[strides[j + 1]];
                }
                data += v_stride;
                standard_dft_data += v_stride;
                shifted_dft_data  += v_stride;
            }
        }
    }

    /**
     * @brief A helper function to convert given input into full complex format,
     *        used in checking the FFT kernel with DFT reference
     *
     * @param input_type test input type
     * @param data input array in half complex/real format
     * @param data_full output array in full complex format
     * @param stride stride of data
     * @param pass_radix radix
     * @param is_half_complex 1 -> half complex format | 0 -> real
     * @param k_in_size size of input array
     */
    VOID convert_to_fullcomplex(aocl_fftz_test_input input_type, T* data,
                                T *data_full, INTP stride, INTP pass_radix,
                                INTP is_half_complex, INTP k_in_size)
    {
        if (is_half_complex)
        {
            convert_halfcomplex_to_fullcomplex(data_full, data, pass_radix,
                                    offset, stride);
        }
        else
        {
            for (INTP i = 0; i < k_in_size; i++)
            {
                data_full[i * 2] = data[i]; // interleaved format
            }
        }
    }

    /**
     * @brief A helper function to calculate the DFT of input into full complex
     *        format, used in comparing the FFT Kernel output with DFT reference
     *
     * @param input_type test input type
     * @param kernel_stride handle for kernel stride
     * @param in_full input array in full complex format
     * @param out_full output array in full complex format
     * @param fc_out_size size of full complex output array
     * @param is_standard_dft 1 -> Input type standard_dft_data |
     *                        0 -> Input type shifted_dft_data
     */
    VOID calculate_dft(aocl_fftz_test_input input_type,
                       aoclfftz_strides_t kernel_stride,
                       T *in_full, T *out_full, INTP fc_out_size,
                       bool is_standard_dft)
    {
        T *out_ref = NULL;
        ALLOC_ALIGN_INIT(out_ref, T, fc_out_size * sizeof(T));

        if (is_complex == false) // is_real
        {
            populate_stride_array_wrapper(kernel_stride.in_strides,
                        in_stride_w_ds * buf_size_multiplier * 2, radix, 0, 0);
            populate_stride_array_wrapper(kernel_stride.out_strides,
                        out_stride_w_ds * buf_size_multiplier * 2, radix, 0, 0);
            kernel_stride.v_in_stride  = in_stride * radix
                                         * buf_size_multiplier * 2;
            kernel_stride.v_out_stride = out_stride * radix
                                         * buf_size_multiplier * 2;
        }

        T e[2]    = {0.0, 0.0};
        INTP sign = is_bwd ? 1.0 : -1.0;

        INTP in_start  = 0;
        INTP out_start = 0;
        // FIXME: remove this memset
        memset(out_ref, 0, fc_out_size * sizeof(T));
        for (INTP b = 0; b < offset; b++)
        {
            // iterate over output points
            for (INTP k = 0; k < radix; k++)
            {
                INTP out_idx = out_start + kernel_stride.out_strides[k];
                // iterate over input points
                for (INTP i = 0; i < radix; i++)
                {
                    INTP in_idx = in_start + kernel_stride.in_strides[i];
                    T angle     = 0.0;
                    if (!is_standard_dft)
                    {
                        if (is_bwd)
                        {
                            angle =
                                sign * AOCLFFTZ_2_PI * (i + 0.5) * k / radix;
                        }
                        else
                        {
                            angle =
                                sign * AOCLFFTZ_2_PI * i * (k + 0.5) / radix;
                        }
                    }
                    else
                    {
                        angle = sign * AOCLFFTZ_2_PI * i * k / radix;
                    }
                    EULER(angle, e);

                    out_ref[out_idx] +=
                        (in_full[in_idx] * e[0]) - (in_full[in_idx + 1] * e[1]);
                    out_ref[out_idx + 1] +=
                        (in_full[in_idx] * e[1]) + (in_full[in_idx + 1] * e[0]);
                }
            }
            in_start += kernel_stride.v_in_stride;
            out_start += kernel_stride.v_out_stride;
        }

        if (!use_special)
        {
            EXPECT_LE(get_error(out_full, out_ref, fc_out_size), tolerance)
                << "Property: dft_reference, seed: " << random_seed << "\n";
        }

        FREE_ALIGN_ALLOCATED_MEM(out_ref);
    }

    /**
     * @brief A functions to check the FFT kernel with DFT reference for
     * for complex and real type defined in complex_kernel_gtest.h and
     * real_kernel_gtest.h respectively
     *
     * @param input_type test input type
     */
    VOID run_dft_reference_test_complex(aocl_fftz_test_input input_type);
    VOID run_dft_reference_test_real(aocl_fftz_test_input input_type);
    VOID run_dft_reference_test(aocl_fftz_test_input input_type)
    {
        if (is_complex)
        {
            run_dft_reference_test_complex(input_type);
        }
        else
        {
            run_dft_reference_test_real(input_type);
        }
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for FLOAT type
 *
 */

class AoclfftzKernelTestFloat : public AoclfftzKernelTestBase<FLOAT>
{
  public:
    AoclfftzKernelTestFloat()
    {
        is_complex          = true;
        data_stride         = 2;
        tolerance           = TOLERANCE_F;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for DOUBLE type
 *
 */
class AoclfftzKernelTestDouble : public AoclfftzKernelTestBase<DOUBLE>
{
  public:
    AoclfftzKernelTestDouble()
    {
        is_complex          = true;
        data_stride         = 2;
        tolerance           = TOLERANCE_D;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzTwiddleKernelTestBase for FLOAT type
 *
 */
class AoclfftzTwiddleKernelTestFloat : public AoclfftzKernelTestBase<FLOAT>
{
  public:
    AoclfftzTwiddleKernelTestFloat()
    {
        is_complex          = true;
        data_stride         = 2;
        tolerance           = TOLERANCE_F;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzTwiddleKernelTestBase for DOUBLE type
 *
 */
class AoclfftzTwiddleKernelTestDouble : public AoclfftzKernelTestBase<DOUBLE>
{
  public:
    AoclfftzTwiddleKernelTestDouble()
    {
        is_complex          = true;
        data_stride         = 2;
        tolerance           = TOLERANCE_D;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for FLOAT type
 *
 */

class AoclfftzKernelTestFloatReal : public AoclfftzKernelTestBase<FLOAT>
{
  public:
    AoclfftzKernelTestFloatReal()
    {
        is_complex          = false;
        data_stride         = 1;
        tolerance           = TOLERANCE_F;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for DOUBLE type
 *
 */
class AoclfftzKernelTestDoubleReal : public AoclfftzKernelTestBase<DOUBLE>
{
  public:
    AoclfftzKernelTestDoubleReal()
    {
        is_complex          = false;
        data_stride         = 1;
        tolerance           = TOLERANCE_D;
        buf_size_multiplier = 1;
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for FLOAT type
 *
 */

class AoclfftzKernelTestFloatFused : public AoclfftzKernelTestBase<FLOAT>
{
  public:
    AoclfftzKernelTestFloatFused()
    {
        is_complex          = false;
        data_stride         = 1;
        tolerance           = TOLERANCE_F;
        buf_size_multiplier = 2;
    }
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for DOUBLE type
 *
 */
class AoclfftzKernelTestDoubleFused : public AoclfftzKernelTestBase<DOUBLE>
{
  public:
    AoclfftzKernelTestDoubleFused()
    {
        is_complex          = false;
        data_stride         = 1;
        tolerance           = TOLERANCE_D;
        buf_size_multiplier = 2;
    }
};
#endif // AOCLFFTZ_KERNEL_GTEST_BASE_H
