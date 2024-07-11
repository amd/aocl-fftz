/**
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

/** @file kernel_gtest_base.h
 *
 * @brief Base file for GTest kernel tests.
 *
 * This file contains the classes, functions and global kernel tables to be used
 * for running kernel unit tests using GTest.
 *
 * @author Srirammaswamy Srinivasan
 *
 */

#ifndef AOCLFFTZ_KERNEL_GTEST_BASE_H
#define AOCLFFTZ_KERNEL_GTEST_BASE_H

#include <gtest/gtest.h>
#include <math.h>
#include <utility>
extern "C"
{
#include "core/common/strides.h"
}
#include "test/gtest/kernel/kernel_gtest_utils.h"
#include "test/gtest/gtest_types.h"
#include "utils/allocator.h"

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
                                      UINT8>>>
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
    UINT8 is_out_of_place;     // result placement, 0 -> inplace, 1 -> outplace
    INTP offset;              // no. of sets (1 offset size = radix)
    UINT32 random_seed;       // seed value of random number generator
    INTP in_stride_w_ds;       // in stride value multiplied with data_stride
    INTP out_stride_w_ds;      // out stride value multiplied with data_stride
    T tolerance; // precision tolerance for output comparison

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
        radix           = std::get<0>(param);
        kernel_type     = std::get<1>(param);
        UINT8 test_type = std::get<2>(param);

        // Each set represents a data of size `radix`
        // hence, data length = radix * offset
        length = radix * offset;
        wrapper_kernel_fp_list *table = get_kernel_table(kernel_type);
        if (table == nullptr)
        {
            GTEST_NONFATAL_FAILURE_(
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
            GTEST_NONFATAL_FAILURE_(
                std::string("Radix" + std::to_string(radix) +
                            " kernel not found in the kernel table")
                    .c_str());
            return;
        }

        input_length  = length * in_stride;
        output_length = length * out_stride;
        // precompute strides with data stride
        in_stride_w_ds  = in_stride * data_stride;
        out_stride_w_ds = out_stride * data_stride;

        if (test_type & aoclfftz_kernel_test_type::LINEARITY)
        {
            run_linearity_test(use_special ?
                               aocl_fftz_test_input::RANDOM_SPECIAL :
                               aocl_fftz_test_input::RANDOM);
        }
        if (test_type & aoclfftz_kernel_test_type::TRANSFORMATION)
        {
            run_unit_impulse_transform_test(use_special ?
                                        aocl_fftz_test_input::RANDOM_SPECIAL :
                                        aocl_fftz_test_input::IMPULSE);
        }
        if (test_type & aoclfftz_kernel_test_type::TIMESHIFT)
        {
            run_timeshift_test(use_special ?
                               aocl_fftz_test_input::RANDOM_SPECIAL :
                               aocl_fftz_test_input::RANDOM);
            // Need to fix signal input generation,
            // so changed signal input to random
        }
        if (test_type & aoclfftz_kernel_test_type::DFT_REFERENCE)
        {
            run_dft_reference_test(use_special ?
                                   aocl_fftz_test_input::RANDOM_SPECIAL :
                                   aocl_fftz_test_input::RANDOM);
        }
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
        case aocl_fftz_test_input::IMPULSE:
            return prepare_impulse_input();
        case aocl_fftz_test_input::SIGNAL:
            return prepare_signal_input();
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
        for (INTP idx = 0; idx < input_length * data_stride; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            input[idx] = (T)((rand() % 2000) / 200.0) - 10.0;
        }
        return input;
    } // prepare_random_input

    /**
     * @brief A function to generate impulse input with random peak value
     *
     * @return input data of size `length`
     */
    T *prepare_impulse_input()
    {
        T *input;
        ALLOC_ALIGN_INIT(input, T, input_length * data_stride * sizeof(T));
        INTP idx = (INTP)(rand() % length) * in_stride;
        // range: [-10.0, 10.0) with 3 decimal precision
        input[idx * data_stride] = (T)((rand() % 2000) / 200.0 - 10.0);
        if (is_complex)
        {
            input[idx * data_stride + 1] = (T)((rand() % 2000) / 200.0 - 10.0);
        }
        return input;
    } // prepare_impulse_input

    /**
     * @brief A function to a cosine signal input with random shift and
     * amplitude values
     *
     * @return input data of size `length`
     */
    T *prepare_signal_input()
    {
        T *input = NULL;
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * data_stride);
        // Sine wave cycles
        INTP cycles = (rand() % (input_length / 2)) + 2;
        T size      = AOCLFFTZ_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, length)
        INTP shift = rand() % input_length;
        // scale the amplitude of the wave by `scale` times, scale range:
        // [0.0 5.0)
        T scale = ((T)rand() / (T)RAND_MAX) * 5.0;
        for (INTP i = 0; i < input_length; i++)
        {
            input[((i + shift) % input_length) * data_stride] =
                sin((T)(i * size) / input_length) * scale;
            if (is_complex)
            {
                input[((i + shift) % input_length) * data_stride + 1] = 0.0;
            }
        }
        return input;
    } // prepare_signal_input

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
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * data_stride);
        for (INTP idx = 0; idx < input_length * data_stride; idx += in_stride)
        {
            input[idx] = get_maybe_special_value<T>(rand());
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
        for (int idx = 0; idx < _length; idx += _stride)
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
     * @brief Expand the buffer from half complex format of size n to
     * full complex format of size n * 2
     *
     * example with n = 6
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
     * example with n = 6
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
     * @brief A function to check the linearity property of the FFT kernel
     *
     * @param input_type test input type
     */
    VOID run_linearity_test(aocl_fftz_test_input input_type)
    {
        T *in1 = prepare_input(input_type);
        T *in2 = prepare_input(input_type);
        if (in1 == nullptr || in2 == nullptr)
        {
            return;
        }
        T *in_combined = NULL;
        ALLOC_ALIGN_UNINIT(in_combined, T,
                           sizeof(T) * input_length * data_stride);
        T *out1 = NULL;
        T *out2 = NULL;
        T *out_combined = NULL;
        T *out_added = NULL;

        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out1, T, output_length * data_stride * sizeof(T));
            ALLOC_ALIGN_INIT(out2, T, output_length * data_stride * sizeof(T));
            ALLOC_ALIGN_INIT(out_combined, T,
                                output_length * data_stride * sizeof(T));
        }
        else
        {
            out1 = in1;
            out2 = in2;
            out_combined = in_combined;
        }

        ALLOC_ALIGN_INIT(out_added, T, output_length * data_stride * sizeof(T));

        if (is_complex)
        {
            // Complex constant multiplier a1 and a2 will range from
            // [-10.0 to 10.0) with one digit precision
            T a1[2] = {(T)((rand() % 200) / 20.0 - 10.0), 0.0};
            T a2[2] = {(T)((rand() % 200) / 20.0 - 10.0), 0.0};

            // Temporary complex variables
            T temp1[2]     = {0.0, 0.0};
            T temp2[2]     = {0.0, 0.0};
            T cmul_temp[2] = {0.0, 0.0};
            for (INTP idx = 0; idx < input_length; ++idx)
            {
                CMUL(a1, in1 + idx * data_stride, temp1, cmul_temp);
                CMUL(a2, in2 + idx * data_stride, temp2, cmul_temp);
                CADD(temp1, temp2, in_combined + idx * data_stride);
            }

            // prepare local stride
            aoclfftz_strides_t k_stride;
            // strides for FFT kernel
            ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP,
                radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
                radix * sizeof(INTP));

            // permuted kernel
            if (kernel_type & 0x1)
            {
                if (is_out_of_place)
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    k_stride.v_in_stride = radix * in_stride_w_ds;
                    k_stride.v_out_stride = out_stride_w_ds;
                }
                else
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    k_stride.v_in_stride = radix * in_stride_w_ds;
                    k_stride.v_out_stride = k_stride.v_in_stride;
                }
            }
            // standard kernel
            else
            {
                if (is_out_of_place)
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    k_stride.v_in_stride  = in_stride_w_ds;
                    k_stride.v_out_stride = radix * out_stride_w_ds;
                }
                else
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    k_stride.v_in_stride = radix * in_stride_w_ds;
                    k_stride.v_out_stride = k_stride.v_in_stride;
                }
            }

            // Initialize kernel input/output variables
            T *in1_r  = (is_bwd) ? (in1 + 1) : (in1);
            T *in1_i  = (is_bwd) ? (in1) : (in1 + 1);
            T *out1_r = (is_bwd) ? (out1 + 1) : (out1);
            T *out1_i = (is_bwd) ? (out1) : (out1 + 1);
            T *in2_r  = (is_bwd) ? (in2 + 1) : (in2);
            T *in2_i  = (is_bwd) ? (in2) : (in2 + 1);
            T *out2_r = (is_bwd) ? (out2 + 1) : (out2);
            T *out2_i = (is_bwd) ? (out2) : (out2 + 1);

            T *in_combined_r  = (is_bwd) ? (in_combined + 1) : (in_combined);
            T *in_combined_i  = (is_bwd) ? (in_combined) : (in_combined + 1);
            T *out_combined_r = (is_bwd) ? (out_combined + 1) : (out_combined);
            T *out_combined_i = (is_bwd) ? (out_combined) : (out_combined + 1);

            fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, is_bwd);
            fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, is_bwd);
            fft_kernel(in_combined_r, in_combined_i, out_combined_r,
                       out_combined_i, offset, &k_stride, is_bwd);

            for (INTP idx = 0; idx < output_length; ++idx)
            {
                CMUL(a1, out1 + idx * data_stride, temp1, cmul_temp);
                CMUL(a2, out2 + idx * data_stride, temp2, cmul_temp);
                CADD(temp1, temp2, out_added + idx * data_stride);
            }

            if (!use_special)
            {
                EXPECT_LE(get_error(out_combined, out_added,
                                    output_length * data_stride),
                          tolerance)
                    << "Property: linearity, seed: " << random_seed << "\n";
            }

            if (is_out_of_place)
            {
                FREE_ALIGN_ALLOCATED_MEM(out1);
                FREE_ALIGN_ALLOCATED_MEM(out2);
                FREE_ALIGN_ALLOCATED_MEM(out_combined);
            }
            FREE_ALIGN_ALLOCATED_MEM(out_added);
            FREE_ALIGN_ALLOCATED_MEM(in1);
            FREE_ALIGN_ALLOCATED_MEM(in2);
            FREE_ALIGN_ALLOCATED_MEM(in_combined);
            FREE_ALIGN_ALLOCATED_MEM(k_stride.in_strides);
            FREE_ALIGN_ALLOCATED_MEM(k_stride.out_strides);
        }
        else
        {
            // Constant multiplier a1 and a2 will range from [-10.0 to 10.0)
            // with one digit precision
            T a1 = (T)((rand() % 200) / 20.0 - 10.0);
            T a2 = (T)((rand() % 200) / 20.0 - 10.0);

            for (INTP idx = 0; idx < input_length; ++idx)
            {
                in_combined[idx] = a1 * in1[idx] + a2 * in2[idx];
            }

            // prepare local stride
            aoclfftz_strides_t k_stride;
            ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
                radix * sizeof(INTP));

            // standard kernel
            if (is_out_of_place)
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride_w_ds * offset, radix,
                    0 /*compute_half_complex*/,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride_w_ds, radix,
                    1 /*compute_half_complex*/,
                    0 /* adjust_to_full_complex */);

                k_stride.v_in_stride  = in_stride_w_ds;
                k_stride.v_out_stride = radix * out_stride_w_ds;
            }
            else
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride_w_ds, radix,
                    0 /*compute_half_complex*/,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    in_stride_w_ds, radix,
                    1 /*compute_half_complex*/,
                    0 /* adjust_to_full_complex */);
                k_stride.v_in_stride = radix * in_stride_w_ds;
                k_stride.v_out_stride = k_stride.v_in_stride;
            }

            fft_kernel(in1, in1, out1, out1, offset, &k_stride, is_bwd);
            fft_kernel(in2, in2, out2, out2, offset, &k_stride, is_bwd);
            fft_kernel(in_combined, in_combined, out_combined, out_combined,
                       offset, &k_stride, is_bwd);

            for (INTP idx = 0; idx < output_length; ++idx)
            {
                out_added[idx] = a1 * out1[idx] + a2 * out2[idx];
            }

            if (!use_special)
            {
                EXPECT_LE(get_error(out_combined, out_added,
                                    output_length * data_stride),
                          tolerance)
                    << "Property: linearity, seed: " << random_seed << "\n";
            }

            if (is_out_of_place)
            {
                FREE_ALIGN_ALLOCATED_MEM(out1);
                FREE_ALIGN_ALLOCATED_MEM(out2);
                FREE_ALIGN_ALLOCATED_MEM(out_combined);
            }
            FREE_ALIGN_ALLOCATED_MEM(in1);
            FREE_ALIGN_ALLOCATED_MEM(in2);
            FREE_ALIGN_ALLOCATED_MEM(in_combined);
            FREE_ALIGN_ALLOCATED_MEM(out_added);
            FREE_ALIGN_ALLOCATED_MEM(k_stride.in_strides);
            FREE_ALIGN_ALLOCATED_MEM(k_stride.out_strides);
        }
    } // run_linearity_test

    /**
     * @brief A function to check the transformation property of the FFT kernel
     * using unit impulse signal
     *
     * @param input_type test input type
     */
    VOID run_unit_impulse_transform_test(aocl_fftz_test_input input_type)
    {
        T *in = prepare_input(input_type);
        if (in == nullptr)
        {
            return;
        }
        T *in_copy;
        ALLOC_ALIGN_INIT(in_copy, T, input_length * data_stride * sizeof(T));
        memcpy(in_copy, in, input_length * data_stride * sizeof(T));
        T *inv_out;
        ALLOC_ALIGN_INIT(inv_out, T, input_length * data_stride * sizeof(T));
        T *perm_out;
        ALLOC_ALIGN_INIT(perm_out, T, output_length * data_stride * sizeof(T));
        T *out, *perm_inv_out;

        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out, T, output_length * data_stride * sizeof(T));
            ALLOC_ALIGN_INIT(perm_inv_out, T,
                             input_length * data_stride * sizeof(T));
        }
        else
        {
            out = in;
            perm_inv_out = perm_out;
        }

        // prepare local strides
        aoclfftz_strides_t k_stride;
        ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));

        aoclfftz_strides_t k_stride_rev;
        ALLOC_ALIGN_UNINIT(k_stride_rev.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(k_stride_rev.out_strides, INTP,
                           radix * sizeof(INTP));

        aoclfftz_strides_t pc_stride;
        ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP, radix * sizeof(INTP));

        aoclfftz_strides_t pc_stride_rev;
        ALLOC_ALIGN_UNINIT(pc_stride_rev.in_strides, INTP,
            radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(pc_stride_rev.out_strides, INTP,
            radix * sizeof(INTP));

        if (is_complex)
        {
            // permuted kernel
            if (kernel_type & 0x1)
            {
                if (is_out_of_place)
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.in_strides,
                        out_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.out_strides,
                        in_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.in_strides,
                        in_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.out_strides,
                        in_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds * radix;
                    k_stride.v_out_stride = out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.v_in_stride = out_stride_w_ds * radix;
                    k_stride_rev.v_out_stride = in_stride_w_ds;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.v_in_stride = out_stride_w_ds;
                    pc_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.v_in_stride = in_stride_w_ds;
                    pc_stride_rev.v_out_stride = in_stride_w_ds * radix;
                }
                else
                {
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.in_strides,
                        out_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.out_strides,
                        in_stride_w_ds, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.in_strides,
                        in_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.out_strides,
                        in_stride_w_ds * offset, radix,
                        0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds * radix;
                    k_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for reverse FFT kernel
                    k_stride_rev.v_in_stride = out_stride_w_ds * radix;
                    k_stride_rev.v_out_stride = in_stride_w_ds * radix;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.v_in_stride = out_stride_w_ds;
                    pc_stride.v_out_stride = out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.v_in_stride = in_stride_w_ds;
                    pc_stride_rev.v_out_stride = in_stride_w_ds;
                }
            }
            // standard kernel
            else
            {
                if (is_out_of_place)
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for reverse FFT kernel
                    populate_stride_array_wrapper(k_stride_rev.in_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.out_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on FFT output
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on reverse FFT output
                    populate_stride_array_wrapper(pc_stride_rev.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.out_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds;
                    k_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for reverse FFT kernel
                    k_stride_rev.v_in_stride = out_stride_w_ds;
                    k_stride_rev.v_out_stride = in_stride_w_ds * radix;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.v_in_stride = out_stride_w_ds * radix;
                    pc_stride.v_out_stride = out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.v_in_stride = in_stride_w_ds * radix;
                    pc_stride_rev.v_out_stride = in_stride_w_ds;
                }
                else
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for reverse FFT kernel
                    populate_stride_array_wrapper(k_stride_rev.in_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride_rev.out_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on FFT output
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on reverse FFT output
                    populate_stride_array_wrapper(pc_stride_rev.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride_rev.out_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds;
                    k_stride.v_out_stride = out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.v_in_stride = out_stride_w_ds;
                    k_stride_rev.v_out_stride = in_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride.v_in_stride = out_stride_w_ds * radix;
                    pc_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for permuted copy on reverse FFT output
                    pc_stride_rev.v_in_stride = in_stride_w_ds * radix;
                    pc_stride_rev.v_out_stride = in_stride_w_ds * radix;
                }
            }

            // Initialize kernel input/output variables

            T *in_r  = (is_bwd) ? (in + 1) : (in);
            T *in_i  = (is_bwd) ? (in) : (in + 1);
            T *out_r = (is_bwd) ? (out + 1) : (out);
            T *out_i = (is_bwd) ? (out) : (out + 1);

            fft_kernel(in_r, in_i, out_r, out_i, offset, &k_stride, is_bwd);
            // convert the FFT kernel output from in-order to out-of-order for
            // standard kernel and vise versa for permuted kernel
            permuted_copy<T>(out, perm_out, offset, radix,
                             &pc_stride, data_stride);

            T *perm_out_r     = (!is_bwd) ? (perm_out + 1) : (perm_out);
            T *perm_out_i     = (!is_bwd) ? (perm_out) : (perm_out + 1);
            T *perm_inv_out_r = (!is_bwd) ? (perm_inv_out + 1) : (perm_inv_out);
            T *perm_inv_out_i = (!is_bwd) ? (perm_inv_out) : (perm_inv_out + 1);

            fft_kernel(perm_out_r, perm_out_i, perm_inv_out_r, perm_inv_out_i,
                       offset, &k_stride_rev, !is_bwd);
            // convert the reverse FFT kernel output from in-order to
            // out-of-order for standard kernel and
            // vise versa for permuted kernel
            permuted_copy<T>(perm_inv_out, inv_out, offset, radix,
                             &pc_stride_rev, data_stride);

            // normalize reverse FFT output
            for (INTP idx = 0; idx < input_length * data_stride; idx++)
            {
                inv_out[idx] /= radix;
            }

            if (!use_special)
            {
                EXPECT_LE(
                    get_error(in_copy, inv_out, input_length * data_stride,
                        true), tolerance)
                    << "Property: transformation, seed: " << random_seed
                    << "\n";
            }
        }
        else
        {
            // standard kernel
            if (is_out_of_place)
            {
                // strides for forward real FFT
                INTP in_str = is_bwd? in_stride_w_ds :
                                      in_stride_w_ds * offset;
                INTP out_str = is_bwd? out_stride_w_ds * offset :
                                       out_stride_w_ds;;
                INTP in_str_rev = is_bwd? out_stride_w_ds * offset :
                                          out_stride_w_ds;
                INTP out_str_rev = is_bwd? in_stride_w_ds :
                                           in_stride_w_ds * offset;
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_str, radix,
                    is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_str, radix,
                    !is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride_rev.in_strides,
                    in_str_rev, radix,
                    !is_bwd /*compute_half_complex*/,
                    0 /*adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride_rev.out_strides,
                    out_str_rev, radix,
                    is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);

                k_stride.v_in_stride = is_bwd ? in_stride_w_ds * radix :
                                                in_stride_w_ds;
                k_stride.v_out_stride = is_bwd ? out_stride_w_ds :
                                                 out_stride_w_ds * radix;
                k_stride_rev.v_in_stride  = is_bwd ? out_stride_w_ds :
                                                     out_stride_w_ds * radix;
                k_stride_rev.v_out_stride = is_bwd ? in_stride_w_ds * radix :
                                                     in_stride_w_ds;
            }
            else
            {
                // strides for forward real FFT
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride_w_ds, radix,
                    is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride_w_ds, radix,
                    !is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride_rev.in_strides,
                    out_stride_w_ds, radix,
                    !is_bwd /*compute_half_complex*/,
                    0 /*adjust_to_half_complex*/);
                populate_stride_array_wrapper(k_stride_rev.out_strides,
                    in_stride_w_ds, radix,
                    is_bwd /*compute_half_complex*/,
                    0 /* adjust_to_half_complex*/);
                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds * radix;
                k_stride_rev.v_in_stride = out_stride_w_ds * radix;
                k_stride_rev.v_out_stride = in_stride_w_ds * radix;
            }

            // Temporary buffers to be used by Bwd-FFT
            T *temp_in = NULL;
            ALLOC_ALIGN_INIT(temp_in, T,
                    input_length * data_stride * sizeof(T));
            T *temp_out = NULL;
            ALLOC_ALIGN_INIT(temp_out, T,
                    input_length * data_stride * sizeof(T));

            /* In case of Bwd-FFT, input points are set by assuming the constant
               stride but will be read in a different order because of
               different strides for each input point.
               So, refactoring the input in order to place the valid set input
               points at the places where it will be read. */
            if (is_bwd)
            {
                for(INTP i=0; i<offset; i++)
                {
                    for(INTP j=0; j<radix; j++)
                    {
                        INTP i_idx = radix*in_stride_w_ds*i
                                        + k_stride.in_strides[j];
                        INTP j_idx = radix*in_stride_w_ds*i
                                        + j*in_stride_w_ds;
                        temp_in[i_idx] = in[j_idx];
                    }
                }
            }
            else
            {
                memcpy(temp_in, in, input_length*sizeof(T));
            }

            // Initialize kernel input/output variables
            fft_kernel(temp_in, temp_in, out, out, offset, &k_stride, is_bwd);
            // convert the FFT kernel output from in-order to out-of-order for
            // standard kernel and vise versa for permuted kernel
            fft_reverse_kernel(out, out, inv_out, inv_out,
                offset, &k_stride_rev, !is_bwd);
            if (is_bwd)
            {
                for(INTP i=0; i<offset; i++)
                {
                    for(INTP j=0; j<radix; j++)
                    {
                        INTP i_idx = radix*in_stride_w_ds*i
                                    + j*in_stride_w_ds;
                        INTP j_idx = radix*in_stride_w_ds*i
                                    + k_stride.in_strides[j];
                        temp_out[i_idx] = inv_out[j_idx];
                    }
                }
            }
            else
            {
                memcpy(temp_out, inv_out,
                        input_length*sizeof(T));
            }
            // normalize reverse FFT output
            for (INTP idx = 0; idx < input_length * data_stride; idx++)
            {
                temp_out[idx * data_stride] /= radix;
            }
            if (!use_special)
            {
                EXPECT_LE(get_error(in_copy, temp_out, input_length, true),
                    tolerance)<< "Property: transformation, seed: " <<
                    random_seed<< "\n";
            }
            // deallocate the temporary buffers
            FREE_ALIGN_ALLOCATED_MEM(temp_in);
            FREE_ALIGN_ALLOCATED_MEM(temp_out);
        }
        if (is_out_of_place)
        {
            FREE_ALIGN_ALLOCATED_MEM(out);
            FREE_ALIGN_ALLOCATED_MEM(perm_inv_out);
        }
        FREE_ALIGN_ALLOCATED_MEM(in);
        FREE_ALIGN_ALLOCATED_MEM(in_copy);
        FREE_ALIGN_ALLOCATED_MEM(inv_out);
        FREE_ALIGN_ALLOCATED_MEM(perm_out);
        FREE_ALIGN_ALLOCATED_MEM(k_stride.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(k_stride.out_strides);
        FREE_ALIGN_ALLOCATED_MEM(k_stride_rev.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(k_stride_rev.out_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride.out_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride_rev.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride_rev.out_strides);
    } // run_unit_impulse_transform_test

    /**
     * @brief A function to check the timeshift property of the FFT kernel
     *
     * @param input_type test input type
     */
    void run_timeshift_test(aocl_fftz_test_input input_type)
    {
        T *in1 = prepare_input(input_type);
        if (in1 == nullptr)
        {
            return;
        }
        T *in2 = NULL;
        ALLOC_ALIGN_UNINIT(in2, T, sizeof(T) * input_length * data_stride);
        T *out1 = NULL;
        T *out2 = NULL;
        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out1, T, output_length * data_stride * sizeof(T));
            ALLOC_ALIGN_INIT(out2, T, output_length * data_stride * sizeof(T));
        }
        else
        {
            out1 = in1;
            out2 = in2;
        }

        T *out_comp;
        ALLOC_ALIGN_INIT(out_comp, T, output_length * data_stride * sizeof(T));
        T *temp;
        ALLOC_ALIGN_INIT(temp, T,
                         (std::max)(input_length, output_length) * data_stride *
                             sizeof(T));

        // prepare local strides
        aoclfftz_strides_t k_stride;
        ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));

        aoclfftz_strides_t pc_stride;
        ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP, radix * sizeof(INTP));

        /* Complex kernels validation (R2C) for forward/backward FFT by the
         * following steps-
         * 1) right circular shift of input by `m` times,
         * 2) stride array population for input and output based on whether
         *    the operation is out-of-place or in-place.
         * 3) Kernel input/output pointer initialization using is_bwd flag
         * 4) FFT kernel execution - if kernel is permuted, it performs FFT
         *    input and then converts the output from out-of-order to in-order
         *    else if kernel is standard, it converts the input from
         *    in-order to out-of-order, performs FFT and the process the output.
         * 6) Phase multiplication of output
         * 7) checking accuracy by comparing the transformed outputs
         *    to ensure they are within a specified tolerance.
         */
        if (is_complex)
        {
            // Perform circular right shift by `m` times
            // range of m => [1, radix)
            INTP m = (INTP)(rand() % (radix - 1)) + 1;
            for (INTP itr = 0; itr < offset; itr++)
            {
                for (INTP idx = 0; idx < radix; idx++)
                {
                    for (INTP is = 0; is < in_stride; is++)
                    {
                        INTP src = (itr * radix * in_stride +
                                    ((idx + (radix - m)) * in_stride) %
                                        (radix * in_stride) +
                                    is) *
                                   data_stride;
                        INTP dst = ((itr * radix + idx) * in_stride + is) *
                                   data_stride;
                        in2[dst]     = in1[src];
                        in2[dst + 1] = in1[src + 1];
                    }
                }
            }

            // permuted kernel
            if (kernel_type & 0x1)
            {
                if (is_out_of_place)
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for reverse FFT kernel
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds * radix;
                    k_stride.v_out_stride = out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride.v_in_stride = out_stride_w_ds;
                    pc_stride.v_out_stride = out_stride_w_ds * radix;
                }
                else
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on FFT output
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds * radix;
                    k_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for permuted copy on FFT output
                    pc_stride.v_in_stride = out_stride_w_ds * radix;
                    pc_stride.v_out_stride = out_stride_w_ds * radix;
                }
            }
            // standard kernel
            else
            {
                if (is_out_of_place)
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on input
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        in_stride_w_ds * offset,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds;
                    k_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for permuted copy on input
                    pc_stride.v_in_stride = in_stride_w_ds * radix;
                    pc_stride.v_out_stride = in_stride_w_ds;
                }
                else
                {
                    // strides for FFT kernel
                    populate_stride_array_wrapper(k_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(k_stride.out_strides,
                        out_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for permuted copy on input
                    populate_stride_array_wrapper(pc_stride.in_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);
                    populate_stride_array_wrapper(pc_stride.out_strides,
                        in_stride_w_ds,
                        radix, 0 /*compute_half_complex*/,
                        0 /* adjust_to_full_complex */);

                    // strides for FFT kernel
                    k_stride.v_in_stride = in_stride_w_ds * radix;
                    k_stride.v_out_stride = out_stride_w_ds * radix;
                    // strides for permuted copy on input
                    pc_stride.v_in_stride = in_stride_w_ds * radix;
                    pc_stride.v_out_stride = in_stride_w_ds * radix;
                }
            }

            // Initialize kernel input/output variables
            T *in1_r  = (is_bwd) ? (in1 + 1) : (in1);
            T *in1_i  = (is_bwd) ? (in1) : (in1 + 1);
            T *out1_r = (is_bwd) ? (out1 + 1) : (out1);
            T *out1_i = (is_bwd) ? (out1) : (out1 + 1);
            T *in2_r  = (is_bwd) ? (in2 + 1) : (in2);
            T *in2_i  = (is_bwd) ? (in2) : (in2 + 1);
            T *out2_r = (is_bwd) ? (out2 + 1) : (out2);
            T *out2_i = (is_bwd) ? (out2) : (out2 + 1);

            // permuted kernel
            if (kernel_type & 0x1)
            {
                fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride,
                           is_bwd);
                fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride,
                           is_bwd);
                // convert the FFT kernel outputs from out-of-order to in-order
                permuted_copy<T>(out1, temp, offset, radix, &pc_stride,
                                 data_stride);
                memcpy(out1, temp, sizeof(T) * output_length * data_stride);
                permuted_copy<T>(out2, temp, offset, radix, &pc_stride,
                                 data_stride);
                memcpy(out2, temp, sizeof(T) * output_length * data_stride);
            }
            // standard kernel
            else
            {
                // convert the inputs from in-order to out-of-order
                permuted_copy<T>(in1, temp, offset, radix, &pc_stride,
                                 data_stride);
                memcpy(in1, temp, sizeof(T) * input_length * data_stride);
                permuted_copy<T>(in2, temp, offset, radix, &pc_stride,
                                 data_stride);
                memcpy(in2, temp, sizeof(T) * input_length * data_stride);
                fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride,
                           is_bwd);
                fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride,
                           is_bwd);
            }

            // complex variables
            T cmul_temp[2] = {0.0, 0.0};
            T e_k[2]       = {1.0, 0.0};
            T sign         = is_bwd ? 1.0 : -1.0;

            for (INTP k = 0; k < length; k++)
            {
                // handle overflow to improve accuracy for larger values
                INTP mk = (m * k) % radix;
                T angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(out1 + k * out_stride_w_ds, e_k,
                     out_comp + k * out_stride_w_ds, cmul_temp);
            }

            if (!use_special)
            {
                EXPECT_LE(
                    get_error(out2, out_comp, output_length * data_stride),
                    tolerance)
                    << "Property: timeshift, seed: " << random_seed << "\n";
            }
        }

        /* Real kernels validation (R2C) for forward FFT by the following steps:
         * 1) right circular shift of input by `m` times,
         * 2) stride array population for input and output based on whether
         *    the operation is out-of-place or in-place.
         * 3) input conversion from in-order to out-of-order using permuted_copy
         * 4) FFT kernel execution
         * 5) complex conversion of output from half complex to full complex
         * 6) Phase multiplication of output
         * 7) Checking accuracy by comparing the transformed outputs to ensure
         *    they are within a specified tolerance.
         */
        else if (!is_bwd)
        {
            // Perform circular right shift by `m` times
            // range of m => [1, radix)
            INTP m = (INTP)(rand() % (radix - 1)) + 1;
            for (INTP itr = 0; itr < offset; itr++)
            {
                for (INTP idx = 0; idx < radix; idx++)
                {
                    for (INTP is = 0; is < in_stride; is++)
                    {
                        INTP src = (itr * radix * in_stride +
                                    ((idx + (radix - m)) * in_stride) %
                                        (radix * in_stride) +
                                    is);
                        INTP dst = ((itr * radix + idx) * in_stride + is);
                        in2[dst] = in1[src];
                    }
                }
            }

            // standard kernel
            if (is_out_of_place)
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride * offset, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride, radix,
                    1 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                populate_stride_array_wrapper(pc_stride.in_strides,
                    in_stride, radix, 0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(pc_stride.out_strides,
                    in_stride * offset, radix, 0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                k_stride.v_in_stride = in_stride_w_ds;
                k_stride.v_out_stride = out_stride_w_ds * radix;
                pc_stride.v_in_stride = in_stride_w_ds * radix;
                pc_stride.v_out_stride = in_stride_w_ds;
            }
            else
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride, radix, 0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride, radix, 1 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                populate_stride_array_wrapper(pc_stride.in_strides,
                    out_stride, radix, 0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(pc_stride.out_strides,
                    out_stride, radix, 0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds * radix;

                pc_stride.v_in_stride = out_stride_w_ds * radix;
                pc_stride.v_out_stride = out_stride_w_ds * radix;
            }

            // convert the inputs from in-order to out-of-order
            permuted_copy<T>(in1, temp, offset, radix,
                             &pc_stride, data_stride);
            memcpy(in1, temp, sizeof(T) * input_length * data_stride);
            permuted_copy<T>(in2, temp, offset, radix,
                             &pc_stride, data_stride);
            memcpy(in2, temp, sizeof(T) * input_length * data_stride);
            fft_kernel(in1, in1, out1, out1, offset, &k_stride,
                       is_bwd);
            fft_kernel(in2, in2, out2, out2, offset, &k_stride,
                       is_bwd);

            // complex variables
            T cmul_temp[2] = {0.0, 0.0};
            T e_k[2]       = {1.0, 0.0};
            T sign         = is_bwd ? 1.0 : -1.0;

            T *out1_full = NULL;
            ALLOC_ALIGN_INIT(out1_full, T, 2 * output_length * sizeof(T));
            T *out2_full = NULL;
            ALLOC_ALIGN_INIT(out2_full, T, 2 * output_length * sizeof(T));
            convert_halfcomplex_to_fullcomplex(out1_full, out1, radix, offset,
                                     out_stride);
            convert_halfcomplex_to_fullcomplex(out2_full, out2, radix, offset,
                                     out_stride);

            for (int k = 0; k < radix * offset; k++)
            {
                INTP mk = (m * k) % radix;
                T angle;
                if (is_fused_kernel(kernel_type))
                    angle = (sign * AOCLFFTZ_2_PI * m * (k + 0.5)) / radix;
                else
                    angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(out1_full + k * out_stride * 2, e_k,
                     out1_full + k * out_stride * 2, cmul_temp);
            }

            if (!use_special)
            {
                EXPECT_LE(get_error(out1_full, out2_full, output_length * 2),
                          tolerance)
                    << "Property: timeshift, seed: " << random_seed << "\n";
            }
            FREE_ALIGN_ALLOCATED_MEM(out1_full);
            FREE_ALIGN_ALLOCATED_MEM(out2_full);
        }

        /* Real kernels validation (C2R) for backward FFT by the following
         * steps-
         * 1) complex conversion of input from half complex to full complex
         *    perfrom phase multiplication, then revert to half complex
         * 2) stride array popultaion for input and output based on whether
         *    the operation is out-of-place or in-place.
         * 3) FFT kernel execution
         * 4) output conversion from out-of-order to inorder using permuted_copy
         * 5) right circular shift of output by `m` times,
         * 6) compare the transformed outputs to ensure they are within a
         *    specified tolerance.
         */
        else
        {
            // complex variables
            T cmul_temp[2] = {0.0, 0.0};
            T e_k[2]       = {1.0, 0.0};
            T sign         = is_bwd ? 1.0 : -1.0;

            T *in1_full = NULL;
            ALLOC_ALIGN_INIT(in1_full, T, 2 * input_length * sizeof(T));
            convert_halfcomplex_to_fullcomplex(in1_full, in1, radix, offset,
                                               in_stride);

            INTP m = (INTP)(rand() % (radix - 1)) + 1;
            for (INTP k = 0; k < radix * offset; k++)
            {
                INTP mk = (m * k) % radix;
                T angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(in1_full + k * in_stride * 2, e_k,
                     in1_full + k * in_stride * 2, cmul_temp);
            }

            convert_fullcomplex_to_halfcomplex(in2, in1_full, radix, offset,
                                               in_stride);

            // standard kernel
            if (is_out_of_place)
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride, radix,
                    1 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride * offset, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                populate_stride_array_wrapper(pc_stride.in_strides,
                    out_stride * offset, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(pc_stride.out_strides,
                    out_stride, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds;
                pc_stride.v_in_stride = out_stride_w_ds;
                pc_stride.v_out_stride = out_stride_w_ds * radix;
            }
            else
            {
                populate_stride_array_wrapper(k_stride.in_strides,
                    in_stride, radix,
                    1 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(k_stride.out_strides,
                    out_stride, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                populate_stride_array_wrapper(pc_stride.in_strides,
                    out_stride, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);
                populate_stride_array_wrapper(pc_stride.out_strides,
                    out_stride, radix,
                    0 /* compute_half_Complex */,
                    0 /* adjust_to_full_complex */);

                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds * radix;
                pc_stride.v_in_stride = out_stride_w_ds * radix;
                pc_stride.v_out_stride = out_stride_w_ds * radix;
            }

            fft_kernel(in1, in1, out1, out1, offset, &k_stride,
                       is_bwd);
            fft_kernel(in2, in2, out2, out2, offset, &k_stride,
                       is_bwd);
            // convert the outputs from out-of-order to in-order
            permuted_copy<T>(out1, temp, offset, radix,
                             &pc_stride, data_stride);
            memcpy(out1, temp, sizeof(T) * output_length * data_stride);
            permuted_copy<T>(out2, temp, offset, radix,
                             &pc_stride, data_stride);
            memcpy(out2, temp, sizeof(T) * output_length * data_stride);


            // Perform circular right shift by `m` times
            // range of m => [1, radix)
            T *out2_temp = NULL;
            ALLOC_ALIGN_INIT(out2_temp, T, output_length * sizeof(T));
            memcpy(out2_temp, out2, output_length * sizeof(T));
            for (INTP itr = 0; itr < offset; itr++)
            {
                for (INTP idx = 0; idx < radix; idx++)
                {
                    for (INTP is = 0; is < out_stride; is++)
                    {
                        INTP src  = (itr * radix * out_stride +
                                    ((idx + (radix - m)) * out_stride) %
                                        (radix * out_stride) +
                                    is);
                        INTP dst  = ((itr * radix + idx) * out_stride + is);
                        out2[dst] = out2_temp[src];
                    }
                }
            }

            if (!use_special)
            {
                EXPECT_LE(get_error(out1, out2, output_length), tolerance)
                    << "Property: timeshift, seed: " << random_seed << "\n";
            }
            FREE_ALIGN_ALLOCATED_MEM(in1_full);
            FREE_ALIGN_ALLOCATED_MEM(out2_temp);
        }

        if (is_out_of_place)
        {
            FREE_ALIGN_ALLOCATED_MEM(out1);
            FREE_ALIGN_ALLOCATED_MEM(out2);
        }
        FREE_ALIGN_ALLOCATED_MEM(in1);
        FREE_ALIGN_ALLOCATED_MEM(in2);
        FREE_ALIGN_ALLOCATED_MEM(out_comp);
        FREE_ALIGN_ALLOCATED_MEM(temp);
        FREE_ALIGN_ALLOCATED_MEM(k_stride.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(k_stride.out_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(pc_stride.out_strides);
    } // run_timeshift_test

    /**
     * @brief A function to check the FFT kernel with DFT reference
     *
     * @param input_type test input type
     */
    void run_dft_reference_test(aocl_fftz_test_input input_type)
    {
        // k_in_size : size of input buffer for kernel(half/full complex form)
        INTP k_in_size = radix * in_stride * data_stride * offset;

        /* fc_in_size : size of input buffer for full complex form
         * For complex -> fc_in_size == k_in_size
         * For real    -> fc_in_size == k_in_size * 2
         */
        INTP fc_in_size = radix * in_stride * offset * 2;

        // k_out_size : size of output buffer for kernel(half/full complex form)
        INTP k_out_size = radix * out_stride * data_stride * offset;

        /* fc_out_size : size of output buffer for full complex form
         * For complex -> fc_out_size == k_out_size
         * For real    -> fc_out_size == k_out_size * 2
         */
        INTP fc_out_size = radix * out_stride * offset * 2;

        T *in = prepare_input(input_type);
        if (in == nullptr)
        {
            return;
        }
        T *in_full = NULL;
        ALLOC_ALIGN_INIT(in_full, T, fc_in_size * sizeof(T));

        // prepare local strides
        aoclfftz_strides_t kernel_stride;

        // strides for FFT kernel
        ALLOC_ALIGN_UNINIT(kernel_stride.in_strides, INTP,
                           radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(kernel_stride.out_strides, INTP,
                           radix * sizeof(INTP));

        T *out = NULL;
        ALLOC_ALIGN_INIT(out, T, k_out_size * sizeof(T));
        T *out_full = NULL;
        ALLOC_ALIGN_INIT(out_full, T, fc_out_size * sizeof(T));
        T *out_ref = NULL;
        ALLOC_ALIGN_INIT(out_ref, T, fc_out_size * sizeof(T));

        kernel_stride.v_in_stride = in_stride_w_ds * radix;
        kernel_stride.v_out_stride = out_stride_w_ds * radix;

        // populating strides and executing FFT for complex kernels
        if (is_complex)
        {
            populate_stride_array_wrapper(kernel_stride.in_strides,
                in_stride_w_ds, radix, 0, 0);
            populate_stride_array_wrapper(kernel_stride.out_strides,
                out_stride_w_ds, radix, 0, 0);
                T *in_r  = (is_bwd) ? (in + 1) : (in);
                T *in_i  = (is_bwd) ? (in) : (in + 1);
                T *out_r = (is_bwd) ? (out + 1) : (out);
                T *out_i = (is_bwd) ? (out) : (out + 1);
            fft_kernel(in_r, in_i, out_r, out_i, offset, &kernel_stride,
                is_bwd);
        }
        // populating strides and executing FFT for real kernels
        else
        {
            populate_stride_array_wrapper(kernel_stride.in_strides,
                in_stride_w_ds, radix, is_bwd, 0);
            populate_stride_array_wrapper(kernel_stride.out_strides,
                out_stride_w_ds, radix, !is_bwd, 0);
            fft_kernel(in, in, out, out, offset, &kernel_stride, is_bwd);
        }

        if (is_complex)
        {
            memcpy(in_full, in, fc_in_size * sizeof(T));
            memcpy(out_full, out, fc_out_size * sizeof(T));
        }
        else
        {
            // for real problems,
            // forward:
            //   input  : real to full-complex
            //   output : half-complex to full-complex
            // backward:
            //   input  : half-complex to full-complex
            //   output : real to full-complex
            if (is_bwd)
            {
                convert_halfcomplex_to_fullcomplex(in_full, in, radix, offset,
                                                   in_stride);
            }
            else
            {
                for (int i = 0; i < k_in_size; i++)
                {
                    in_full[i * 2] = in[i]; // interleaved format
                }
            }
            // re-arrange the output
            if (is_bwd)
            {
                for (INTP i = 0; i < k_out_size; i++)
                {
                    out_full[i * 2] = out[i]; // interleaved format
                }
            }
            else
            {
                convert_halfcomplex_to_fullcomplex(out_full, out, radix, offset,
                                                   out_stride);
            }
        }

        if (is_complex == false) // is_real
        {
            populate_stride_array_wrapper(kernel_stride.in_strides,
                in_stride_w_ds * 2, radix, 0, 0);
            populate_stride_array_wrapper(kernel_stride.out_strides,
                                  out_stride_w_ds * 2, radix, 0, 0);
            kernel_stride.v_in_stride = in_stride * radix * 2;
            kernel_stride.v_out_stride = out_stride * radix * 2;
        }

        T e[2]    = {0.0, 0.0};     // complex variable
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
                    if (is_fused_kernel(kernel_type))
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

        FREE_ALIGN_ALLOCATED_MEM(in);
        FREE_ALIGN_ALLOCATED_MEM(in_full);
        FREE_ALIGN_ALLOCATED_MEM(out);
        FREE_ALIGN_ALLOCATED_MEM(out_full);
        FREE_ALIGN_ALLOCATED_MEM(out_ref);
        FREE_ALIGN_ALLOCATED_MEM(kernel_stride.in_strides);
        FREE_ALIGN_ALLOCATED_MEM(kernel_stride.out_strides);
    } // run_dft_reference_test
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
        is_complex  = true;
        data_stride = 2;
        tolerance   = TOLERANCE_F;
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
        is_complex  = true;
        data_stride = 2;
        tolerance   = TOLERANCE_D;
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
        is_complex  = false;
        data_stride = 1;
        tolerance   = TOLERANCE_F;
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
        is_complex  = false;
        data_stride = 1;
        tolerance   = TOLERANCE_D;
    }
};

#endif // AOCLFFTZ_KERNEL_GTEST_BASE_H
