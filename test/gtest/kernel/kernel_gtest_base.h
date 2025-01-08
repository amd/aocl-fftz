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
#include "test/gtest/kernel/kernel_gtest_utils.h"
#include "test/gtest/gtest_types.h"
#include "utils/allocator.h"

/**
 * @brief Base class for the AOCLFFTZ Kernel Tests
 *
 * @tparam T type of the input / output. (Supported types: FLOAT, DOUBLE)
 */
template <class T>
class AoclfftzKernelTestBase
    : public ::testing::TestWithParam<std::tuple<aoclfftz_kernel_test_params_t,
                                      std::tuple<INTP, INTP, INTP, UINT8,
                                      UINT8>>>
{
  protected:
    bool use_special;          // whether to use special inputs or not
    UINT32 radix;              // radix of the FFT kernel
    INTP length;               // no. of points in the data or data length
    INTP input_length;         // length with input strides
    INTP output_length;        // length with output strides
    kfft_ fft_kernel;          // pointer to the kernel function
    INTP in_stride;            // input stride value
    INTP out_stride;           // output stride value
    UINT8 kernel_type;         // kernel type
    UINT8 is_bwd;              // direction, 1 -> BWD, 0 -> FWD
    UINT8 is_out_of_place;     // result placement, 0 -> inplace, 1 -> outplace
    INTP offset;               // no. of sets (1 offset size = radix)
    UINT32 random_seed;        // seed value of random number generator
    INTP in_stride_w_ds;       // in stride value multiplied with data_stride
    INTP out_stride_w_ds;      // out stride value multiplied with data_stride

    void SetUp() override
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
    void run_kernel_test(bool special = false)
    {
        use_special = special;
        auto param = std::get<0>(GetParam());
        auto io_param = std::get<1>(GetParam());
        in_stride = std::get<0>(io_param);
        out_stride = std::get<1>(io_param);
        offset = std::get<2>(io_param);
        is_bwd = std::get<3>(io_param);
        is_out_of_place = std::get<4>(io_param);
        radix  = std::get<0>(param);
        kernel_type = std::get<1>(param);
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
        fft_kernel = get_kernel<T>(table, radix);
        if (fft_kernel == nullptr)
        {
            GTEST_NONFATAL_FAILURE_(
                std::string("Radix" + std::to_string(radix) +
                            " kernel not found in the kernel table")
                    .c_str());
            return;
        }

        input_length = length * in_stride;
        in_stride_w_ds = in_stride * DATA_STRIDE;
        output_length = length * out_stride;
        out_stride_w_ds = out_stride * DATA_STRIDE;

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
                               aocl_fftz_test_input::SIGNAL);
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
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * DATA_STRIDE);
        for (INTP idx = 0; idx < input_length * DATA_STRIDE; ++idx)
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
        ALLOC_ALIGN_INIT(input, T, input_length * DATA_STRIDE * sizeof(T));
        INTP idx = (INTP)(rand() % length) * in_stride;
        // range: [-10.0, 10.0) with 3 decimal precision
        input[idx * DATA_STRIDE] = (T)((rand() % 2000) / 200.0 - 10.0);
        input[idx * DATA_STRIDE + 1] = (T)((rand() % 2000) / 200.0 - 10.0);
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
        // Sine wave cycles
        INTP cycles = (rand() % (input_length / 2)) + 2;
        T size = AOCLFFTZ_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, length)
        INTP shift = rand() % input_length;
        // scale the amplitude of the wave by `scale` times, scale range:
        // [0.0 5.0)
        T scale = ((T)rand() / (T)RAND_MAX) * 5.0;
        T *input = NULL;
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * DATA_STRIDE);
        for (INTP i = 0; i < input_length; i++)
        {
            input[((i + shift) % input_length) * DATA_STRIDE] =
                sin((T)(i * size) / input_length) * scale;
            input[((i + shift) % input_length) * DATA_STRIDE + 1] = 0.0;
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
        ALLOC_ALIGN_UNINIT(input, T, sizeof(T) * input_length * DATA_STRIDE);
        for (INTP idx = 0; idx < length; idx += in_stride)
        {
            input[idx * DATA_STRIDE] = get_maybe_special_value<T>(rand());
            input[idx * DATA_STRIDE + 1] = get_maybe_special_value<T>(rand());
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
    T get_error(T *a, T *b, bool use_input_params = false)
    {
        T max_e = 0.0;
        T max_mag = 0.0;
        INTP _length = use_input_params ? input_length : output_length;
        INT32 _stride = use_input_params ? in_stride : out_stride;
        for (int idx = 0; idx < _length; idx += _stride)
        {
            T e = (std::max)(
                std::abs(a[idx * DATA_STRIDE] - b[idx * DATA_STRIDE]),
                std::abs(a[idx * DATA_STRIDE + 1] - b[idx * DATA_STRIDE + 1]));
            T mag = (std::min)((std::max)(std::abs(a[idx * DATA_STRIDE]),
                                          std::abs(a[idx * DATA_STRIDE + 1])),
                               (std::max)(std::abs(b[idx * DATA_STRIDE]),
                                          std::abs(b[idx * DATA_STRIDE + 1])));
            max_e = (std::max)(max_e, e);
            max_mag = (std::max)(max_mag, mag);
        }
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
     * @brief A function to check the linearity property of the FFT kernel
     *
     * @param input_type test input type
     */
    void run_linearity_test(aocl_fftz_test_input input_type)
    {
        T *in1 = prepare_input(input_type);
        T *in2 = prepare_input(input_type);
        if (in1 == nullptr || in2 == nullptr)
        {
            return;
        }
        T *in_combined = NULL;
        ALLOC_ALIGN_UNINIT(in_combined, T,
                           sizeof(T) * input_length * DATA_STRIDE);
        T *out1, *out2, *out_combined, *out_added;

        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out1, T, output_length * DATA_STRIDE * sizeof(T));
            ALLOC_ALIGN_INIT(out2, T, output_length * DATA_STRIDE * sizeof(T));
            ALLOC_ALIGN_INIT(out_combined, T,
                                output_length * DATA_STRIDE * sizeof(T));
        }
        else
        {
            out1 = in1;
            out2 = in2;
            out_combined = in_combined;
        }

        ALLOC_ALIGN_INIT(out_added, T, output_length * DATA_STRIDE * sizeof(T));

        // Constant multiplier a1 and a2 will range from [-10.0 to 10.0)
        // with one digit precision
        T a1[DATA_STRIDE] = {(T)((rand() % 200) / 20.0 - 10.0), 0.0};
        T a2[DATA_STRIDE] = {(T)((rand() % 200) / 20.0 - 10.0), 0.0};

        T temp1[DATA_STRIDE] = {0.0, 0.0};
        T temp2[DATA_STRIDE] = {0.0, 0.0};
        T cmul_temp[DATA_STRIDE] = {0.0, 0.0};
        for (INTP idx = 0; idx < input_length; ++idx)
        {
            CMUL(a1, in1 + idx * DATA_STRIDE, temp1, cmul_temp);
            CMUL(a2, in2 + idx * DATA_STRIDE, temp2, cmul_temp);
            CADD(temp1, temp2, in_combined + idx * DATA_STRIDE);
        }

        // prepare local stride
        aoclfftz_strides_t k_stride;
        ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));

        // permuted kernel
        if (kernel_type & 0x1)
        {
            if (is_out_of_place)
            {
                for (INTP i = 0; i < radix; i++)
                {
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = offset * i * out_stride_w_ds;
                }
                k_stride.v_in_stride = radix * in_stride_w_ds;
                k_stride.v_out_stride = out_stride_w_ds;
            }
            else
            {
                for (INTP i = 0; i < radix; i++)
                {
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = k_stride.in_strides[i];
                }
                k_stride.v_in_stride = radix * in_stride_w_ds;
                k_stride.v_out_stride = k_stride.v_in_stride;
            }
        }
        // standard kernel
        else
        {
            if (is_out_of_place)
            {
                for (INTP i = 0; i < radix; i++)
                {
                    k_stride.in_strides[i] = offset * i * in_stride_w_ds;
                    k_stride.out_strides[i] = i * out_stride_w_ds;
                }
                k_stride.v_in_stride = in_stride_w_ds;
                k_stride.v_out_stride = radix * out_stride_w_ds;
            }
            else
            {
                for (INTP i = 0; i < radix; i++)
                {
                    k_stride.in_strides[i] = offset * i * in_stride_w_ds;
                    k_stride.out_strides[i] = k_stride.in_strides[i];
                }
                k_stride.v_in_stride = in_stride_w_ds;
                k_stride.v_out_stride = k_stride.v_in_stride;
            }
        }

        // Initialize kernel input/output variables
        T *in1_r = (is_bwd) ? (in1 + 1) : (in1);
        T *in1_i = (is_bwd) ? (in1) : (in1 + 1);
        T *out1_r = (is_bwd) ? (out1 + 1) : (out1);
        T *out1_i = (is_bwd) ? (out1) : (out1 + 1);
        T *in2_r = (is_bwd) ? (in2 + 1) : (in2);
        T *in2_i = (is_bwd) ? (in2) : (in2 + 1);
        T *out2_r = (is_bwd) ? (out2 + 1) : (out2);
        T *out2_i = (is_bwd) ? (out2) : (out2 + 1);

        T *in_combined_r = (is_bwd) ? (in_combined + 1) : (in_combined);
        T *in_combined_i = (is_bwd) ? (in_combined) : (in_combined + 1);
        T *out_combined_r = (is_bwd) ? (out_combined + 1) : (out_combined);
        T *out_combined_i = (is_bwd) ? (out_combined) : (out_combined + 1);

        fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, is_bwd);
        fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, is_bwd);
        fft_kernel(in_combined_r, in_combined_i, out_combined_r, out_combined_i,
                   offset, &k_stride, is_bwd);

        for (INTP idx = 0; idx < output_length; ++idx)
        {
            CMUL(a1, out1 + idx * DATA_STRIDE, temp1, cmul_temp);
            CMUL(a2, out2 + idx * DATA_STRIDE, temp2, cmul_temp);
            CADD(temp1, temp2, out_added + idx * DATA_STRIDE);
        }

        if (!use_special)
        {
            if (typeid(T) == typeid(FLOAT))
            {
                EXPECT_LE(get_error(out_combined, out_added), TOLERANCE_F)
                    << "Property: linearity, seed: " << random_seed << "\n";
            }
            else // typeid(T) == typeid(DOUBLE)
            {
                EXPECT_LE(get_error(out_combined, out_added), TOLERANCE_D)
                    << "Property: linearity, seed: " << random_seed << "\n";
            }
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
    } // run_linearity_test

    /**
     * @brief A function to check the transformation property of the FFT kernel
     * using unit impulse signal
     *
     * @param input_type test input type
     */
    void run_unit_impulse_transform_test(aocl_fftz_test_input input_type)
    {
        T *in = prepare_input(input_type);
        if (in == nullptr)
        {
            return;
        }
        T *in_copy;
        ALLOC_ALIGN_INIT(in_copy, T, input_length * DATA_STRIDE * sizeof(T));
        memcpy(in_copy, in, sizeof(T) * input_length * DATA_STRIDE);
        T *inv_out;
        ALLOC_ALIGN_INIT(inv_out, T, input_length * DATA_STRIDE * sizeof(T));
        T *perm_out;
        ALLOC_ALIGN_INIT(perm_out, T, output_length * DATA_STRIDE * sizeof(T));
        T *out, *perm_inv_out;

        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out, T, output_length * DATA_STRIDE * sizeof(T));
            ALLOC_ALIGN_INIT(perm_inv_out, T,
                             input_length * DATA_STRIDE * sizeof(T));
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

        if (kernel_type & 0x1)
        {
            if (is_out_of_place)
            {
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = offset * i * out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.in_strides[i] = i * out_stride_w_ds;
                    k_stride_rev.out_strides[i] = offset * i * in_stride_w_ds;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.in_strides[i] = offset * i * out_stride_w_ds;
                    pc_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.in_strides[i] = offset * i * in_stride_w_ds;
                    pc_stride_rev.out_strides[i] = i * in_stride_w_ds;
                }
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
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.in_strides[i] = i * out_stride_w_ds;
                    k_stride_rev.out_strides[i] = i * in_stride_w_ds;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.in_strides[i] = offset * i * out_stride_w_ds;
                    pc_stride.out_strides[i] = offset * i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.in_strides[i] = offset * i * in_stride_w_ds;
                    pc_stride_rev.out_strides[i] = offset*i * in_stride_w_ds;
                }
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
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = offset * i * in_stride_w_ds;
                    k_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.in_strides[i] = offset * i * out_stride_w_ds;
                    k_stride_rev.out_strides[i] = i * in_stride_w_ds;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.in_strides[i] = i * out_stride_w_ds;
                    pc_stride.out_strides[i] = offset * i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.in_strides[i] = i * in_stride_w_ds;
                    pc_stride_rev.out_strides[i] = offset * i * in_stride_w_ds;
                }
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
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = offset * i * in_stride_w_ds;
                    k_stride.out_strides[i] = offset * i * out_stride_w_ds;
                    // strides for reverse FFT kernel
                    k_stride_rev.in_strides[i] = offset * i * out_stride_w_ds;
                    k_stride_rev.out_strides[i] = offset* i * in_stride_w_ds;
                    // strides for permuted copy on reverse FFT output
                    pc_stride.in_strides[i] = i * out_stride_w_ds;
                    pc_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride_rev.in_strides[i] = i * in_stride_w_ds;
                    pc_stride_rev.out_strides[i] = i * in_stride_w_ds;
                }
                // strides for FFT kernel
                k_stride.v_in_stride = in_stride_w_ds;
                k_stride.v_out_stride = out_stride_w_ds;
                // strides for reverse FFT kernel
                k_stride_rev.v_in_stride = out_stride_w_ds;
                k_stride_rev.v_out_stride = in_stride_w_ds;
                // strides for permuted copy on reverse FFT output
                pc_stride.v_in_stride = out_stride_w_ds * radix;
                pc_stride.v_out_stride = out_stride_w_ds * radix;
                // strides for permuted copy on FFT output
                pc_stride_rev.v_in_stride = in_stride_w_ds * radix;
                pc_stride_rev.v_out_stride = in_stride_w_ds * radix;
            }
        }

        // Initialize kernel input/output variables

        T *in_r = (is_bwd) ? (in + 1) : (in);
        T *in_i = (is_bwd) ? (in) : (in + 1);
        T *out_r = (is_bwd) ? (out + 1) : (out);
        T *out_i = (is_bwd) ? (out) : (out + 1);

        fft_kernel(in_r, in_i, out_r, out_i, offset, &k_stride, is_bwd);
        // convert the FFT kernel output from in-order to out-of-order for
        // standard kernel and vise versa for permuted kernel
        permuted_copy<T>(out, perm_out, offset, radix, &pc_stride);

        T *perm_out_r = (!is_bwd) ? (perm_out + 1) : (perm_out);
        T *perm_out_i = (!is_bwd) ? (perm_out) : (perm_out + 1);
        T *perm_inv_out_r = (!is_bwd) ? (perm_inv_out + 1) : (perm_inv_out);
        T *perm_inv_out_i = (!is_bwd) ? (perm_inv_out) : (perm_inv_out + 1);

        fft_kernel(perm_out_r, perm_out_i, perm_inv_out_r, perm_inv_out_i,
                   offset, &k_stride_rev, !is_bwd);
        // convert the reverse FFT kernel output from in-order to out-of-order
        // for standard kernel and vise versa for permuted kernel
        permuted_copy<T>(perm_inv_out, inv_out, offset, radix, &pc_stride_rev);

        // normalize reverse FFT output
        for (INTP idx = 0; idx < input_length; idx++)
        {
            inv_out[idx * DATA_STRIDE] /= radix;
            inv_out[idx * DATA_STRIDE + 1] /= radix;
        }
        if (!use_special)
        {
            if (typeid(T) == typeid(FLOAT))
            {
                EXPECT_LE(get_error(in_copy, inv_out, true), TOLERANCE_F)
                    << "Property: transformation, seed: " << random_seed
                    << "\n";
            }
            else // typeid(T) == typeid(DOUBLE)
            {
                EXPECT_LE(get_error(in_copy, inv_out, true), TOLERANCE_D)
                    << "Property: transformation, seed: " << random_seed
                    << "\n";
            }
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
        ALLOC_ALIGN_UNINIT(in2, T, sizeof(T) * input_length * DATA_STRIDE);
        T *out1, *out2;
        if (is_out_of_place)
        {
            ALLOC_ALIGN_INIT(out1, T, output_length * DATA_STRIDE * sizeof(T));
            ALLOC_ALIGN_INIT(out2, T, output_length * DATA_STRIDE * sizeof(T));
        }
        else
        {
            out1 = in1;
            out2 = in2;
        }

        T *out_comp;
        ALLOC_ALIGN_INIT(out_comp, T, output_length * DATA_STRIDE * sizeof(T));
        T *temp;
        ALLOC_ALIGN_INIT(temp, T,
            (std::max)(input_length, output_length) * DATA_STRIDE * sizeof(T));

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
                               DATA_STRIDE;
                    INTP dst = ((itr * radix + idx) * in_stride + is) *
                               DATA_STRIDE;
                    in2[dst] = in1[src];
                    in2[dst + 1] = in1[src + 1];
                }
            }
        }

        // prepare local strides
        aoclfftz_strides_t k_stride;
        ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));
        aoclfftz_strides_t pc_stride;
        ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP, radix * sizeof(INTP));

        // permuted kernel
        if (kernel_type & 0x1)
        {
            if (is_out_of_place)
            {
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = offset * i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride.in_strides[i] = offset * i * out_stride_w_ds;
                    pc_stride.out_strides[i] = i * out_stride_w_ds;
                }
                // strides for FFT kernel
                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds;
                // strides for permuted copy on FFT output
                pc_stride.v_in_stride = out_stride_w_ds;
                pc_stride.v_out_stride = out_stride_w_ds * radix;
            }
            else
            {
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = i * in_stride_w_ds;
                    k_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for permuted copy on FFT output
                    pc_stride.in_strides[i] = i * out_stride_w_ds;
                    pc_stride.out_strides[i] = i * out_stride_w_ds;
                }
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
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] = offset * i * in_stride_w_ds;
                    k_stride.out_strides[i] = i * out_stride_w_ds;
                    // strides for permuted copy on input
                    pc_stride.in_strides[i] = i * in_stride_w_ds;
                    pc_stride.out_strides[i] = offset * i * in_stride_w_ds;
                }
                // strides for FFT kernel
                k_stride.v_in_stride = in_stride_w_ds;
                k_stride.v_out_stride = out_stride_w_ds * radix;
                // strides for permuted copy on input
                pc_stride.v_in_stride = in_stride_w_ds * radix;
                pc_stride.v_out_stride = in_stride_w_ds;
            }
            else
            {
                for (INTP i = 0; i < radix; i++)
                {
                    // strides for FFT kernel
                    k_stride.in_strides[i] =  i * in_stride_w_ds;
                    k_stride.out_strides[i] =  i * out_stride_w_ds;
                    // strides for permuted copy on input
                    pc_stride.in_strides[i] = i * in_stride_w_ds;
                    pc_stride.out_strides[i] = i * in_stride_w_ds;
                }
                // strides for FFT kernel
                k_stride.v_in_stride = in_stride_w_ds * radix;
                k_stride.v_out_stride = out_stride_w_ds * radix;
                // strides for permuted copy on input
                pc_stride.v_in_stride = in_stride_w_ds * radix;
                pc_stride.v_out_stride = in_stride_w_ds * radix;
            }
        }

        // Initialize kernel input/output variables
        T *in1_r = (is_bwd) ? (in1 + 1) : (in1);
        T *in1_i = (is_bwd) ? (in1) : (in1 + 1);
        T *out1_r = (is_bwd) ? (out1 + 1) : (out1);
        T *out1_i = (is_bwd) ? (out1) : (out1 + 1);
        T *in2_r = (is_bwd) ? (in2 + 1) : (in2);
        T *in2_i = (is_bwd) ? (in2) : (in2 + 1);
        T *out2_r = (is_bwd) ? (out2 + 1) : (out2);
        T *out2_i = (is_bwd) ? (out2) : (out2 + 1);

        // permuted kernel
        if (kernel_type & 0x1)
        {
            fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, is_bwd);
            fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, is_bwd);
            // convert the FFT kernel outputs from out-of-order to in-order
            permuted_copy<T>(out1, temp, offset, radix, &pc_stride);
            memcpy(out1, temp, sizeof(T) * output_length * DATA_STRIDE);
            permuted_copy<T>(out2, temp, offset, radix, &pc_stride);
            memcpy(out2, temp, sizeof(T) * output_length * DATA_STRIDE);
        }
        // standard kernel
        else
        {
            // convert the inputs from in-order to out-of-order
            permuted_copy<T>(in1, temp, offset, radix, &pc_stride);
            memcpy(in1, temp, sizeof(T) * input_length * DATA_STRIDE);
            permuted_copy<T>(in2, temp, offset, radix, &pc_stride);
            memcpy(in2, temp, sizeof(T) * input_length * DATA_STRIDE);
            fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, is_bwd);
            fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, is_bwd);
        }

        T cmul_temp[DATA_STRIDE] = {0.0, 0.0};
        T e_k[DATA_STRIDE] = {1.0, 0.0};
        T sign = is_bwd ? 1.0 : -1.0;

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
            if (typeid(T) == typeid(FLOAT))
            {
                EXPECT_LE(get_error(out2, out_comp), TOLERANCE_F)
                    << "Property: timeshift, seed: " << random_seed << "\n";
            }
            else // typeid(T) == typeid(DOUBLE)
            {
                EXPECT_LE(get_error(out2, out_comp), TOLERANCE_D)
                    << "Property: timeshift, seed: " << random_seed << "\n";
            }
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
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for FLOAT type
 *
 */

class AoclfftzKernelTestFloat : public AoclfftzKernelTestBase<FLOAT>
{
};

/**
 * @brief A derived class from AoclfftzKernelTestBase for DOUBLE type
 *
 */
class AoclfftzKernelTestDouble : public AoclfftzKernelTestBase<DOUBLE>
{
};

#endif // AOCLFFTZ_KERNEL_GTEST_BASE_H
