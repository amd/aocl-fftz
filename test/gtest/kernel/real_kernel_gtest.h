/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file complex_kernel_gtest.h
 *
 * @brief file for DFT property function's definition(r2hc type) for gtest.
 *
 * This file contains the definitions of member functions of DFT properties
 * to be used for running real kernel unit tests using GTest.
 *
 * @author Partiksha
 *
 */

#ifndef AOCLFFTZ_REAL_KERNEL_GTEST_H
#define AOCLFFTZ_REAL_KERNEL_GTEST_H

#include "test/gtest/kernel/kernel_gtest_base.h"

template <class T>
VOID AoclfftzKernelTestBase<T>::run_linearity_test_real(
    aocl_fftz_test_input input_type)
{
    T *in1 = prepare_input(input_type);
    T *in2 = prepare_input(input_type);
    if (in1 == nullptr || in2 == nullptr)
    {
        return;
    }
    bool is_fused = is_fused_kernel(kernel_type);

    T *in_combined = NULL;
    ALLOC_ALIGN_UNINIT(in_combined, T, sizeof(T) * input_length * data_stride);
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
    ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP,
        radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
        radix * buf_size_multiplier * sizeof(INTP));

    // standard kernel
    if (is_out_of_place)
    {
        populate_stride_array_wrapper(k_stride.in_strides,
            in_stride_w_ds * offset * buf_size_multiplier, radix,
            0 /* compute_half_complex */,
            0 /* adjust_to_full_complex */);
        populate_stride_array_wrapper(k_stride.out_strides,
            out_stride_w_ds, radix * buf_size_multiplier,
            1 /* compute_half_complex */,
            0 /* adjust_to_full_complex */);

        if (is_fused)
        {
            prepare_fused_kernel_strides_wrapper(k_stride.in_strides,
                                         radix, in_stride_w_ds);
        }
        k_stride.v_in_stride  = in_stride_w_ds * buf_size_multiplier;
        k_stride.v_out_stride = radix * out_stride_w_ds * buf_size_multiplier;
    }
    else
    {
        populate_stride_array_wrapper(k_stride.in_strides,
            in_stride_w_ds, radix * buf_size_multiplier,
            0 /* compute_half_complex */,
            0 /* adjust_to_full_complex */);
        populate_stride_array_wrapper(k_stride.out_strides,
            in_stride_w_ds, radix * buf_size_multiplier,
            1 /* compute_half_complex */,
            0 /* adjust_to_full_complex */);
        k_stride.v_in_stride = radix * in_stride_w_ds
                               * buf_size_multiplier;
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
            output_length * data_stride), tolerance)
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
} // run_linearity_test

/**
* @brief A function to check the transformation property of the FFT kernel
* using unit impulse signal
*
* @param input_type test input type
*/
template <class T>
VOID AoclfftzKernelTestBase<T>::run_unit_impulse_transform_test_real(
    aocl_fftz_test_input input_type)
{
    T *in = prepare_input(input_type);
    if (in == nullptr)
    {
        return;
    }
    bool is_fused = is_fused_kernel(kernel_type);

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
    ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));

    aoclfftz_strides_t k_stride_rev;
    ALLOC_ALIGN_UNINIT(k_stride_rev.in_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride_rev.out_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));

    aoclfftz_strides_t pc_stride;
    ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));

    aoclfftz_strides_t pc_stride_rev;
    ALLOC_ALIGN_UNINIT(pc_stride_rev.in_strides, INTP,
        radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride_rev.out_strides, INTP,
        radix * buf_size_multiplier * sizeof(INTP));


    // standard kernel
    if (is_out_of_place)
    {
        if (!is_bwd)
        {
            // strides for forward real FFT
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds * offset * buf_size_multiplier, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds, radix * buf_size_multiplier,
                1 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);

            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds, radix * buf_size_multiplier,
                1 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds * offset * buf_size_multiplier, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);

            if (is_fused)
            {
                prepare_fused_kernel_strides_wrapper( k_stride.in_strides,
                    radix, in_stride_w_ds);
                prepare_fused_kernel_strides_wrapper(k_stride_rev.out_strides,
                    radix, in_stride_w_ds);
            }
        }
        else
        {
            // strides for forward real FFT
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds, radix * buf_size_multiplier,
                1 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds * offset * buf_size_multiplier, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);

            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds * offset * buf_size_multiplier, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds, radix * buf_size_multiplier,
                1 /* compute_half_complex */,
                0 /* adjust_to_half_complex*/);
            if (is_fused)
            {
                prepare_fused_kernel_strides_wrapper(k_stride.out_strides,
                    radix, out_stride_w_ds);
                prepare_fused_kernel_strides_wrapper(k_stride_rev.in_strides,
                    radix, out_stride_w_ds);
            }
        }

        k_stride.v_in_stride = is_bwd ?
                in_stride_w_ds * radix * buf_size_multiplier :
                in_stride_w_ds * buf_size_multiplier;
        k_stride.v_out_stride = is_bwd ?
                out_stride_w_ds * buf_size_multiplier :
                out_stride_w_ds * radix * buf_size_multiplier;

        k_stride_rev.v_in_stride  = is_bwd ?
                out_stride_w_ds * buf_size_multiplier :
                out_stride_w_ds * buf_size_multiplier * radix;
        k_stride_rev.v_out_stride = is_bwd ?
                in_stride_w_ds * buf_size_multiplier * radix :
                in_stride_w_ds * buf_size_multiplier;
    }
    else
    {
        // strides for forward real FFT
        populate_stride_array_wrapper(k_stride.in_strides,
            in_stride_w_ds, radix * buf_size_multiplier,
            is_bwd /* compute_half_complex */,
            0 /* adjust_to_half_complex*/);
        populate_stride_array_wrapper(k_stride.out_strides,
            out_stride_w_ds, radix * buf_size_multiplier,
            !is_bwd /* compute_half_complex */,
            0 /* adjust_to_half_complex*/);
        populate_stride_array_wrapper(k_stride_rev.in_strides,
            out_stride_w_ds, radix * buf_size_multiplier,
            !is_bwd /* compute_half_complex */,
            0 /*adjust_to_half_complex*/);
        populate_stride_array_wrapper(k_stride_rev.out_strides,
            in_stride_w_ds, radix * buf_size_multiplier,
            is_bwd /* compute_half_complex */,
            0 /* adjust_to_half_complex*/);
        k_stride.v_in_stride = in_stride_w_ds * radix
                               * buf_size_multiplier;
        k_stride.v_out_stride = out_stride_w_ds * radix
                                * buf_size_multiplier;
        k_stride_rev.v_in_stride = out_stride_w_ds * radix
                                   * buf_size_multiplier;
        k_stride_rev.v_out_stride = in_stride_w_ds * radix
                                    * buf_size_multiplier;
    }

    // Temporary buffers to be used by Bwd-FFT
    T *temp_in = NULL;
    ALLOC_ALIGN_INIT(temp_in, T, input_length * data_stride * sizeof(T));
    T *temp_out = NULL;
    ALLOC_ALIGN_INIT(temp_out, T, input_length * data_stride * sizeof(T));

    /* In case of Bwd-FFT, input points are set by assuming the constant
       stride but will be read in a different order because of
       different strides for each input point.
       So, refactoring the input in order to place the valid set input
       points at the places where it will be read. */
    if (is_bwd)
    {
        for (INTP i = 0; i < offset; i++)
        {
            for (INTP j = 0; j < radix * buf_size_multiplier; j++)
            {
                INTP i_idx = radix * buf_size_multiplier * in_stride_w_ds * i
                             + k_stride.in_strides[j];
                INTP j_idx = radix * buf_size_multiplier * in_stride_w_ds * i
                             + j * in_stride_w_ds;
                temp_in[i_idx] = in[j_idx];
            }
        }
    }
    else
    {
        memcpy(temp_in, in, input_length *sizeof(T));
    }

    // Initialize kernel input/output variables
    fft_kernel(temp_in, temp_in, out, out, offset, &k_stride, is_bwd);
    // convert the FFT kernel output from in-order to out-of-order for
    // standard kernel and vise versa for permuted kernel
    fft_reverse_kernel(out, out, inv_out, inv_out, offset, &k_stride_rev,
                       !is_bwd);
    if (is_bwd)
    {
        for (INTP i = 0; i < offset; i++)
        {
            for (INTP j = 0; j < radix * buf_size_multiplier;  j++)
             {
                INTP i_idx = radix * buf_size_multiplier * in_stride_w_ds * i
                             + j * in_stride_w_ds;
                INTP j_idx = radix * buf_size_multiplier * in_stride_w_ds * i
                             + k_stride.in_strides[j];
                temp_out[i_idx] = inv_out[j_idx];
            }
        }
    }
    else
    {
        memcpy(temp_out, inv_out, input_length * sizeof(T));
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
template <class T>
VOID AoclfftzKernelTestBase<T>::run_timeshift_test_real(
    aocl_fftz_test_input input_type)
{
    T *in1 = prepare_input(input_type);
    if (in1 == nullptr)
    {
        return;
    }
    bool is_fused = is_fused_kernel(kernel_type);

    // Pointers to store standard DFT and shifted DFT data
    T *in1_standard_dft = NULL;
    ALLOC_ALIGN_UNINIT(in1_standard_dft, T,
        sizeof(T) * input_length * data_stride);
    T *in1_shifted_dft = NULL;
    ALLOC_ALIGN_UNINIT(in1_shifted_dft, T,
        sizeof(T) * input_length * data_stride);

    T *in2 = NULL;
    ALLOC_ALIGN_UNINIT(in2, T, sizeof(T) * input_length * data_stride);
    T *in2_standard_dft = NULL;
    ALLOC_ALIGN_UNINIT(in2_standard_dft, T,
        sizeof(T) * input_length * data_stride);
    T *in2_shifted_data = NULL;
    ALLOC_ALIGN_UNINIT(in2_shifted_data, T,
        sizeof(T) * input_length * data_stride);


    T *out1_standard_dft = NULL;
    ALLOC_ALIGN_INIT(out1_standard_dft, T,
        output_length * data_stride * sizeof(T));
    T *out1_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(out1_shifted_dft, T,
        output_length * data_stride * sizeof(T));
    T *out2_standard_dft = NULL;
    ALLOC_ALIGN_INIT(out2_standard_dft, T,
        output_length * data_stride * sizeof(T));
    T *out2_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(out2_shifted_dft, T,
        output_length * data_stride * sizeof(T));


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
    ALLOC_ALIGN_INIT(temp, T, (std::max)(input_length, output_length)
                     * data_stride * sizeof(T));

    // prepare local strides
    aoclfftz_strides_t k_stride;
    ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));

    aoclfftz_strides_t pc_stride;
    ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP,
                       radix * buf_size_multiplier * sizeof(INTP));


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
    if (!is_bwd)
    {
        // standard kernel
        if (is_out_of_place)
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride * offset * buf_size_multiplier, radix,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                1 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            populate_stride_array_wrapper(pc_stride.in_strides,
                in_stride * buf_size_multiplier, radix,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                in_stride * offset * buf_size_multiplier, radix,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            if (is_fused)
            {
                prepare_fused_kernel_strides_wrapper(k_stride.in_strides,
                    radix, in_stride_w_ds);
                prepare_fused_kernel_strides_wrapper(
                    pc_stride.in_strides, radix, in_stride_w_ds);
                prepare_fused_kernel_strides_wrapper(
                    pc_stride.out_strides, radix, in_stride_w_ds);
            }
            k_stride.v_in_stride = in_stride_w_ds * buf_size_multiplier;
            k_stride.v_out_stride = out_stride_w_ds * radix
                                    * buf_size_multiplier;
            pc_stride.v_in_stride = in_stride_w_ds * radix
                                    * buf_size_multiplier;
            pc_stride.v_out_stride = in_stride_w_ds * buf_size_multiplier;
        }
        else
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                1 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            k_stride.v_in_stride   = in_stride_w_ds * radix
                                     * buf_size_multiplier;
            k_stride.v_out_stride  = out_stride_w_ds * radix
                                     * buf_size_multiplier;
            pc_stride.v_in_stride  = out_stride_w_ds * radix
                                     * buf_size_multiplier;
            pc_stride.v_out_stride = out_stride_w_ds * radix
                                     * buf_size_multiplier;
        }

        // Perform circular right shift by `m` times
        // range of m => [1, radix)
        INTP m = (INTP)(rand() % (radix - 1)) + 1;
        if (is_fused)
        {
            memset(in1_shifted_dft, 0, input_length * sizeof(T));
            memset(in1_standard_dft, 0, input_length * sizeof(T));
            split_r2hcf_data(input_type, in1, in1_standard_dft,
                in1_shifted_dft, 0 /* is_half_complex */,
                k_stride.in_strides, k_stride.v_in_stride);

            // shift for standard_dft inputs
            for (INTP itr = 0; itr < offset; itr++)
            {
                for (INTP idx = 0; idx < radix; idx++)
                {
                    for (INTP is = 0; is < in_stride * buf_size_multiplier;
                         is++)
                    {
                        INTP src = (itr * radix * in_stride
                                    * buf_size_multiplier
                                    + ((idx + (radix - m)) * in_stride
                                    * buf_size_multiplier)
                                    % (radix * in_stride
                                    * buf_size_multiplier) + is);
                        INTP dst = ((itr * radix + idx) * in_stride
                                    * buf_size_multiplier + is);

                        // shifting standard_dft inputs
                        in2_standard_dft[dst] = in1_standard_dft[src];
                        // shifting shifted_dft inputs
                        in2_shifted_data[dst] = in1_shifted_dft[src];
                    }
                }
            }
            combine_data_for_r2hcf(input_type, in2, in2_standard_dft,
                in2_shifted_data, 0 /*is_half_complex*/, k_stride.in_strides,
                k_stride.v_in_stride);
        }
        else
        {
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
        }

        // convert the inputs from in-order to out-of-order
        permuted_copy_fused<T>(in1, temp, offset, radix,
            &pc_stride, data_stride, buf_size_multiplier);
        memcpy(in1, temp, sizeof(T) * input_length * data_stride);
        permuted_copy_fused<T>(in2, temp, offset, radix,
            &pc_stride, data_stride, buf_size_multiplier);
        memcpy(in2, temp, sizeof(T) * input_length * data_stride);

        fft_kernel(in1, in1, out1, out1, offset, &k_stride, is_bwd);
        fft_kernel(in2, in2, out2, out2, offset, &k_stride, is_bwd);

        if (is_fused)
        {
            split_r2hcf_data(input_type, out1, out1_standard_dft,
                out1_shifted_dft, 1/* is_half_complex */,
                k_stride.out_strides, k_stride.v_out_stride);
            split_r2hcf_data(input_type, out2, out2_standard_dft,
                out2_shifted_dft, 1/* is_half_complex */,
                k_stride.out_strides, k_stride.v_out_stride);
        }
        else
        {
            memcpy(out1_standard_dft, out1, output_length * sizeof(T));
            memcpy(out2_standard_dft, out2, output_length * sizeof(T));
        }

        // complex variables
        T cmul_temp[2] = {0.0, 0.0};
        T e_k[2]       = {1.0, 0.0};
        T sign         = is_bwd ? 1.0 : -1.0;

        T *out1_standard_dft_full = NULL;
        ALLOC_ALIGN_INIT(out1_standard_dft_full, T,
            2 * output_length * sizeof(T));
        T *out2_standard_dft_full = NULL;
        ALLOC_ALIGN_INIT(out2_standard_dft_full, T,
            2 * output_length * sizeof(T));
        convert_halfcomplex_to_fullcomplex(out1_standard_dft_full,
            out1_standard_dft, radix * buf_size_multiplier,
            offset, out_stride);

        convert_halfcomplex_to_fullcomplex(out2_standard_dft_full,
            out2_standard_dft, radix * buf_size_multiplier,
            offset, out_stride);

        // Euler Calculations
        for (INTP k = 0; k < radix * offset; k++)
        {
            INTP mk = (m * k) % radix;
            T angle = sign * AOCLFFTZ_2_PI * mk / radix;
            EULER(angle, e_k);
            CMUL(out1_standard_dft_full + k * out_stride * 2
                 * buf_size_multiplier, e_k,
                 out1_standard_dft_full + k * out_stride * 2
                 * buf_size_multiplier, cmul_temp);
        }

        if (!use_special)
        {
            EXPECT_LE(get_error(out1_standard_dft_full,
                out2_standard_dft_full, output_length * 2), tolerance)
                << "Property: timeshift, seed: " << random_seed << "\n";
        }
        FREE_ALIGN_ALLOCATED_MEM(out1_standard_dft_full);
        FREE_ALIGN_ALLOCATED_MEM(out2_standard_dft_full);
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

        if (is_out_of_place)
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride, radix * buf_size_multiplier,
                1 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride * offset * buf_size_multiplier, radix,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride * offset * buf_size_multiplier, radix,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            if (is_fused)
            {
                prepare_fused_kernel_strides_wrapper(pc_stride.in_strides,
                                             radix, in_stride_w_ds);
                prepare_fused_kernel_strides_wrapper(k_stride.out_strides,
                                            radix, out_stride_w_ds);
            }

            k_stride.v_in_stride   = in_stride_w_ds * radix
                                     * buf_size_multiplier;
            k_stride.v_out_stride  = out_stride_w_ds * buf_size_multiplier;
            pc_stride.v_in_stride  = out_stride_w_ds * buf_size_multiplier;
            pc_stride.v_out_stride = out_stride_w_ds * radix
                                     * buf_size_multiplier;
        }
        else
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride, radix * buf_size_multiplier,
                1 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride, radix * buf_size_multiplier,
                0 /* compute_half_Complex */,
                0 /* adjust_to_full_complex */);

            k_stride.v_in_stride   = in_stride_w_ds * radix
                                     * buf_size_multiplier;
            k_stride.v_out_stride  = out_stride_w_ds * radix
                                     * buf_size_multiplier;
            pc_stride.v_in_stride  = out_stride_w_ds * radix
                                     * buf_size_multiplier;
            pc_stride.v_out_stride = out_stride_w_ds * radix
                                     * buf_size_multiplier;
        }

        INTP m = (INTP)(rand() % (radix - 1)) + 1;
        if (is_fused)
        {
            memset(in1_standard_dft, 0, input_length * sizeof(T));
            memset(in1_shifted_dft, 0, input_length * sizeof(T));
            memset(in2_standard_dft, 0, input_length * sizeof(T));
            memset(in2_shifted_data, 0, input_length * sizeof(T));
            split_r2hcf_data(input_type, in1, in1_standard_dft,
                in1_shifted_dft, 1/*is_half_complex*/, k_stride.in_strides,
                k_stride.v_in_stride);

            T *in1_standard_dft_full = NULL;
            ALLOC_ALIGN_INIT(in1_standard_dft_full, T,
                2 * input_length * sizeof(T));
            convert_halfcomplex_to_fullcomplex(in1_standard_dft_full,
                in1_standard_dft, radix * buf_size_multiplier, offset,
                in_stride);

            T *in1_shifted_dft_full = NULL;
            ALLOC_ALIGN_INIT(in1_shifted_dft_full, T,
                2 * input_length * sizeof(T));
            convert_halfcomplex_to_fullcomplex(in1_shifted_dft_full,
                in1_shifted_dft, radix * buf_size_multiplier, offset,
                in_stride);

            for (INTP k = 0; k < radix * offset; k++)
            {
                INTP mk = (m * k) % radix;
                T angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(in1_standard_dft_full + k * in_stride * 2
                     * buf_size_multiplier, e_k,
                     in1_standard_dft_full + k * in_stride * 2
                     * buf_size_multiplier, cmul_temp);
            }

            for (INTP k = 0; k < radix * offset; k++)
            {
                INTP mk = (m * k) % radix;
                T angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(in1_shifted_dft_full + k * in_stride * 2
                     * buf_size_multiplier, e_k,
                     in1_shifted_dft_full + k * in_stride * 2
                     * buf_size_multiplier, cmul_temp);
            }

            convert_fullcomplex_to_halfcomplex(in2_standard_dft,
                in1_standard_dft_full, radix * buf_size_multiplier, offset,
                in_stride);
            convert_fullcomplex_to_halfcomplex(in2_shifted_data,
                in1_shifted_dft_full, radix * buf_size_multiplier, offset,
                in_stride);
            combine_data_for_r2hcf(input_type, in2, in2_standard_dft,
                in2_shifted_data, 1/*is_half_complex*/, k_stride.in_strides,
                k_stride.v_in_stride);

            FREE_ALIGN_ALLOCATED_MEM(in1_standard_dft_full);
            FREE_ALIGN_ALLOCATED_MEM(in1_shifted_dft_full);
        }
        else
        {
            T *in1_full = NULL;
            ALLOC_ALIGN_INIT(in1_full, T, 2 * input_length * sizeof(T));
            convert_halfcomplex_to_fullcomplex(in1_full, in1,
                radix * buf_size_multiplier, offset, in_stride);
            for (INTP k = 0; k < length; k++)
            {
                INTP mk = (m * k) % radix;
                T angle = sign * AOCLFFTZ_2_PI * mk / radix;
                EULER(angle, e_k);
                CMUL(in1_full + k * in_stride * 2, e_k,
                    in1_full + k * in_stride * 2, cmul_temp);
            }

            memset(in2, 0, input_length * sizeof(T));
            convert_fullcomplex_to_halfcomplex(in2, in1_full,
                radix * buf_size_multiplier, offset, in_stride);

            FREE_ALIGN_ALLOCATED_MEM(in1_full);
        }


        // convert the inputs from in-order to out-of-order
        fft_kernel(in1, in1, out1, out1, offset, &k_stride,
                   is_bwd);
        fft_kernel(in2, in2, out2, out2, offset, &k_stride,
                   is_bwd);
        // convert the outputs from out-of-order to in-order
        permuted_copy<T>(out1, temp, offset, radix,
                         &pc_stride, data_stride);
        memcpy(out1, temp, sizeof(T) * (output_length / buf_size_multiplier)
               * data_stride);
        permuted_copy<T>(out2, temp, offset, radix, &pc_stride,
                         data_stride);
        memcpy(out2, temp, sizeof(T) * (output_length / buf_size_multiplier)
               * data_stride);


        // Perform circular right shift by `m` times
        // range of m => [1, radix)
        T *out2_temp = NULL;
        ALLOC_ALIGN_INIT(out2_temp, T, output_length * sizeof(T));
        memcpy(out2_temp, out2_standard_dft, output_length * sizeof(T));
        for (INTP itr = 0; itr < offset; itr++)
        {
            for (INTP idx = 0; idx < radix; idx++)
            {
                for (INTP is = 0; is < out_stride * buf_size_multiplier;
                     is++)
                {
                    INTP src  = (itr * radix * out_stride
                                * buf_size_multiplier
                                + ((idx + (radix - m)) * out_stride
                                * buf_size_multiplier) % (radix * out_stride
                                * buf_size_multiplier) + is);
                    INTP dst  = ((itr * radix + idx) * out_stride
                                * buf_size_multiplier + is);
                    out2_standard_dft[dst] = out2_temp[src];
                }
            }
        }

        if (!use_special)
        {
            EXPECT_LE(get_error(out1_standard_dft, out2_standard_dft,
                output_length), tolerance)
                << "Property: timeshift, seed: " << random_seed << "\n";
        }

        FREE_ALIGN_ALLOCATED_MEM(out2_temp);
    }

    if (is_out_of_place)
    {
        FREE_ALIGN_ALLOCATED_MEM(out1);
        FREE_ALIGN_ALLOCATED_MEM(out2);
    }
    FREE_ALIGN_ALLOCATED_MEM(in1);
    FREE_ALIGN_ALLOCATED_MEM(in1_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in1_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(in2);
    FREE_ALIGN_ALLOCATED_MEM(in2_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in2_shifted_data);
    FREE_ALIGN_ALLOCATED_MEM(out1_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out1_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(out2_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out2_shifted_dft);
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
template <class T>
VOID AoclfftzKernelTestBase<T>::run_dft_reference_test_real(
    aocl_fftz_test_input input_type)
{
    // k_in_size : size of input buffer for kernel(half/full complex form)
    INTP k_in_size = radix * in_stride * data_stride * buf_size_multiplier
                     * offset;

    // k_out_size : size of output buffer for kernel(half/full complex form)
    INTP k_out_size = radix * out_stride * data_stride * buf_size_multiplier
                      * offset;

    /* fc_in_size : size of input buffer for full complex form
     * For real    -> fc_in_size == k_in_size * 2
     */
    INTP fc_in_size = radix * in_stride * offset * buf_size_multiplier * 2;

    /* fc_out_size : size of output buffer for full complex form
     * For real    -> fc_out_size == k_out_size * 2
     */
    INTP fc_out_size = radix * out_stride * offset * buf_size_multiplier * 2;

    T *in = prepare_input(input_type);
    if (in == nullptr)
    {
        return;
    }
    bool is_fused = is_fused_kernel(kernel_type);

    T *in_full = NULL;
    ALLOC_ALIGN_INIT(in_full, T, fc_in_size * sizeof(T));
    T *in_full_standard_dft = NULL;
    ALLOC_ALIGN_INIT(in_full_standard_dft, T, fc_in_size * sizeof(T));
    T *in_full_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(in_full_shifted_dft, T,
                    (fc_in_size + in_stride * 2) * sizeof(T));

    T *out = NULL;
    ALLOC_ALIGN_INIT(out, T, k_out_size * sizeof(T));
    T *out_full = NULL;
    ALLOC_ALIGN_INIT(out_full, T, fc_out_size * sizeof(T));
    T *out_full_standard_dft = NULL;
    ALLOC_ALIGN_INIT(out_full_standard_dft, T, fc_out_size * sizeof(T));
    T *out_full_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(out_full_shifted_dft, T,
                    (fc_out_size + out_stride * 2) * sizeof(T));

    T *in_standard_dft = NULL;
    ALLOC_ALIGN_INIT(in_standard_dft, T, k_in_size * sizeof(T));
    T *in_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(in_shifted_dft, T, k_in_size * sizeof(T));
    T *out_standard_dft = NULL;
    ALLOC_ALIGN_INIT(out_standard_dft, T, k_out_size * sizeof(T));
    T *out_shifted_dft = NULL;
    ALLOC_ALIGN_INIT(out_shifted_dft, T, k_out_size * sizeof(T));

    // prepare local strides
    aoclfftz_strides_t kernel_stride;

    // strides for FFT kernel
    ALLOC_ALIGN_UNINIT(kernel_stride.in_strides, INTP,
                       buf_size_multiplier * radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(kernel_stride.out_strides, INTP,
                       buf_size_multiplier * radix * sizeof(INTP));

    kernel_stride.v_in_stride  = in_stride_w_ds * buf_size_multiplier * radix;
    kernel_stride.v_out_stride = out_stride_w_ds * buf_size_multiplier * radix;

    // populating strides and executing FFT for real kernels
    populate_stride_array_wrapper(kernel_stride.in_strides,
        in_stride_w_ds, buf_size_multiplier * radix, is_bwd, 0);
    populate_stride_array_wrapper(kernel_stride.out_strides,
        out_stride_w_ds, buf_size_multiplier * radix, !is_bwd, 0);
    fft_kernel(in, in, out, out, offset, &kernel_stride, is_bwd);

    // for real problems,
    // forward:
    //   input  : real to full-complex
    //   output : half-complex to full-complex
    // backward:
    //   input  : half-complex to full-complex
    //   output : real to full-complex

    // Full input --> in_full
    convert_to_fullcomplex(input_type, in, in_full, in_stride,
                    radix * buf_size_multiplier, is_bwd, k_in_size);
    // Full output --> out_full
    convert_to_fullcomplex(input_type, out, out_full, out_stride,
                    radix * buf_size_multiplier, !is_bwd, k_out_size);

    if (is_fused)
    {
        // split in and out for standard DFT and shifted DFT
        split_r2hcf_data(input_type, in, in_standard_dft,
            in_shifted_dft, is_bwd, kernel_stride.in_strides,
            kernel_stride.v_in_stride);

        split_r2hcf_data(input_type, out, out_standard_dft,
            out_shifted_dft, !is_bwd, kernel_stride.out_strides,
            kernel_stride.v_out_stride);

        // re-arrange the standard DFT input to covert into full complex format
        convert_to_fullcomplex(input_type, in_standard_dft,
            in_full_standard_dft, in_stride, radix * 2, is_bwd,
            k_in_size);

        // re-arrange the standard DFT output to covert into full complex format
        convert_to_fullcomplex(input_type, out_standard_dft,
            out_full_standard_dft, out_stride, radix * 2, !is_bwd,
            k_out_size);

        // re-arrange the shifted DFT input to covert into full complex format
        convert_to_fullcomplex(input_type, in_shifted_dft,
            in_full_shifted_dft, in_stride, radix * 2, is_bwd,
            k_in_size);

        // re-arrange the shifted DFT output to covert into full complex format
        convert_to_fullcomplex(input_type, out_shifted_dft,
            out_full_shifted_dft, out_stride, radix * 2, !is_bwd,
            k_out_size);
    }

    if (is_fused)
    {
        calculate_dft(input_type, kernel_stride, in_full_standard_dft,
                out_full_standard_dft, fc_out_size, true /* is_standard_dft */);
        calculate_dft(input_type, kernel_stride,
                in_full_shifted_dft + in_stride * 2,
                out_full_shifted_dft + out_stride * 2, fc_out_size,
                false /* is_standard_dft */);
    }
    else{
        calculate_dft(input_type, kernel_stride, in_full,
                out_full, fc_out_size, true /* is_standard_dft */);
    }

    FREE_ALIGN_ALLOCATED_MEM(in);
    FREE_ALIGN_ALLOCATED_MEM(in_full);
    FREE_ALIGN_ALLOCATED_MEM(out);
    FREE_ALIGN_ALLOCATED_MEM(out_full);
    FREE_ALIGN_ALLOCATED_MEM(in_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_full_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_full_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_full_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_full_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.in_strides);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.out_strides);
} // run_dft_reference_test

#endif // AOCL_REAL_KERNEL_GTEST_H
