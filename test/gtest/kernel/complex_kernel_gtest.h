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
 * @brief file for DFT property function's definition(complex type) for gtest.
 *
 * This file contains the definitions of member functions of DFT properties
 * to be used for running complex kernel unit tests using GTest.
 *
 * @author Partiksha
 *
 */

#ifndef AOCLFFTZ_COMPLEX_KERNEL_GTEST_H
#define AOCLFFTZ_COMPLEX_KERNEL_GTEST_H

#include "test/gtest/kernel/kernel_gtest_base.h"

template <class T>
VOID AoclfftzKernelTestBase<T>::run_linearity_test_complex(
    aocl_fftz_test_input input_type)
{
    T *in1 = prepare_input(input_type);
    T *in2 = prepare_input(input_type);
    if (in1 == nullptr || in2 == nullptr)
    {
        return;
    }

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

    // prepare local strides for FFT kernel
    aoclfftz_strides_t k_stride;
    ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));

    // permuted kernel
    if (kernel_type & 0x1)
    {
        if (is_out_of_place)
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            k_stride.v_in_stride = radix * in_stride_w_ds;
            k_stride.v_out_stride = out_stride_w_ds;
        }
        else
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            k_stride.v_in_stride  = in_stride_w_ds;
            k_stride.v_out_stride = radix * out_stride_w_ds;
        }
        else
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            k_stride.v_in_stride = radix * in_stride_w_ds;
            k_stride.v_out_stride = k_stride.v_in_stride;
        }
    }

    // Initialize kernel input/output variables
    VOID *twid = NULL; // For twiddle kernels, this needd to be updated with pre-computed twiddle values
    T *in1_r  = in1;
    T *in1_i  = in1 + 1;
    T *out1_r = out1;
    T *out1_i = out1 + 1;
    T *in2_r  = in2;
    T *in2_i  = in2 + 1;
    T *out2_r = out2;
    T *out2_i = out2 + 1;

    T *in_combined_r  = in_combined;
    T *in_combined_i  = in_combined + 1;
    T *out_combined_r = out_combined;
    T *out_combined_i = out_combined + 1;

    fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, twid, is_bwd);
    fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, twid, is_bwd);
    fft_kernel(in_combined_r, in_combined_i, out_combined_r,
               out_combined_i, offset, &k_stride, twid, is_bwd);

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
} // run_linearity_test

/**
 * @brief A function to check the transformation property of the FFT kernel
 * using unit impulse signal
 *
 * @param input_type test input type
 */
template <class T>
VOID AoclfftzKernelTestBase<T>::run_unit_impulse_transform_test_complex(
    aocl_fftz_test_input input_type)
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
    ALLOC_ALIGN_UNINIT(k_stride_rev.out_strides, INTP, radix * sizeof(INTP));

    aoclfftz_strides_t pc_stride;
    ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP, radix * sizeof(INTP));

    aoclfftz_strides_t pc_stride_rev;
    ALLOC_ALIGN_UNINIT(pc_stride_rev.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride_rev.out_strides, INTP, radix * sizeof(INTP));

    // permuted kernel
    if (kernel_type & 0x1)
    {
        if (is_out_of_place)
        {
            populate_stride_array_wrapper(k_stride.in_strides,
                in_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.in_strides,
                in_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.out_strides,
                in_stride_w_ds, radix,
                0 /* compute_half_complex */,
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
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.in_strides,
                in_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.out_strides,
                in_stride_w_ds * offset, radix,
                0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for reverse FFT kernel
            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on FFT output
            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on reverse FFT output
            populate_stride_array_wrapper(pc_stride_rev.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.out_strides,
                in_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for reverse FFT kernel
            populate_stride_array_wrapper(k_stride_rev.in_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride_rev.out_strides,
                in_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on FFT output
            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on reverse FFT output
            populate_stride_array_wrapper(pc_stride_rev.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride_rev.out_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
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
    VOID *twid = NULL; // For twiddle kernels, this needd to be updated with pre-computed twiddle values
    T *in_r  = in;
    T *in_i  = in + 1;
    T *out_r = out;
    T *out_i = out + 1;

    fft_kernel(in_r, in_i, out_r, out_i, offset, &k_stride, twid, is_bwd);
    // convert the FFT kernel output from in-order to out-of-order for
    // standard kernel and vise versa for permuted kernel
    permuted_copy<T>(out, perm_out, offset, radix,
                     &pc_stride, data_stride);

    T *perm_out_r     = perm_out;
    T *perm_out_i     = perm_out + 1;
    T *perm_inv_out_r = perm_inv_out;
    T *perm_inv_out_i = perm_inv_out + 1;

    fft_kernel(perm_out_r, perm_out_i, perm_inv_out_r, perm_inv_out_i,
               offset, &k_stride_rev, twid, !is_bwd);
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
VOID AoclfftzKernelTestBase<T>::run_timeshift_test_complex(
    aocl_fftz_test_input input_type)
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
    ALLOC_ALIGN_INIT(temp, T, (std::max)(input_length, output_length)
                     * data_stride * sizeof(T));

    // prepare local strides
    aoclfftz_strides_t k_stride;
    ALLOC_ALIGN_UNINIT(k_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(k_stride.out_strides, INTP, radix * sizeof(INTP));

    aoclfftz_strides_t pc_stride;
    ALLOC_ALIGN_UNINIT(pc_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(pc_stride.out_strides, INTP, radix * sizeof(INTP));

    /* Complex kernels validation (C2C) for forward/backward FFT by the
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

    // Perform circular right shift by `m` times range of m => [1, radix)
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
                            is) * data_stride;
                INTP dst = ((itr * radix + idx) * in_stride + is) * data_stride;
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            // strides for reverse FFT kernel
            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on FFT output
            populate_stride_array_wrapper(pc_stride.in_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on input
            populate_stride_array_wrapper(pc_stride.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                in_stride_w_ds * offset,
                radix, 0 /* compute_half_complex */,
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
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(k_stride.out_strides,
                out_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);

            // strides for permuted copy on input
            populate_stride_array_wrapper(pc_stride.in_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
                0 /* adjust_to_full_complex */);
            populate_stride_array_wrapper(pc_stride.out_strides,
                in_stride_w_ds,
                radix, 0 /* compute_half_complex */,
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
    VOID *twid = NULL; // For twiddle kernels, this needd to be updated with pre-computed twiddle values
    T *in1_r  = in1;
    T *in1_i  = in1 + 1;
    T *out1_r = out1;
    T *out1_i = out1 + 1;
    T *in2_r  = in2;
    T *in2_i  = in2 + 1;
    T *out2_r = out2;
    T *out2_i = out2 + 1;

    // permuted kernel
    if (kernel_type & 0x1)
    {
        fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, twid, is_bwd);
        fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, twid, is_bwd);

        // convert the FFT kernel outputs from out-of-order to in-order
        permuted_copy<T>(out1, temp, offset, radix, &pc_stride, data_stride);
        memcpy(out1, temp, sizeof(T) * output_length * data_stride);
        permuted_copy<T>(out2, temp, offset, radix, &pc_stride, data_stride);
        memcpy(out2, temp, sizeof(T) * output_length * data_stride);
    }
    // standard kernel
    else
    {
        // convert the inputs from in-order to out-of-order
        permuted_copy<T>(in1, temp, offset, radix, &pc_stride, data_stride);
        memcpy(in1, temp, sizeof(T) * input_length * data_stride);
        permuted_copy<T>(in2, temp, offset, radix, &pc_stride, data_stride);
        memcpy(in2, temp, sizeof(T) * input_length * data_stride);

        fft_kernel(in1_r, in1_i, out1_r, out1_i, offset, &k_stride, twid, is_bwd);
        fft_kernel(in2_r, in2_i, out2_r, out2_i, offset, &k_stride, twid, is_bwd);
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
template <class T>
VOID AoclfftzKernelTestBase<T>::run_dft_reference_test_complex(
    aocl_fftz_test_input input_type)
{
    // k_in_size : size of input buffer for kernel(half/full complex form)
    INTP k_in_size = radix * in_stride * data_stride * offset;

    // k_out_size : size of output buffer for kernel(half/full complex form)
    INTP k_out_size = radix * out_stride * data_stride * offset;

    /* fc_in_size : size of input buffer for full complex form
     * For complex -> fc_in_size == k_in_size
     */
    INTP fc_in_size = radix * in_stride * offset * 2;

    /* fc_out_size : size of output buffer for full complex form
     * For complex -> fc_out_size == k_out_size
     */
    INTP fc_out_size = radix * out_stride * offset * 2;

    T *in = prepare_input(input_type);
    if (in == nullptr)
    {
        return;
    }

    T *in_full = NULL;
    ALLOC_ALIGN_INIT(in_full, T, fc_in_size * sizeof(T));
    T *out = NULL;
    ALLOC_ALIGN_INIT(out, T, k_out_size * sizeof(T));
    T *out_full = NULL;
    ALLOC_ALIGN_INIT(out_full, T, fc_out_size * sizeof(T));

    // prepare local strides for FFT kernel
    VOID *twid = NULL; // For twiddle kernels, this needd to be updated with pre-computed twiddle values
    aoclfftz_strides_t kernel_stride;
    ALLOC_ALIGN_UNINIT(kernel_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(kernel_stride.out_strides, INTP, radix * sizeof(INTP));

    kernel_stride.v_in_stride  = in_stride_w_ds * radix;
    kernel_stride.v_out_stride = out_stride_w_ds * radix;

    // populating strides and executing FFT for complex kernels
    populate_stride_array_wrapper(kernel_stride.in_strides,
        in_stride_w_ds, radix, 0, 0);
    populate_stride_array_wrapper(kernel_stride.out_strides,
        out_stride_w_ds, radix, 0, 0);
    T *in_r  = in;
    T *in_i  = in + 1;
    T *out_r = out;
    T *out_i = out + 1;
    fft_kernel(in_r, in_i, out_r, out_i, offset, &kernel_stride, twid, is_bwd);

    memcpy(in_full, in, fc_in_size * sizeof(T));
    memcpy(out_full, out, fc_out_size * sizeof(T));
    calculate_dft(input_type, kernel_stride, in_full,
            out_full, fc_out_size, true /* is_standard_dft */);

    FREE_ALIGN_ALLOCATED_MEM(in);
    FREE_ALIGN_ALLOCATED_MEM(in_full);
    FREE_ALIGN_ALLOCATED_MEM(out);
    FREE_ALIGN_ALLOCATED_MEM(out_full);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.in_strides);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.out_strides);
} // run_dft_reference_test

#endif // AOCL_COMPLEX_GTEST_H
