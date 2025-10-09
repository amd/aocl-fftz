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

/** @file dft_reference.h
 *
 * @brief DFT reference test implementation for complex and real FFT kernels.
 *
 * This file contains the DFT reference test function implementation
 * for validating complex and real FFT kernel outputs against DFT calculations
 * using gtest.
 *
 * @author Partiksha
 * @author Jeevanantham N
 *
 */

#ifndef AOCLFFTZ_DFT_REFERENCE_H
#define AOCLFFTZ_DFT_REFERENCE_H

#include "test/gtest/kernel/kernel_gtest_base.h"

/**
 * @brief A function to check the accuracy of Complex FFT kernel
 * against DFT reference
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

    T    *out      = NULL;
    VOID *twid     = NULL; // For twiddle kernels, this need to be updated with pre-computed twiddle values
    
    aoclfftz_strides_t kernel_stride;
    kernel_stride.in_strides  = NULL;
    kernel_stride.out_strides = NULL;
    
    T *in_r  = NULL;
    T *in_i  = NULL;
    T *out_r = NULL;
    T *out_i = NULL;
    
    T *in = prepare_input(input_type);
    if (in == nullptr)
    {
        GTEST_FATAL_FAILURE_("Input preparation failed in complex "
                             "dft reference test");
        goto cleanup_complex;
    }

    ALLOC_ALIGN_INIT(out, T, k_out_size * sizeof(T));

    if (out == nullptr)
    {
        GTEST_FATAL_FAILURE_("Memory allocation failed in complex "
                             "dft reference test");
        goto cleanup_complex;
    }

    // prepare local strides for FFT kernel
    ALLOC_ALIGN_UNINIT(kernel_stride.in_strides, INTP, radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(kernel_stride.out_strides, INTP, radix * sizeof(INTP));

    if (kernel_stride.in_strides == nullptr ||
        kernel_stride.out_strides == nullptr)
    {
        GTEST_FATAL_FAILURE_("Memory allocation failed in complex "
                             "dft reference test");
        goto cleanup_complex;
    }

    kernel_stride.v_in_stride  = in_stride_w_ds * radix;
    kernel_stride.v_out_stride = out_stride_w_ds * radix;

    // populating strides and executing FFT for complex kernels
    populate_stride_array_wrapper(kernel_stride.in_strides,
        in_stride_w_ds, radix, 0, 0);
    populate_stride_array_wrapper(kernel_stride.out_strides,
        out_stride_w_ds, radix, 0, 0);
    
    in_r  = in;
    in_i  = in + 1;
    out_r = out;
    out_i = out + 1;
    fft_kernel(in_r, in_i, out_r, out_i, offset, &kernel_stride, twid, is_bwd);

    calculate_dft(input_type, kernel_stride, in,
            out, fc_out_size, true /* is_standard_dft */);

cleanup_complex:
    FREE_ALIGN_ALLOCATED_MEM(in);
    FREE_ALIGN_ALLOCATED_MEM(out);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.in_strides);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.out_strides);
} // run_dft_reference_test


/**
 * @brief A function to check the accuracy of Real FFT kernel
 * against DFT reference
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

    T    *in                      = NULL;
    T    *in_full                 = NULL;
    T    *in_full_standard_dft    = NULL;
    T    *in_full_shifted_dft     = NULL;
    T    *out                     = NULL;
    T    *out_full                = NULL;
    T    *out_full_standard_dft   = NULL;
    T    *out_full_shifted_dft    = NULL;
    T    *in_standard_dft         = NULL;
    T    *in_shifted_dft          = NULL;
    T    *out_standard_dft        = NULL;
    T    *out_shifted_dft         = NULL;
    VOID *twid                    = NULL; // For twiddle kernels, this need to be updated with pre-computed twiddle values

    aoclfftz_strides_t kernel_stride;
    kernel_stride.in_strides  = NULL;
    kernel_stride.out_strides = NULL;

    bool is_fused = is_fused_kernel(kernel_type);

    in = prepare_input(input_type);
    if (in == nullptr)
    {
        GTEST_FATAL_FAILURE_("Input preparation failed in real "
                             "dft reference test");
        goto cleanup_real;
    }

    ALLOC_ALIGN_INIT(in_full, T, fc_in_size * sizeof(T));
    ALLOC_ALIGN_INIT(in_full_standard_dft, T, fc_in_size * sizeof(T));
    ALLOC_ALIGN_INIT(in_full_shifted_dft, T,
                     (fc_in_size + in_stride * 2) * sizeof(T));
    ALLOC_ALIGN_INIT(out, T, k_out_size * sizeof(T));
    ALLOC_ALIGN_INIT(out_full, T, fc_out_size * sizeof(T));
    ALLOC_ALIGN_INIT(out_full_standard_dft, T, fc_out_size * sizeof(T));
    ALLOC_ALIGN_INIT(out_full_shifted_dft, T,
                     (fc_out_size + out_stride * 2) * sizeof(T));
    ALLOC_ALIGN_INIT(in_standard_dft, T, k_in_size * sizeof(T));
    ALLOC_ALIGN_INIT(in_shifted_dft, T, k_in_size * sizeof(T));
    ALLOC_ALIGN_INIT(out_standard_dft, T, k_out_size * sizeof(T));
    ALLOC_ALIGN_INIT(out_shifted_dft, T, k_out_size * sizeof(T));

    if (in_full == nullptr || in_full_standard_dft == nullptr ||
        in_full_shifted_dft == nullptr || out == nullptr ||
        out_full == nullptr || out_full_standard_dft == nullptr ||
        out_full_shifted_dft == nullptr || in_standard_dft == nullptr ||
        in_shifted_dft == nullptr || out_standard_dft == nullptr ||
        out_shifted_dft == nullptr)
    {
        GTEST_FATAL_FAILURE_("Memory allocation failed in real "
                             "dft reference test");
        goto cleanup_real;
    }

    // prepare local strides for FFT kernel
    ALLOC_ALIGN_UNINIT(kernel_stride.in_strides, INTP,
                       buf_size_multiplier * radix * sizeof(INTP));
    ALLOC_ALIGN_UNINIT(kernel_stride.out_strides, INTP,
                       buf_size_multiplier * radix * sizeof(INTP));

    if (kernel_stride.in_strides == nullptr ||
        kernel_stride.out_strides == nullptr)
    {
        GTEST_FATAL_FAILURE_("Memory allocation failed in real "
                             "dft reference test");
        goto cleanup_real;
    }

    kernel_stride.v_in_stride  = in_stride_w_ds * buf_size_multiplier * radix;
    kernel_stride.v_out_stride = out_stride_w_ds * buf_size_multiplier * radix;

    // populating strides and executing FFT for real kernels
    populate_stride_array_wrapper(kernel_stride.in_strides,
        in_stride_w_ds, buf_size_multiplier * radix, is_bwd, 0);
    populate_stride_array_wrapper(kernel_stride.out_strides,
        out_stride_w_ds, buf_size_multiplier * radix, !is_bwd, 0);

    fft_kernel(in, in, out, out, offset, &kernel_stride, twid, is_bwd);

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
    else
    {
        calculate_dft(input_type, kernel_stride, in_full,
            out_full, fc_out_size, true /* is_standard_dft */);
    }

cleanup_real:
    FREE_ALIGN_ALLOCATED_MEM(in);
    FREE_ALIGN_ALLOCATED_MEM(in_full);
    FREE_ALIGN_ALLOCATED_MEM(in_full_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_full_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(out);
    FREE_ALIGN_ALLOCATED_MEM(out_full);
    FREE_ALIGN_ALLOCATED_MEM(out_full_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_full_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(in_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_standard_dft);
    FREE_ALIGN_ALLOCATED_MEM(out_shifted_dft);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.in_strides);
    FREE_ALIGN_ALLOCATED_MEM(kernel_stride.out_strides);
} // run_dft_reference_test

#endif // AOCLFFTZ_DFT_REFERENCE_H
