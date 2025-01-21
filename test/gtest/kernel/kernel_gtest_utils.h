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

/** @file kernel_gtest_utils.h
 *
 * @brief Utility functions for GTest kernel tests.
 *
 * This file contains the utility functions used to add and run gtests
 * for FFT kernels.
 *
 * @author Srirammaswamy Srinivasan
 *
 */

#ifndef AOCLFFTZ_KERNEL_GTEST_UTILS_H
#define AOCLFFTZ_KERNEL_GTEST_UTILS_H

#include <string>
#include <typeinfo>
extern "C"
{
#include "test/gtest/aoclfftz_core_wrapper.h"
}
#include "test/gtest/gtest_types.h"
#include "utils/complex_utils.h"

#define TOLERANCE_F 1E-3
#define TOLERANCE_D 1E-10

#ifndef DBL_TRUE_MIN
#define DBL_TRUE_MIN 4.9406564584124654e-324
#endif
#ifndef FLT_TRUE_MIN
#define FLT_TRUE_MIN 1.40129846e-45F
#endif
/**
 * @brief Get the kernel object from the kernel table based on the given radix
 *
 * @param kernel_table wrapper_kernel_fp_list to search the kernel
 * @param dir direction (forward / backward)
 * @param radix radix of the FFT kernel
 * @return kffft_ a kernel pointer; returns nullptr if kernel not found
 */
template <class T>
kfft_ get_kernel(const wrapper_kernel_fp_list *kernel_table, const INT32 dir,
                 const UINT32 radix)
{
    while (kernel_table->k_register_kernel != nullptr)
    {
        if (kernel_table->radix == radix)
        {
            if (typeid(float) == typeid(T))
            {
                kfft_ kernel = kernel_table->k_register_kernel(DT_FLOAT, dir);
                return kernel;
            }
            else
            {
                kfft_ kernel = kernel_table->k_register_kernel(DT_DOUBLE, dir);
                return kernel;
            }
        }
        kernel_table++;
    }
    return nullptr;
}

/**
 * @brief Get the global kernel table vector based on the kernel type
 *
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @return wrapper_kernel_fp_list
 */
wrapper_kernel_fp_list *get_kernel_table(UINT8 kernel_type)
{
    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::STANDARD_C2C_C:
    case aocl_fftz_kernel_type::PERMUTED_C2C_C:
        return wrapper_kernels_c2c_c;
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX128:
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX128:
        return wrapper_kernels_c2c_avx128;
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX256:
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX256:
        return wrapper_kernels_c2c_avx256;
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX512:
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX512:
        return wrapper_kernels_c2c_avx512;
#endif
    case aocl_fftz_kernel_type::STANDARD_R2HC_C:
    case aocl_fftz_kernel_type::PERMUTED_R2HC_C:
        return wrapper_kernels_r2hc_c;
    case aocl_fftz_kernel_type::STANDARD_R2HCF_C:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_C:
        return wrapper_kernels_r2hcf_c;
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX128:
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128:
        return wrapper_kernels_r2hc_avx128;
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX128:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX128:
        return wrapper_kernels_r2hcf_avx128;
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX256:
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256:
        return wrapper_kernels_r2hc_avx256;
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX256:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX256:
        return wrapper_kernels_r2hcf_avx256;
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX512:
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512:
        return wrapper_kernels_r2hc_avx512;
#endif
    default:
        return {};
    }
}

/**
 * @brief Get the kernel type as a string (used for naming the the test case)
 *
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @return std::string name of the kernel type
 */
std::string get_kernel_type_as_string(UINT8 kernel_type)
{
    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::STANDARD_C2C_C:
        return "_STANDARD_C2C_C";
    case aocl_fftz_kernel_type::PERMUTED_C2C_C:
        return "_PERMUTED_C2C_C";
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX128:
        return "_STANDARD_C2C_AVX128";
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX128:
        return "_PERMUTED_C2C_AVX128";
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX256:
        return "_STANDARD_C2C_AVX256";
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX256:
        return "_PERMUTED_C2C_AVX256";
    case aocl_fftz_kernel_type::STANDARD_C2C_AVX512:
        return "_STANDARD_C2C_AVX512";
    case aocl_fftz_kernel_type::PERMUTED_C2C_AVX512:
        return "_PERMUTED_C2C_AVX512";
    case aocl_fftz_kernel_type::STANDARD_R2HC_C:
        return "_STANDARD_R2HC_C";
    case aocl_fftz_kernel_type::PERMUTED_R2HC_C:
        return "_PERMUTED_R2HC_C";
    case aocl_fftz_kernel_type::STANDARD_R2HCF_C:
        return "_STANDARD_R2HCF_C";
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_C:
        return "_PERMUTED_R2HCF_C";
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX128:
        return "_STANDARD_R2HC_AVX128";
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128:
        return "_PERMUTED_R2HC_AVX128";
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX128:
        return "_STANDARD_R2HCF_AVX128";
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX128:
        return "_PERMUTED_R2HCF_AVX128";
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX256:
        return "_STANDARD_R2HC_AVX256";
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256:
        return "_PERMUTED_R2HC_AVX256";
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX256:
        return "_STANDARD_R2HCF_AVX256";
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX256:
        return "_PERMUTED_R2HCF_AVX256";
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::STANDARD_R2HC_AVX512:
        return "_STANDARD_R2HC_AVX2512";
    case aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512:
        return "_PERMUTED_R2HC_AVX2512";
#endif
    default:
        return "_UNKNOWN";
    }
}

/**
 * @brief Helper function to check whether the given kernel is of fused or not
 * @param kernel_type aocl_fftz_kernel_type kernel type
 *
 */
bool is_fused_kernel(UINT8 kernel_type)
{
    switch (kernel_type)
    {
    // TODO: add avx variants
    case aocl_fftz_kernel_type::STANDARD_R2HCF_C:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_C:
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX256:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX256:
    case aocl_fftz_kernel_type::STANDARD_R2HCF_AVX128:
    case aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX128:
        return true;
    default:
        return false;
    }
}

/**
 * @brief A template wrapper function for the permuted_copy kernel
 *        TODO: Remove this function, register permuted copy in a kernel table
 * and use it instead.
 *
 * @tparam T data type (float32 or float64)
 * @param in input data array
 * @param out output array to store the permuted output
 * @param n no. of sets (or) offset
 * @param size size of each offset (or) n
 * @param strides data stride values (in-stride for dir 0, out-stride for dir 1)
 */
template <class T>
void permuted_copy(T *in, T *out, INTP n, INTP size,
                   aoclfftz_strides_t *strides, UINT8 data_stride)
{
    if (typeid(T) == typeid(FLOAT32))
    {
        permuted_copy_c_fp32_wrapper((FLOAT32 *)in, (FLOAT32 *)out, n, size,
                                     strides, data_stride);
    }
    else if (typeid(T) == typeid(FLOAT64))
    {
        permuted_copy_c_fp64_wrapper((FLOAT64 *)in, (FLOAT64 *)out, n, size,
                                     strides, data_stride);
    }
}

/**
 * @brief This function similar to permuted_copy kernel with additional
 *        paramter `buf_size_multiplier` which is necessary to permute the
 *        fused part-2 data in case of fused kernels.
 *        TODO: Remove this function, modify the permuted_copy kernel
 *        and use that instead
 *
 * @tparam T data type (float32 or float64)
 * @param in input data array
 * @param out output array to store the permuted output
 * @param n no. of sets (or) offset
 * @param size size of each offset (or) n
 * @param strides data stride values (in-stride for dir 0, out-stride for dir 1)
 * @param data_stride data_stride -> 1 for real-fft and 2 for complex-fft
 * @param buf_size_multiplier buf_size_multiplier used here to permute fused
 *                            part-2 data.
 */
template <class T>
VOID permuted_copy_fused(T *in, T *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides, INT32 data_stride,
                          INT8 buf_size_multiplier)
{
    INTP in_stride = strides->in_strides[0] ?
             strides->in_strides[0] * data_stride : strides->in_strides[1];

    INTP in_stride_fused_part1 = strides->in_strides[0] ?
                                strides->in_strides[0] * data_stride :
                                strides->in_strides[buf_size_multiplier];
    INTP out_stride_fused_part1 = strides->out_strides[0] ?
                                strides->out_strides[0] * data_stride :
                                strides->out_strides[buf_size_multiplier];

    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    // iterates over the number of offsets (n)
    for (INTP i = 0; i < n; i++)
    {
        // iterates over the size of each offset (size)
        for (INTP j = 0; j < size; j++)
        {
            for (INTP k = 0; k < data_stride; k++)
            {
                out[j * out_stride_fused_part1 + k] = in[j *
                                        in_stride_fused_part1 + k];

                if (buf_size_multiplier == 2) // for fused kernel
                {
                    out[j * out_stride_fused_part1 + k + in_stride] = in[j *
                                        in_stride_fused_part1 + k + in_stride];
                }
            }
        }

        in += v_in_stride;
        out += v_out_stride;
    }
}

/**
 * @brief An utility function to print the complex array
 *
 * @tparam T array type (float32 or float64)
 * @param arr complex array
 * @param n size of the array (actual size = size * 2 [real + complex])
 */
template <class T> inline void print_carray(T *arr, INTP n)
{
    if (typeid(T) == typeid(FLOAT32))
    {
        PRINT_CARRAY_FP32((FLOAT32 *)arr, n);
    }
    else if (typeid(T) == typeid(FLOAT64))
    {
        PRINT_CARRAY_FP64((FLOAT64 *)arr, n);
    }
}

/**
 * @brief An utility function to print the complex value
 *
 * @tparam T data type (float32 or float64)
 * @param val complex value (array)
 */
template <class T> inline void print_complex(T *val)
{
    if (typeid(T) == typeid(FLOAT32))
    {
        PRINT_COMPLEX_FP32((FLOAT32 *)val);
    }
    else if (typeid(T) == typeid(FLOAT64))
    {
        PRINT_COMPLEX_FP64((FLOAT64 *)val);
    }
}

/**
 * @brief Get the special values like NaN, infinity, negative infinity
 * floating point min/max/subnormal-min values in positive and negative range.
 * This function may generate the above special values based on the given
 * `value` parameter. value mod 20 is used instead of 10 to limit these
 * special values to 50% propability. Normal random numbers will be in range
 * [-10.0, 10.0) with 3 decimal precision.
 *
 * @tparam T data type (float32 or float64)
 * @param value input value to pick the type of value
 * @return T return the normal or special value
 */
template <class T> T get_maybe_special_value(INT32 value)
{
    if (typeid(T) == typeid(FLOAT64))
    {
        switch (value % 20)
        {
        case 0:
            return 0.0L;
        case 1:
            return DBL_TRUE_MIN;
        case 2:
            return -DBL_TRUE_MIN;
        case 3:
            return DBL_MIN;
        case 4:
            return -DBL_MIN;
        case 5:
            return DBL_MAX;
        case 6:
            return -DBL_MAX;
        case 7:
            return INFINITY;
        case 8:
            return -INFINITY;
        case 9:
            return NAN;
        default:
            return ((rand() % 2000) / 200.0) - 10.0;
        }
    }
    else if (typeid(T) == typeid(FLOAT32))
    {
        switch (value % 20)
        {
        case 0:
            return 0.0F;
        case 1:
            return FLT_TRUE_MIN;
        case 2:
            return -FLT_TRUE_MIN;
        case 3:
            return FLT_MIN;
        case 4:
            return -FLT_MIN;
        case 5:
            return FLT_MAX;
        case 6:
            return -FLT_MAX;
        case 7:
            return INFINITY;
        case 8:
            return -INFINITY;
        case 9:
            return NAN;
        default:
            return ((rand() % 2000) / 200.0) - 10.0;
        }
    }
    return 0.0;
}

#endif // AOCLFFTZ_KERNEL_GTEST_UTILS_H
