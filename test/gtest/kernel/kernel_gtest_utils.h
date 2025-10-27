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

/** @file kernel_gtest_utils.h
 *
 * @brief Utility functions for GTest kernel tests.
 *
 * This file contains the utility functions used to add and run gtests
 * for FFT kernels.
 *
 * @author Srirammaswamy Srinivasan
 * @author Jeevanantham N
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
#include "test/utils/complex_utils.h"
#include "test/gtest/common_gtest_utils.h"

#define TOLERANCE_F 1E-3
#define TOLERANCE_D 1E-10

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
 * @brief Get the twiddle kernel pointer for the given radix
 *
 * @tparam T data type (float or double)
 * @param radix radix of the FFT kernel
 * @param dir direction (forward / backward)
 * @param kernel_type kernel type - C2C_TWID_ [C|AVX128|AVX256|AVX512]
 * @return kffft_ a kernel pointer; returns nullptr if kernel not found
 */
template <class T>
kfft_ get_twiddle_kernel(const UINT32 radix, const INT32 dir,
                            const UINT8 kernel_type)
{
    UINT8 prec;

    if (typeid(T) == typeid(float))
    {
        prec = DT_FLOAT;
    }
    else
    {
        prec = DT_DOUBLE;
    }

    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::C2C_TWID_C:
        switch (radix)
        {
        case 2:
            return register_kernel_twid_fft2c_wrapper(prec, dir);
        case 3:
            return register_kernel_twid_fft3c_wrapper(prec, dir);
        case 4:
            return register_kernel_twid_fft4c_wrapper(prec, dir);
        case 5:
            return register_kernel_twid_fft5c_wrapper(prec, dir);
        case 6:
            return register_kernel_twid_fft6c_wrapper(prec, dir);
        case 7:
            return register_kernel_twid_fft7c_wrapper(prec, dir);
        case 8:
            return register_kernel_twid_fft8c_wrapper(prec, dir);
        case 9:
            return register_kernel_twid_fft9c_wrapper(prec, dir);
        case 10:
            return register_kernel_twid_fft10c_wrapper(prec, dir);
        case 11:
            return register_kernel_twid_fft11c_wrapper(prec, dir);
        case 12:
            return register_kernel_twid_fft12c_wrapper(prec, dir);
        case 13:
            return register_kernel_twid_fft13c_wrapper(prec, dir);
        case 14:
            return register_kernel_twid_fft14c_wrapper(prec, dir);
        case 15:
            return register_kernel_twid_fft15c_wrapper(prec, dir);
        case 16:
            return register_kernel_twid_fft16c_wrapper(prec, dir);
        }
        break;

#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_AVX128:
        switch (radix)
        {
        case 2:
            return register_kernel_twid_fft2avx128_wrapper(prec, dir);
        case 3:
            return register_kernel_twid_fft3avx128_wrapper(prec, dir);
        case 4:
            return register_kernel_twid_fft4avx128_wrapper(prec, dir);
        case 5:
            return register_kernel_twid_fft5avx128_wrapper(prec, dir);
        case 6:
            return register_kernel_twid_fft6avx128_wrapper(prec, dir);
        case 7:
            return register_kernel_twid_fft7avx128_wrapper(prec, dir);
        case 8:
            return register_kernel_twid_fft8avx128_wrapper(prec, dir);
        case 9:
            return register_kernel_twid_fft9avx128_wrapper(prec, dir);
        case 10:
            return register_kernel_twid_fft10avx128_wrapper(prec, dir);
        case 11:
            return register_kernel_twid_fft11avx128_wrapper(prec, dir);
        case 12:
            return register_kernel_twid_fft12avx128_wrapper(prec, dir);
        case 13:
            return register_kernel_twid_fft13avx128_wrapper(prec, dir);
        case 14:
            return register_kernel_twid_fft14avx128_wrapper(prec, dir);
        case 15:
            return register_kernel_twid_fft15avx128_wrapper(prec, dir);
        case 16:
            return register_kernel_twid_fft16avx128_wrapper(prec, dir);
        }
        break;
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_AVX256:
        switch (radix)
        {
        case 2:
            return register_kernel_twid_fft2avx256_wrapper(prec, dir);
        case 3:
            return register_kernel_twid_fft3avx256_wrapper(prec, dir);
        case 4:
            return register_kernel_twid_fft4avx256_wrapper(prec, dir);
        case 5:
            return register_kernel_twid_fft5avx256_wrapper(prec, dir);
        case 6:
            return register_kernel_twid_fft6avx256_wrapper(prec, dir);
        case 7:
            return register_kernel_twid_fft7avx256_wrapper(prec, dir);
        case 8:
            return register_kernel_twid_fft8avx256_wrapper(prec, dir);
        case 9:
            return register_kernel_twid_fft9avx256_wrapper(prec, dir);
        case 10:
            return register_kernel_twid_fft10avx256_wrapper(prec, dir);
        case 11:
            return register_kernel_twid_fft11avx256_wrapper(prec, dir);
        case 12:
            return register_kernel_twid_fft12avx256_wrapper(prec, dir);
        case 13:
            return register_kernel_twid_fft13avx256_wrapper(prec, dir);
        case 14:
            return register_kernel_twid_fft14avx256_wrapper(prec, dir);
        case 15:
            return register_kernel_twid_fft15avx256_wrapper(prec, dir);
        case 16:
            return register_kernel_twid_fft16avx256_wrapper(prec, dir);
        }
        break;
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_AVX512:
        switch (radix)
        {
        case 2:
            return register_kernel_twid_fft2avx512_wrapper(prec, dir);
        case 3:
            return register_kernel_twid_fft3avx512_wrapper(prec, dir);
        case 4:
            return register_kernel_twid_fft4avx512_wrapper(prec, dir);
        case 5:
            return register_kernel_twid_fft5avx512_wrapper(prec, dir);
        case 6:
            return register_kernel_twid_fft6avx512_wrapper(prec, dir);
        case 7:
            return register_kernel_twid_fft7avx512_wrapper(prec, dir);
        case 8:
            return register_kernel_twid_fft8avx512_wrapper(prec, dir);
        case 9:
            return register_kernel_twid_fft9avx512_wrapper(prec, dir);
        case 10:
            return register_kernel_twid_fft10avx512_wrapper(prec, dir);
        case 11:
            return register_kernel_twid_fft11avx512_wrapper(prec, dir);
        case 12:
            return register_kernel_twid_fft12avx512_wrapper(prec, dir);
        case 13:
            return register_kernel_twid_fft13avx512_wrapper(prec, dir);
        case 14:
            return register_kernel_twid_fft14avx512_wrapper(prec, dir);
        case 15:
            return register_kernel_twid_fft15avx512_wrapper(prec, dir);
        case 16:
            return register_kernel_twid_fft16avx512_wrapper(prec, dir);
        }
        break;
#endif
    }
    return nullptr;
}

/**
 * @brief Run the twiddle multiplication without transpose
 * 
 * @tparam T data type (float or double)
 * @param in_real input real buffer
 * @param in_imag input imaginary buffer
 * @param radix radix of the FFT kernel
 * @param sets number of sets (or) offset
 * @param in_stride in-stride of the kernel
 * @param v_in_stride virtual in-stride of the kernel
 * @param twiddle_buffer twiddle buffer
 *
 * @return INT32 1 if successful, 0 if failed
 */
template <typename T>
INT32 gtest_twiddle_multiplier_no_transpose(T *in_real, T *in_imag, INTP radix,
                                            INTP sets, INTP in_stride,
                                            INTP v_in_stride,
                                            VOID *twiddle_buffer)
{
    T *twiddle_buffer_real = (T *)twiddle_buffer;
    T *twiddle_buffer_imag = twiddle_buffer_real + 1;

    for (INTP r = 1; r < radix; r++)
    {
        INTP in_index = r * in_stride + v_in_stride;
        INTP tw_in_index =  DATA_STRIDE * (r * sets + 1);

        for (INTP s = 1; s < sets; s++)
        {
            T TW_real = twiddle_buffer_real[tw_in_index];
            T TW_imag = twiddle_buffer_imag[tw_in_index];

            T real = in_real[in_index];
            T imag = in_imag[in_index];

            T result_real = real * TW_real - imag * TW_imag;
            T result_imag = real * TW_imag + imag * TW_real;

            in_real[in_index] = result_real;
            in_imag[in_index] = result_imag;

            in_index += v_in_stride;
            tw_in_index += DATA_STRIDE;
        }
    }
    return 1;
}

template <typename T>
VOID compute_twiddle_buffer_wrapper(VOID *twiddle_buffer, INTP r, INTP m);

template <>
VOID compute_twiddle_buffer_wrapper<FLOAT>(VOID *twiddle_buffer, INTP r, INTP m)
{
    compute_twiddle_buffer_float_wrapper(twiddle_buffer, r, m);
}

template <>
VOID compute_twiddle_buffer_wrapper<DOUBLE>(VOID *twiddle_buffer, INTP r,
                                            INTP m)
{
    compute_twiddle_buffer_double_wrapper(twiddle_buffer, r, m);
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
    case aocl_fftz_kernel_type::C2C_C:
        return wrapper_kernels_c2c_c;
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_AVX128:
        return wrapper_kernels_c2c_avx128;
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_AVX256:
        return wrapper_kernels_c2c_avx256;
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_AVX512:
        return wrapper_kernels_c2c_avx512;
#endif
    case aocl_fftz_kernel_type::R2HC_C:
        return wrapper_kernels_r2hc_c;
    case aocl_fftz_kernel_type::R2HCF_C:
        return wrapper_kernels_r2hcf_c;
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::R2HC_AVX128:
        return wrapper_kernels_r2hc_avx128;
    case aocl_fftz_kernel_type::R2HCF_AVX128:
        return wrapper_kernels_r2hcf_avx128;
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::R2HC_AVX256:
        return wrapper_kernels_r2hc_avx256;
    case aocl_fftz_kernel_type::R2HCF_AVX256:
        return wrapper_kernels_r2hcf_avx256;
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::R2HC_AVX512:
        return wrapper_kernels_r2hc_avx512;
    case aocl_fftz_kernel_type::R2HCF_AVX512:
        return wrapper_kernels_r2hcf_avx512;
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
    case aocl_fftz_kernel_type::C2C_C:
        return "_C2C_C";
    case aocl_fftz_kernel_type::C2C_TWID_C:
        return "_C2C_TWID_C";
    case aocl_fftz_kernel_type::R2HC_C:
        return "_R2HC_C";
    case aocl_fftz_kernel_type::R2HCF_C:
        return "_R2HCF_C";
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_AVX128:
        return "_C2C_AVX128";
    case aocl_fftz_kernel_type::C2C_TWID_AVX128:
        return "_C2C_TWID_AVX128";
    case aocl_fftz_kernel_type::R2HC_AVX128:
        return "_R2HC_AVX128";
    case aocl_fftz_kernel_type::R2HCF_AVX128:
        return "_R2HCF_AVX128";
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_AVX256:
        return "_C2C_AVX256";
    case aocl_fftz_kernel_type::C2C_TWID_AVX256:
        return "_C2C_TWID_AVX256";
    case aocl_fftz_kernel_type::R2HC_AVX256:
        return "_R2HC_AVX256";
    case aocl_fftz_kernel_type::R2HCF_AVX256:
        return "_R2HCF_AVX256";
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_AVX512:
        return "_C2C_AVX512";
    case aocl_fftz_kernel_type::R2HC_AVX512:
        return "_R2HC_AVX512";
    case aocl_fftz_kernel_type::R2HCF_AVX512:
        return "_R2HCF_AVX512";
    case aocl_fftz_kernel_type::C2C_TWID_AVX512:
        return "_C2C_TWID_AVX512";
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
    case aocl_fftz_kernel_type::R2HCF_C:
    case aocl_fftz_kernel_type::R2HCF_AVX256:
    case aocl_fftz_kernel_type::R2HCF_AVX128:
    case aocl_fftz_kernel_type::R2HCF_AVX512:
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

#endif // AOCLFFTZ_KERNEL_GTEST_UTILS_H
