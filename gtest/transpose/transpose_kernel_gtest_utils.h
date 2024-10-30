/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

/** @file transpose_kernel_gtest_utils.h
 *
 * @brief Utilities for transpose kernel tests.
 *
 * This file contains utilities for testing transpose kernels.
 *
 * @author Ashwin K. Godbole
 *
 */

#ifndef TRANSPOSE_KERNEL_GTEST_UTILS_H
#define TRANSPOSE_KERNEL_GTEST_UTILS_H

#include <string>
#include <gtest/gtest.h>

extern "C"
{
#include "api/aoclfftz_internal.h"
#include "gtest/aoclfftz_core_wrapper.h"
}

// -----------------------------------------------------------------------------
template <class T> static bool data_equal(T a, T b)
{
    return a == b;
}

template <> bool data_equal(aoclfftz_complex_f_t a, aoclfftz_complex_f_t b)
{
    return (a.real == b.real) && (a.imag == b.imag);
}

template <> bool data_equal(aoclfftz_complex_d_t a, aoclfftz_complex_d_t b)
{
    return (a.real == b.real) && (a.imag == b.imag);
}

// -----------------------------------------------------------------------------
template <class T> static auto compare_data_string(T recieved, T expected)
{
    return "Got( " + std::to_string(recieved) + " ), Expected( " +
           std::to_string(expected) + " )";
}

template <>
auto compare_data_string(aoclfftz_complex_f_t recieved,
                         aoclfftz_complex_f_t expected)
{
    return "Got( " + std::to_string(recieved.real) + "+" +
           std::to_string(recieved.imag) + "i" + " ), Expected( " +
           std::to_string(expected.real) + "+" + std::to_string(expected.imag) +
           "i" + " )";
}

template <>
auto compare_data_string(aoclfftz_complex_d_t recieved,
                         aoclfftz_complex_d_t expected)
{
    return "Got( " + std::to_string(recieved.real) + "+" +
           std::to_string(recieved.imag) + "i" + " ), Expected( " +
           std::to_string(expected.real) + "+" + std::to_string(expected.imag) +
           "i" + " )";
}

// -----------------------------------------------------------------------------
#define GENERATE_TRANSPOSE_KERNEL_TABLE(type, isa)                             \
    static aoclfftz_transpose_kernel transpose_kernel_table_##type##_##isa[] = \
    {                                                                          \
        /* 0 */ CONCAT(FUNC(tiq_iterative, type, isa), _wrapper),              \
        /* 1 */ CONCAT(FUNC(tisq_iterative, type, isa), _wrapper),             \
        /* 2 */ CONCAT(FUNC(tiq_recursive_buf, type, isa), _wrapper),          \
    }

GENERATE_TRANSPOSE_KERNEL_TABLE(FLOAT, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(DOUBLE, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(aoclfftz_complex_f_t, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(aoclfftz_complex_d_t, c);

static std::string transpose_kernel_names_table[] =
{
    "tiq_iterative",
    "tisq_iterative",
    "tiq_recursive_buf",
};

// -----------------------------------------------------------------------------
template <typename T>
static aoclfftz_transpose_kernel *get_transpose_kernels_c();

template <> aoclfftz_transpose_kernel *get_transpose_kernels_c<FLOAT>()
{
    return transpose_kernel_table_FLOAT_c;
}

template <> aoclfftz_transpose_kernel *get_transpose_kernels_c<DOUBLE>()
{
    return transpose_kernel_table_DOUBLE_c;
}

template <>
aoclfftz_transpose_kernel *get_transpose_kernels_c<aoclfftz_complex_f_t>()
{
    return transpose_kernel_table_aoclfftz_complex_f_t_c;
}

template <>
aoclfftz_transpose_kernel *get_transpose_kernels_c<aoclfftz_complex_d_t>()
{
    return transpose_kernel_table_aoclfftz_complex_d_t_c;
}

// -----------------------------------------------------------------------------
template <typename T> static T get_value(INTP value)
{
    return static_cast<T>(value);
}

template <> aoclfftz_complex_f_t get_value(INTP value)
{
    return aoclfftz_complex_f_t{static_cast<FLOAT>(value), 1.0f};
}

template <> aoclfftz_complex_d_t get_value(INTP value)
{
    return aoclfftz_complex_d_t{static_cast<DOUBLE>(value), 1.0};
}

template <typename T>
VOID matrix_init(T *matrix, INTP rows, INTP cols, INTP stride)
{
    INTP leading_dim = stride * cols;
    INTP value = 1;

    for (INTP i = 0; i < rows; i++)
    {
        for (INTP j = 0; j < cols; j++)
        {
            matrix[(i * leading_dim) + (j * stride)] = get_value<T>(value);
            value++;
        }
    }
}

#endif // TRANSPOSE_KERNEL_GTEST_UTILS_H
