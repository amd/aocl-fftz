// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#ifdef MULTI_THREADING
#include <omp.h>
#endif

extern "C"
{
#include "api/aoclfftz_internal.h"
#include "test/gtest/aoclfftz_core_wrapper.h"
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
        /* 3 */ CONCAT(FUNC(tir_cycles, type, isa), _wrapper),                 \
        /* 4 */ CONCAT(FUNC(tisr_cycles, type, isa), _wrapper),                \
        /* 5 */ CONCAT(FUNC(tos_iterative, type, isa), _wrapper),              \
        /* 6 */ CONCAT(FUNC(tos_blocked, type, isa), _wrapper),                \
    }

GENERATE_TRANSPOSE_KERNEL_TABLE(FFTZ_FLOAT, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(FFTZ_DOUBLE, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(aoclfftz_complex_f_t, c);
GENERATE_TRANSPOSE_KERNEL_TABLE(aoclfftz_complex_d_t, c);

static std::string transpose_kernel_names_table[] =
{
    "tiq_iterative",
    "tisq_iterative",
    "tiq_recursive_buf",
    "tir_cycles",
    "tisr_cycles",
    "tos_iterative",
    "tos_blocked",
};

// -----------------------------------------------------------------------------
template <typename T>
static aoclfftz_transpose_kernel *get_transpose_kernels_c();

template <> aoclfftz_transpose_kernel *get_transpose_kernels_c<FFTZ_FLOAT>()
{
    return transpose_kernel_table_FFTZ_FLOAT_c;
}

template <> aoclfftz_transpose_kernel *get_transpose_kernels_c<FFTZ_DOUBLE>()
{
    return transpose_kernel_table_FFTZ_DOUBLE_c;
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
template <typename T> static T get_value(FFTZ_INTP value)
{
    return static_cast<T>(value);
}

template <> aoclfftz_complex_f_t get_value(FFTZ_INTP value)
{
    return aoclfftz_complex_f_t{static_cast<FFTZ_FLOAT>(value), 1.0f};
}

template <> aoclfftz_complex_d_t get_value(FFTZ_INTP value)
{
    return aoclfftz_complex_d_t{static_cast<FFTZ_DOUBLE>(value), 1.0};
}

template <typename T>
FFTZ_VOID matrix_init(T *matrix, FFTZ_INTP rows, FFTZ_INTP cols,
                      FFTZ_INTP stride)
{
    FFTZ_INTP leading_dim = stride * cols;
    FFTZ_INTP value = 1;

    for (FFTZ_INTP i = 0; i < rows; i++)
    {
        for (FFTZ_INTP j = 0; j < cols; j++)
        {
            matrix[(i * leading_dim) + (j * stride)] = get_value<T>(value);
            value++;
        }
    }
}

// performs an out-of-place transpose
template <typename T>
void transpose_reference(T *in, T *out, FFTZ_INTP rows, FFTZ_INTP cols,
                         FFTZ_INTP in_stride, FFTZ_INTP out_stride)
{
    FFTZ_INTP old_leading_dim = in_stride * cols;
    FFTZ_INTP new_leading_dim = out_stride * rows;

    for (FFTZ_INTP i = 0; i < rows; ++i)
    {
        for (FFTZ_INTP j = 0; j < cols; ++j)
        {
            out[(j * new_leading_dim) + (i * out_stride)] =
                in[(i * old_leading_dim) + (j * in_stride)];
        }
    }
}

#endif // TRANSPOSE_KERNEL_GTEST_UTILS_H
