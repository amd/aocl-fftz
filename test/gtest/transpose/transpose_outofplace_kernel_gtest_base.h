// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_outofplace_kernel_gtest_base.h
 *
 * @brief Base file for out-of-place transpose kernel tests.
 *
 * This file contains the classes and functions to be used for running transpose
 * kernel unit tests using GTest.
 *
 * @author Ashwin K. Godbole
 *
 */

#ifndef TRANSPOSE_OUTOFPLACE_KERNEL_GTEST_BASE_H
#define TRANSPOSE_OUTOFPLACE_KERNEL_GTEST_BASE_H

#include <complex>
#include <iostream>
#include <gtest/gtest.h>
#include "api/aoclfftz_internal.h"
#include "test/gtest/transpose/transpose_kernel_gtest_utils.h"

/**
 * @brief Base class for the transpose kernel tests
 *
 * @tparam T type of the input / output
 */
template <class T>
class AoclfftzOutOfPlaceTransposeTestBase
    : public ::testing::TestWithParam<
                                std::tuple<
                                    std::tuple<INTP, INTP>, /* rows, cols */
                                    INTP,                   /* in stride */
                                    INTP,                   /* out stride */
                                    INT32>>                 /* kernel index */
{
  protected:
    INTP rows;
    INTP cols;
    INTP in_stride;
    INTP out_stride;
    aoclfftz_transpose_kernel kernel;

    void expect_matrix_equal(T *in, T *out, INTP rows, INTP cols, INTP stride)
    {
        INTP leading_dim = cols * stride;

        for (INTP i = 0; i < rows; ++i)
        {
            for (INTP j = 0; j < cols; ++j)
            {
                // since this is checking an out of place transpose, we don't
                // have to check if the elements in the stride area are equal or
                // not.
                EXPECT_EQ(
                    data_equal(in[LINEAR_IDX_2D(i, j, stride, leading_dim)],
                               out[LINEAR_IDX_2D(i, j, stride, leading_dim)]),
                    1)
                    << "Mismatch "
                    << compare_data_string(
                           in[LINEAR_IDX_2D(i, j, stride, leading_dim)],
                           out[LINEAR_IDX_2D(i, j, stride, leading_dim)])
                    << " at location: (" << i << ", " << j << ")\n";
            }
        }
    }

    void test_kernel()
    {
        rows = std::get<0>(std::get<0>(GetParam()));
        cols = std::get<1>(std::get<0>(GetParam()));
        in_stride = std::get<1>(GetParam());
        out_stride = std::get<2>(GetParam());
        int kernel_idx = std::get<3>(GetParam());
        aoclfftz_transpose_kernel *kernel_table = get_transpose_kernels_c<T>();
        kernel = kernel_table[kernel_idx];

        INTP in_size = in_stride * cols * rows;
        INTP out_size = out_stride * cols * rows;

        T* in = (T*) calloc(in_size, sizeof(T));
        T* out_ref = (T*) calloc(out_size, sizeof(T));
        T* out_ker = (T*) calloc(out_size, sizeof(T));

        // initialize the input matrix
        matrix_init(in, rows, cols, in_stride);

        // transpose using default (reference) transpose
        transpose_reference(in, out_ref, rows, cols, in_stride, out_stride);

        // setup transpose metadata
        aoclfftz_dim_t_64_ row_m, col_m;

        row_m.n = rows;
        row_m.in_stride = cols * in_stride;   /* old leading_dim */
        row_m.out_stride = rows * out_stride; /* new leading_dim */

        col_m.n = cols;
        col_m.in_stride = in_stride;
        col_m.out_stride = out_stride;

        // transpose using kernel
        kernel((VOID *)in, (VOID *)out_ker, row_m, col_m, NULL /* aux-mem */);

        // check if matrices are equal
        expect_matrix_equal(out_ref, out_ker, cols, rows, out_stride);

        free(in);
        free(out_ref);
        free(out_ker);
    }
};

/**
 * @brief A derived class from AoclfftzOutOfPlaceTransposeTestBase for FLOAT
 * type
 *
 */

class AoclfftzOutOfPlaceTransposeKernelTestF32
    : public AoclfftzOutOfPlaceTransposeTestBase<FLOAT>
{
};

/**
 * @brief A derived class from AoclfftzOutOfPlaceTransposeTestBase for DOUBLE
 * type
 *
 */
class AoclfftzOutOfPlaceTransposeKernelTestF64
    : public AoclfftzOutOfPlaceTransposeTestBase<DOUBLE>
{
};

/**
 * @brief A derived class from AoclfftzOutOfPlaceTransposeTestBase for
 * Complex(FLOAT) type
 *
 */

class AoclfftzOutOfPlaceTransposeKernelTestF32C
    : public AoclfftzOutOfPlaceTransposeTestBase<aoclfftz_complex_f_t>
{
};

/**
 * @brief A derived class from AoclfftzOutOfPlaceTransposeTestBase for
 * Complex(DOUBLE) type
 *
 */
class AoclfftzOutOfPlaceTransposeKernelTestF64C
    : public AoclfftzOutOfPlaceTransposeTestBase<aoclfftz_complex_d_t>
{
};

#endif // TRANSPOSE_OUTOFPLACE_KERNEL_GTEST_BASE_H
