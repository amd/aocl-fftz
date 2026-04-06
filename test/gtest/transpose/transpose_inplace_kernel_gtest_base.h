// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_kernel_gtest_base.h
 *
 * @brief Base file for transpose kernel tests.
 *
 * This file contains the classes and functions to be used for running transpose
 * kernel unit tests using GTest.
 *
 * @author Ashwin K. Godbole
 *
 */

#ifndef TRANSPOSE_INPLACE_KERNEL_GTEST_BASE_H
#define TRANSPOSE_INPLACE_KERNEL_GTEST_BASE_H

#include <gtest/gtest.h>
#include "api/aoclfftz_internal.h"
#include "test/gtest/transpose/transpose_kernel_gtest_utils.h"

/**
 * @brief Base class for the transpose kernel tests
 *
 * @tparam T type of the input / output
 */
template <class T>
class AoclfftzInplaceTransposeTestBase
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<INTP, INTP>, /* rows, cols */
                     INTP,                   /* stride */
                     INT32>>                 /* kernel index */
{
  protected:
    INTP rows;
    INTP cols;
    INTP stride;
    aoclfftz_transpose_kernel kernel;

    VOID expect_matrix_equal(T* in, T* out, INTP rows, INTP cols, INTP stride)
    {
        // iterating over the entire matrix helps ensure that the elements that are
        // skipped during strided access have not been modified
        for (INTP i = 0; i < rows * cols * stride; ++i)
        {
            EXPECT_EQ(data_equal(in[i], out[i]), true)
                << "Mismatch " << compare_data_string(in[i], out[i])
                << " at offset: " << i << "\n";
        }
    }

    VOID test_kernel()
    {
        rows = std::get<0>(std::get<0>(GetParam()));
        cols = std::get<1>(std::get<0>(GetParam()));
        stride = std::get<1>(GetParam());
        INT32 kernel_idx = std::get<2>(GetParam());
        aoclfftz_transpose_kernel *kernel_table = get_transpose_kernels_c<T>();
        kernel = kernel_table[kernel_idx];

        INTP n_elems = rows * cols * stride;

        T *in, *out;
        ALLOC_UNALIGN_INIT(in, T, n_elems * sizeof(T));
        ALLOC_UNALIGN_INIT(out, T, n_elems * sizeof(T));

        aoclfftz_transpose_aux_mem_t aux_mem;
        aux_mem.size = rows * cols;
        ALLOC_UNALIGN_INIT(aux_mem.data, UINT8, aux_mem.size * sizeof(UINT8));

        // initialize the input matrix
        matrix_init(in, rows, cols, stride);

        // transpose using default (reference) transpose
        transpose_reference(in, out, rows, cols, stride, stride);

        // setup transpose metadata
        aoclfftz_dim_t_64_ row_m, col_m;

        row_m.n = rows;
        row_m.in_stride = cols * stride;  /* old leading_dim */
        row_m.out_stride = rows * stride; /* new leading_dim */

        col_m.n = cols;
        col_m.in_stride = stride;
        col_m.out_stride = stride;

        // transpose using kernel
        kernel((VOID *)in, (VOID *)in, row_m, col_m, &aux_mem);

        // check if matrices are equal (rows and cols are interchanged because
        // this is a post transpose comparison)
        expect_matrix_equal(out, in, cols, rows, stride);

        free(in);
        free(out);
        free(aux_mem.data);
    }
};

/**
 * @brief A derived class from AoclfftzInplaceTransposeTestBase for FLOAT type
 *
 */

class AoclfftzInplaceTransposeKernelTestFloat
    : public AoclfftzInplaceTransposeTestBase<FLOAT>
{
};

/**
 * @brief A derived class from AoclfftzInplaceTransposeTestBase for DOUBLE type
 *
 */
class AoclfftzInplaceTransposeKernelTestDouble
    : public AoclfftzInplaceTransposeTestBase<DOUBLE>
{
};

/**
 * @brief A derived class from AoclfftzInplaceTransposeTestBase for
 * Complex(FLOAT) type
 *
 */

class AoclfftzInplaceTransposeKernelTestFloatComplex
    : public AoclfftzInplaceTransposeTestBase<aoclfftz_complex_f_t>
{
};

/**
 * @brief A derived class from AoclfftzInplaceTransposeTestBase for
 * Complex(DOUBLE) type
 *
 */
class AoclfftzInplaceTransposeKernelTestDoubleComplex
    : public AoclfftzInplaceTransposeTestBase<aoclfftz_complex_d_t>
{
};

#endif // TRANSPOSE_INPLACE_KERNEL_GTEST_BASE_H
