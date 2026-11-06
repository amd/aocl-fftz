// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_memory_gtest.cpp
 * @brief FFTW wrapper memory allocation and free tests.
 */

#include <cstdint>
#include "fftw_wrapper_test_utils.h"

class FftwWrapperMemoryTest : public ::testing::Test
{
};

/* ===== malloc / free ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_MALLOC_VALID_SIZE)
{
    void *ptr = fftw_malloc(1024);
    ASSERT_NE(ptr, nullptr);
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTWF_MALLOC_VALID_SIZE)
{
    void *ptr = fftwf_malloc(1024);
    ASSERT_NE(ptr, nullptr);
    fftwf_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_MALLOC_ZERO_SIZE)
{
    void *ptr = fftw_malloc(0);
    /* posix_memalign with size 0 may return NULL or a valid pointer
     * that can be freed -- just verify no crash. */
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTWF_MALLOC_ZERO_SIZE)
{
    void *ptr = fftwf_malloc(0);
    fftwf_free(ptr);
}

/* ===== alloc_real ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_REAL_DOUBLE)
{
    double *ptr = fftw_alloc_real(64);
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 1.0;
    ptr[63] = 2.0;
    EXPECT_EQ(ptr[0], 1.0);
    EXPECT_EQ(ptr[63], 2.0);
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_REAL_FLOAT)
{
    float *ptr = fftwf_alloc_real(64);
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 1.0f;
    ptr[63] = 2.0f;
    EXPECT_EQ(ptr[0], 1.0f);
    EXPECT_EQ(ptr[63], 2.0f);
    fftwf_free(ptr);
}

/* ===== alloc_complex ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_COMPLEX_DOUBLE)
{
    fftw_complex *ptr = fftw_alloc_complex(32);
    ASSERT_NE(ptr, nullptr);
    ptr[0][0] = 1.0;
    ptr[0][1] = 2.0;
    ptr[31][0] = 3.0;
    ptr[31][1] = 4.0;
    EXPECT_EQ(ptr[0][0], 1.0);
    EXPECT_EQ(ptr[31][1], 4.0);
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_COMPLEX_FLOAT)
{
    fftwf_complex *ptr = fftwf_alloc_complex(32);
    ASSERT_NE(ptr, nullptr);
    ptr[0][0] = 1.0f;
    ptr[0][1] = 2.0f;
    ptr[31][0] = 3.0f;
    ptr[31][1] = 4.0f;
    EXPECT_EQ(ptr[0][0], 1.0f);
    EXPECT_EQ(ptr[31][1], 4.0f);
    fftwf_free(ptr);
}

/* ===== Alignment verification ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_REAL_ALIGNMENT)
{
    double *ptr = fftw_alloc_real(64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(fftw_alignment_of(ptr), 0)
        << "alloc_real should return 16-byte aligned memory";
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALLOC_COMPLEX_ALIGNMENT)
{
    fftw_complex *ptr = fftw_alloc_complex(32);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(fftw_alignment_of((double *)ptr), 0)
        << "alloc_complex should return 16-byte aligned memory";
    fftw_free(ptr);
}

/* ===== free edge cases ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_FREE_VALID_PTR)
{
    void *ptr = fftw_malloc(256);
    ASSERT_NE(ptr, nullptr);
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_FREE_NULL_PTR)
{
    fftw_free(nullptr);
    fftwf_free(nullptr);
}

/* fftw_sprint_plan returns a static empty string; fftw_free has a guard
 * for this pointer so freeing it is a safe no-op. */
TEST_F(FftwWrapperMemoryTest,
       KNOWN_DIVERGENCE_FFTW_FREE_SPRINT_PLAN_STATIC_PTR)
{
    fftw_complex *in = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    char *s = fftw_sprint_plan(p);
    ASSERT_NE(s, nullptr);
    fftw_free(s);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

/* fftw_export_wisdom_to_string also returns the same static pointer. */
TEST_F(FftwWrapperMemoryTest, TEST_FFTW_FREE_WISDOM_STRING_PTR)
{
    char *s = fftw_export_wisdom_to_string();
    ASSERT_NE(s, nullptr);
    fftw_free(s);
}

/* ===== alignment_of ===== */

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALIGNMENT_OF_DOUBLE)
{
    double *ptr = fftw_alloc_real(128);
    ASSERT_NE(ptr, nullptr);
    int a1 = fftw_alignment_of(ptr);
    int a2 = fftw_alignment_of(ptr);
    EXPECT_EQ(a1, a2) << "alignment_of should be deterministic";
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_FFTW_ALIGNMENT_OF_FLOAT)
{
    float *ptr = fftwf_alloc_real(128);
    ASSERT_NE(ptr, nullptr);
    int a1 = fftwf_alignment_of(ptr);
    int a2 = fftwf_alignment_of(ptr);
    EXPECT_EQ(a1, a2);
    fftwf_free(ptr);
}

/* ===== write and read back ===== */

TEST_F(FftwWrapperMemoryTest, TEST_ALLOCATED_MEMORY_WRITABLE)
{
    const size_t N = 256;
    double *ptr = (double *)fftw_malloc(N * sizeof(double));
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < N; i++)
    {
        ptr[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(ptr[i], static_cast<double>(i));
    }
    fftw_free(ptr);
}

TEST_F(FftwWrapperMemoryTest, TEST_ALLOCATED_MEMORY_WRITABLE_FLOAT)
{
    const size_t N = 256;
    float *ptr = (float *)fftwf_malloc(N * sizeof(float));
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < N; i++)
    {
        ptr[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(ptr[i], static_cast<float>(i));
    }
    fftwf_free(ptr);
}

/* ===== Allocate, use in plan, free ===== */

TEST_F(FftwWrapperMemoryTest, TEST_ALLOC_AND_USE_IN_PLAN)
{
    const int N = 16;
    fftw_complex *in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *ref = fftw_alloc_complex(N);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);
    ASSERT_NE(ref, nullptr);

    fill_complex_real_iota(in, N, 0.0);

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftw_execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(
        ref, out, N, dft_tolerance<FftwTypes<double>>(N));

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    fftw_free(ref);
}

TEST_F(FftwWrapperMemoryTest, TEST_ALLOC_AND_USE_IN_PLAN_FLOAT)
{
    const int N = 16;
    fftwf_complex *in  = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_complex *out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_complex *ref = fftwf_alloc_complex(N);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);
    ASSERT_NE(ref, nullptr);

    fill_complex_real_iota(in, N, 0.0f);

    fftwf_plan p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftwf_execute(p);

    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(
        ref, out, N, dft_tolerance<FftwTypes<float>>(N));

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(ref);
}
