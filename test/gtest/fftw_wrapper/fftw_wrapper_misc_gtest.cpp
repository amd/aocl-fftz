// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_misc_gtest.cpp
 * @brief GTest cases for FFTW wrapper miscellaneous APIs.
 */

#include <cstdio>
#include <cstring>
#include "fftw_wrapper_test_utils.h"

class FftwWrapperMiscTest : public ::testing::Test
{
};

/* ===== Version string ===== */

TEST_F(FftwWrapperMiscTest, TEST_VERSION_CONTAINS_FFTW_COMPAT)
{
    const char *v = fftw_version;
    ASSERT_NE(v, nullptr);
    EXPECT_GT(std::strlen(v), 0u);
    EXPECT_NE(std::strstr(v, "FFTW compatible"), nullptr)
        << "fftw_version should contain 'FFTW compatible', got: " << v;
}

TEST_F(FftwWrapperMiscTest, TEST_VERSION_F_CONTAINS_FFTW_COMPAT)
{
    const char *v = fftwf_version;
    ASSERT_NE(v, nullptr);
    EXPECT_GT(std::strlen(v), 0u);
    EXPECT_NE(std::strstr(v, "FFTW compatible"), nullptr)
        << "fftwf_version should contain 'FFTW compatible', got: " << v;
}

/* ===== cleanup (empty stubs) ===== */

TEST_F(FftwWrapperMiscTest, TEST_CLEANUP)
{
    fftw_cleanup();
}

TEST_F(FftwWrapperMiscTest, TEST_CLEANUP_F)
{
    fftwf_cleanup();
}

/* ===== flops: no-op that does not touch the output pointers ===== */

TEST_F(FftwWrapperMiscTest, TEST_FLOPS_NO_CRASH)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    double add = 42.0, mul = 42.0, fma = 42.0;
    fftw_flops(p, &add, &mul, &fma);
    EXPECT_EQ(add, 42.0) << "flops is a no-op; values should be unchanged";
    EXPECT_EQ(mul, 42.0);
    EXPECT_EQ(fma, 42.0);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_FLOPS_F_NO_CRASH)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    double add = 42.0, mul = 42.0, fma = 42.0;
    fftwf_flops(p, &add, &mul, &fma);
    EXPECT_EQ(add, 42.0);
    EXPECT_EQ(mul, 42.0);
    EXPECT_EQ(fma, 42.0);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

/* ===== estimate_cost / cost: hardcoded to return 0.0 ===== */

TEST_F(FftwWrapperMiscTest, TEST_ESTIMATE_COST_RETURNS_ZERO)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(fftw_estimate_cost(p), 0.0);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_ESTIMATE_COST_F_RETURNS_ZERO)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(fftwf_estimate_cost(p), 0.0);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_COST_RETURNS_ZERO)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(fftw_cost(p), 0.0);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_COST_F_RETURNS_ZERO)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(fftwf_cost(p), 0.0);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

/* ===== print_plan / fprint_plan / sprint_plan (no-op stubs) ===== */

TEST_F(FftwWrapperMiscTest, TEST_PRINT_PLAN)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftw_print_plan(p);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_PRINT_PLAN_F)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftwf_print_plan(p);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_FPRINT_PLAN)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftw_fprint_plan(p, stdout);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_FPRINT_PLAN_F)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(
        8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);
    fftwf_fprint_plan(p, stdout);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_SPRINT_PLAN_RETURNS_NON_NULL)
{
    fftw_complex *in  = fftw_alloc_complex(8);
    fftw_complex *out = fftw_alloc_complex(8);
    fftw_plan p = fftw_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    char *s = fftw_sprint_plan(p);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::strlen(s), 0u);
    /* FFTW contract: the caller frees the sprint_plan string with fftw_free.
     * The wrapper returns a guarded static empty string, so this is a safe
     * no-op today, but freeing keeps the test aligned with the API. */
    fftw_free(s);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

TEST_F(FftwWrapperMiscTest, TEST_SPRINT_PLAN_F_RETURNS_NON_NULL)
{
    fftwf_complex *in  = fftwf_alloc_complex(8);
    fftwf_complex *out = fftwf_alloc_complex(8);
    fftwf_plan p = fftwf_plan_dft_1d(8, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p, nullptr);

    char *s = fftwf_sprint_plan(p);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::strlen(s), 0u);
    /* FFTW contract: pair sprint_plan with the matching free (fftwf_free). */
    fftwf_free(s);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}
