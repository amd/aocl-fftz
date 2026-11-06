// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftw_wrapper_wisdom_gtest.cpp
 * @brief GTest cases for FFTW wrapper wisdom APIs (no-op stubs).
 */

/*
 * All wisdom APIs in the FFTW wrapper are no-op stubs.
 * Export functions return 1 (success), import functions return 0 (failure).
 * No wisdom is ever stored, loaded or persisted.
 * These tests verify the stub return values and no-crash behaviour.
 */

/* std::tmpfile() is flagged as "deprecated/unsafe" by the MSVC CRT headers,
 * which makes clang-cl emit -Wdeprecated-declarations on Windows. Silence that
 * CRT-specific deprecation here (no effect on POSIX toolchains). */
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <cstring>
#include "fftw_wrapper_test_utils.h"

class FftwWrapperWisdomTest : public FftwWrapperTestBase<double>
{
};

/* ===== forget_wisdom (empty no-op) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_FORGET_WISDOM)
{
    fftw_forget_wisdom();
}

TEST_F(FftwWrapperWisdomTest, TEST_FORGET_WISDOM_F)
{
    fftwf_forget_wisdom();
}

/* ===== export_wisdom_to_filename (returns 1, no file written) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_TO_FILENAME)
{
    int ret = fftw_export_wisdom_to_filename("fftw_test_wisdom.wis");
    EXPECT_EQ(ret, 1);
}

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_TO_FILENAME_F)
{
    int ret = fftwf_export_wisdom_to_filename("fftwf_test_wisdom.wis");
    EXPECT_EQ(ret, 1);
}

/* ===== export_wisdom_to_string (returns non-NULL empty string) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_TO_STRING)
{
    char *s = fftw_export_wisdom_to_string();
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::strlen(s), 0u);
    /* FFTW contract: caller frees the returned string. Safe with the wrapper,
     * which returns a sentinel empty string that fftw_free treats as a no-op. */
    fftw_free(s);
}

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_TO_STRING_F)
{
    char *s = fftwf_export_wisdom_to_string();
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::strlen(s), 0u);
    /* FFTW contract: caller frees the returned string. Safe with the wrapper,
     * which returns a sentinel empty string that fftwf_free treats as a no-op. */
    fftwf_free(s);
}

/* ===== export_wisdom_to_file (no-op) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_TO_FILE)
{
    FILE *f = std::tmpfile();
    if (f)
    {
        fftw_export_wisdom_to_file(f);
        std::fclose(f);
    }
}

/* ===== export_wisdom with callback (no-op, callback never invoked) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_EXPORT_WISDOM_CALLBACK)
{
    fftw_export_wisdom(nullptr, nullptr);
    fftwf_export_wisdom(nullptr, nullptr);
}

/* ===== import_wisdom_from_filename (always returns 0) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_WISDOM_FROM_FILENAME)
{
    int ret = fftw_import_wisdom_from_filename("nonexistent.wis");
    EXPECT_EQ(ret, 0);
}

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_WISDOM_FROM_FILENAME_F)
{
    int ret = fftwf_import_wisdom_from_filename("nonexistent.wis");
    EXPECT_EQ(ret, 0);
}

/* ===== import_wisdom_from_file (always returns 0) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_WISDOM_FROM_FILE)
{
    FILE *f = std::tmpfile();
    if (f)
    {
        int ret = fftw_import_wisdom_from_file(f);
        EXPECT_EQ(ret, 0);
        std::fclose(f);
    }
}

/* ===== import_wisdom_from_string (always returns 0) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_WISDOM_FROM_STRING)
{
    int ret = fftw_import_wisdom_from_string("(fftw-3.3.10 fftw_wisdom)");
    EXPECT_EQ(ret, 0);
}

/* ===== import_wisdom callback (always returns 0) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_WISDOM_CALLBACK)
{
    int ret = fftw_import_wisdom(nullptr, nullptr);
    EXPECT_EQ(ret, 0);
    ret = fftwf_import_wisdom(nullptr, nullptr);
    EXPECT_EQ(ret, 0);
}

/* ===== import_system_wisdom (returns 0, no crash) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_IMPORT_SYSTEM_WISDOM)
{
    int ret = fftw_import_system_wisdom();
    EXPECT_EQ(ret, 0);
    ret = fftwf_import_system_wisdom();
    EXPECT_EQ(ret, 0);
}

/* ===== make_planner_thread_safe (empty no-op) ===== */

TEST_F(FftwWrapperWisdomTest, TEST_MAKE_PLANNER_THREAD_SAFE)
{
    fftw_make_planner_thread_safe();
    fftwf_make_planner_thread_safe();
}

/* ===== Wisdom no-ops should not corrupt planner state ===== */

TEST_F(FftwWrapperWisdomTest, TEST_WISDOM_STUB_PLAN_STILL_WORKS)
{
    const int N = 16;
    fftw_complex *in  = fftw_alloc_complex(N);
    fftw_complex *out = fftw_alloc_complex(N);
    fftw_complex *ref = fftw_alloc_complex(N);

    this->init_complex(in, N);

    fftw_plan p1 = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p1, nullptr);
    fftw_execute(p1);
    dft_reference_1d(in, ref, N, FFTW_FORWARD);
    compare_complex_arrays(ref, out, N, dft_tolerance<FftwTypes<double>>(N));
    fftw_destroy_plan(p1);

    /* AOCL wisdom is a documented no-op: export reports success (1) but
     * import reports failure (0) because nothing is persisted.  Pin that
     * contract instead of discarding the return values.  The plan is rebuilt
     * from scratch below regardless, so the transform must still be correct. */
    EXPECT_EQ(fftw_export_wisdom_to_filename("dummy.wis"), 1);
    fftw_forget_wisdom();
    EXPECT_EQ(fftw_import_wisdom_from_filename("dummy.wis"), 0);

    fftw_plan p2 = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(p2, nullptr);
    fftw_execute(p2);
    compare_complex_arrays(ref, out, N, dft_tolerance<FftwTypes<double>>(N));
    fftw_destroy_plan(p2);

    fftw_free(in);
    fftw_free(out);
    fftw_free(ref);
}
