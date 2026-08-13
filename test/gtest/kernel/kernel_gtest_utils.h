// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#ifdef MULTI_THREADING
#include <omp.h>
#endif
#include <string>
#include <typeinfo>
#include <vector>
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
 * @brief The four twiddle kernel kinds exercised by
 * AoclfftzKernelTestBase::run_twiddle_kernel_test()
 *
 * All four operate on complex data only (unlike the R2HC/R2HCF real
 * kernels, there is no real/half-complex packing to account for here):
 * fwd/bwd are the plain C2C forward/backward twiddle-on-load kernels, r2c
 * additionally H2-conjugates its output and c2r H2-conjugates its input and
 * twiddles its output instead of its input.
 *
 */
enum class twiddle_kind
{
    FWD,
    BWD,
    R2C,
    C2R
};

/**
 * @brief Classify a kernel_type into its twiddle_kind
 *
 * Centralizes the mapping from each of the 16 direction-specific twiddle
 * kernel enums (C2C_TWID_FWD_*, C2C_TWID_BWD_*, C2C_TWID_R2C_*,
 * C2C_TWID_C2R_*) to its twiddle_kind, so callers (the test runner, the
 * gtest name generator) do not each need to re-enumerate every ISA variant.
 *
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @param kind [out] the decoded twiddle_kind, only set when this function
 * returns true
 * @return true if `kernel_type` is one of the 16 twiddle kernel enums,
 * false otherwise
 */
inline bool twiddle_kind_from_kernel_type(FFTZ_UINT8 kernel_type,
                                          twiddle_kind &kind)
{
    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::C2C_TWID_FWD_C:
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128:
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256:
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512:
#endif
        kind = twiddle_kind::FWD;
        return true;

    case aocl_fftz_kernel_type::C2C_TWID_BWD_C:
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128:
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256:
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512:
#endif
        kind = twiddle_kind::BWD;
        return true;

    case aocl_fftz_kernel_type::C2C_TWID_R2C_C:
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128:
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256:
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512:
#endif
        kind = twiddle_kind::R2C;
        return true;

    case aocl_fftz_kernel_type::C2C_TWID_C2R_C:
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128:
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256:
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512:
#endif
        kind = twiddle_kind::C2R;
        return true;

    default:
        return false;
    }
}

/**
 * @brief Whether a twiddle_kind is validated against the backward-direction
 * reference path
 *
 * BWD and C2R both run their internal butterfly with backward-direction
 * algebra (see AoclfftzKernelTestBase::run_twiddle_kernel_test()); FWD and
 * R2C both run it forward.
 *
 * @param kind the twiddle_kind to check
 * @return FFTZ_UINT8 1 if `kind` is BWD or C2R, 0 otherwise
 */
inline FFTZ_UINT8 twiddle_kind_is_bwd(twiddle_kind kind)
{
    return (kind == twiddle_kind::BWD || kind == twiddle_kind::C2R) ? 1 : 0;
}

/**
 * @brief Get the kernel object from the kernel table based on the given radix
 *
 * @param kernel_table wrapper_kernel_fp_list to search the kernel
 * @param dir direction (forward / backward)
 * @param radix radix of the FFT kernel
 * @return kffft_ a kernel pointer; returns nullptr if kernel not found
 */
template <class T>
kfft_ get_kernel(const wrapper_kernel_fp_list *kernel_table,
                 const FFTZ_INT32 dir, const FFTZ_UINT32 radix)
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

// All four twiddle kernel kinds (fwd, bwd, r2c, c2r) are inherently fixed to
// one direction each -- register_kernel_twid_<kind>_fft<radix><isa> ignores
// `dir` internally (it is accepted for signature uniformity only, matching
// e.g. register_kernel_twid_fwd_fft2c's `direction /* unused */` parameter).
// The `kind` chosen at the call site below (via kernel_type) is therefore
// what actually selects the direction, not `dir`.
#define REGISTER_TWID_KERNEL_CASE(kind, radix, isa)                           \
    case radix:                                                              \
        return register_kernel_twid_##kind##_fft##radix##isa##_wrapper(      \
            prec, dir);

#define REGISTER_TWID_KERNEL_CASES(kind, isa)                                 \
    REGISTER_TWID_KERNEL_CASE(kind, 2, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 3, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 4, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 5, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 6, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 7, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 8, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 9, isa)                                   \
    REGISTER_TWID_KERNEL_CASE(kind, 10, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 11, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 12, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 13, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 14, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 15, isa)                                  \
    REGISTER_TWID_KERNEL_CASE(kind, 16, isa)

/**
 * @brief Get the twiddle kernel pointer for the given radix
 *
 * @tparam T data type (float or double)
 * @param radix radix of the FFT kernel
 * @param dir direction (forward / backward) -- ignored by every twiddle
 * kernel kind; direction is instead selected via `kernel_type`
 * @param kernel_type kernel type - C2C_TWID_FWD_ / C2C_TWID_BWD_ /
 * C2C_TWID_R2C_ / C2C_TWID_C2R_ [C|AVX128|AVX256|AVX512]
 * @return kffft_ a kernel pointer; returns nullptr if kernel not found
 */
template <class T>
kfft_ get_twiddle_kernel(const FFTZ_UINT32 radix, const FFTZ_INT32 dir,
                         const FFTZ_UINT8 kernel_type)
{
    FFTZ_UINT8 prec;
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
    case aocl_fftz_kernel_type::C2C_TWID_FWD_C:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(fwd, c)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_BWD_C:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(bwd, c)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_R2C_C:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(r2c, c)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_C2R_C:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(c2r, c)
        }
        break;

#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(fwd, avx128)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(bwd, avx128)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(r2c, avx128)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(c2r, avx128)
        }
        break;
#endif

#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(fwd, avx256)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(bwd, avx256)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(r2c, avx256)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(c2r, avx256)
        }
        break;
#endif

#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(fwd, avx512)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(bwd, avx512)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(r2c, avx512)
        }
        break;

    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512:
        switch (radix)
        {
            REGISTER_TWID_KERNEL_CASES(c2r, avx512)
        }
        break;
#endif
    }
    return nullptr;
}

#undef REGISTER_TWID_KERNEL_CASES
#undef REGISTER_TWID_KERNEL_CASE

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
 * @return FFTZ_INT32 1 if successful, 0 if failed
 */
template <typename T>
FFTZ_INT32 gtest_twiddle_multiplier_no_transpose(
    T *in_real, T *in_imag, FFTZ_INTP radix, FFTZ_INTP sets,
    FFTZ_INTP in_stride, FFTZ_INTP v_in_stride, FFTZ_INTP load_multi_cols)
{
    const double TWO_PI = 6.28318530717958647692;
    const double angle_base = -TWO_PI / (double)(radix * sets);

    for (FFTZ_INTP r = 1; r < radix; r++)
    {
        FFTZ_INTP in_index = r * in_stride + v_in_stride;

        for (FFTZ_INTP s = 1; s < sets; s++)
        {
            const double angle = angle_base * (double)r * (double)s;
            const T TW_real = (T)cos(angle);
            const T TW_imag = (T)sin(angle);

            T real = in_real[in_index];
            T imag = in_imag[in_index];

            T result_real = real * TW_real - imag * TW_imag;
            T result_imag = real * TW_imag + imag * TW_real;

            in_real[in_index] = result_real;
            in_imag[in_index] = result_imag;

            in_index += v_in_stride;
        }
    }
    return 1;
}

/**
 * @brief H2-conjugate the upper-half points of a buffer, in place
 *
 * The C2R and R2C twiddle kernels' H1/H2 packed format stores upper-half
 * points (index i with i >= ceil(radix/2)) as the complex conjugate of the
 * logical spectral value: C2R twiddle kernels negate their input's
 * imaginary component internally to recover it before running the
 * backward butterfly, while R2C twiddle kernels negate their output's
 * imaginary component after running the forward butterfly to produce it.
 * This helper reproduces that same H2 conjugation on a separate buffer so
 * it can be used as ground truth when validating either kernel kind (see
 * AoclfftzKernelTestBase::run_twiddle_kernel_test()).
 *
 * @tparam T data type (float or double)
 * @param real real buffer
 * @param imag imaginary buffer
 * @param radix radix of the FFT kernel
 * @param sets number of sets (or) offset
 * @param strides per-point stride array (as populated for the kernel call)
 * @param v_stride virtual stride of the kernel (per-set stride)
 */
template <typename T>
FFTZ_VOID gtest_conjugate_h2_points(T *real, T *imag, FFTZ_UINT32 radix,
                                    FFTZ_INTP sets, const FFTZ_INTP *strides,
                                    FFTZ_INTP v_stride)
{
    FFTZ_UINT32 h2_start = (radix + 1) / 2; // ceil(radix / 2)

    for (FFTZ_INTP s = 0; s < sets; s++)
    {
        FFTZ_INTP base = s * v_stride;
        for (FFTZ_UINT32 i = h2_start; i < radix; i++)
        {
            imag[base + strides[i]] = -imag[base + strides[i]];
        }
    }
}

template <typename T>
FFTZ_VOID compute_twiddle_buffer_wrapper(FFTZ_VOID *twiddle_buffer, FFTZ_INTP r,
                                         FFTZ_INTP m, FFTZ_INTP register_width,
                                         FFTZ_INTP load_multi_cols);

template <>
FFTZ_VOID compute_twiddle_buffer_wrapper<FFTZ_FLOAT>(FFTZ_VOID *twiddle_buffer,
                                                     FFTZ_INTP r, FFTZ_INTP m,
                                                     FFTZ_INTP register_width,
                                                     FFTZ_INTP load_multi_cols)
{
    compute_twiddle_buffer_float_wrapper(twiddle_buffer, r, m, register_width,
                                         load_multi_cols);
}

template <>
FFTZ_VOID compute_twiddle_buffer_wrapper<FFTZ_DOUBLE>(FFTZ_VOID *twiddle_buffer,
                                                      FFTZ_INTP r, FFTZ_INTP m,
                                                      FFTZ_INTP register_width,
                                                      FFTZ_INTP load_multi_cols)
{
    compute_twiddle_buffer_double_wrapper(twiddle_buffer, r, m, register_width,
                                          load_multi_cols);
}

// Map a twiddle-kernel ISA enum to the kernel's NUM_SETS_D (the
// register width in complex pairs that the producer needs to emit the
// correct linear/tile-packed layout). Falls back to AVX-512 width for
// any unknown variant (legacy default).
template <typename T>
FFTZ_INTP twiddle_kernel_register_width(aocl_fftz_kernel_type kt)
{
    FFTZ_INTP data_per_lane =
        static_cast<FFTZ_INTP>(64 / sizeof(T) / 2); // doubles: 4, floats: 8
    switch (kt)
    {
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512:
        return data_per_lane;
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256:
        return data_per_lane / 2;
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128:
        return data_per_lane / 4;
    case aocl_fftz_kernel_type::C2C_TWID_FWD_C:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_C:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_C:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_C:
        return 1;
    default:
        return data_per_lane;
    }
}

/**
 * @brief Get the global kernel table vector based on the kernel type
 *
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @return wrapper_kernel_fp_list
 */
wrapper_kernel_fp_list *get_kernel_table(FFTZ_UINT8 kernel_type)
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
 * @brief Check whether a kernel of the given radix and kernel type is
 * registered for the given data type
 *
 * Used to filter a shared (precision-independent) parameter table down to
 * the subset of entries actually available for `T`, so a radix registered
 * for only one of float/double does not need a separate table per type.
 *
 * @tparam T data type (float or double)
 * @param radix radix of the FFT kernel
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @return true if the kernel is registered for `T`, false otherwise
 */
template <class T>
bool kernel_exists_for_type(FFTZ_UINT32 radix, FFTZ_UINT8 kernel_type)
{
    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::C2C_TWID_FWD_C:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_C:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_C:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_C:
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128:
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256:
#endif
#ifdef ENABLE_AVX512
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512:
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512:
#endif
        // `dir` is ignored by every twiddle kernel kind (direction is
        // selected by kernel_type itself), so a single probe suffices.
        return get_twiddle_kernel<T>(radix, 0, kernel_type) != nullptr;

    default:
    {
        wrapper_kernel_fp_list *table = get_kernel_table(kernel_type);
        // Non-twiddle kernels may be looked up in either direction
        // (see AoclfftzKernelTestBase::run_kernel_test()), so both must
        // be registered for the kernel to be usable.
        return table != nullptr &&
               get_kernel<T>(table, 0, radix) != nullptr &&
               get_kernel<T>(table, 1, radix) != nullptr;
    }
    }
}

/**
 * @brief Filter a shared (precision-independent) kernel parameter table down
 * to the entries actually registered for the given data type
 *
 * @tparam T data type (float or double)
 * @param table the shared table to filter
 * @return the subset of `table` usable with `T`
 */
template <class T>
std::vector<aoclfftz_kernel_test_params_t> filter_kernels_for_type(
    const std::vector<aoclfftz_kernel_test_params_t> &table)
{
    std::vector<aoclfftz_kernel_test_params_t> out;
    out.reserve(table.size());
    for (const auto &entry : table)
    {
        if (kernel_exists_for_type<T>(std::get<0>(entry), std::get<1>(entry)))
        {
            out.push_back(entry);
        }
    }
    return out;
}

/**
 * @brief Get the kernel type as a string (used for naming the the test case)
 *
 * @param kernel_type kernel type (as an unsigned 8-bit integer)
 * @return std::string name of the kernel type
 */
std::string get_kernel_type_as_string(FFTZ_UINT8 kernel_type)
{
    switch (kernel_type)
    {
    case aocl_fftz_kernel_type::C2C_C:
        return "_C2C_C";
    case aocl_fftz_kernel_type::C2C_TWID_FWD_C:
        return "_C2C_TWID_FWD_C";
    case aocl_fftz_kernel_type::C2C_TWID_BWD_C:
        return "_C2C_TWID_BWD_C";
    case aocl_fftz_kernel_type::R2HC_C:
        return "_R2HC_C";
    case aocl_fftz_kernel_type::R2HCF_C:
        return "_R2HCF_C";
    case aocl_fftz_kernel_type::C2C_TWID_R2C_C:
        return "_C2C_TWID_R2C_C";
    case aocl_fftz_kernel_type::C2C_TWID_C2R_C:
        return "_C2C_TWID_C2R_C";
#ifdef ENABLE_AVX128
    case aocl_fftz_kernel_type::C2C_AVX128:
        return "_C2C_AVX128";
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128:
        return "_C2C_TWID_FWD_AVX128";
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128:
        return "_C2C_TWID_BWD_AVX128";
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128:
        return "_C2C_TWID_R2C_AVX128";
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128:
        return "_C2C_TWID_C2R_AVX128";
    case aocl_fftz_kernel_type::R2HC_AVX128:
        return "_R2HC_AVX128";
    case aocl_fftz_kernel_type::R2HCF_AVX128:
        return "_R2HCF_AVX128";
#endif
#ifdef ENABLE_AVX256
    case aocl_fftz_kernel_type::C2C_AVX256:
        return "_C2C_AVX256";
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256:
        return "_C2C_TWID_FWD_AVX256";
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256:
        return "_C2C_TWID_BWD_AVX256";
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256:
        return "_C2C_TWID_R2C_AVX256";
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256:
        return "_C2C_TWID_C2R_AVX256";
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
    case aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512:
        return "_C2C_TWID_FWD_AVX512";
    case aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512:
        return "_C2C_TWID_BWD_AVX512";
    case aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512:
        return "_C2C_TWID_R2C_AVX512";
    case aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512:
        return "_C2C_TWID_C2R_AVX512";
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
bool is_fused_kernel(FFTZ_UINT8 kernel_type)
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
 * @param in_stride input data stride
 * @param out_stride output data stride
 * @param v_in_stride vectorized input stride
 * @param v_out_stride vectorized output stride
 */
template <class T>
void permuted_copy(T *in, T *out, FFTZ_INTP n, FFTZ_INTP size,
                   FFTZ_INTP in_stride, FFTZ_INTP out_stride,
                   FFTZ_INTP v_in_stride, FFTZ_INTP v_out_stride)
{
    if (typeid(T) == typeid(FFTZ_FLOAT32))
    {
        permuted_copy_c_fp32_wrapper((FFTZ_FLOAT32 *)in, (FFTZ_FLOAT32 *)out, n,
                                     size, in_stride, out_stride, v_in_stride,
                                     v_out_stride);
    }
    else if (typeid(T) == typeid(FFTZ_FLOAT64))
    {
        permuted_copy_c_fp64_wrapper((FFTZ_FLOAT64 *)in, (FFTZ_FLOAT64 *)out, n,
                                     size, in_stride, out_stride, v_in_stride,
                                     v_out_stride);
    }
}

/**
 * @brief An utility function to print the complex array
 *
 * @tparam T array type (float32 or float64)
 * @param arr complex array
 * @param n size of the array (actual size = size * 2 [real + complex])
 */
template <class T> inline void print_carray(T *arr, FFTZ_INTP n)
{
    if (typeid(T) == typeid(FFTZ_FLOAT32))
    {
        PRINT_CARRAY_FP32((FFTZ_FLOAT32 *)arr, n);
    }
    else if (typeid(T) == typeid(FFTZ_FLOAT64))
    {
        PRINT_CARRAY_FP64((FFTZ_FLOAT64 *)arr, n);
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
    if (typeid(T) == typeid(FFTZ_FLOAT32))
    {
        PRINT_COMPLEX_FP32((FFTZ_FLOAT32 *)val);
    }
    else if (typeid(T) == typeid(FFTZ_FLOAT64))
    {
        PRINT_COMPLEX_FP64((FFTZ_FLOAT64 *)val);
    }
}

#endif // AOCLFFTZ_KERNEL_GTEST_UTILS_H
