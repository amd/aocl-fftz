// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_r2c_fft6c.c
 *
 *  @brief R2C fused twiddle (forward twiddle + conjugate output) Radix-6 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-6 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 6

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 46, 34, 0, 0},
                                                     {0, 28, 46, 34, 0, 0}};

ops_cycles_t get_ops_cnt_twid_r2c_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

static FFTZ_VOID twid_r2c_fft6c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        +0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_6_2 =
        +0.866025403784438646763723170752936183471402627;

    FFTZ_DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_DOUBLE twr, twi;

    in_h1_r = (FFTZ_DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
            tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
            tv2ii, tv1ii;
        // Input point 1: x(0)
        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h2_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h2_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h2_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h2_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h2_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_h1_r = av1rr + av8rr;
        *out_h1_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_h2_r[out_strides[3]] = av4rr + av7rr;
        out_h2_i[out_strides[3]] = -(av4ii + av7ii);

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_h1_r[out_strides[1]] = tv1rr + tv1ii;
        out_h1_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_h2_r[out_strides[5]] = tv1rr - tv1ii;
        out_h2_i[out_strides[5]] = -(tv2ii + tv2rr);

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_h1_r[out_strides[2]] = tv1rr + tv1ii;
        out_h1_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_h2_r[out_strides[4]] = tv1rr - tv1ii;
        out_h2_i[out_strides[4]] = -(tv2ii - tv2rr);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID twid_r2c_fft6c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        +0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_6_2 =
        +0.866025403784438646763723170752936183471402627f;

    FFTZ_FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_FLOAT twr, twi;

    in_h1_r = (FFTZ_FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
            tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
            tv2ii, tv1ii;
        // Input point 1: x(0)
        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h2_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h2_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h2_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h2_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h2_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_h1_r = av1rr + av8rr;
        *out_h1_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_h2_r[out_strides[3]] = av4rr + av7rr;
        out_h2_i[out_strides[3]] = -(av4ii + av7ii);

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_h1_r[out_strides[1]] = tv1rr + tv1ii;
        out_h1_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_h2_r[out_strides[5]] = tv1rr - tv1ii;
        out_h2_i[out_strides[5]] = -(tv2ii + tv2rr);

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_h1_r[out_strides[2]] = tv1rr + tv1ii;
        out_h1_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_h2_r[out_strides[4]] = tv1rr - tv1ii;
        out_h2_i[out_strides[4]] = -(tv2ii - tv2rr);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_r2c_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_r2c_fft6c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_r2c_fft6c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
