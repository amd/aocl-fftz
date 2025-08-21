/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file twid_fft6c.c
 *
 *  @brief Twiddle Radix-6 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-6 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 5 * (4 muls, 2 adds, 2 movs [loads])
//              = (20, 10, 10)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 46, 34, 0, 0},
                                                     {0, 28, 46, 34, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft6c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft6c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_6_1 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_6_2 = +0.866025403784438646763723170752936183471402627;

    DOUBLE *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    DOUBLE twr, twi;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (DOUBLE *)in_imag;
        in_i = (DOUBLE *)in_real;
        out_r = (DOUBLE *)out_imag;
        out_i = (DOUBLE *)out_real;
    }
    else
    {
        in_r = (DOUBLE *)in_real;
        in_i = (DOUBLE *)in_imag;
        out_r = (DOUBLE *)out_real;
        out_i = (DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
            tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
            tv2ii, tv1ii;
        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_r[in_strides[5]];
        DOUBLE v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
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
        *out_r = av1rr + av8rr;
        *out_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_r[out_strides[3]] = av4rr + av7rr;
        out_i[out_strides[3]] = av4ii + av7ii;

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = tv1rr + tv1ii;
        out_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_r[out_strides[5]] = tv1rr - tv1ii;
        out_i[out_strides[5]] = tv2ii + tv2rr;

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_r[out_strides[2]] = tv1rr + tv1ii;
        out_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_r[out_strides[4]] = tv1rr - tv1ii;
        out_i[out_strides[4]] = tv2ii - tv2rr;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft6c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_6_1 = +0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_6_2 = +0.866025403784438646763723170752936183471402627f;

    FLOAT *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    FLOAT twr, twi;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FLOAT *)in_imag;
        in_i = (FLOAT *)in_real;
        out_r = (FLOAT *)out_imag;
        out_i = (FLOAT *)out_real;
    }
    else
    {
        in_r = (FLOAT *)in_real;
        in_i = (FLOAT *)in_imag;
        out_r = (FLOAT *)out_real;
        out_i = (FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, av1rr,
            av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr, tv2rr,
            av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii, tv2ii,
            tv1ii;
        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FLOAT v6r_t = in_r[in_strides[5]];
        FLOAT v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
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
        *out_r = av1rr + av8rr;
        *out_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_r[out_strides[3]] = av4rr + av7rr;
        out_i[out_strides[3]] = av4ii + av7ii;

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = tv1rr + tv1ii;
        out_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_r[out_strides[5]] = tv1rr - tv1ii;
        out_i[out_strides[5]] = tv2ii + tv2rr;

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_r[out_strides[2]] = tv1rr + tv1ii;
        out_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_r[out_strides[4]] = tv1rr - tv1ii;
        out_i[out_strides[4]] = tv2ii - tv2rr;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft6c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft6c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft6c_fp64;
    }
    else
    {
        return NULL;
    }
}
