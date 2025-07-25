/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file fft9c.c
 *
 *  @brief Radix-9 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-9 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author Prasandh Sankarankutty
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 40, 80, 36, 0, 0},
                                                     {0, 40, 80, 36, 0, 0}};

ops_cycles_t get_ops_cnt_fft9c(UINT8 precision, UINT8 direction)
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

static VOID fft9c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_9_1 = +0.939692620785908384054109277324731469936208134;
    const DOUBLE CRTM_9_2 = +0.342020143325668733044099614682259580763083368;
    const DOUBLE CRTM_9_3 = +0.984807753012208059366743024589523013670643252;
    const DOUBLE CRTM_9_4 = +0.173648177666930348851716626769314796000375677;
    const DOUBLE CRTM_9_5 = +0.642787609686539326322643409907263432907559884;
    const DOUBLE CRTM_9_6 = +0.766044443118978035202392650555416673935832457;
    const DOUBLE CRTM_9_7 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_9_8 = +0.866025403784438646763723170752936183471402627;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v1rr, v1ii, tv1rr, tv2rr, tv3rr, tv1ii,
            tv2ii, tv3ii, tv3ir, tv4ir, tv3ir1, tv4ir1, tv1rr1, tv2rr1, tv1ii1,
            tv2ii1, tv1rr2, tv2rr2, tv1ii2, tv2ii2, tv1rr3, tv2rr3, tv1ii3,
            tv2ii3, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii, cv4rr, cv4ii,
            cv5rr, cv5ii, cv6rr, cv6ii, cv7rr, cv7ii, cv8rr, cv8ii, cv9rr,
            cv9ii;

        // Input points - Real : 0, 3, 6
        v4r = in_r[in_strides[3]];
        v7r = in_r[in_strides[6]];
        v1rr = v4r + v7r;
        tv1rr3 = CRTM_9_8 * (v7r - v4r);
        v1r = *in_r;
        cv1rr = v1r + v1rr;
        tv2rr3 = v1r - (CRTM_9_7 * v1rr);

        // Input points - Real : 1, 4, 7
        v5r = in_r[in_strides[4]];
        v8r = in_r[in_strides[7]];
        v1rr = v5r + v8r;
        tv1rr1 = CRTM_9_8 * (v8r - v5r);
        v2r = in_r[in_strides[1]];
        cv4rr = v2r + v1rr;
        tv2rr1 = v2r - (CRTM_9_7 * v1rr);

        // Input points - Real : 2, 5, 8
        v6r = in_r[in_strides[5]];
        v9r = in_r[in_strides[8]];
        v1rr = v6r + v9r;
        tv1rr2 = CRTM_9_8 * (v9r - v6r);
        v3r = in_r[in_strides[2]];
        cv7rr = v3r + v1rr;
        tv2rr2 = v3r - (CRTM_9_7 * v1rr);

        // Input points - Imag : 0, 3, 6
        v4i = in_i[in_strides[3]];
        v7i = in_i[in_strides[6]];
        v1ii = v4i + v7i;
        tv1ii3 = CRTM_9_8 * (v4i - v7i);
        v1i = *in_i;
        cv1ii = v1i + v1ii;
        tv2ii3 = v1i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 1, 4, 7
        v5i = in_i[in_strides[4]];
        v8i = in_i[in_strides[7]];
        v1ii = v5i + v8i;
        tv1ii1 = CRTM_9_8 * (v5i - v8i);
        v2i = in_i[in_strides[1]];
        cv4ii = v2i + v1ii;
        tv2ii1 = v2i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 2, 5, 8
        v6i = in_i[in_strides[5]];
        v9i = in_i[in_strides[8]];
        v1ii = v6i + v9i;
        tv1ii2 = CRTM_9_8 * (v6i - v9i);
        v3i = in_i[in_strides[2]];
        cv7ii = v3i + v1ii;
        tv2ii2 = v3i - (CRTM_9_7 * v1ii);

        // Output point : 0
        v1rr = cv4rr + cv7rr;
        tv1rr = CRTM_9_8 * (cv7rr - cv4rr);
        v1ii = cv4ii + cv7ii;
        tv1ii = CRTM_9_8 * (cv4ii - cv7ii);

        *out_r = cv1rr + v1rr;
        tv2rr = cv1rr - (CRTM_9_7 * v1rr);
        *out_i = cv1ii + v1ii;
        tv2ii = cv1ii - (CRTM_9_7 * v1ii);

        // Output points : 3, 6
        out_r[out_strides[3]] = tv2rr + tv1ii;
        out_r[out_strides[6]] = tv2rr - tv1ii;

        out_i[out_strides[3]] = tv2ii + tv1rr;
        out_i[out_strides[6]] = tv2ii - tv1rr;

        cv5rr = tv2rr1 + tv1ii1;
        cv6rr = tv2rr1 - tv1ii1;
        cv5ii = tv2ii1 + tv1rr1;
        cv6ii = tv2ii1 - tv1rr1;
        tv3ir = (CRTM_9_6 * cv5rr) + (CRTM_9_5 * cv5ii);
        tv3ir1 = (CRTM_9_6 * cv5ii) - (CRTM_9_5 * cv5rr);

        cv8rr = tv2rr2 + tv1ii2;
        cv9rr = tv2rr2 - tv1ii2;
        cv8ii = tv2ii2 + tv1rr2;
        cv9ii = tv2ii2 - tv1rr2;
        tv4ir = (CRTM_9_4 * cv8rr) + (CRTM_9_3 * cv8ii);
        tv4ir1 = (CRTM_9_4 * cv8ii) - (CRTM_9_3 * cv8rr);

        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 - tv4ir1);
        tv1ii = tv3ir1 + tv4ir1;
        cv3rr = tv2rr3 - tv1ii3;
        cv3ii = tv2ii3 - tv1rr3;
        cv2rr = tv2rr3 + tv1ii3;
        cv2ii = tv2ii3 + tv1rr3;

        // Output point : 1
        out_r[out_strides[1]] = cv2rr + tv1rr;
        out_i[out_strides[1]] = cv2ii + tv1ii;

        tv2rr = cv2rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv2ii - (CRTM_9_7 * tv1ii);

        // Output point : 4
        out_r[out_strides[4]] = tv2rr + tv3ii;
        out_i[out_strides[4]] = tv2ii + tv3rr;

        // Output point : 7
        out_r[out_strides[7]] = tv2rr - tv3ii;
        out_i[out_strides[7]] = tv2ii - tv3rr;

        tv3ir = (CRTM_9_4 * cv6rr) + (CRTM_9_3 * cv6ii);
        tv3ir1 = (CRTM_9_4 * cv6ii) - (CRTM_9_3 * cv6rr);
        tv4ir = (CRTM_9_2 * cv9ii) - (CRTM_9_1 * cv9rr);
        tv4ir1 = (CRTM_9_1 * cv9ii) + (CRTM_9_2 * cv9rr);
        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 + tv4ir1);
        tv1ii = tv3ir1 - tv4ir1;

        // Output point : 2
        out_r[out_strides[2]] = cv3rr + tv1rr;
        out_i[out_strides[2]] = cv3ii + tv1ii;

        tv2rr = cv3rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv3ii - (CRTM_9_7 * tv1ii);

        // Output point : 5
        out_r[out_strides[5]] = tv2rr + tv3ii;
        out_i[out_strides[5]] = tv2ii + tv3rr;

        // Output point : 8
        out_r[out_strides[8]] = tv2rr - tv3ii;
        out_i[out_strides[8]] = tv2ii - tv3rr;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft9c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_9_1 = +0.939692620785908384054109277324731469936208134;
    const FLOAT CRTM_9_2 = +0.342020143325668733044099614682259580763083368;
    const FLOAT CRTM_9_3 = +0.984807753012208059366743024589523013670643252;
    const FLOAT CRTM_9_4 = +0.173648177666930348851716626769314796000375677;
    const FLOAT CRTM_9_5 = +0.642787609686539326322643409907263432907559884;
    const FLOAT CRTM_9_6 = +0.766044443118978035202392650555416673935832457;
    const FLOAT CRTM_9_7 = +0.500000000000000000000000000000000000000000000;
    const FLOAT CRTM_9_8 = +0.866025403784438646763723170752936183471402627;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v1rr, v1ii, tv1rr, tv2rr, tv3rr, tv1ii,
            tv2ii, tv3ii, tv3ir, tv4ir, tv3ir1, tv4ir1, tv1rr1, tv2rr1, tv1ii1,
            tv2ii1, tv1rr2, tv2rr2, tv1ii2, tv2ii2, tv1rr3, tv2rr3, tv1ii3,
            tv2ii3, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii, cv4rr, cv4ii,
            cv5rr, cv5ii, cv6rr, cv6ii, cv7rr, cv7ii, cv8rr, cv8ii, cv9rr,
            cv9ii;

        // Input points - Real : 0, 3, 6
        v4r = in_r[in_strides[3]];
        v7r = in_r[in_strides[6]];
        v1rr = v4r + v7r;
        tv1rr3 = CRTM_9_8 * (v7r - v4r);
        v1r = *in_r;
        cv1rr = v1r + v1rr;
        tv2rr3 = v1r - (CRTM_9_7 * v1rr);

        // Input points - Real : 1, 4, 7
        v5r = in_r[in_strides[4]];
        v8r = in_r[in_strides[7]];
        v1rr = v5r + v8r;
        tv1rr1 = CRTM_9_8 * (v8r - v5r);
        v2r = in_r[in_strides[1]];
        cv4rr = v2r + v1rr;
        tv2rr1 = v2r - (CRTM_9_7 * v1rr);

        // Input points - Real : 2, 5, 8
        v6r = in_r[in_strides[5]];
        v9r = in_r[in_strides[8]];
        v1rr = v6r + v9r;
        tv1rr2 = CRTM_9_8 * (v9r - v6r);
        v3r = in_r[in_strides[2]];
        cv7rr = v3r + v1rr;
        tv2rr2 = v3r - (CRTM_9_7 * v1rr);

        // Input points - Imag : 0, 3, 6
        v4i = in_i[in_strides[3]];
        v7i = in_i[in_strides[6]];
        v1ii = v4i + v7i;
        tv1ii3 = CRTM_9_8 * (v4i - v7i);
        v1i = *in_i;
        cv1ii = v1i + v1ii;
        tv2ii3 = v1i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 1, 4, 7
        v5i = in_i[in_strides[4]];
        v8i = in_i[in_strides[7]];
        v1ii = v5i + v8i;
        tv1ii1 = CRTM_9_8 * (v5i - v8i);
        v2i = in_i[in_strides[1]];
        cv4ii = v2i + v1ii;
        tv2ii1 = v2i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 2, 5, 8
        v6i = in_i[in_strides[5]];
        v9i = in_i[in_strides[8]];
        v1ii = v6i + v9i;
        tv1ii2 = CRTM_9_8 * (v6i - v9i);
        v3i = in_i[in_strides[2]];
        cv7ii = v3i + v1ii;
        tv2ii2 = v3i - (CRTM_9_7 * v1ii);

        // Output point : 0
        v1rr = cv4rr + cv7rr;
        tv1rr = CRTM_9_8 * (cv7rr - cv4rr);
        v1ii = cv4ii + cv7ii;
        tv1ii = CRTM_9_8 * (cv4ii - cv7ii);

        *out_r = cv1rr + v1rr;
        tv2rr = cv1rr - (CRTM_9_7 * v1rr);
        *out_i = cv1ii + v1ii;
        tv2ii = cv1ii - (CRTM_9_7 * v1ii);

        // Output points : 3, 6
        out_r[out_strides[3]] = tv2rr + tv1ii;
        out_r[out_strides[6]] = tv2rr - tv1ii;

        out_i[out_strides[3]] = tv2ii + tv1rr;
        out_i[out_strides[6]] = tv2ii - tv1rr;

        cv5rr = tv2rr1 + tv1ii1;
        cv6rr = tv2rr1 - tv1ii1;
        cv5ii = tv2ii1 + tv1rr1;
        cv6ii = tv2ii1 - tv1rr1;
        tv3ir = (CRTM_9_6 * cv5rr) + (CRTM_9_5 * cv5ii);
        tv3ir1 = (CRTM_9_6 * cv5ii) - (CRTM_9_5 * cv5rr);

        cv8rr = tv2rr2 + tv1ii2;
        cv9rr = tv2rr2 - tv1ii2;
        cv8ii = tv2ii2 + tv1rr2;
        cv9ii = tv2ii2 - tv1rr2;
        tv4ir = (CRTM_9_4 * cv8rr) + (CRTM_9_3 * cv8ii);
        tv4ir1 = (CRTM_9_4 * cv8ii) - (CRTM_9_3 * cv8rr);

        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 - tv4ir1);
        tv1ii = tv3ir1 + tv4ir1;
        cv3rr = tv2rr3 - tv1ii3;
        cv3ii = tv2ii3 - tv1rr3;
        cv2rr = tv2rr3 + tv1ii3;
        cv2ii = tv2ii3 + tv1rr3;

        // Output point : 1
        out_r[out_strides[1]] = cv2rr + tv1rr;
        out_i[out_strides[1]] = cv2ii + tv1ii;

        tv2rr = cv2rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv2ii - (CRTM_9_7 * tv1ii);

        // Output point : 4
        out_r[out_strides[4]] = tv2rr + tv3ii;
        out_i[out_strides[4]] = tv2ii + tv3rr;

        // Output point : 7
        out_r[out_strides[7]] = tv2rr - tv3ii;
        out_i[out_strides[7]] = tv2ii - tv3rr;

        tv3ir = (CRTM_9_4 * cv6rr) + (CRTM_9_3 * cv6ii);
        tv3ir1 = (CRTM_9_4 * cv6ii) - (CRTM_9_3 * cv6rr);
        tv4ir = (CRTM_9_2 * cv9ii) - (CRTM_9_1 * cv9rr);
        tv4ir1 = (CRTM_9_1 * cv9ii) + (CRTM_9_2 * cv9rr);
        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 + tv4ir1);
        tv1ii = tv3ir1 - tv4ir1;

        // Output point : 2
        out_r[out_strides[2]] = cv3rr + tv1rr;
        out_i[out_strides[2]] = cv3ii + tv1ii;

        tv2rr = cv3rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv3ii - (CRTM_9_7 * tv1ii);

        // Output point : 5
        out_r[out_strides[5]] = tv2rr + tv3ii;
        out_i[out_strides[5]] = tv2ii + tv3rr;

        // Output point : 8
        out_r[out_strides[8]] = tv2rr - tv3ii;
        out_i[out_strides[8]] = tv2ii - tv3rr;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft9c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft9c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft9c_fp64;
    }
    else
    {
        return NULL;
    }
}
