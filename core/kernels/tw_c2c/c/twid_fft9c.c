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

/** @file twid_fft9c.c
 *
 *  @brief Twiddle Radix-9 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-9 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 8 * (4 muls, 2 adds, 2 movs [loads])
//              = (32, 16, 16)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 72, 96, 52, 0, 0},
                                                     {0, 72, 96, 52, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft9c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft9c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    DOUBLE twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i;
        DOUBLE v1rr, v1ii, tv1rr, tv1ii, tv2rr, tv2ii, cv1rr, cv1ii, cv2rr,
            cv2ii, cv3rr, cv3ii, cv4rr, cv4ii, cv5rr, cv5ii, cv6rr, cv6ii,
            cv7rr, cv7ii, cv8rr, cv8ii, cv9rr, cv9ii;
        DOUBLE tv1rr1, tv1rr2, tv1rr3, tv1ii1, tv1ii2, tv1ii3;
        DOUBLE tv2rr1, tv2rr2, tv2rr3, tv2ii1, tv2ii2, tv2ii3;
        DOUBLE tv3rr, tv3ii, tv3ir, tv3ir1, tv4ir, tv4ir1;

        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_r[in_strides[5]];
        DOUBLE v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_r[in_strides[6]];
        DOUBLE v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_r[in_strides[7]];
        DOUBLE v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        DOUBLE v9r_t = in_r[in_strides[8]];
        DOUBLE v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input points - Real : 0, 3, 6
        v1rr = v4r + v7r;
        tv1rr3 = CRTM_9_8 * (v7r - v4r);
        cv1rr = v1r + v1rr;
        tv2rr3 = v1r - (CRTM_9_7 * v1rr);

        // Input points - Real : 1, 4, 7
        v1rr = v5r + v8r;
        tv1rr1 = CRTM_9_8 * (v8r - v5r);
        cv4rr = v2r + v1rr;
        tv2rr1 = v2r - (CRTM_9_7 * v1rr);

        // Input points - Real : 2, 5, 8
        v1rr = v6r + v9r;
        tv1rr2 = CRTM_9_8 * (v9r - v6r);
        cv7rr = v3r + v1rr;
        tv2rr2 = v3r - (CRTM_9_7 * v1rr);

        // Input points - Imag : 0, 3, 6
        v1ii = v4i + v7i;
        tv1ii3 = CRTM_9_8 * (v4i - v7i);
        cv1ii = v1i + v1ii;
        tv2ii3 = v1i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 1, 4, 7
        v1ii = v5i + v8i;
        tv1ii1 = CRTM_9_8 * (v5i - v8i);
        cv4ii = v2i + v1ii;
        tv2ii1 = v2i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 2, 5, 8
        v1ii = v6i + v9i;
        tv1ii2 = CRTM_9_8 * (v6i - v9i);
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

static VOID twid_fft9c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_9_1 = +0.939692620785908384054109277324731469936208134f;
    const FLOAT CRTM_9_2 = +0.342020143325668733044099614682259580763083368f;
    const FLOAT CRTM_9_3 = +0.984807753012208059366743024589523013670643252f;
    const FLOAT CRTM_9_4 = +0.173648177666930348851716626769314796000375677f;
    const FLOAT CRTM_9_5 = +0.642787609686539326322643409907263432907559884f;
    const FLOAT CRTM_9_6 = +0.766044443118978035202392650555416673935832457f;
    const FLOAT CRTM_9_7 = +0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_9_8 = +0.866025403784438646763723170752936183471402627f;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    FLOAT twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i;
        FLOAT v1rr, v1ii, tv1rr, tv1ii, tv2rr, tv2ii, cv1rr, cv1ii, cv2rr,
            cv2ii, cv3rr, cv3ii, cv4rr, cv4ii, cv5rr, cv5ii, cv6rr, cv6ii,
            cv7rr, cv7ii, cv8rr, cv8ii, cv9rr, cv9ii;
        FLOAT tv1rr1, tv1rr2, tv1rr3, tv1ii1, tv1ii2, tv1ii3;
        FLOAT tv2rr1, tv2rr2, tv2rr3, tv2ii1, tv2ii2, tv2ii3;
        FLOAT tv3rr, tv3ii, tv3ir, tv3ir1, tv4ir, tv4ir1;

        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FLOAT v6r_t = in_r[in_strides[5]];
        FLOAT v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FLOAT v7r_t = in_r[in_strides[6]];
        FLOAT v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FLOAT v8r_t = in_r[in_strides[7]];
        FLOAT v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FLOAT v9r_t = in_r[in_strides[8]];
        FLOAT v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input points - Real : 0, 3, 6
        v1rr = v4r + v7r;
        tv1rr3 = CRTM_9_8 * (v7r - v4r);
        cv1rr = v1r + v1rr;
        tv2rr3 = v1r - (CRTM_9_7 * v1rr);

        // Input points - Real : 1, 4, 7
        v1rr = v5r + v8r;
        tv1rr1 = CRTM_9_8 * (v8r - v5r);
        cv4rr = v2r + v1rr;
        tv2rr1 = v2r - (CRTM_9_7 * v1rr);

        // Input points - Real : 2, 5, 8
        v1rr = v6r + v9r;
        tv1rr2 = CRTM_9_8 * (v9r - v6r);
        cv7rr = v3r + v1rr;
        tv2rr2 = v3r - (CRTM_9_7 * v1rr);

        // Input points - Imag : 0, 3, 6
        v1ii = v4i + v7i;
        tv1ii3 = CRTM_9_8 * (v4i - v7i);
        cv1ii = v1i + v1ii;
        tv2ii3 = v1i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 1, 4, 7
        v1ii = v5i + v8i;
        tv1ii1 = CRTM_9_8 * (v5i - v8i);
        cv4ii = v2i + v1ii;
        tv2ii1 = v2i - (CRTM_9_7 * v1ii);

        // Input points - Imag : 2, 5, 8
        v1ii = v6i + v9i;
        tv1ii2 = CRTM_9_8 * (v6i - v9i);
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

kfft_ register_kernel_twid_fft9c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft9c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft9c_fp64;
    }
    else
    {
        return NULL;
    }
}
