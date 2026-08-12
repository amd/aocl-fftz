// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

ops_cycles_t get_ops_cnt_fft9c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft9c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_9_1 =
        +0.939692620785908384054109277324731469936208134;
    const FFTZ_DOUBLE CRTM_9_2 =
        +0.342020143325668733044099614682259580763083368;
    const FFTZ_DOUBLE CRTM_9_3 =
        +0.984807753012208059366743024589523013670643252;
    const FFTZ_DOUBLE CRTM_9_4 =
        +0.173648177666930348851716626769314796000375677;
    const FFTZ_DOUBLE CRTM_9_5 =
        +0.642787609686539326322643409907263432907559884;
    const FFTZ_DOUBLE CRTM_9_6 =
        +0.766044443118978035202392650555416673935832457;
    const FFTZ_DOUBLE CRTM_9_7 =
        +0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_9_8 =
        +0.866025403784438646763723170752936183471402627;

    FFTZ_DOUBLE *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_DOUBLE *)in_imag;
        in_i = (FFTZ_DOUBLE *)in_real;
        out_r = (FFTZ_DOUBLE *)out_imag;
        out_i = (FFTZ_DOUBLE *)out_real;
    }
    else
    {
        in_r = (FFTZ_DOUBLE *)in_real;
        in_i = (FFTZ_DOUBLE *)in_imag;
        out_r = (FFTZ_DOUBLE *)out_real;
        out_i = (FFTZ_DOUBLE *)out_imag;
    }

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v9r, v9i, v1rr, v1ii, tv1rr, tv2rr, tv3rr,
            tv1ii, tv2ii, tv3ii, tv3ir, tv4ir, tv3ir1, tv4ir1, tv1rr1, tv2rr1,
            tv1ii1, tv2ii1, tv1rr2, tv2rr2, tv1ii2, tv2ii2, tv1rr3, tv2rr3,
            tv1ii3, tv2ii3, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii, cv4rr,
            cv4ii, cv5rr, cv5ii, cv6rr, cv6ii, cv7rr, cv7ii, cv8rr, cv8ii,
            cv9rr, cv9ii;

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
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft9c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_9_1 =
        +0.939692620785908384054109277324731469936208134;
    const FFTZ_FLOAT CRTM_9_2 =
        +0.342020143325668733044099614682259580763083368;
    const FFTZ_FLOAT CRTM_9_3 =
        +0.984807753012208059366743024589523013670643252;
    const FFTZ_FLOAT CRTM_9_4 =
        +0.173648177666930348851716626769314796000375677;
    const FFTZ_FLOAT CRTM_9_5 =
        +0.642787609686539326322643409907263432907559884;
    const FFTZ_FLOAT CRTM_9_6 =
        +0.766044443118978035202392650555416673935832457;
    const FFTZ_FLOAT CRTM_9_7 =
        +0.500000000000000000000000000000000000000000000;
    const FFTZ_FLOAT CRTM_9_8 =
        +0.866025403784438646763723170752936183471402627;

    FFTZ_FLOAT *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_FLOAT *)in_imag;
        in_i = (FFTZ_FLOAT *)in_real;
        out_r = (FFTZ_FLOAT *)out_imag;
        out_i = (FFTZ_FLOAT *)out_real;
    }
    else
    {
        in_r = (FFTZ_FLOAT *)in_real;
        in_i = (FFTZ_FLOAT *)in_imag;
        out_r = (FFTZ_FLOAT *)out_real;
        out_i = (FFTZ_FLOAT *)out_imag;
    }

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v9r, v9i, v1rr, v1ii, tv1rr, tv2rr, tv3rr,
            tv1ii, tv2ii, tv3ii, tv3ir, tv4ir, tv3ir1, tv4ir1, tv1rr1, tv2rr1,
            tv1ii1, tv2ii1, tv1rr2, tv2rr2, tv1ii2, tv2ii2, tv1rr3, tv2rr3,
            tv1ii3, tv2ii3, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii, cv4rr,
            cv4ii, cv5rr, cv5ii, cv6rr, cv6ii, cv7rr, cv7ii, cv8rr, cv8ii,
            cv9rr, cv9ii;

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
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft9c(FFTZ_UINT8 precision,
                            FFTZ_UINT8 direction /* unused */)
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
