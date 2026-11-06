// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fwd_fft9c.c
 *
 *  @brief Forward-only twiddle Radix-9 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-9 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 9

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 72, 96, 52, 0, 0},
                                                     {0, 72, 96, 52, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fwd_fft9c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_fwd_fft9c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
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

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v9r, v9i;
        FFTZ_DOUBLE v1rr, v1ii, tv1rr, tv1ii, tv2rr, tv2ii, cv1rr, cv1ii, cv2rr,
            cv2ii, cv3rr, cv3ii, cv4rr, cv4ii, cv5rr, cv5ii, cv6rr, cv6ii,
            cv7rr, cv7ii, cv8rr, cv8ii, cv9rr, cv9ii;
        FFTZ_DOUBLE tv1rr1, tv1rr2, tv1rr3, tv1ii1, tv1ii2, tv1ii3;
        FFTZ_DOUBLE tv2rr1, tv2rr2, tv2rr3, tv2ii1, tv2ii2, tv2ii3;
        FFTZ_DOUBLE tv3rr, tv3ii, tv3ir, tv3ir1, tv4ir, tv4ir1;

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
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h1_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h1_i[in_strides[4]];
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

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_DOUBLE v9r_t = in_h2_r[in_strides[8]];
        FFTZ_DOUBLE v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
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

        *out_h1_r = cv1rr + v1rr;
        tv2rr = cv1rr - (CRTM_9_7 * v1rr);
        *out_h1_i = cv1ii + v1ii;
        tv2ii = cv1ii - (CRTM_9_7 * v1ii);

        // Output points : 3, 6
        out_h1_r[out_strides[3]] = tv2rr + tv1ii;
        out_h2_r[out_strides[6]] = tv2rr - tv1ii;

        out_h1_i[out_strides[3]] = tv2ii + tv1rr;
        out_h2_i[out_strides[6]] = tv2ii - tv1rr;

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
        out_h1_r[out_strides[1]] = cv2rr + tv1rr;
        out_h1_i[out_strides[1]] = cv2ii + tv1ii;

        tv2rr = cv2rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv2ii - (CRTM_9_7 * tv1ii);

        // Output point : 4
        out_h1_r[out_strides[4]] = tv2rr + tv3ii;
        out_h1_i[out_strides[4]] = tv2ii + tv3rr;

        // Output point : 7
        out_h2_r[out_strides[7]] = tv2rr - tv3ii;
        out_h2_i[out_strides[7]] = tv2ii - tv3rr;

        tv3ir = (CRTM_9_4 * cv6rr) + (CRTM_9_3 * cv6ii);
        tv3ir1 = (CRTM_9_4 * cv6ii) - (CRTM_9_3 * cv6rr);
        tv4ir = (CRTM_9_2 * cv9ii) - (CRTM_9_1 * cv9rr);
        tv4ir1 = (CRTM_9_1 * cv9ii) + (CRTM_9_2 * cv9rr);
        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 + tv4ir1);
        tv1ii = tv3ir1 - tv4ir1;

        // Output point : 2
        out_h1_r[out_strides[2]] = cv3rr + tv1rr;
        out_h1_i[out_strides[2]] = cv3ii + tv1ii;

        tv2rr = cv3rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv3ii - (CRTM_9_7 * tv1ii);

        // Output point : 5
        out_h2_r[out_strides[5]] = tv2rr + tv3ii;
        out_h2_i[out_strides[5]] = tv2ii + tv3rr;

        // Output point : 8
        out_h2_r[out_strides[8]] = tv2rr - tv3ii;
        out_h2_i[out_strides[8]] = tv2ii - tv3rr;

        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID twid_fwd_fft9c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_9_1 =
        +0.939692620785908384054109277324731469936208134f;
    const FFTZ_FLOAT CRTM_9_2 =
        +0.342020143325668733044099614682259580763083368f;
    const FFTZ_FLOAT CRTM_9_3 =
        +0.984807753012208059366743024589523013670643252f;
    const FFTZ_FLOAT CRTM_9_4 =
        +0.173648177666930348851716626769314796000375677f;
    const FFTZ_FLOAT CRTM_9_5 =
        +0.642787609686539326322643409907263432907559884f;
    const FFTZ_FLOAT CRTM_9_6 =
        +0.766044443118978035202392650555416673935832457f;
    const FFTZ_FLOAT CRTM_9_7 =
        +0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_9_8 =
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

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v9r, v9i;
        FFTZ_FLOAT v1rr, v1ii, tv1rr, tv1ii, tv2rr, tv2ii, cv1rr, cv1ii, cv2rr,
            cv2ii, cv3rr, cv3ii, cv4rr, cv4ii, cv5rr, cv5ii, cv6rr, cv6ii,
            cv7rr, cv7ii, cv8rr, cv8ii, cv9rr, cv9ii;
        FFTZ_FLOAT tv1rr1, tv1rr2, tv1rr3, tv1ii1, tv1ii2, tv1ii3;
        FFTZ_FLOAT tv2rr1, tv2rr2, tv2rr3, tv2ii1, tv2ii2, tv2ii3;
        FFTZ_FLOAT tv3rr, tv3ii, tv3ir, tv3ir1, tv4ir, tv4ir1;

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
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h1_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h1_i[in_strides[4]];
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

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_FLOAT v9r_t = in_h2_r[in_strides[8]];
        FFTZ_FLOAT v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
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

        *out_h1_r = cv1rr + v1rr;
        tv2rr = cv1rr - (CRTM_9_7 * v1rr);
        *out_h1_i = cv1ii + v1ii;
        tv2ii = cv1ii - (CRTM_9_7 * v1ii);

        // Output points : 3, 6
        out_h1_r[out_strides[3]] = tv2rr + tv1ii;
        out_h2_r[out_strides[6]] = tv2rr - tv1ii;

        out_h1_i[out_strides[3]] = tv2ii + tv1rr;
        out_h2_i[out_strides[6]] = tv2ii - tv1rr;

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
        out_h1_r[out_strides[1]] = cv2rr + tv1rr;
        out_h1_i[out_strides[1]] = cv2ii + tv1ii;

        tv2rr = cv2rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv2ii - (CRTM_9_7 * tv1ii);

        // Output point : 4
        out_h1_r[out_strides[4]] = tv2rr + tv3ii;
        out_h1_i[out_strides[4]] = tv2ii + tv3rr;

        // Output point : 7
        out_h2_r[out_strides[7]] = tv2rr - tv3ii;
        out_h2_i[out_strides[7]] = tv2ii - tv3rr;

        tv3ir = (CRTM_9_4 * cv6rr) + (CRTM_9_3 * cv6ii);
        tv3ir1 = (CRTM_9_4 * cv6ii) - (CRTM_9_3 * cv6rr);
        tv4ir = (CRTM_9_2 * cv9ii) - (CRTM_9_1 * cv9rr);
        tv4ir1 = (CRTM_9_1 * cv9ii) + (CRTM_9_2 * cv9rr);
        tv3rr = CRTM_9_8 * (tv4ir - tv3ir);
        tv1rr = tv3ir + tv4ir;
        tv3ii = CRTM_9_8 * (tv3ir1 + tv4ir1);
        tv1ii = tv3ir1 - tv4ir1;

        // Output point : 2
        out_h1_r[out_strides[2]] = cv3rr + tv1rr;
        out_h1_i[out_strides[2]] = cv3ii + tv1ii;

        tv2rr = cv3rr - (CRTM_9_7 * tv1rr);
        tv2ii = cv3ii - (CRTM_9_7 * tv1ii);

        // Output point : 5
        out_h2_r[out_strides[5]] = tv2rr + tv3ii;
        out_h2_i[out_strides[5]] = tv2ii + tv3rr;

        // Output point : 8
        out_h2_r[out_strides[8]] = tv2rr - tv3ii;
        out_h2_i[out_strides[8]] = tv2ii - tv3rr;

        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_fwd_fft9c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fwd_fft9c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fwd_fft9c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
