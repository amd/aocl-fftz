// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_r2c_fft15c.c
 *
 *  @brief R2C fused twiddle (forward twiddle + conjugate output) Radix-15 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-15 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author Ashwin K. Godbole
 *
 */
#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 106, 184, 88, 0, 0},
                                                     {0, 106, 184, 88, 0, 0}};

ops_cycles_t get_ops_cnt_twid_r2c_fft15c(UINT8 precision, UINT8 direction)
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

static VOID twid_r2c_fft15c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978;
    const DOUBLE CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const DOUBLE CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633;

    DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6, av7,
            av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
            av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12,
            cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11,
            cv12, cv13, cv14, cv15, cv16;

        {
            UINTP twid_addr1 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            DOUBLE twid_r1 = tw[twid_addr1];
            DOUBLE twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            DOUBLE twid_r2 = tw[twid_addr2];
            DOUBLE twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 =
                DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
            DOUBLE twid_r3 = tw[twid_addr3];
            DOUBLE twid_i3 = tw[1 + twid_addr3];

            DOUBLE v1r_temp = in_h1_r[in_strides[2]];
            DOUBLE v1i_temp = in_h1_i[in_strides[2]];
            DOUBLE v2r_temp = in_h1_r[in_strides[7]];
            DOUBLE v2i_temp = in_h1_i[in_strides[7]];
            DOUBLE v3r_temp = in_h2_r[in_strides[12]];
            DOUBLE v3i_temp = in_h2_i[in_strides[12]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av2 = v1r - v2r;
            av3 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av5 = v1i - v2i;
            av6 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {
            UINTP twid_addr1 = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            DOUBLE twid_r1 = tw[twid_addr1];
            DOUBLE twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (13 * cols + cnt * load_multi_cols);
            DOUBLE twid_r2 = tw[twid_addr2];
            DOUBLE twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            DOUBLE twid_r3 = tw[twid_addr3];
            DOUBLE twid_i3 = tw[1 + twid_addr3];

            DOUBLE v1r_temp = in_h2_r[in_strides[8]];
            DOUBLE v1i_temp = in_h2_i[in_strides[8]];
            DOUBLE v2r_temp = in_h2_r[in_strides[13]];
            DOUBLE v2i_temp = in_h2_i[in_strides[13]];
            DOUBLE v3r_temp = in_h1_r[in_strides[3]];
            DOUBLE v3i_temp = in_h1_i[in_strides[3]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av9 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av11 = v2i - v1i;
            av12 = v3i + av1;
            av13 = v3i - (CRTM_15_5 * av1);
        }

        cv1 = av10 + av4;
        cv2 = av10 - av4;
        cv3 = av13 + av7;
        cv4 = av13 - av7;
        cv5 = av8 + av2;
        cv6 = av8 - av2;
        cv7 = av11 + av5;
        cv8 = av11 - av5;

        {
            UINTP twid_addr1 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            DOUBLE twid_r1 = tw[twid_addr1];
            DOUBLE twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            DOUBLE twid_r2 = tw[twid_addr2];
            DOUBLE twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            DOUBLE twid_r3 = tw[twid_addr3];
            DOUBLE twid_i3 = tw[1 + twid_addr3];

            DOUBLE v1r_temp = in_h1_r[in_strides[1]];
            DOUBLE v1i_temp = in_h1_i[in_strides[1]];
            DOUBLE v2r_temp = in_h2_r[in_strides[11]];
            DOUBLE v2i_temp = in_h2_i[in_strides[11]];
            DOUBLE v3r_temp = in_h1_r[in_strides[6]];
            DOUBLE v3i_temp = in_h1_i[in_strides[6]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av2 = v2r - v1r;
            av14 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av5 = v2i - v1i;
            av15 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {
            UINTP twid_addr1 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            DOUBLE twid_r1 = tw[twid_addr1];
            DOUBLE twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (14 * cols + cnt * load_multi_cols);
            DOUBLE twid_r2 = tw[twid_addr2];
            DOUBLE twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            DOUBLE twid_r3 = tw[twid_addr3];
            DOUBLE twid_i3 = tw[1 + twid_addr3];

            DOUBLE v1r_temp = in_h1_r[in_strides[4]];
            DOUBLE v1i_temp = in_h1_i[in_strides[4]];
            DOUBLE v2r_temp = in_h2_r[in_strides[14]];
            DOUBLE v2i_temp = in_h2_i[in_strides[14]];
            DOUBLE v3r_temp = in_h2_r[in_strides[9]];
            DOUBLE v3i_temp = in_h2_i[in_strides[9]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av16 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av11 = v2i - v1i;
            av17 = v3i + av1;
            av13 = v3i - (CRTM_15_5 * av1);
        }

        cv9 = av10 + av4;
        cv10 = av4 - av10;
        cv11 = av13 + av7;
        cv12 = av7 - av13;
        cv13 = av8 + av2;
        cv14 = av2 - av8;
        cv15 = av11 + av5;
        cv16 = av5 - av11;

        {
            UINTP twid_addr1 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            DOUBLE twid_r1 = tw[twid_addr1];
            DOUBLE twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            DOUBLE twid_r2 = tw[twid_addr2];
            DOUBLE twid_i2 = tw[1 + twid_addr2];

            DOUBLE v1r_temp = in_h1_r[in_strides[5]];
            DOUBLE v1i_temp = in_h1_i[in_strides[5]];
            DOUBLE v2r_temp = in_h2_r[in_strides[10]];
            DOUBLE v2i_temp = in_h2_i[in_strides[10]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = *in_h1_r;

            av1 = v1r + v2r;
            av2 = CRTM_15_6 * (v2r - v1r);
            av18 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = *in_h1_i;

            av1 = v1i + v2i;
            av5 = CRTM_15_6 * (v2i - v1i);
            av19 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        tv1 = av16 + av14;
        tv2 = av3 + av9;
        tv11 = av9 - av3;
        tv12 = av16 - av14;
        tv3 = tv1 + tv2;
        tv4 = CRTM_15_1 * (tv1 - tv2);
        tv5 = av18 - (CRTM_15_2 * tv3);

        *out_h1_r = av18 + tv3;

        tv6 = av6 - av12;
        tv8 = av6 + av12;
        tv1 = av15 - av17;
        tv7 = av17 + av15;
        tv2 = (CRTM_15_3 * tv1) + (CRTM_15_4 * tv6);
        cv1r = tv5 + tv4;

        out_h1_r[out_strides[3]] = cv1r + tv2;
        out_h2_r[out_strides[12]] = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_h1_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        out_h1_i[out_strides[3]] = cv1i + tv8;
        out_h2_i[out_strides[12]] = -(cv1i - tv8);

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        out_h1_r[out_strides[6]] = cv1r + tv2;
        out_h2_r[out_strides[9]] = cv1r - tv2;
        out_h1_i[out_strides[6]] = cv1i + tv8;
        out_h2_i[out_strides[9]] = -(cv1i - tv8);

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        out_h1_r[out_strides[5]] = tv5 + tv6;
        out_h2_r[out_strides[10]] = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        out_h1_i[out_strides[5]] = tv5 + tv6;
        out_h2_i[out_strides[10]] = -(tv5 - tv6);

        tv5 = av4 - (CRTM_15_2 * tv3);
        tv6 = (CRTM_15_10 * tv4) - av5;
        tv11 = av7 - (CRTM_15_2 * tv9);
        tv12 = av2 + (CRTM_15_10 * tv10);
        tv3 = tv6 + tv5;
        tv4 = tv5 - tv6;
        tv9 = tv2 - tv1;
        tv10 = tv1 + tv2;

        cv1r = tv3 - tv10;
        cv1 = (CRTM_15_4 * cv12) + (CRTM_15_3 * cv4);
        cv3 = (CRTM_15_8 * cv5) - (CRTM_15_7 * cv14);
        cv6 = cv1 + cv3;
        cv8 = cv1 - cv3;

        out_h1_r[out_strides[1]] = cv1r + cv6;
        out_h1_r[out_strides[4]] = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        out_h1_r[out_strides[7]] = cv1r + cv6;
        out_h2_r[out_strides[13]] = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        out_h1_i[out_strides[1]] = cv1i + cv6;
        out_h1_i[out_strides[4]] = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        out_h1_i[out_strides[7]] = cv1i + tv3;
        out_h2_i[out_strides[13]] = -(cv1i - tv3);

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        out_h1_r[out_strides[2]] = cv1r + cv6;
        out_h2_r[out_strides[8]] = cv1r - cv6;

        cv1i = tv2 + tv5;
        out_h1_i[out_strides[2]] = cv1i + tv3;
        out_h2_i[out_strides[8]] = -(cv1i - tv3);

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        out_h2_r[out_strides[11]] = cv1r + cv8;
        out_h2_i[out_strides[11]] = -(cv1i - tv7);
        out_h2_r[out_strides[14]] = cv1r - cv8;
        out_h2_i[out_strides[14]] = -(cv1i + tv7);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID twid_r2c_fft15c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978f;
    const FLOAT CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FLOAT CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633f;

    FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6, av7,
            av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
            av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12,
            cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11,
            cv12, cv13, cv14, cv15, cv16;

        {
            UINTP twid_addr1 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FLOAT twid_r1 = tw[twid_addr1];
            FLOAT twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            FLOAT twid_r2 = tw[twid_addr2];
            FLOAT twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 =
                DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
            FLOAT twid_r3 = tw[twid_addr3];
            FLOAT twid_i3 = tw[1 + twid_addr3];

            FLOAT v1r_temp = in_h1_r[in_strides[2]];
            FLOAT v1i_temp = in_h1_i[in_strides[2]];
            FLOAT v2r_temp = in_h1_r[in_strides[7]];
            FLOAT v2i_temp = in_h1_i[in_strides[7]];
            FLOAT v3r_temp = in_h2_r[in_strides[12]];
            FLOAT v3i_temp = in_h2_i[in_strides[12]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av2 = v1r - v2r;
            av3 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av5 = v1i - v2i;
            av6 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {
            UINTP twid_addr1 = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            FLOAT twid_r1 = tw[twid_addr1];
            FLOAT twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (13 * cols + cnt * load_multi_cols);
            FLOAT twid_r2 = tw[twid_addr2];
            FLOAT twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FLOAT twid_r3 = tw[twid_addr3];
            FLOAT twid_i3 = tw[1 + twid_addr3];

            FLOAT v1r_temp = in_h2_r[in_strides[8]];
            FLOAT v1i_temp = in_h2_i[in_strides[8]];
            FLOAT v2r_temp = in_h2_r[in_strides[13]];
            FLOAT v2i_temp = in_h2_i[in_strides[13]];
            FLOAT v3r_temp = in_h1_r[in_strides[3]];
            FLOAT v3i_temp = in_h1_i[in_strides[3]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av9 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av11 = v2i - v1i;
            av12 = v3i + av1;
            av13 = v3i - (CRTM_15_5 * av1);
        }

        cv1 = av10 + av4;
        cv2 = av10 - av4;
        cv3 = av13 + av7;
        cv4 = av13 - av7;
        cv5 = av8 + av2;
        cv6 = av8 - av2;
        cv7 = av11 + av5;
        cv8 = av11 - av5;

        {
            UINTP twid_addr1 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FLOAT twid_r1 = tw[twid_addr1];
            FLOAT twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            FLOAT twid_r2 = tw[twid_addr2];
            FLOAT twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            FLOAT twid_r3 = tw[twid_addr3];
            FLOAT twid_i3 = tw[1 + twid_addr3];

            FLOAT v1r_temp = in_h1_r[in_strides[1]];
            FLOAT v1i_temp = in_h1_i[in_strides[1]];
            FLOAT v2r_temp = in_h2_r[in_strides[11]];
            FLOAT v2i_temp = in_h2_i[in_strides[11]];
            FLOAT v3r_temp = in_h1_r[in_strides[6]];
            FLOAT v3i_temp = in_h1_i[in_strides[6]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av2 = v2r - v1r;
            av14 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av5 = v2i - v1i;
            av15 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {
            UINTP twid_addr1 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FLOAT twid_r1 = tw[twid_addr1];
            FLOAT twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (14 * cols + cnt * load_multi_cols);
            FLOAT twid_r2 = tw[twid_addr2];
            FLOAT twid_i2 = tw[1 + twid_addr2];
            UINTP twid_addr3 = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            FLOAT twid_r3 = tw[twid_addr3];
            FLOAT twid_i3 = tw[1 + twid_addr3];

            FLOAT v1r_temp = in_h1_r[in_strides[4]];
            FLOAT v1i_temp = in_h1_i[in_strides[4]];
            FLOAT v2r_temp = in_h2_r[in_strides[14]];
            FLOAT v2i_temp = in_h2_i[in_strides[14]];
            FLOAT v3r_temp = in_h2_r[in_strides[9]];
            FLOAT v3i_temp = in_h2_i[in_strides[9]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = v3r_temp * twid_r3 - v3i_temp * twid_i3;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av16 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = v3r_temp * twid_i3 + v3i_temp * twid_r3;

            av1 = v1i + v2i;
            av11 = v2i - v1i;
            av17 = v3i + av1;
            av13 = v3i - (CRTM_15_5 * av1);
        }

        cv9 = av10 + av4;
        cv10 = av4 - av10;
        cv11 = av13 + av7;
        cv12 = av7 - av13;
        cv13 = av8 + av2;
        cv14 = av2 - av8;
        cv15 = av11 + av5;
        cv16 = av5 - av11;

        {
            UINTP twid_addr1 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            FLOAT twid_r1 = tw[twid_addr1];
            FLOAT twid_i1 = tw[1 + twid_addr1];
            UINTP twid_addr2 =
                DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            FLOAT twid_r2 = tw[twid_addr2];
            FLOAT twid_i2 = tw[1 + twid_addr2];

            FLOAT v1r_temp = in_h1_r[in_strides[5]];
            FLOAT v1i_temp = in_h1_i[in_strides[5]];
            FLOAT v2r_temp = in_h2_r[in_strides[10]];
            FLOAT v2i_temp = in_h2_i[in_strides[10]];

            v1r = v1r_temp * twid_r1 - v1i_temp * twid_i1;
            v2r = v2r_temp * twid_r2 - v2i_temp * twid_i2;
            v3r = *in_h1_r;

            av1 = v1r + v2r;
            av2 = CRTM_15_6 * (v2r - v1r);
            av18 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1r_temp * twid_i1 + v1i_temp * twid_r1;
            v2i = v2r_temp * twid_i2 + v2i_temp * twid_r2;
            v3i = *in_h1_i;

            av1 = v1i + v2i;
            av5 = CRTM_15_6 * (v2i - v1i);
            av19 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        tv1 = av16 + av14;
        tv2 = av3 + av9;
        tv11 = av9 - av3;
        tv12 = av16 - av14;
        tv3 = tv1 + tv2;
        tv4 = CRTM_15_1 * (tv1 - tv2);
        tv5 = av18 - (CRTM_15_2 * tv3);

        *out_h1_r = av18 + tv3;

        tv6 = av6 - av12;
        tv8 = av6 + av12;
        tv1 = av15 - av17;
        tv7 = av17 + av15;
        tv2 = (CRTM_15_3 * tv1) + (CRTM_15_4 * tv6);
        cv1r = tv5 + tv4;

        out_h1_r[out_strides[3]] = cv1r + tv2;
        out_h2_r[out_strides[12]] = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_h1_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        out_h1_i[out_strides[3]] = cv1i + tv8;
        out_h2_i[out_strides[12]] = -(cv1i - tv8);

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        out_h1_r[out_strides[6]] = cv1r + tv2;
        out_h2_r[out_strides[9]] = cv1r - tv2;
        out_h1_i[out_strides[6]] = cv1i + tv8;
        out_h2_i[out_strides[9]] = -(cv1i - tv8);

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        out_h1_r[out_strides[5]] = tv5 + tv6;
        out_h2_r[out_strides[10]] = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        out_h1_i[out_strides[5]] = tv5 + tv6;
        out_h2_i[out_strides[10]] = -(tv5 - tv6);

        tv5 = av4 - (CRTM_15_2 * tv3);
        tv6 = (CRTM_15_10 * tv4) - av5;
        tv11 = av7 - (CRTM_15_2 * tv9);
        tv12 = av2 + (CRTM_15_10 * tv10);
        tv3 = tv6 + tv5;
        tv4 = tv5 - tv6;
        tv9 = tv2 - tv1;
        tv10 = tv1 + tv2;

        cv1r = tv3 - tv10;
        cv1 = (CRTM_15_4 * cv12) + (CRTM_15_3 * cv4);
        cv3 = (CRTM_15_8 * cv5) - (CRTM_15_7 * cv14);
        cv6 = cv1 + cv3;
        cv8 = cv1 - cv3;

        out_h1_r[out_strides[1]] = cv1r + cv6;
        out_h1_r[out_strides[4]] = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        out_h1_r[out_strides[7]] = cv1r + cv6;
        out_h2_r[out_strides[13]] = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        out_h1_i[out_strides[1]] = cv1i + cv6;
        out_h1_i[out_strides[4]] = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        out_h1_i[out_strides[7]] = cv1i + tv3;
        out_h2_i[out_strides[13]] = -(cv1i - tv3);

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        out_h1_r[out_strides[2]] = cv1r + cv6;
        out_h2_r[out_strides[8]] = cv1r - cv6;

        cv1i = tv2 + tv5;
        out_h1_i[out_strides[2]] = cv1i + tv3;
        out_h2_i[out_strides[8]] = -(cv1i - tv3);

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        out_h2_r[out_strides[11]] = cv1r + cv8;
        out_h2_i[out_strides[11]] = -(cv1i - tv7);
        out_h2_r[out_strides[14]] = cv1r - cv8;
        out_h2_i[out_strides[14]] = -(cv1i + tv7);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_r2c_fft15c(UINT8 precision,
                                      UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_r2c_fft15c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_r2c_fft15c_fp64;
    }
    else
    {
        return NULL;
    }
}
