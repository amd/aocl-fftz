// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft15c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-15 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-15 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author Ashwin K. Godbole
 *
 */
#include "core/kernels/kernel.h"

#define RADIX 15

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 106, 184, 88, 0, 0},
                                                     {0, 106, 184, 88, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft15c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_c2r_fft15c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000;
    const FFTZ_DOUBLE CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978;
    const FFTZ_DOUBLE CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FFTZ_DOUBLE CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FFTZ_DOUBLE CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FFTZ_DOUBLE CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FFTZ_DOUBLE CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633;

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
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6,
            av7, av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
            av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12,
            cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11,
            cv12, cv13, cv14, cv15, cv16;

        {

            FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[2]];
            FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[2]];
            FFTZ_DOUBLE v2r_temp = in_h1_r[in_strides[7]];
            FFTZ_DOUBLE v2i_temp = in_h1_i[in_strides[7]];
            FFTZ_DOUBLE v3r_temp = in_h2_r[in_strides[12]];
            FFTZ_DOUBLE v3i_temp = in_h2_i[in_strides[12]];
            v3i_temp = -v3i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av2 = v1r - v2r;
            av3 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

            av1 = v1i + v2i;
            av5 = v1i - v2i;
            av6 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {

            FFTZ_DOUBLE v1r_temp = in_h2_r[in_strides[8]];
            FFTZ_DOUBLE v1i_temp = in_h2_i[in_strides[8]];
            v1i_temp = -v1i_temp;
            FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[13]];
            FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[13]];
            v2i_temp = -v2i_temp;
            FFTZ_DOUBLE v3r_temp = in_h1_r[in_strides[3]];
            FFTZ_DOUBLE v3i_temp = in_h1_i[in_strides[3]];

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av9 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

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

            FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[1]];
            FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[1]];
            FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[11]];
            FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[11]];
            v2i_temp = -v2i_temp;
            FFTZ_DOUBLE v3r_temp = in_h1_r[in_strides[6]];
            FFTZ_DOUBLE v3i_temp = in_h1_i[in_strides[6]];

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av2 = v2r - v1r;
            av14 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

            av1 = v1i + v2i;
            av5 = v2i - v1i;
            av15 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {

            FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[4]];
            FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[4]];
            FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[14]];
            FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[14]];
            v2i_temp = -v2i_temp;
            FFTZ_DOUBLE v3r_temp = in_h2_r[in_strides[9]];
            FFTZ_DOUBLE v3i_temp = in_h2_i[in_strides[9]];
            v3i_temp = -v3i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av16 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

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

            FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[5]];
            FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[5]];
            FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[10]];
            FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[10]];
            v2i_temp = -v2i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = *in_h1_r;

            av1 = v1r + v2r;
            av2 = CRTM_15_6 * (v2r - v1r);
            av18 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
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

        FFTZ_DOUBLE _or_3 = cv1r + tv2;
        FFTZ_DOUBLE _or_12 = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_h1_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        FFTZ_DOUBLE _oi_3 = cv1i + tv8;
        FFTZ_DOUBLE _oi_12 = cv1i - tv8;
        {
            FFTZ_DOUBLE _twr = tw_ptr[11 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[11 * DATA_STRIDE + 1];
            out_h2_r[out_strides[12]] = _or_3 * _twr - _oi_3 * _twi;
            out_h2_i[out_strides[12]] = _or_3 * _twi + _oi_3 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[2 * DATA_STRIDE + 1];
            out_h1_r[out_strides[3]] = _or_12 * _twr - _oi_12 * _twi;
            out_h1_i[out_strides[3]] = _or_12 * _twi + _oi_12 * _twr;
        }

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        FFTZ_DOUBLE _or_6 = cv1r + tv2;
        FFTZ_DOUBLE _or_9 = cv1r - tv2;
        FFTZ_DOUBLE _oi_6 = cv1i + tv8;
        FFTZ_DOUBLE _oi_9 = cv1i - tv8;
        {
            FFTZ_DOUBLE _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[8 * DATA_STRIDE + 1];
            out_h2_r[out_strides[9]] = _or_6 * _twr - _oi_6 * _twi;
            out_h2_i[out_strides[9]] = _or_6 * _twi + _oi_6 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[5 * DATA_STRIDE + 1];
            out_h1_r[out_strides[6]] = _or_9 * _twr - _oi_9 * _twi;
            out_h1_i[out_strides[6]] = _or_9 * _twi + _oi_9 * _twr;
        }

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        FFTZ_DOUBLE _or_5 = tv5 + tv6;
        FFTZ_DOUBLE _or_10 = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        FFTZ_DOUBLE _oi_5 = tv5 + tv6;
        FFTZ_DOUBLE _oi_10 = tv5 - tv6;
        {
            FFTZ_DOUBLE _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[9 * DATA_STRIDE + 1];
            out_h2_r[out_strides[10]] = _or_5 * _twr - _oi_5 * _twi;
            out_h2_i[out_strides[10]] = _or_5 * _twi + _oi_5 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[4 * DATA_STRIDE + 1];
            out_h1_r[out_strides[5]] = _or_10 * _twr - _oi_10 * _twi;
            out_h1_i[out_strides[5]] = _or_10 * _twi + _oi_10 * _twr;
        }

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

        FFTZ_DOUBLE _or_1 = cv1r + cv6;
        FFTZ_DOUBLE _or_4 = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        FFTZ_DOUBLE _or_7 = cv1r + cv6;
        FFTZ_DOUBLE _or_13 = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        FFTZ_DOUBLE _oi_1 = cv1i + cv6;
        FFTZ_DOUBLE _oi_4 = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        FFTZ_DOUBLE _oi_7 = cv1i + tv3;
        FFTZ_DOUBLE _oi_13 = cv1i - tv3;

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        FFTZ_DOUBLE _or_2 = cv1r + cv6;
        FFTZ_DOUBLE _or_8 = cv1r - cv6;

        cv1i = tv2 + tv5;
        FFTZ_DOUBLE _oi_2 = cv1i + tv3;
        FFTZ_DOUBLE _oi_8 = cv1i - tv3;

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        FFTZ_DOUBLE _or_11 = cv1r + cv8;
        FFTZ_DOUBLE _oi_11 = cv1i - tv7;
        FFTZ_DOUBLE _or_14 = cv1r - cv8;
        FFTZ_DOUBLE _oi_14 = cv1i + tv7;

        {
            FFTZ_DOUBLE _twr = tw_ptr[13 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[13 * DATA_STRIDE + 1];
            out_h2_r[out_strides[14]] = _or_1 * _twr - _oi_1 * _twi;
            out_h2_i[out_strides[14]] = _or_1 * _twi + _oi_1 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[10 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[10 * DATA_STRIDE + 1];
            out_h2_r[out_strides[11]] = _or_4 * _twr - _oi_4 * _twi;
            out_h2_i[out_strides[11]] = _or_4 * _twi + _oi_4 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[7 * DATA_STRIDE + 1];
            out_h2_r[out_strides[8]] = _or_7 * _twr - _oi_7 * _twi;
            out_h2_i[out_strides[8]] = _or_7 * _twi + _oi_7 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[DATA_STRIDE + 1];
            out_h1_r[out_strides[2]] = _or_13 * _twr - _oi_13 * _twi;
            out_h1_i[out_strides[2]] = _or_13 * _twi + _oi_13 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[12 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[12 * DATA_STRIDE + 1];
            out_h2_r[out_strides[13]] = _or_2 * _twr - _oi_2 * _twi;
            out_h2_i[out_strides[13]] = _or_2 * _twi + _oi_2 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[6 * DATA_STRIDE + 1];
            out_h1_r[out_strides[7]] = _or_8 * _twr - _oi_8 * _twi;
            out_h1_i[out_strides[7]] = _or_8 * _twi + _oi_8 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[3 * DATA_STRIDE + 1];
            out_h1_r[out_strides[4]] = _or_11 * _twr - _oi_11 * _twi;
            out_h1_i[out_strides[4]] = _or_11 * _twi + _oi_11 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[0];
            FFTZ_DOUBLE _twi = tw_ptr[1];
            out_h1_r[out_strides[1]] = _or_14 * _twr - _oi_14 * _twi;
            out_h1_i[out_strides[1]] = _or_14 * _twi + _oi_14 * _twr;
        }

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

static FFTZ_VOID twid_c2r_fft15c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_15_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FFTZ_FLOAT CRTM_15_2 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_15_3 =
        +0.95105651629515357211643933337938214340569863400000f;
    const FFTZ_FLOAT CRTM_15_4 =
        +0.58778525229247301629891039327884007596190389052978f;
    const FFTZ_FLOAT CRTM_15_5 =
        +0.50000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_15_6 =
        +0.86602540378443864676372317075293618347140262690519f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FFTZ_FLOAT CRTM_15_7 =
        +0.50903696045512706468216979248996715975105181034577f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FFTZ_FLOAT CRTM_15_8 =
        +0.82363910354633184270744116161596601637855195182647f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FFTZ_FLOAT CRTM_15_9 =
        +0.48412291827592710612024388657479988457787393064252f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FFTZ_FLOAT CRTM_15_10 =
        +0.21650635094610964914707551542960572987794876098633f;

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
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, av1, av2, av3, av4, av5, av6,
            av7, av8, av9, av10, av11, av12, av13, av14, av15, av16, av17, av18,
            av19, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12,
            cv1r, cv1i, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11,
            cv12, cv13, cv14, cv15, cv16;

        {

            FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[2]];
            FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[2]];
            FFTZ_FLOAT v2r_temp = in_h1_r[in_strides[7]];
            FFTZ_FLOAT v2i_temp = in_h1_i[in_strides[7]];
            FFTZ_FLOAT v3r_temp = in_h2_r[in_strides[12]];
            FFTZ_FLOAT v3i_temp = in_h2_i[in_strides[12]];
            v3i_temp = -v3i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av2 = v1r - v2r;
            av3 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

            av1 = v1i + v2i;
            av5 = v1i - v2i;
            av6 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {

            FFTZ_FLOAT v1r_temp = in_h2_r[in_strides[8]];
            FFTZ_FLOAT v1i_temp = in_h2_i[in_strides[8]];
            v1i_temp = -v1i_temp;
            FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[13]];
            FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[13]];
            v2i_temp = -v2i_temp;
            FFTZ_FLOAT v3r_temp = in_h1_r[in_strides[3]];
            FFTZ_FLOAT v3i_temp = in_h1_i[in_strides[3]];

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av9 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

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

            FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[1]];
            FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[1]];
            FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[11]];
            FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[11]];
            v2i_temp = -v2i_temp;
            FFTZ_FLOAT v3r_temp = in_h1_r[in_strides[6]];
            FFTZ_FLOAT v3i_temp = in_h1_i[in_strides[6]];

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av2 = v2r - v1r;
            av14 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

            av1 = v1i + v2i;
            av5 = v2i - v1i;
            av15 = v3i + av1;
            av7 = v3i - (CRTM_15_5 * av1);
        }

        {

            FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[4]];
            FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[4]];
            FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[14]];
            FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[14]];
            v2i_temp = -v2i_temp;
            FFTZ_FLOAT v3r_temp = in_h2_r[in_strides[9]];
            FFTZ_FLOAT v3i_temp = in_h2_i[in_strides[9]];
            v3i_temp = -v3i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = v3r_temp;

            av1 = v1r + v2r;
            av8 = v2r - v1r;
            av16 = v3r + av1;
            av10 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
            v3i = v3i_temp;

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

            FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[5]];
            FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[5]];
            FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[10]];
            FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[10]];
            v2i_temp = -v2i_temp;

            v1r = v1r_temp;
            v2r = v2r_temp;
            v3r = *in_h1_r;

            av1 = v1r + v2r;
            av2 = CRTM_15_6 * (v2r - v1r);
            av18 = v3r + av1;
            av4 = v3r - (CRTM_15_5 * av1);

            v1i = v1i_temp;
            v2i = v2i_temp;
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

        FFTZ_FLOAT _or_3 = cv1r + tv2;
        FFTZ_FLOAT _or_12 = cv1r - tv2;

        tv3 = tv7 + tv8;
        tv9 = CRTM_15_1 * (tv7 - tv8);
        tv10 = av19 - (CRTM_15_2 * tv3);

        *out_h1_i = av19 + tv3;

        tv8 = (CRTM_15_3 * tv12) + (CRTM_15_4 * tv11);
        cv1i = tv10 + tv9;
        FFTZ_FLOAT _oi_3 = cv1i + tv8;
        FFTZ_FLOAT _oi_12 = cv1i - tv8;
        {
            FFTZ_FLOAT _twr = tw_ptr[11 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[11 * DATA_STRIDE + 1];
            out_h2_r[out_strides[12]] = _or_3 * _twr - _oi_3 * _twi;
            out_h2_i[out_strides[12]] = _or_3 * _twi + _oi_3 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[2 * DATA_STRIDE + 1];
            out_h1_r[out_strides[3]] = _or_12 * _twr - _oi_12 * _twi;
            out_h1_i[out_strides[3]] = _or_12 * _twi + _oi_12 * _twr;
        }

        cv1r = tv5 - tv4;
        tv2 = (CRTM_15_4 * tv1) - (CRTM_15_3 * tv6);
        cv1i = tv10 - tv9;
        tv8 = (CRTM_15_4 * tv12) - (CRTM_15_3 * tv11);

        FFTZ_FLOAT _or_6 = cv1r + tv2;
        FFTZ_FLOAT _or_9 = cv1r - tv2;
        FFTZ_FLOAT _oi_6 = cv1i + tv8;
        FFTZ_FLOAT _oi_9 = cv1i - tv8;
        {
            FFTZ_FLOAT _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[8 * DATA_STRIDE + 1];
            out_h2_r[out_strides[9]] = _or_6 * _twr - _oi_6 * _twi;
            out_h2_i[out_strides[9]] = _or_6 * _twi + _oi_6 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[5 * DATA_STRIDE + 1];
            out_h1_r[out_strides[6]] = _or_9 * _twr - _oi_9 * _twi;
            out_h1_i[out_strides[6]] = _or_9 * _twi + _oi_9 * _twr;
        }

        tv1 = CRTM_15_1 * (cv9 - cv1);
        tv2 = CRTM_15_9 * (cv15 + cv8);
        tv3 = cv9 + cv1;
        tv4 = cv8 - cv15;
        tv5 = tv3 + av4;
        tv6 = CRTM_15_6 * tv4 + av5;

        FFTZ_FLOAT _or_5 = tv5 + tv6;
        FFTZ_FLOAT _or_10 = tv5 - tv6;

        tv7 = CRTM_15_1 * (cv11 - cv3);
        tv8 = CRTM_15_9 * (cv13 + cv6);
        tv9 = cv11 + cv3;
        tv10 = cv13 - cv6;
        tv5 = tv9 + av7;
        tv6 = CRTM_15_6 * tv10 - av2;

        FFTZ_FLOAT _oi_5 = tv5 + tv6;
        FFTZ_FLOAT _oi_10 = tv5 - tv6;
        {
            FFTZ_FLOAT _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[9 * DATA_STRIDE + 1];
            out_h2_r[out_strides[10]] = _or_5 * _twr - _oi_5 * _twi;
            out_h2_i[out_strides[10]] = _or_5 * _twi + _oi_5 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[4 * DATA_STRIDE + 1];
            out_h1_r[out_strides[5]] = _or_10 * _twr - _oi_10 * _twi;
            out_h1_i[out_strides[5]] = _or_10 * _twi + _oi_10 * _twr;
        }

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

        FFTZ_FLOAT _or_1 = cv1r + cv6;
        FFTZ_FLOAT _or_4 = cv1r - cv6;

        cv1r = tv3 + tv10;
        cv1 = (CRTM_15_8 * cv14) + (CRTM_15_7 * cv5);
        cv3 = (CRTM_15_4 * cv4) - (CRTM_15_3 * cv12);
        cv6 = cv3 + cv1;

        FFTZ_FLOAT _or_7 = cv1r + cv6;
        FFTZ_FLOAT _or_13 = cv1r - cv6;

        tv1 = tv12 + tv11;
        tv2 = tv11 - tv12;
        tv3 = tv7 - tv8;
        tv5 = tv8 + tv7;
        cv1i = tv1 - tv3;

        tv6 = (CRTM_15_4 * cv10) + (CRTM_15_3 * cv2);
        tv10 = (CRTM_15_8 * cv7) - (CRTM_15_7 * cv16);
        cv6 = tv10 - tv6;
        tv7 = tv6 + tv10;

        FFTZ_FLOAT _oi_1 = cv1i + cv6;
        FFTZ_FLOAT _oi_4 = cv1i - cv6;

        cv1i = tv1 + tv3;
        tv8 = (CRTM_15_8 * cv16) + (CRTM_15_7 * cv7);
        tv1 = (CRTM_15_3 * cv10) - (CRTM_15_4 * cv2);
        tv3 = tv1 + tv8;

        FFTZ_FLOAT _oi_7 = cv1i + tv3;
        FFTZ_FLOAT _oi_13 = cv1i - tv3;

        tv3 = tv1 - tv8;
        cv1r = tv4 - tv9;
        cv6 = cv3 - cv1;
        FFTZ_FLOAT _or_2 = cv1r + cv6;
        FFTZ_FLOAT _or_8 = cv1r - cv6;

        cv1i = tv2 + tv5;
        FFTZ_FLOAT _oi_2 = cv1i + tv3;
        FFTZ_FLOAT _oi_8 = cv1i - tv3;

        cv1r = tv4 + tv9;
        cv1i = tv2 - tv5;

        FFTZ_FLOAT _or_11 = cv1r + cv8;
        FFTZ_FLOAT _oi_11 = cv1i - tv7;
        FFTZ_FLOAT _or_14 = cv1r - cv8;
        FFTZ_FLOAT _oi_14 = cv1i + tv7;

        {
            FFTZ_FLOAT _twr = tw_ptr[13 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[13 * DATA_STRIDE + 1];
            out_h2_r[out_strides[14]] = _or_1 * _twr - _oi_1 * _twi;
            out_h2_i[out_strides[14]] = _or_1 * _twi + _oi_1 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[10 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[10 * DATA_STRIDE + 1];
            out_h2_r[out_strides[11]] = _or_4 * _twr - _oi_4 * _twi;
            out_h2_i[out_strides[11]] = _or_4 * _twi + _oi_4 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[7 * DATA_STRIDE + 1];
            out_h2_r[out_strides[8]] = _or_7 * _twr - _oi_7 * _twi;
            out_h2_i[out_strides[8]] = _or_7 * _twi + _oi_7 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[DATA_STRIDE + 1];
            out_h1_r[out_strides[2]] = _or_13 * _twr - _oi_13 * _twi;
            out_h1_i[out_strides[2]] = _or_13 * _twi + _oi_13 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[12 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[12 * DATA_STRIDE + 1];
            out_h2_r[out_strides[13]] = _or_2 * _twr - _oi_2 * _twi;
            out_h2_i[out_strides[13]] = _or_2 * _twi + _oi_2 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[6 * DATA_STRIDE + 1];
            out_h1_r[out_strides[7]] = _or_8 * _twr - _oi_8 * _twi;
            out_h1_i[out_strides[7]] = _or_8 * _twi + _oi_8 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[3 * DATA_STRIDE + 1];
            out_h1_r[out_strides[4]] = _or_11 * _twr - _oi_11 * _twi;
            out_h1_i[out_strides[4]] = _or_11 * _twi + _oi_11 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[0];
            FFTZ_FLOAT _twi = tw_ptr[1];
            out_h1_r[out_strides[1]] = _or_14 * _twr - _oi_14 * _twi;
            out_h1_i[out_strides[1]] = _or_14 * _twi + _oi_14 * _twr;
        }

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

kfft_ register_kernel_twid_c2r_fft15c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft15c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft15c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
