// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft14c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-14 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-14 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 124, 174, 82, 0, 0},
                                                     {0, 124, 174, 82, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft14c(UINT8 precision, UINT8 direction)
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

static VOID twid_c2r_fft14c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000;
    const DOUBLE CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829;
    const DOUBLE CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372;
    const DOUBLE CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327;
    const DOUBLE CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720;
    const DOUBLE CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000;

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

    in_h1_r = (DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r, v313r,
            v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i, v97i,
            v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r, v124r, v115r,
            v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i, cv2r, cv2i, cv3r,
            cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii, tv1ri, tv2ri, tv1ir,
            tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir, tv4ir, tv5ir, tv6ir;

        DOUBLE t_v1r = in_h1_r[in_strides[1]];
        DOUBLE t_v1i = in_h1_i[in_strides[1]];
        v1r = t_v1r;
        v1i = t_v1i;

        DOUBLE t_v2r = in_h2_r[in_strides[13]];
        DOUBLE t_v2i = in_h2_i[in_strides[13]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[3]];
        t_v1i = in_h1_i[in_strides[3]];
        v1r = t_v1r;
        v1i = t_v1i;

        t_v2r = in_h2_r[in_strides[11]];
        t_v2i = in_h2_i[in_strides[11]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[5]];
        t_v1i = in_h1_i[in_strides[5]];
        v1r = t_v1r;
        v1i = t_v1i;

        t_v2r = in_h2_r[in_strides[9]];
        t_v2i = in_h2_i[in_strides[9]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        DOUBLE t_v8r = in_h2_r[in_strides[7]];
        DOUBLE t_v8i = in_h2_i[in_strides[7]];
        t_v8r = -t_v8r;
        v8r = t_v8r;
        v8i = t_v8i;

        DOUBLE t_v3r = in_h1_r[in_strides[2]];
        DOUBLE t_v3i = in_h1_i[in_strides[2]];
        v3r = t_v3r;
        v3i = t_v3i;

        DOUBLE t_v4r = in_h2_r[in_strides[12]];
        DOUBLE t_v4i = in_h2_i[in_strides[12]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        t_v3r = in_h1_r[in_strides[4]];
        t_v3i = in_h1_i[in_strides[4]];
        v3r = t_v3r;
        v3i = t_v3i;

        t_v4r = in_h2_r[in_strides[10]];
        t_v4i = in_h2_i[in_strides[10]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v115i = v3i + v4i;
        v511i = v3i - v4i;

        v511r = v3r + v4r;
        v115r = v3r - v4r;

        tv1rr = (CRTM_14_1 * v214r) + (CRTM_14_5 * v412r) -
                (CRTM_14_3 * v610r) - v8r;
        tv3rr = (CRTM_14_3 * v214r) - (CRTM_14_1 * v412r) -
                (CRTM_14_5 * v610r) + v8r;
        tv5rr = (CRTM_14_5 * v214r) - (CRTM_14_3 * v412r) +
                (CRTM_14_1 * v610r) - v8r;
        tv1ii = (CRTM_14_1 * v214i) + (CRTM_14_5 * v412i) -
                (CRTM_14_3 * v610i) - v8i;
        tv3ii = (CRTM_14_3 * v214i) - (CRTM_14_1 * v412i) -
                (CRTM_14_5 * v610i) + v8i;
        tv5ii = (CRTM_14_5 * v214i) - (CRTM_14_3 * v412i) +
                (CRTM_14_1 * v610i) - v8i;

        tv1ir = (CRTM_14_2 * v142r) + (CRTM_14_6 * v124r) + (CRTM_14_4 * v106r);
        tv3ir = (CRTM_14_4 * v142r) + (CRTM_14_2 * v124r) - (CRTM_14_6 * v106r);
        tv5ir = (CRTM_14_6 * v142r) - (CRTM_14_4 * v124r) + (CRTM_14_2 * v106r);

        t_v3r = in_h1_r[in_strides[6]];
        t_v3i = in_h1_i[in_strides[6]];
        v3r = t_v3r;
        v3i = t_v3i;

        t_v4r = in_h2_r[in_strides[8]];
        t_v4i = in_h2_i[in_strides[8]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v79i = v3i + v4i;
        v97i = v4i - v3i;

        v79r = v3r + v4r;
        v97r = v4r - v3r;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        tv2ir = (CRTM_14_4 * v133r) - (CRTM_14_6 * v115r) + (CRTM_14_2 * v97r);
        tv4ir = (CRTM_14_2 * v115r) + (CRTM_14_6 * v133r) - (CRTM_14_4 * v97r);
        tv6ir = (CRTM_14_4 * v115r) + (CRTM_14_2 * v133r) + (CRTM_14_6 * v97r);
        tv1ri = (CRTM_14_2 * v142i) + (CRTM_14_6 * v124i) + (CRTM_14_4 * v106i);
        tv3ri = (CRTM_14_4 * v142i) + (CRTM_14_2 * v124i) - (CRTM_14_6 * v106i);
        tv5ri = (CRTM_14_6 * v142i) - (CRTM_14_4 * v124i) + (CRTM_14_2 * v106i);

        tvrr = v1r + v313r + v511r + v79r;
        tvri = v214r + v412r + v610r + v8r;
        tvir = v214i + v412i + v610i + v8i;
        tvii = v1i + v313i + v115i + v79i;

        *out_h1_r = tvrr + tvri;
        *out_h1_i = tvir + tvii;

        DOUBLE _or_7 = tvrr - tvri;
        {
            UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
        }

        cv1r = v1r + (CRTM_14_3 * v313r) - (CRTM_14_5 * v511r) -
               (CRTM_14_1 * v79r);
        cv2r = v1r + (CRTM_14_3 * v79r) -
               ((CRTM_14_1 * v511r) + (CRTM_14_5 * v313r));
        cv3r = v1r + (CRTM_14_3 * v511r) - (CRTM_14_1 * v313r) -
               (CRTM_14_5 * v79r);

        tv2ri = (CRTM_14_4 * v133i) - (CRTM_14_6 * v511i) + (CRTM_14_2 * v97i);
        tv4ri = (CRTM_14_6 * v133i) + (CRTM_14_2 * v511i) - (CRTM_14_4 * v97i);
        tv6ri = (CRTM_14_2 * v133i) + (CRTM_14_4 * v511i) + (CRTM_14_6 * v97i);

        cv1i = v1i + (CRTM_14_3 * v313i) - (CRTM_14_5 * v115i) -
               (CRTM_14_1 * v79i);
        cv2i = v1i + (CRTM_14_3 * v79i) -
               ((CRTM_14_5 * v313i) + (CRTM_14_1 * v115i));
        cv3i = v1i + (CRTM_14_3 * v115i) - (CRTM_14_1 * v313i) -
               (CRTM_14_5 * v79i);

        //-------------------------------------------

        tvrr = cv1r + tv1rr;
        tvri = tv1ri + tv2ri;

        tvir = tv1ir + tv2ir;
        tvii = cv1i + tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_1 = tvrr - tvri;
            DOUBLE _oi_1 = tvir + tvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        DOUBLE _or_13 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (13 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[13]] = _or_13 * _twr + _oi * _twi;
            out_h2_i[out_strides[13]] = _oi * _twr - _or_13 * _twi;
        }

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_6 = tvrr - tvri;
            DOUBLE _oi_6 = tvir + tvii;
            out_h1_r[out_strides[6]] = _or_6 * _twr + _oi_6 * _twi;
            out_h1_i[out_strides[6]] = _oi_6 * _twr - _or_6 * _twi;
        }
        DOUBLE _or_8 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[8]] = _or_8 * _twr + _oi * _twi;
            out_h2_i[out_strides[8]] = _oi * _twr - _or_8 * _twi;
        }

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_2 = tvrr - tvri;
            DOUBLE _oi_2 = tvir + tvii;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        DOUBLE _or_12 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[12]] = _or_12 * _twr + _oi * _twi;
            out_h2_i[out_strides[12]] = _oi * _twr - _or_12 * _twi;
        }

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        {
            UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_5 = tvrr - tvri;
            DOUBLE _oi_5 = tvir + tvii;
            out_h1_r[out_strides[5]] = _or_5 * _twr + _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _oi_5 * _twr - _or_5 * _twi;
        }
        DOUBLE _or_9 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[9]] = _or_9 * _twr + _oi * _twi;
            out_h2_i[out_strides[9]] = _oi * _twr - _or_9 * _twi;
        }

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_3 = tvrr - tvri;
            DOUBLE _oi_3 = tvir + tvii;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        DOUBLE _or_11 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[11]] = _or_11 * _twr + _oi * _twi;
            out_h2_i[out_strides[11]] = _oi * _twr - _or_11 * _twi;
        }

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_4 = tvrr - tvri;
            DOUBLE _oi_4 = tvir + tvii;
            out_h1_r[out_strides[4]] = _or_4 * _twr + _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _oi_4 * _twr - _or_4 * _twi;
        }
        DOUBLE _or_10 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[10]] = _or_10 * _twr + _oi * _twi;
            out_h2_i[out_strides[10]] = _oi * _twr - _or_10 * _twi;
        }

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

static VOID twid_c2r_fft14c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000f;
    const FLOAT CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829f;
    const FLOAT CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372f;
    const FLOAT CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327f;
    const FLOAT CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720f;
    const FLOAT CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000f;

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

    in_h1_r = (FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r, v313r,
            v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i, v97i,
            v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r, v124r, v115r,
            v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i, cv2r, cv2i, cv3r,
            cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii, tv1ri, tv2ri, tv1ir,
            tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir, tv4ir, tv5ir, tv6ir;

        FLOAT t_v1r = in_h1_r[in_strides[1]];
        FLOAT t_v1i = in_h1_i[in_strides[1]];
        v1r = t_v1r;
        v1i = t_v1i;

        FLOAT t_v2r = in_h2_r[in_strides[13]];
        FLOAT t_v2i = in_h2_i[in_strides[13]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[3]];
        t_v1i = in_h1_i[in_strides[3]];
        v1r = t_v1r;
        v1i = t_v1i;

        t_v2r = in_h2_r[in_strides[11]];
        t_v2i = in_h2_i[in_strides[11]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[5]];
        t_v1i = in_h1_i[in_strides[5]];
        v1r = t_v1r;
        v1i = t_v1i;

        t_v2r = in_h2_r[in_strides[9]];
        t_v2i = in_h2_i[in_strides[9]];
        t_v2r = -t_v2r;
        v2r = t_v2r;
        v2i = t_v2i;

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        FLOAT t_v8r = in_h2_r[in_strides[7]];
        FLOAT t_v8i = in_h2_i[in_strides[7]];
        t_v8r = -t_v8r;
        v8r = t_v8r;
        v8i = t_v8i;

        FLOAT t_v3r = in_h1_r[in_strides[2]];
        FLOAT t_v3i = in_h1_i[in_strides[2]];
        v3r = t_v3r;
        v3i = t_v3i;

        FLOAT t_v4r = in_h2_r[in_strides[12]];
        FLOAT t_v4i = in_h2_i[in_strides[12]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        t_v3r = in_h1_r[in_strides[4]];
        t_v3i = in_h1_i[in_strides[4]];
        v3r = t_v3r;
        v3i = t_v3i;

        t_v4r = in_h2_r[in_strides[10]];
        t_v4i = in_h2_i[in_strides[10]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v115i = v3i + v4i;
        v511i = v3i - v4i;

        v511r = v3r + v4r;
        v115r = v3r - v4r;

        tv1rr = (CRTM_14_1 * v214r) + (CRTM_14_5 * v412r) -
                (CRTM_14_3 * v610r) - v8r;
        tv3rr = (CRTM_14_3 * v214r) - (CRTM_14_1 * v412r) -
                (CRTM_14_5 * v610r) + v8r;
        tv5rr = (CRTM_14_5 * v214r) - (CRTM_14_3 * v412r) +
                (CRTM_14_1 * v610r) - v8r;
        tv1ii = (CRTM_14_1 * v214i) + (CRTM_14_5 * v412i) -
                (CRTM_14_3 * v610i) - v8i;
        tv3ii = (CRTM_14_3 * v214i) - (CRTM_14_1 * v412i) -
                (CRTM_14_5 * v610i) + v8i;
        tv5ii = (CRTM_14_5 * v214i) - (CRTM_14_3 * v412i) +
                (CRTM_14_1 * v610i) - v8i;

        tv1ir = (CRTM_14_2 * v142r) + (CRTM_14_6 * v124r) + (CRTM_14_4 * v106r);
        tv3ir = (CRTM_14_4 * v142r) + (CRTM_14_2 * v124r) - (CRTM_14_6 * v106r);
        tv5ir = (CRTM_14_6 * v142r) - (CRTM_14_4 * v124r) + (CRTM_14_2 * v106r);

        t_v3r = in_h1_r[in_strides[6]];
        t_v3i = in_h1_i[in_strides[6]];
        v3r = t_v3r;
        v3i = t_v3i;

        t_v4r = in_h2_r[in_strides[8]];
        t_v4i = in_h2_i[in_strides[8]];
        t_v4r = -t_v4r;
        v4r = t_v4r;
        v4i = t_v4i;

        v79i = v3i + v4i;
        v97i = v4i - v3i;

        v79r = v3r + v4r;
        v97r = v4r - v3r;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        tv2ir = (CRTM_14_4 * v133r) - (CRTM_14_6 * v115r) + (CRTM_14_2 * v97r);
        tv4ir = (CRTM_14_2 * v115r) + (CRTM_14_6 * v133r) - (CRTM_14_4 * v97r);
        tv6ir = (CRTM_14_4 * v115r) + (CRTM_14_2 * v133r) + (CRTM_14_6 * v97r);
        tv1ri = (CRTM_14_2 * v142i) + (CRTM_14_6 * v124i) + (CRTM_14_4 * v106i);
        tv3ri = (CRTM_14_4 * v142i) + (CRTM_14_2 * v124i) - (CRTM_14_6 * v106i);
        tv5ri = (CRTM_14_6 * v142i) - (CRTM_14_4 * v124i) + (CRTM_14_2 * v106i);

        tvrr = v1r + v313r + v511r + v79r;
        tvri = v214r + v412r + v610r + v8r;
        tvir = v214i + v412i + v610i + v8i;
        tvii = v1i + v313i + v115i + v79i;

        *out_h1_r = tvrr + tvri;
        *out_h1_i = tvir + tvii;

        FLOAT _or_7 = tvrr - tvri;
        {
            UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
        }

        cv1r = v1r + (CRTM_14_3 * v313r) - (CRTM_14_5 * v511r) -
               (CRTM_14_1 * v79r);
        cv2r = v1r + (CRTM_14_3 * v79r) -
               ((CRTM_14_1 * v511r) + (CRTM_14_5 * v313r));
        cv3r = v1r + (CRTM_14_3 * v511r) - (CRTM_14_1 * v313r) -
               (CRTM_14_5 * v79r);

        tv2ri = (CRTM_14_4 * v133i) - (CRTM_14_6 * v511i) + (CRTM_14_2 * v97i);
        tv4ri = (CRTM_14_6 * v133i) + (CRTM_14_2 * v511i) - (CRTM_14_4 * v97i);
        tv6ri = (CRTM_14_2 * v133i) + (CRTM_14_4 * v511i) + (CRTM_14_6 * v97i);

        cv1i = v1i + (CRTM_14_3 * v313i) - (CRTM_14_5 * v115i) -
               (CRTM_14_1 * v79i);
        cv2i = v1i + (CRTM_14_3 * v79i) -
               ((CRTM_14_5 * v313i) + (CRTM_14_1 * v115i));
        cv3i = v1i + (CRTM_14_3 * v115i) - (CRTM_14_1 * v313i) -
               (CRTM_14_5 * v79i);

        //-------------------------------------------

        tvrr = cv1r + tv1rr;
        tvri = tv1ri + tv2ri;

        tvir = tv1ir + tv2ir;
        tvii = cv1i + tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_1 = tvrr - tvri;
            FLOAT _oi_1 = tvir + tvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        FLOAT _or_13 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (13 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[13]] = _or_13 * _twr + _oi * _twi;
            out_h2_i[out_strides[13]] = _oi * _twr - _or_13 * _twi;
        }

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_6 = tvrr - tvri;
            FLOAT _oi_6 = tvir + tvii;
            out_h1_r[out_strides[6]] = _or_6 * _twr + _oi_6 * _twi;
            out_h1_i[out_strides[6]] = _oi_6 * _twr - _or_6 * _twi;
        }
        FLOAT _or_8 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[8]] = _or_8 * _twr + _oi * _twi;
            out_h2_i[out_strides[8]] = _oi * _twr - _or_8 * _twi;
        }

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_2 = tvrr - tvri;
            FLOAT _oi_2 = tvir + tvii;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        FLOAT _or_12 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[12]] = _or_12 * _twr + _oi * _twi;
            out_h2_i[out_strides[12]] = _oi * _twr - _or_12 * _twi;
        }

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        {
            UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_5 = tvrr - tvri;
            FLOAT _oi_5 = tvir + tvii;
            out_h1_r[out_strides[5]] = _or_5 * _twr + _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _oi_5 * _twr - _or_5 * _twi;
        }
        FLOAT _or_9 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[9]] = _or_9 * _twr + _oi * _twi;
            out_h2_i[out_strides[9]] = _oi * _twr - _or_9 * _twi;
        }

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_3 = tvrr - tvri;
            FLOAT _oi_3 = tvir + tvii;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        FLOAT _or_11 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[11]] = _or_11 * _twr + _oi * _twi;
            out_h2_i[out_strides[11]] = _oi * _twr - _or_11 * _twi;
        }

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_4 = tvrr - tvri;
            FLOAT _oi_4 = tvir + tvii;
            out_h1_r[out_strides[4]] = _or_4 * _twr + _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _oi_4 * _twr - _or_4 * _twi;
        }
        FLOAT _or_10 = tvrr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[10]] = _or_10 * _twr + _oi * _twi;
            out_h2_i[out_strides[10]] = _oi * _twr - _or_10 * _twi;
        }

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

kfft_ register_kernel_twid_c2r_fft14c(UINT8 precision,
                                      UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft14c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft14c_fp64;
    }
    else
    {
        return NULL;
    }
}
