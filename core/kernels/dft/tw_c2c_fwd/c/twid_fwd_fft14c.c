// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fwd_fft14c.c
 *
 *  @brief Forward-only twiddle Radix-14 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-14 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 14

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 124, 174, 82, 0, 0},
                                                     {0, 124, 174, 82, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fwd_fft14c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_fwd_fft14c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000;
    const FFTZ_DOUBLE CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829;
    const FFTZ_DOUBLE CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372;
    const FFTZ_DOUBLE CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327;
    const FFTZ_DOUBLE CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720;
    const FFTZ_DOUBLE CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000;

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
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
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
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r,
            v313r, v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i,
            v97i, v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r, v124r,
            v115r, v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i, cv2r, cv2i,
            cv3r, cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii, tv1ri, tv2ri,
            tv1ir, tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir, tv4ir, tv5ir,
            tv6ir;

        FFTZ_DOUBLE t_v1r = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE t_v1i = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        FFTZ_DOUBLE t_v2r = in_h2_r[in_strides[13]];
        FFTZ_DOUBLE t_v2i = in_h2_i[in_strides[13]];
        twr = tw_ptr[12 * DATA_STRIDE];
        twi = tw_ptr[12 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[3]];
        t_v1i = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        t_v2r = in_h2_r[in_strides[11]];
        t_v2i = in_h2_i[in_strides[11]];
        twr = tw_ptr[10 * DATA_STRIDE];
        twi = tw_ptr[10 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[5]];
        t_v1i = in_h1_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        t_v2r = in_h2_r[in_strides[9]];
        t_v2i = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        FFTZ_DOUBLE t_v8r = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE t_v8i = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = t_v8r * twr - t_v8i * twi;
        v8i = t_v8r * twi + t_v8i * twr;

        FFTZ_DOUBLE t_v3r = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE t_v3i = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        FFTZ_DOUBLE t_v4r = in_h2_r[in_strides[12]];
        FFTZ_DOUBLE t_v4i = in_h2_i[in_strides[12]];
        twr = tw_ptr[11 * DATA_STRIDE];
        twi = tw_ptr[11 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        t_v3r = in_h1_r[in_strides[4]];
        t_v3i = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        t_v4r = in_h2_r[in_strides[10]];
        t_v4i = in_h2_i[in_strides[10]];
        twr = tw_ptr[9 * DATA_STRIDE];
        twi = tw_ptr[9 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

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
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        t_v4r = in_h2_r[in_strides[8]];
        t_v4i = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

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

        out_h2_r[out_strides[7]] = tvrr - tvri;
        out_h2_i[out_strides[7]] = tvii - tvir;

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

        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[13]] = tvrr + tvri;
        out_h1_i[out_strides[1]] = tvir + tvii;

        out_h2_i[out_strides[13]] = tvii - tvir;

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        out_h1_r[out_strides[6]] = tvrr - tvri;
        out_h2_r[out_strides[8]] = tvrr + tvri;
        out_h1_i[out_strides[6]] = tvir + tvii;

        out_h2_i[out_strides[8]] = tvii - tvir;

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        out_h1_r[out_strides[2]] = tvrr - tvri;
        out_h2_r[out_strides[12]] = tvrr + tvri;
        out_h1_i[out_strides[2]] = tvir + tvii;

        out_h2_i[out_strides[12]] = tvii - tvir;

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        out_h1_r[out_strides[5]] = tvrr - tvri;
        out_h2_r[out_strides[9]] = tvrr + tvri;
        out_h1_i[out_strides[5]] = tvir + tvii;

        out_h2_i[out_strides[9]] = tvii - tvir;

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[11]] = tvrr + tvri;
        out_h1_i[out_strides[3]] = tvir + tvii;

        out_h2_i[out_strides[11]] = tvii - tvir;

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        out_h1_r[out_strides[4]] = tvrr - tvri;
        out_h2_r[out_strides[10]] = tvrr + tvri;
        out_h1_i[out_strides[4]] = tvir + tvii;

        out_h2_i[out_strides[10]] = tvii - tvir;

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

static FFTZ_VOID twid_fwd_fft14c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000f;
    const FFTZ_FLOAT CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829f;
    const FFTZ_FLOAT CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372f;
    const FFTZ_FLOAT CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327f;
    const FFTZ_FLOAT CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720f;
    const FFTZ_FLOAT CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000f;

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
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
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
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r,
            v313r, v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i,
            v97i, v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r, v124r,
            v115r, v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i, cv2r, cv2i,
            cv3r, cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii, tv1ri, tv2ri,
            tv1ir, tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir, tv4ir, tv5ir,
            tv6ir;

        FFTZ_FLOAT t_v1r = in_h1_r[in_strides[1]];
        FFTZ_FLOAT t_v1i = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        FFTZ_FLOAT t_v2r = in_h2_r[in_strides[13]];
        FFTZ_FLOAT t_v2i = in_h2_i[in_strides[13]];
        twr = tw_ptr[12 * DATA_STRIDE];
        twi = tw_ptr[12 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[3]];
        t_v1i = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        t_v2r = in_h2_r[in_strides[11]];
        t_v2i = in_h2_i[in_strides[11]];
        twr = tw_ptr[10 * DATA_STRIDE];
        twi = tw_ptr[10 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        t_v1r = in_h1_r[in_strides[5]];
        t_v1i = in_h1_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v1r = t_v1r * twr - t_v1i * twi;
        v1i = t_v1r * twi + t_v1i * twr;

        t_v2r = in_h2_r[in_strides[9]];
        t_v2i = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
        v2r = t_v2r * twr - t_v2i * twi;
        v2i = t_v2r * twi + t_v2i * twr;

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        FFTZ_FLOAT t_v8r = in_h2_r[in_strides[7]];
        FFTZ_FLOAT t_v8i = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = t_v8r * twr - t_v8i * twi;
        v8i = t_v8r * twi + t_v8i * twr;

        FFTZ_FLOAT t_v3r = in_h1_r[in_strides[2]];
        FFTZ_FLOAT t_v3i = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        FFTZ_FLOAT t_v4r = in_h2_r[in_strides[12]];
        FFTZ_FLOAT t_v4i = in_h2_i[in_strides[12]];
        twr = tw_ptr[11 * DATA_STRIDE];
        twi = tw_ptr[11 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        t_v3r = in_h1_r[in_strides[4]];
        t_v3i = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        t_v4r = in_h2_r[in_strides[10]];
        t_v4i = in_h2_i[in_strides[10]];
        twr = tw_ptr[9 * DATA_STRIDE];
        twi = tw_ptr[9 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

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
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v3r = t_v3r * twr - t_v3i * twi;
        v3i = t_v3r * twi + t_v3i * twr;

        t_v4r = in_h2_r[in_strides[8]];
        t_v4i = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        v4r = t_v4r * twr - t_v4i * twi;
        v4i = t_v4r * twi + t_v4i * twr;

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

        out_h2_r[out_strides[7]] = tvrr - tvri;
        out_h2_i[out_strides[7]] = tvii - tvir;

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

        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[13]] = tvrr + tvri;
        out_h1_i[out_strides[1]] = tvir + tvii;

        out_h2_i[out_strides[13]] = tvii - tvir;

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        out_h1_r[out_strides[6]] = tvrr - tvri;
        out_h2_r[out_strides[8]] = tvrr + tvri;
        out_h1_i[out_strides[6]] = tvir + tvii;

        out_h2_i[out_strides[8]] = tvii - tvir;

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        out_h1_r[out_strides[2]] = tvrr - tvri;
        out_h2_r[out_strides[12]] = tvrr + tvri;
        out_h1_i[out_strides[2]] = tvir + tvii;

        out_h2_i[out_strides[12]] = tvii - tvir;

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        out_h1_r[out_strides[5]] = tvrr - tvri;
        out_h2_r[out_strides[9]] = tvrr + tvri;
        out_h1_i[out_strides[5]] = tvir + tvii;

        out_h2_i[out_strides[9]] = tvii - tvir;

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[11]] = tvrr + tvri;
        out_h1_i[out_strides[3]] = tvir + tvii;

        out_h2_i[out_strides[11]] = tvii - tvir;

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        out_h1_r[out_strides[4]] = tvrr - tvri;
        out_h2_r[out_strides[10]] = tvrr + tvri;
        out_h1_i[out_strides[4]] = tvir + tvii;

        out_h2_i[out_strides[10]] = tvii - tvir;

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

kfft_ register_kernel_twid_fwd_fft14c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fwd_fft14c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fwd_fft14c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
