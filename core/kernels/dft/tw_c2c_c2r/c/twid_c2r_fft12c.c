// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft12c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-12 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-12 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 118, 70, 0, 0},
                                                     {0, 60, 118, 70, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft12c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_c2r_fft12c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_12_1 =
        +0.86602540378443864676372317075293618347140262700000;
    const FFTZ_DOUBLE CRTM_12_2 =
        +0.50000000000000000000000000000000000000000000000000;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP cols = tws->cols;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5,
            sbr1, sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
            tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15, sbr51,
            sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2, tvir2,
            v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i, cv1r, cv2i,
            cv2r, icv1i, icv1r, icv2i, icv2r;

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        FFTZ_DOUBLE v1r = *in_h1_r;
        FFTZ_DOUBLE v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        FFTZ_DOUBLE v2r = v2r_t;
        FFTZ_DOUBLE v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        FFTZ_DOUBLE v3r = v3r_t;
        FFTZ_DOUBLE v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        FFTZ_DOUBLE v4r = v4r_t;
        FFTZ_DOUBLE v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h1_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h1_i[in_strides[4]];
        FFTZ_DOUBLE v5r = v5r_t;
        FFTZ_DOUBLE v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h1_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h1_i[in_strides[5]];
        FFTZ_DOUBLE v6r = v6r_t;
        FFTZ_DOUBLE v6i = v6i_t;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        FFTZ_DOUBLE v7r = v7r_t;
        FFTZ_DOUBLE v7i = v7i_t;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        v8r_t = -v8r_t;
        FFTZ_DOUBLE v8r = v8r_t;
        FFTZ_DOUBLE v8i = v8i_t;

        // Input point 9: x(8)
        FFTZ_DOUBLE v9r_t = in_h2_r[in_strides[8]];
        FFTZ_DOUBLE v9i_t = in_h2_i[in_strides[8]];
        v9r_t = -v9r_t;
        FFTZ_DOUBLE v9r = v9r_t;
        FFTZ_DOUBLE v9i = v9i_t;

        // Input point 10: x(9)
        FFTZ_DOUBLE v10r_t = in_h2_r[in_strides[9]];
        FFTZ_DOUBLE v10i_t = in_h2_i[in_strides[9]];
        v10r_t = -v10r_t;
        FFTZ_DOUBLE v10r = v10r_t;
        FFTZ_DOUBLE v10i = v10i_t;

        // Input point 11: x(10)
        FFTZ_DOUBLE v11r_t = in_h2_r[in_strides[10]];
        FFTZ_DOUBLE v11i_t = in_h2_i[in_strides[10]];
        v11r_t = -v11r_t;
        FFTZ_DOUBLE v11r = v11r_t;
        FFTZ_DOUBLE v11i = v11i_t;

        // Input point 12: x(11)
        FFTZ_DOUBLE v12r_t = in_h2_r[in_strides[11]];
        FFTZ_DOUBLE v12i_t = in_h2_i[in_strides[11]];
        v12r_t = -v12r_t;
        FFTZ_DOUBLE v12r = v12r_t;
        FFTZ_DOUBLE v12i = v12i_t;

        // Process FFT using twiddle-modified input values

        v3 = v5r;
        v4 = v9r;
        ad2 = v3 + v4;
        sbr4 = v4 - v3;

        v1 = v3r;
        v2 = v11r;
        ad1 = v1 + v2;
        sbr2 = v2 - v1;

        ad24 = ad1 + ad2;
        adr42 = ad1 - ad2;
        sbr24 = sbr2 - sbr4;
        tv5 = CRTM_12_1 * (sbr2 + sbr4);

        v1 = v2r;
        v2 = v12r;
        ad1 = v1 + v2;
        sbr1 = v2 - v1;

        v3 = v6r;
        v4 = v8r;
        ad2 = v3 + v4;
        sbr5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv7 = CRTM_12_1 * (ad1 - ad2);
        sbr15 = sbr1 + sbr5;
        sbr51 = sbr1 - sbr5;

        cv1i = ad15 + ad24;
        cv2i = ad15 - ad24;

        v1 = v4r;
        v2 = v10r;
        ad3 = v1 + v2;
        sbr3 = v2 - v1;

        tvir = sbr15 - sbr3;
        tv4 = CRTM_12_2 * sbr15 + sbr3;

        v4 = v7r;
        v3 = v1r;
        v17 = v3 + v4;
        v71r = v3 - v4;

        cv1r = v17 + ad3;
        cv2r = v17 - ad3;
        tvrr48 = cv1r - (CRTM_12_2 * cv1i);
        tvrr210 = cv2r + (CRTM_12_2 * cv2i);

        *out_h1_r = cv1r + cv1i;
        FFTZ_DOUBLE _or_6 = cv2r - cv2i;

        v3 = v5i;
        v4 = v9i;
        ad2 = v3 + v4;
        sbi4 = v4 - v3;

        v1 = v3i;
        v2 = v11i;
        ad1 = v1 + v2;
        sbi2 = v2 - v1;

        ad24 = ad1 + ad2;
        adi42 = ad1 - ad2;
        sbi24 = sbi2 - sbi4;
        tv3 = CRTM_12_1 * (sbi2 + sbi4);

        v1 = v2i;
        v2 = v12i;
        ad1 = v1 + v2;
        sbi1 = v2 - v1;

        v3 = v6i;
        v4 = v8i;
        ad2 = v3 + v4;
        sbi5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv8 = CRTM_12_1 * (ad1 - ad2);
        sbi15 = sbi1 + sbi5;
        sbi51 = sbi1 - sbi5;

        tvri = CRTM_12_1 * (sbi51 + sbi24);
        FFTZ_DOUBLE _or_2 = tvrr210 - tvri;
        FFTZ_DOUBLE _or_10 = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        FFTZ_DOUBLE _or_4 = tvrr48 - tvri2;
        FFTZ_DOUBLE _or_8 = tvrr48 + tvri2;

        icv1i = ad15 + ad24;
        icv2i = ad15 - ad24;

        v3 = v4i;
        v4 = v10i;
        ad3 = v3 + v4;
        sbi3 = v4 - v3;

        tvri = sbi15 - sbi3;
        tv2 = CRTM_12_2 * sbi15 + sbi3;

        v1 = v7i;
        v2 = v1i;
        v17 = v2 + v1;
        v71i = v2 - v1;

        icv1r = v17 + ad3;
        icv2r = v17 - ad3;

        *out_h1_i = icv1i + icv1r;

        tvii48 = icv1r - (CRTM_12_2 * icv1i);
        tvii210 = icv2r + (CRTM_12_2 * icv2i);

        {
            FFTZ_UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = icv2r - icv2i;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        FFTZ_DOUBLE _oi_2 = tvii210 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = tvii210 - tvir2;
            out_h2_r[out_strides[10]] = _or_10 * _twr + _oi * _twi;
            out_h2_i[out_strides[10]] = _oi * _twr - _or_10 * _twi;
        }

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        FFTZ_DOUBLE _oi_4 = tvii48 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            out_h1_r[out_strides[4]] = _or_4 * _twr + _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _oi_4 * _twr - _or_4 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = tvii48 - tvir2;
            out_h2_r[out_strides[8]] = _or_8 * _twr + _oi * _twi;
            out_h2_i[out_strides[8]] = _oi * _twr - _or_8 * _twi;
        }

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        FFTZ_DOUBLE _or_3 = tvrr - tvri;
        FFTZ_DOUBLE _or_9 = tvrr + tvri;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi_3 = tvii + tvir;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[9]] = _or_9 * _twr + _oi * _twi;
            out_h2_i[out_strides[9]] = _oi * _twr - _or_9 * _twi;
        }

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        FFTZ_DOUBLE _or_1 = tvrr - tvri;
        FFTZ_DOUBLE _or_11 = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        FFTZ_DOUBLE _or_5 = tvrr2 - tvri2;
        FFTZ_DOUBLE _or_7 = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        FFTZ_DOUBLE _oi_1 = tvii + tvir;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[11]] = _or_11 * _twr + _oi * _twi;
            out_h2_i[out_strides[11]] = _oi * _twr - _or_11 * _twi;
        }

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        FFTZ_DOUBLE _oi_5 = tvii2 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            out_h1_r[out_strides[5]] = _or_5 * _twr + _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _oi_5 * _twr - _or_5 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            FFTZ_DOUBLE _twr = tw[_twa];
            FFTZ_DOUBLE _twi = tw[1 + _twa];
            FFTZ_DOUBLE _oi = tvii2 - tvir2;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
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

static FFTZ_VOID twid_c2r_fft12c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_12_1 =
        +0.86602540378443864676372317075293618347140262700000f;
    const FFTZ_FLOAT CRTM_12_2 =
        +0.50000000000000000000000000000000000000000000000000f;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP cols = tws->cols;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_real;
    out_h2_i = out_h1_i;

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5, sbr1,
            sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
            tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15, sbr51,
            sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2, tvir2,
            v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i, cv1r, cv2i,
            cv2r, icv1i, icv1r, icv2i, icv2r;

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        FFTZ_FLOAT v1r = *in_h1_r;
        FFTZ_FLOAT v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        FFTZ_FLOAT v2r = v2r_t;
        FFTZ_FLOAT v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        FFTZ_FLOAT v3r = v3r_t;
        FFTZ_FLOAT v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        FFTZ_FLOAT v4r = v4r_t;
        FFTZ_FLOAT v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h1_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h1_i[in_strides[4]];
        FFTZ_FLOAT v5r = v5r_t;
        FFTZ_FLOAT v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h1_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h1_i[in_strides[5]];
        FFTZ_FLOAT v6r = v6r_t;
        FFTZ_FLOAT v6i = v6i_t;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        FFTZ_FLOAT v7r = v7r_t;
        FFTZ_FLOAT v7i = v7i_t;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        v8r_t = -v8r_t;
        FFTZ_FLOAT v8r = v8r_t;
        FFTZ_FLOAT v8i = v8i_t;

        // Input point 9: x(8)
        FFTZ_FLOAT v9r_t = in_h2_r[in_strides[8]];
        FFTZ_FLOAT v9i_t = in_h2_i[in_strides[8]];
        v9r_t = -v9r_t;
        FFTZ_FLOAT v9r = v9r_t;
        FFTZ_FLOAT v9i = v9i_t;

        // Input point 10: x(9)
        FFTZ_FLOAT v10r_t = in_h2_r[in_strides[9]];
        FFTZ_FLOAT v10i_t = in_h2_i[in_strides[9]];
        v10r_t = -v10r_t;
        FFTZ_FLOAT v10r = v10r_t;
        FFTZ_FLOAT v10i = v10i_t;

        // Input point 11: x(10)
        FFTZ_FLOAT v11r_t = in_h2_r[in_strides[10]];
        FFTZ_FLOAT v11i_t = in_h2_i[in_strides[10]];
        v11r_t = -v11r_t;
        FFTZ_FLOAT v11r = v11r_t;
        FFTZ_FLOAT v11i = v11i_t;

        // Input point 12: x(11)
        FFTZ_FLOAT v12r_t = in_h2_r[in_strides[11]];
        FFTZ_FLOAT v12i_t = in_h2_i[in_strides[11]];
        v12r_t = -v12r_t;
        FFTZ_FLOAT v12r = v12r_t;
        FFTZ_FLOAT v12i = v12i_t;

        // Process FFT using twiddle-modified input values

        v3 = v5r;
        v4 = v9r;
        ad2 = v3 + v4;
        sbr4 = v4 - v3;

        v1 = v3r;
        v2 = v11r;
        ad1 = v1 + v2;
        sbr2 = v2 - v1;

        ad24 = ad1 + ad2;
        adr42 = ad1 - ad2;
        sbr24 = sbr2 - sbr4;
        tv5 = CRTM_12_1 * (sbr2 + sbr4);

        v1 = v2r;
        v2 = v12r;
        ad1 = v1 + v2;
        sbr1 = v2 - v1;

        v3 = v6r;
        v4 = v8r;
        ad2 = v3 + v4;
        sbr5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv7 = CRTM_12_1 * (ad1 - ad2);
        sbr15 = sbr1 + sbr5;
        sbr51 = sbr1 - sbr5;

        cv1i = ad15 + ad24;
        cv2i = ad15 - ad24;

        v1 = v4r;
        v2 = v10r;
        ad3 = v1 + v2;
        sbr3 = v2 - v1;

        tvir = sbr15 - sbr3;
        tv4 = CRTM_12_2 * sbr15 + sbr3;

        v4 = v7r;
        v3 = v1r;
        v17 = v3 + v4;
        v71r = v3 - v4;

        cv1r = v17 + ad3;
        cv2r = v17 - ad3;
        tvrr48 = cv1r - (CRTM_12_2 * cv1i);
        tvrr210 = cv2r + (CRTM_12_2 * cv2i);

        *out_h1_r = cv1r + cv1i;
        FFTZ_FLOAT _or_6 = cv2r - cv2i;

        v3 = v5i;
        v4 = v9i;
        ad2 = v3 + v4;
        sbi4 = v4 - v3;

        v1 = v3i;
        v2 = v11i;
        ad1 = v1 + v2;
        sbi2 = v2 - v1;

        ad24 = ad1 + ad2;
        adi42 = ad1 - ad2;
        sbi24 = sbi2 - sbi4;
        tv3 = CRTM_12_1 * (sbi2 + sbi4);

        v1 = v2i;
        v2 = v12i;
        ad1 = v1 + v2;
        sbi1 = v2 - v1;

        v3 = v6i;
        v4 = v8i;
        ad2 = v3 + v4;
        sbi5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv8 = CRTM_12_1 * (ad1 - ad2);
        sbi15 = sbi1 + sbi5;
        sbi51 = sbi1 - sbi5;

        tvri = CRTM_12_1 * (sbi51 + sbi24);
        FFTZ_FLOAT _or_2 = tvrr210 - tvri;
        FFTZ_FLOAT _or_10 = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        FFTZ_FLOAT _or_4 = tvrr48 - tvri2;
        FFTZ_FLOAT _or_8 = tvrr48 + tvri2;

        icv1i = ad15 + ad24;
        icv2i = ad15 - ad24;

        v3 = v4i;
        v4 = v10i;
        ad3 = v3 + v4;
        sbi3 = v4 - v3;

        tvri = sbi15 - sbi3;
        tv2 = CRTM_12_2 * sbi15 + sbi3;

        v1 = v7i;
        v2 = v1i;
        v17 = v2 + v1;
        v71i = v2 - v1;

        icv1r = v17 + ad3;
        icv2r = v17 - ad3;

        *out_h1_i = icv1i + icv1r;

        tvii48 = icv1r - (CRTM_12_2 * icv1i);
        tvii210 = icv2r + (CRTM_12_2 * icv2i);

        {
            FFTZ_UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = icv2r - icv2i;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        FFTZ_FLOAT _oi_2 = tvii210 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = tvii210 - tvir2;
            out_h2_r[out_strides[10]] = _or_10 * _twr + _oi * _twi;
            out_h2_i[out_strides[10]] = _oi * _twr - _or_10 * _twi;
        }

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        FFTZ_FLOAT _oi_4 = tvii48 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            out_h1_r[out_strides[4]] = _or_4 * _twr + _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _oi_4 * _twr - _or_4 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = tvii48 - tvir2;
            out_h2_r[out_strides[8]] = _or_8 * _twr + _oi * _twi;
            out_h2_i[out_strides[8]] = _oi * _twr - _or_8 * _twi;
        }

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        FFTZ_FLOAT _or_3 = tvrr - tvri;
        FFTZ_FLOAT _or_9 = tvrr + tvri;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi_3 = tvii + tvir;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[9]] = _or_9 * _twr + _oi * _twi;
            out_h2_i[out_strides[9]] = _oi * _twr - _or_9 * _twi;
        }

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        FFTZ_FLOAT _or_1 = tvrr - tvri;
        FFTZ_FLOAT _or_11 = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        FFTZ_FLOAT _or_5 = tvrr2 - tvri2;
        FFTZ_FLOAT _or_7 = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        FFTZ_FLOAT _oi_1 = tvii + tvir;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[11]] = _or_11 * _twr + _oi * _twi;
            out_h2_i[out_strides[11]] = _oi * _twr - _or_11 * _twi;
        }

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        FFTZ_FLOAT _oi_5 = tvii2 + tvir2;
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            out_h1_r[out_strides[5]] = _or_5 * _twr + _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _oi_5 * _twr - _or_5 * _twi;
        }
        {
            FFTZ_UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            FFTZ_FLOAT _twr = tw[_twa];
            FFTZ_FLOAT _twi = tw[1 + _twa];
            FFTZ_FLOAT _oi = tvii2 - tvir2;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
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

kfft_ register_kernel_twid_c2r_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft12c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft12c_fp64;
    }
    else
    {
        return NULL;
    }
}
