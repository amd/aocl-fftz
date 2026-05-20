// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_bwd_fft12c.c
 *
 *  @brief Backward-only twiddle Radix-12 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-12 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 12

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 118, 70, 0, 0},
                                                     {0, 60, 118, 70, 0, 0}};

ops_cycles_t get_ops_cnt_twid_bwd_fft12c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_bwd_fft12c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
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
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_DOUBLE twr, twi;

    in_h1_r = (FFTZ_DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

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
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        FFTZ_DOUBLE v2r = v2r_t * twr - v2i_t * twi;
        FFTZ_DOUBLE v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        FFTZ_DOUBLE v3r = v3r_t * twr - v3i_t * twi;
        FFTZ_DOUBLE v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v4r = v4r_t * twr - v4i_t * twi;
        FFTZ_DOUBLE v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h1_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v5r = v5r_t * twr - v5i_t * twi;
        FFTZ_DOUBLE v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h1_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h1_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v6r = v6r_t * twr - v6i_t * twi;
        FFTZ_DOUBLE v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v7r = v7r_t * twr - v7i_t * twi;
        FFTZ_DOUBLE v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v8r = v8r_t * twr - v8i_t * twi;
        FFTZ_DOUBLE v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_DOUBLE v9r_t = in_h2_r[in_strides[8]];
        FFTZ_DOUBLE v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v9r = v9r_t * twr - v9i_t * twi;
        FFTZ_DOUBLE v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FFTZ_DOUBLE v10r_t = in_h2_r[in_strides[9]];
        FFTZ_DOUBLE v10i_t = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v10r = v10r_t * twr - v10i_t * twi;
        FFTZ_DOUBLE v10i = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        FFTZ_DOUBLE v11r_t = in_h2_r[in_strides[10]];
        FFTZ_DOUBLE v11i_t = in_h2_i[in_strides[10]];
        twr = tw_ptr[9 * DATA_STRIDE];
        twi = tw_ptr[9 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v11r = v11r_t * twr - v11i_t * twi;
        FFTZ_DOUBLE v11i = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        FFTZ_DOUBLE v12r_t = in_h2_r[in_strides[11]];
        FFTZ_DOUBLE v12i_t = in_h2_i[in_strides[11]];
        twr = tw_ptr[10 * DATA_STRIDE];
        twi = tw_ptr[10 * DATA_STRIDE + 1];
        FFTZ_DOUBLE v12r = v12r_t * twr - v12i_t * twi;
        FFTZ_DOUBLE v12i = v12r_t * twi + v12i_t * twr;

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
        out_h2_r[out_strides[6]] = cv2r - cv2i;

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
        out_h1_r[out_strides[2]] = tvrr210 - tvri;
        out_h2_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_h1_r[out_strides[4]] = tvrr48 - tvri2;
        out_h2_r[out_strides[8]] = tvrr48 + tvri2;

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

        out_h2_i[out_strides[6]] = icv2r - icv2i;

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        out_h1_i[out_strides[2]] = tvii210 + tvir2;
        out_h2_i[out_strides[10]] = tvii210 - tvir2;

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        out_h1_i[out_strides[4]] = tvii48 + tvir2;
        out_h2_i[out_strides[8]] = tvii48 - tvir2;

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[9]] = tvrr + tvri;
        out_h1_i[out_strides[3]] = tvii + tvir;
        out_h2_i[out_strides[9]] = tvii - tvir;

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[11]] = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        out_h1_r[out_strides[5]] = tvrr2 - tvri2;
        out_h2_r[out_strides[7]] = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        out_h1_i[out_strides[1]] = tvii + tvir;
        out_h2_i[out_strides[11]] = tvii - tvir;

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        out_h1_i[out_strides[5]] = tvii2 + tvir2;
        out_h2_i[out_strides[7]] = tvii2 - tvir2;

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

static FFTZ_VOID twid_bwd_fft12c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
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
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_FLOAT twr, twi;

    in_h1_r = (FFTZ_FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_real;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

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
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        FFTZ_FLOAT v2r = v2r_t * twr - v2i_t * twi;
        FFTZ_FLOAT v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        FFTZ_FLOAT v3r = v3r_t * twr - v3i_t * twi;
        FFTZ_FLOAT v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        FFTZ_FLOAT v4r = v4r_t * twr - v4i_t * twi;
        FFTZ_FLOAT v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h1_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        FFTZ_FLOAT v5r = v5r_t * twr - v5i_t * twi;
        FFTZ_FLOAT v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h1_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h1_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        FFTZ_FLOAT v6r = v6r_t * twr - v6i_t * twi;
        FFTZ_FLOAT v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        FFTZ_FLOAT v7r = v7r_t * twr - v7i_t * twi;
        FFTZ_FLOAT v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        FFTZ_FLOAT v8r = v8r_t * twr - v8i_t * twi;
        FFTZ_FLOAT v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_FLOAT v9r_t = in_h2_r[in_strides[8]];
        FFTZ_FLOAT v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        FFTZ_FLOAT v9r = v9r_t * twr - v9i_t * twi;
        FFTZ_FLOAT v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FFTZ_FLOAT v10r_t = in_h2_r[in_strides[9]];
        FFTZ_FLOAT v10i_t = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
        FFTZ_FLOAT v10r = v10r_t * twr - v10i_t * twi;
        FFTZ_FLOAT v10i = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        FFTZ_FLOAT v11r_t = in_h2_r[in_strides[10]];
        FFTZ_FLOAT v11i_t = in_h2_i[in_strides[10]];
        twr = tw_ptr[9 * DATA_STRIDE];
        twi = tw_ptr[9 * DATA_STRIDE + 1];
        FFTZ_FLOAT v11r = v11r_t * twr - v11i_t * twi;
        FFTZ_FLOAT v11i = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        FFTZ_FLOAT v12r_t = in_h2_r[in_strides[11]];
        FFTZ_FLOAT v12i_t = in_h2_i[in_strides[11]];
        twr = tw_ptr[10 * DATA_STRIDE];
        twi = tw_ptr[10 * DATA_STRIDE + 1];
        FFTZ_FLOAT v12r = v12r_t * twr - v12i_t * twi;
        FFTZ_FLOAT v12i = v12r_t * twi + v12i_t * twr;

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
        out_h2_r[out_strides[6]] = cv2r - cv2i;

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
        out_h1_r[out_strides[2]] = tvrr210 - tvri;
        out_h2_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_h1_r[out_strides[4]] = tvrr48 - tvri2;
        out_h2_r[out_strides[8]] = tvrr48 + tvri2;

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

        out_h2_i[out_strides[6]] = icv2r - icv2i;

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        out_h1_i[out_strides[2]] = tvii210 + tvir2;
        out_h2_i[out_strides[10]] = tvii210 - tvir2;

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        out_h1_i[out_strides[4]] = tvii48 + tvir2;
        out_h2_i[out_strides[8]] = tvii48 - tvir2;

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[9]] = tvrr + tvri;
        out_h1_i[out_strides[3]] = tvii + tvir;
        out_h2_i[out_strides[9]] = tvii - tvir;

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[11]] = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        out_h1_r[out_strides[5]] = tvrr2 - tvri2;
        out_h2_r[out_strides[7]] = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        out_h1_i[out_strides[1]] = tvii + tvir;
        out_h2_i[out_strides[11]] = tvii - tvir;

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        out_h1_i[out_strides[5]] = tvii2 + tvir2;
        out_h2_i[out_strides[7]] = tvii2 - tvir2;

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

kfft_ register_kernel_twid_bwd_fft12c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_bwd_fft12c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_bwd_fft12c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
