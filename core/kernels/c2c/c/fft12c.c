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

/** @file fft12c.c
 *
 *  @brief Radix-12 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-12 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 16, 96, 48, 0, 0},
                                                     {0, 16, 96, 48, 0, 0}};

ops_cycles_t get_ops_cnt_fft12c(UINT8 precision, UINT8 direction)
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

static VOID fft12c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_12_1 =
        +0.86602540378443864676372317075293618347140262700000;
    const DOUBLE CRTM_12_2 =
        +0.50000000000000000000000000000000000000000000000000;

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
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5, sbr1,
               sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
               tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15,
               sbr51, sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2,
               tvir2, v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i,
               cv1r, cv2i, cv2r, icv1i, icv1r, icv2i, icv2r;

        v3 = in_r[in_strides[4]];
        v4 = in_r[in_strides[8]];
        ad2 = v3 + v4;
        sbr4 = v4 - v3;

        v1 = in_r[in_strides[2]];
        v2 = in_r[in_strides[10]];
        ad1 = v1 + v2;
        sbr2 = v2 - v1;

        ad24 = ad1 + ad2;
        adr42 = ad1 - ad2;
        sbr24 = sbr2 - sbr4;
        tv5 = CRTM_12_1 * (sbr2 + sbr4);

        v1 = in_r[in_strides[1]];
        v2 = in_r[in_strides[11]];
        ad1 = v1 + v2;
        sbr1 = v2 - v1;

        v3 = in_r[in_strides[5]];
        v4 = in_r[in_strides[7]];
        ad2 = v3 + v4;
        sbr5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv7 = CRTM_12_1 * (ad1 - ad2);
        sbr15 = sbr1 + sbr5;
        sbr51 = sbr1 - sbr5;

        cv1i = ad15 + ad24;
        cv2i = ad15 - ad24;

        v1 = in_r[in_strides[3]];
        v2 = in_r[in_strides[9]];
        ad3 = v1 + v2;
        sbr3 = v2 - v1;

        tvir = sbr15 - sbr3;
        tv4 = CRTM_12_2 * sbr15 + sbr3;

        v4 = in_r[in_strides[6]];
        v3 = *in_r;
        v17 = v3 + v4;
        v71r = v3 - v4;

        cv1r = v17 + ad3;
        cv2r = v17 - ad3;
        tvrr48 = cv1r - (CRTM_12_2 * cv1i);
        tvrr210 = cv2r + (CRTM_12_2 * cv2i);

        *out_r = cv1r + cv1i;
        out_r[out_strides[6]] = cv2r - cv2i;

        v3 = in_i[in_strides[4]];
        v4 = in_i[in_strides[8]];
        ad2 = v3 + v4;
        sbi4 = v4 - v3;

        v1 = in_i[in_strides[2]];
        v2 = in_i[in_strides[10]];
        ad1 = v1 + v2;
        sbi2 = v2 - v1;

        ad24 = ad1 + ad2;
        adi42 = ad1 - ad2;
        sbi24 = sbi2 - sbi4;
        tv3 = CRTM_12_1 * (sbi2 + sbi4);

        v1 = in_i[in_strides[1]];
        v2 = in_i[in_strides[11]];
        ad1 = v1 + v2;
        sbi1 = v2 - v1;

        v3 = in_i[in_strides[5]];
        v4 = in_i[in_strides[7]];
        ad2 = v3 + v4;
        sbi5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv8 = CRTM_12_1 * (ad1 - ad2);
        sbi15 = sbi1 + sbi5;
        sbi51 = sbi1 - sbi5;

        tvri = CRTM_12_1 * (sbi51 + sbi24);
        out_r[out_strides[2]] = tvrr210 - tvri;
        out_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_r[out_strides[4]] = tvrr48 - tvri2;
        out_r[out_strides[8]] = tvrr48 + tvri2;

        icv1i = ad15 + ad24;
        icv2i = ad15 - ad24;

        v3 = in_i[in_strides[3]];
        v4 = in_i[in_strides[9]];
        ad3 = v3 + v4;
        sbi3 = v4 - v3;

        tvri = sbi15 - sbi3;
        tv2 = CRTM_12_2 * sbi15 + sbi3;

        v1 = in_i[in_strides[6]];
        v2 = *in_i;
        v17 = v2 + v1;
        v71i = v2 - v1;

        icv1r = v17 + ad3;
        icv2r = v17 - ad3;

        *out_i = icv1i + icv1r;

        tvii48 = icv1r - (CRTM_12_2 * icv1i);
        tvii210 = icv2r + (CRTM_12_2 * icv2i);

        out_i[out_strides[6]] = icv2r - icv2i;

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        out_i[out_strides[2]] = tvii210 + tvir2;
        out_i[out_strides[10]] = tvii210 - tvir2;

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        out_i[out_strides[4]] = tvii48 + tvir2;
        out_i[out_strides[8]] = tvii48 - tvir2;

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii + tvir;
        out_i[out_strides[9]] = tvii - tvir;

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        out_r[out_strides[5]] = tvrr2 - tvri2;
        out_r[out_strides[7]] = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        out_i[out_strides[1]] = tvii + tvir;
        out_i[out_strides[11]] = tvii - tvir;

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        out_i[out_strides[5]] = tvii2 + tvir2;
        out_i[out_strides[7]] = tvii2 - tvir2;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft12c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_12_1 =
        +0.86602540378443864676372317075293618347140262700000;
    const FLOAT CRTM_12_2 =
        +0.50000000000000000000000000000000000000000000000000;

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
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5, sbr1,
              sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
              tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15,
              sbr51, sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2,
              tvir2, v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i, cv1r,
              cv2i, cv2r, icv1i, icv1r, icv2i, icv2r;

        v3 = in_r[in_strides[4]];
        v4 = in_r[in_strides[8]];
        ad2 = v3 + v4;
        sbr4 = v4 - v3;

        v1 = in_r[in_strides[2]];
        v2 = in_r[in_strides[10]];
        ad1 = v1 + v2;
        sbr2 = v2 - v1;

        ad24 = ad1 + ad2;
        adr42 = ad1 - ad2;
        sbr24 = sbr2 - sbr4;
        tv5 = CRTM_12_1 * (sbr2 + sbr4);

        v1 = in_r[in_strides[1]];
        v2 = in_r[in_strides[11]];
        ad1 = v1 + v2;
        sbr1 = v2 - v1;

        v3 = in_r[in_strides[5]];
        v4 = in_r[in_strides[7]];
        ad2 = v3 + v4;
        sbr5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv7 = CRTM_12_1 * (ad1 - ad2);
        sbr15 = sbr1 + sbr5;
        sbr51 = sbr1 - sbr5;

        cv1i = ad15 + ad24;
        cv2i = ad15 - ad24;

        v1 = in_r[in_strides[3]];
        v2 = in_r[in_strides[9]];
        ad3 = v1 + v2;
        sbr3 = v2 - v1;

        tvir = sbr15 - sbr3;
        tv4 = CRTM_12_2 * sbr15 + sbr3;

        v4 = in_r[in_strides[6]];
        v3 = *in_r;
        v17 = v3 + v4;
        v71r = v3 - v4;

        cv1r = v17 + ad3;
        cv2r = v17 - ad3;
        tvrr48 = cv1r - (CRTM_12_2 * cv1i);
        tvrr210 = cv2r + (CRTM_12_2 * cv2i);

        *out_r = cv1r + cv1i;
        out_r[out_strides[6]] = cv2r - cv2i;

        v3 = in_i[in_strides[4]];
        v4 = in_i[in_strides[8]];
        ad2 = v3 + v4;
        sbi4 = v4 - v3;

        v1 = in_i[in_strides[2]];
        v2 = in_i[in_strides[10]];
        ad1 = v1 + v2;
        sbi2 = v2 - v1;

        ad24 = ad1 + ad2;
        adi42 = ad1 - ad2;
        sbi24 = sbi2 - sbi4;
        tv3 = CRTM_12_1 * (sbi2 + sbi4);

        v1 = in_i[in_strides[1]];
        v2 = in_i[in_strides[11]];
        ad1 = v1 + v2;
        sbi1 = v2 - v1;

        v3 = in_i[in_strides[5]];
        v4 = in_i[in_strides[7]];
        ad2 = v3 + v4;
        sbi5 = v4 - v3;

        ad15 = ad1 + ad2;
        tv8 = CRTM_12_1 * (ad1 - ad2);
        sbi15 = sbi1 + sbi5;
        sbi51 = sbi1 - sbi5;

        tvri = CRTM_12_1 * (sbi51 + sbi24);
        out_r[out_strides[2]] = tvrr210 - tvri;
        out_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_r[out_strides[4]] = tvrr48 - tvri2;
        out_r[out_strides[8]] = tvrr48 + tvri2;

        icv1i = ad15 + ad24;
        icv2i = ad15 - ad24;

        v3 = in_i[in_strides[3]];
        v4 = in_i[in_strides[9]];
        ad3 = v3 + v4;
        sbi3 = v4 - v3;

        tvri = sbi15 - sbi3;
        tv2 = CRTM_12_2 * sbi15 + sbi3;

        v1 = in_i[in_strides[6]];
        v2 = *in_i;
        v17 = v2 + v1;
        v71i = v2 - v1;

        icv1r = v17 + ad3;
        icv2r = v17 - ad3;

        *out_i = icv1i + icv1r;

        tvii48 = icv1r - (CRTM_12_2 * icv1i);
        tvii210 = icv2r + (CRTM_12_2 * icv2i);

        out_i[out_strides[6]] = icv2r - icv2i;

        tvir2 = CRTM_12_1 * (sbr51 + sbr24);
        out_i[out_strides[2]] = tvii210 + tvir2;
        out_i[out_strides[10]] = tvii210 - tvir2;

        tvir2 = CRTM_12_1 * (sbr51 - sbr24);
        out_i[out_strides[4]] = tvii48 + tvir2;
        out_i[out_strides[8]] = tvii48 - tvir2;

        tvrr = v71r - adr42;
        tvii = v71i - adi42;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii + tvir;
        out_i[out_strides[9]] = tvii - tvir;

        v711r = v71r + CRTM_12_2 * adr42;
        v711i = v71i + CRTM_12_2 * adi42;

        tvrr = v711r + tv7;
        tvri = tv2 + tv3;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;

        tvrr2 = v711r - tv7;
        tvri2 = tv2 - tv3;
        out_r[out_strides[5]] = tvrr2 - tvri2;
        out_r[out_strides[7]] = tvrr2 + tvri2;

        tvir = tv4 + tv5;
        tvii = v711i + tv8;
        out_i[out_strides[1]] = tvii + tvir;
        out_i[out_strides[11]] = tvii - tvir;

        tvir2 = tv4 - tv5;
        tvii2 = v711i - tv8;
        out_i[out_strides[5]] = tvii2 + tvir2;
        out_i[out_strides[7]] = tvii2 - tvir2;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft12c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft12c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft12c_fp64;
    }
    else
    {
        return NULL;
    }
}
