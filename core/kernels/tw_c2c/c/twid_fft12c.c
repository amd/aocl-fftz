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

/** @file twid_fft12c.c
 *
 *  @brief Twiddle Radix-12 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-12 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 11 * (4 muls, 2 adds, 2 movs [loads])
//              = (44, 22, 22)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 118, 70, 0, 0},
                                                     {0, 60, 118, 70, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft12c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft12c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
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

    DOUBLE *tw = (DOUBLE *)twd;
    DOUBLE twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5, sbr1,
            sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
            tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15, sbr51,
            sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2, tvir2,
            v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i, cv1r, cv2i,
            cv2r, icv1i, icv1r, icv2i, icv2r;

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        DOUBLE v1r = *in_r;
        DOUBLE v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        DOUBLE v2r = v2r_t * twr - v2i_t * twi;
        DOUBLE v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        DOUBLE v3r = v3r_t * twr - v3i_t * twi;
        DOUBLE v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        DOUBLE v4r = v4r_t * twr - v4i_t * twi;
        DOUBLE v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        DOUBLE v5r = v5r_t * twr - v5i_t * twi;
        DOUBLE v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_r[in_strides[5]];
        DOUBLE v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * n + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        DOUBLE v6r = v6r_t * twr - v6i_t * twi;
        DOUBLE v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_r[in_strides[6]];
        DOUBLE v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * n + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        DOUBLE v7r = v7r_t * twr - v7i_t * twi;
        DOUBLE v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_r[in_strides[7]];
        DOUBLE v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * n + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        DOUBLE v8r = v8r_t * twr - v8i_t * twi;
        DOUBLE v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        DOUBLE v9r_t = in_r[in_strides[8]];
        DOUBLE v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * n + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        DOUBLE v9r = v9r_t * twr - v9i_t * twi;
        DOUBLE v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        DOUBLE v10r_t = in_r[in_strides[9]];
        DOUBLE v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * n + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        DOUBLE v10r = v10r_t * twr - v10i_t * twi;
        DOUBLE v10i = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        DOUBLE v11r_t = in_r[in_strides[10]];
        DOUBLE v11i_t = in_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * n + cnt);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        DOUBLE v11r = v11r_t * twr - v11i_t * twi;
        DOUBLE v11i = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        DOUBLE v12r_t = in_r[in_strides[11]];
        DOUBLE v12i_t = in_i[in_strides[11]];
        UINTP twid_addr12 = DATA_STRIDE * (11 * n + cnt);
        twr = tw[twid_addr12];
        twi = tw[1 + twid_addr12];
        DOUBLE v12r = v12r_t * twr - v12i_t * twi;
        DOUBLE v12i = v12r_t * twi + v12i_t * twr;

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

        *out_r = cv1r + cv1i;
        out_r[out_strides[6]] = cv2r - cv2i;

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
        out_r[out_strides[2]] = tvrr210 - tvri;
        out_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_r[out_strides[4]] = tvrr48 - tvri2;
        out_r[out_strides[8]] = tvrr48 + tvri2;

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

static VOID twid_fft12c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
                             UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_12_1 =
        +0.86602540378443864676372317075293618347140262700000f;
    const FLOAT CRTM_12_2 =
        +0.50000000000000000000000000000000000000000000000000f;

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

    FLOAT *tw = (FLOAT *)twd;
    FLOAT twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v71r, v71i, v711r, v711i, sbi1, sbi2, sbi3, sbi4, sbi5, sbr1,
            sbr2, sbr3, sbr4, sbr5, tv2, tv3, tv4, tv5, tv7, tv8, tvrr48,
            tvii48, tvrr210, tvii210, adr42, adi42, sbi15, sbi51, sbr15, sbr51,
            sbi24, sbr24, tvrr, tvri, tvii, tvir, tvrr2, tvri2, tvii2, tvir2,
            v1, v2, v3, v4, ad1, ad2, ad3, v17, ad15, ad24, cv1i, cv1r, cv2i,
            cv2r, icv1i, icv1r, icv2i, icv2r;

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        FLOAT v1r = *in_r;
        FLOAT v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        FLOAT v2r = v2r_t * twr - v2i_t * twi;
        FLOAT v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        FLOAT v3r = v3r_t * twr - v3i_t * twi;
        FLOAT v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        FLOAT v4r = v4r_t * twr - v4i_t * twi;
        FLOAT v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        FLOAT v5r = v5r_t * twr - v5i_t * twi;
        FLOAT v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FLOAT v6r_t = in_r[in_strides[5]];
        FLOAT v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * n + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        FLOAT v6r = v6r_t * twr - v6i_t * twi;
        FLOAT v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FLOAT v7r_t = in_r[in_strides[6]];
        FLOAT v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * n + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        FLOAT v7r = v7r_t * twr - v7i_t * twi;
        FLOAT v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FLOAT v8r_t = in_r[in_strides[7]];
        FLOAT v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * n + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        FLOAT v8r = v8r_t * twr - v8i_t * twi;
        FLOAT v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FLOAT v9r_t = in_r[in_strides[8]];
        FLOAT v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * n + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        FLOAT v9r = v9r_t * twr - v9i_t * twi;
        FLOAT v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FLOAT v10r_t = in_r[in_strides[9]];
        FLOAT v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * n + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        FLOAT v10r = v10r_t * twr - v10i_t * twi;
        FLOAT v10i = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        FLOAT v11r_t = in_r[in_strides[10]];
        FLOAT v11i_t = in_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * n + cnt);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        FLOAT v11r = v11r_t * twr - v11i_t * twi;
        FLOAT v11i = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        FLOAT v12r_t = in_r[in_strides[11]];
        FLOAT v12i_t = in_i[in_strides[11]];
        UINTP twid_addr12 = DATA_STRIDE * (11 * n + cnt);
        twr = tw[twid_addr12];
        twi = tw[1 + twid_addr12];
        FLOAT v12r = v12r_t * twr - v12i_t * twi;
        FLOAT v12i = v12r_t * twi + v12i_t * twr;

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

        *out_r = cv1r + cv1i;
        out_r[out_strides[6]] = cv2r - cv2i;

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
        out_r[out_strides[2]] = tvrr210 - tvri;
        out_r[out_strides[10]] = tvrr210 + tvri;

        tvri2 = CRTM_12_1 * (sbi51 - sbi24);
        out_r[out_strides[4]] = tvrr48 - tvri2;
        out_r[out_strides[8]] = tvrr48 + tvri2;

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

kfft_ register_kernel_twid_fft12c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft12c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft12c_fp64;
    }
    else
    {
        return NULL;
    }
}
