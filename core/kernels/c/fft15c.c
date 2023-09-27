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

/** @file fft15c.c
 *
 *  @brief Radix-15 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-15 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author S. Biplab Raut
 * @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft15c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft15c_fp32;
    else if (precision == DT_DOUBLE)
        return fft15c_fp64;
    else
        return NULL;
}
#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
// TODO: update ops_cnt
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 104, 203, 60, 0, 0},
                                                     {0, 104, 203, 60, 0, 0}};

ops_cycles_t get_ops_cnt_fft15c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

VOID fft15c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    const DOUBLE CRTM_15_1 =
        +0.91354545764260089786653411260175763014630651476859;
    const DOUBLE CRTM_15_2 =
        +0.40673664307580020244344195165731987476994255841557;
    const DOUBLE CRTM_15_3 =
        +0.66913060635885822246624475148920014459869481182487;
    const DOUBLE CRTM_15_4 =
        +0.74314482547739422723523183906600387567804039038099;
    const DOUBLE CRTM_15_5 =
        +0.30901699437494735775909215694883353691106407759980;
    const DOUBLE CRTM_15_6 =
        +0.95105651629515359367265213397317432075121805913368;
    const DOUBLE CRTM_15_7 =
        +0.10452846326765344827475689190358280057779765406449;
    const DOUBLE CRTM_15_8 =
        +0.99452189536827333935323550615724320908650656433540;
    const DOUBLE CRTM_15_9 =
        +0.49999999999999989931390918883503052434117283719579;
    const DOUBLE CRTM_15_10 =
        +0.86602540378443870489486480423012865512801223847686;
    const DOUBLE CRTM_15_11 =
        +0.80901699437494750610700001978173170344740065252139;
    const DOUBLE CRTM_15_12 =
        +0.58778525229247301629891039327884007596190389052978;
    const DOUBLE CRTM_15_13 =
        +0.97814760073380564759748190481590945657945761737292;
    const DOUBLE CRTM_15_14 =
        +0.20791169081775929161307291104291640870739670391251;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v14r, v14i, v15r, v15i,

            adr1, adr2, adr3, adr4, adr5, adr6, adr7, adi1, adi2, adi3, adi4,
            adi5, adi6, adi7, sbi1, sbi2, sbi3, sbi4, sbi5, sbi6, sbi7, sbr1,
            sbr2, sbr3, sbr4, sbr5, sbr6, sbr7, adr12, adi12,

            tvrr, tvri, tvir, tvii,

            tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13,
            tv14, tv15, tv16, tv17, tv18, tv19, tv20, adr9, adr11, sbi8, adi10,
            adi11, sbr8, adr8, adi9, adr10, adi8, sbr9, sbi9, tv21, tv22, tv23,
            tv24, tv25, tv26, tv27, tv28, tv29, tv30;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];

        // Input point 3: x(2)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        // Input point 4: x(3)
        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        // Input point 6: x(5)
        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        // Input point 7: x(6)
        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        // Input point 8: x(7)
        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        // Input point 10: x(9)
        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        // Input point 11: x(10)
        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // Input point 12: x(11)
        v12r = in_r[in_stride * 11];
        v12i = in_i[in_stride * 11];

        // Input point 13: x(12)
        v13r = in_r[in_stride * 12];
        v13i = in_i[in_stride * 12];

        // Input point 14: x(13)
        v14r = in_r[in_stride * 13];
        v14i = in_i[in_stride * 13];

        // Input point 15: x(14)
        v15r = in_r[in_stride * 14];
        v15i = in_i[in_stride * 14];

        // common calculations
        adr1 = v2r + v15r;
        adr2 = v3r + v14r;
        adr3 = v4r + v13r;
        adr4 = v5r + v12r;
        adr5 = v6r + v11r;
        adr6 = v7r + v10r;
        adr7 = v8r + v9r;
        adr8 = adr1 + adr4;
        adr9 = adr8 + adr6;
        adr10 = adr2 + adr7;
        adr11 = adr10 + adr3;
        adr12 = v1r + adr5;

        sbi1 = v15i - v2i;
        sbi2 = v14i - v3i;
        sbi3 = v13i - v4i;
        sbi4 = v12i - v5i;
        sbi5 = v11i - v6i;
        sbi6 = v10i - v7i;
        sbi7 = v9i - v8i;
        sbi8 = sbi1 - sbi4 + sbi6;
        sbi9 = sbi2 + sbi7;

        sbr1 = v15r - v2r;
        sbr2 = v14r - v3r;
        sbr3 = v13r - v4r;
        sbr4 = v12r - v5r;
        sbr5 = v11r - v6r;
        sbr6 = v10r - v7r;
        sbr7 = v9r - v8r;
        sbr8 = sbr1 - sbr4 + sbr6;
        sbr9 = sbr2 + sbr7;

        adi1 = v2i + v15i;
        adi2 = v3i + v14i;
        adi3 = v4i + v13i;
        adi4 = v5i + v12i;
        adi5 = v6i + v11i;
        adi6 = v7i + v10i;
        adi7 = v8i + v9i;
        adi8 = adi2 + adi7;
        adi9 = adi1 + adi4;
        adi10 = adi9 + adi6;
        adi11 = adi2 + adi3 + adi7;
        adi12 = v1i + adi5;

        tv1 = CRTM_15_5 * adr3;
        tv2 = CRTM_15_9 * adr5;
        tv3 = CRTM_15_11 * adr6;
        tv4 = CRTM_15_6 * sbi3;
        tv5 = CRTM_15_10 * sbi5;
        tv6 = CRTM_15_12 * sbi6;
        tv7 = CRTM_15_6 * sbr3;
        tv8 = CRTM_15_10 * sbr5;
        tv9 = CRTM_15_12 * sbr6;
        tv10 = CRTM_15_5 * adi3;
        tv11 = CRTM_15_9 * adi5;
        tv12 = CRTM_15_11 * adi6;
        tv13 = CRTM_15_11 * adr3;
        tv14 = CRTM_15_5 * adr6;
        tv15 = CRTM_15_12 * sbi3;
        tv16 = CRTM_15_6 * sbi6;
        tv17 = CRTM_15_12 * sbr3;
        tv18 = CRTM_15_6 * sbr6;
        tv19 = CRTM_15_11 * adi3;
        tv20 = CRTM_15_5 * adi6;

        tv21 = tv7 + tv9;
        tv22 = tv4 + tv6;
        tv23 = tv2 + tv3;
        tv24 = tv11 + tv12;
        tv25 = v1r + tv14 - tv13 - tv2;
        tv26 = tv15 - tv16;
        tv27 = tv17 - tv18;
        tv28 = v1i + tv20 - tv19 - tv11;
        tv29 = v1r + tv1 - tv23;
        tv30 = v1i + tv10 - tv24;

        // Output point 1: X(0)
        *out_r = adr12 + adr9 + adr11;
        *out_i = adi12 + adi10 + adi11;

        // Output point 2: X(1)
        tvrr = (CRTM_15_1 * adr1);
        tvrr += (CRTM_15_3 * adr2);
        tvrr += tv29;
        tvrr -= (CRTM_15_7 * adr4);
        tvrr -= (CRTM_15_13 * adr7);
        tvri = (CRTM_15_2 * sbi1);
        tvri += (CRTM_15_4 * sbi2);
        tvri += tv22;
        tvri += (CRTM_15_8 * sbi4);
        tvri += tv5;
        tvri += (CRTM_15_14 * sbi7);

        tvir = (CRTM_15_2 * sbr1);
        tvir += (CRTM_15_4 * sbr2);
        tvir += tv21;
        tvir += (CRTM_15_8 * sbr4);
        tvir += tv8;
        tvir += (CRTM_15_14 * sbr7);
        tvii = (CRTM_15_1 * adi1);
        tvii += (CRTM_15_3 * adi2);
        tvii += tv30;
        tvii -= (CRTM_15_7 * adi4);
        tvii -= (CRTM_15_13 * adi7);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvii + tvir;

        // Output point 15: X(14)
        out_r[out_stride * 14] = tvrr + tvri;
        out_i[out_stride * 14] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = (CRTM_15_3 * adr1);
        tvrr -= (CRTM_15_7 * adr2);
        tvrr -= (CRTM_15_13 * adr4);
        tvrr += tv25;
        tvrr += (CRTM_15_1 * adr7);
        tvri = (CRTM_15_4 * sbi1);
        tvri += (CRTM_15_8 * sbi2);
        tvri += tv26;
        tvri -= (CRTM_15_14 * sbi4);
        tvri -= tv5;
        tvri -= (CRTM_15_2 * sbi7);

        tvir = (CRTM_15_4 * sbr1);
        tvir += (CRTM_15_8 * sbr2);
        tvir += tv27;
        tvir -= (CRTM_15_14 * sbr4);
        tvir -= tv8;
        tvir -= (CRTM_15_2 * sbr7);
        tvii = (CRTM_15_3 * adi1);
        tvii -= (CRTM_15_7 * adi2);
        tvii -= (CRTM_15_13 * adi4);
        tvii += tv28;
        tvii += (CRTM_15_1 * adi7);

        out_r[out_stride * 2] = tvrr - tvri;
        out_i[out_stride * 2] = tvii + tvir;

        // Output point 14: X(13)
        out_r[out_stride * 13] = tvrr + tvri;
        out_i[out_stride * 13] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = adr12;
        tvrr += CRTM_15_5 * (adr9);
        tvrr -= CRTM_15_11 * (adr11);
        tvri = CRTM_15_6 * (sbi8);
        tvri += CRTM_15_12 * (sbi9 - sbi3);

        tvir = CRTM_15_6 * (sbr8);
        tvir += CRTM_15_12 * (sbr9 - sbr3);
        tvii = adi12;
        tvii += CRTM_15_5 * (adi10);
        tvii -= CRTM_15_11 * (adi11);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvii + tvir;

        // Output point 13: X(12)
        out_r[out_stride * 12] = tvrr + tvri;
        out_i[out_stride * 12] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = (CRTM_15_1 * adr4);
        tvrr -= (CRTM_15_7 * adr1);
        tvrr -= (CRTM_15_13 * adr2);
        tvrr += tv29;
        tvrr += (CRTM_15_3 * adr7);
        tvri = (CRTM_15_8 * sbi1);
        tvri -= (CRTM_15_14 * sbi2);
        tvri -= tv22;
        tvri += (CRTM_15_2 * sbi4);
        tvri += tv5;
        tvri -= (CRTM_15_4 * sbi7);

        tvir = (CRTM_15_8 * sbr1);
        tvir -= (CRTM_15_14 * sbr2);
        tvir -= tv21;
        tvir += (CRTM_15_2 * sbr4);
        tvir += tv8;
        tvir -= (CRTM_15_4 * sbr7);
        tvii = (CRTM_15_1 * adi4);
        tvii -= (CRTM_15_7 * adi1);
        tvii -= (CRTM_15_13 * adi2);
        tvii += tv30;
        tvii += (CRTM_15_3 * adi7);

        out_r[out_stride * 4] = tvrr - tvri;
        out_i[out_stride * 4] = tvii + tvir;

        // Output point 12: X(11)
        out_r[out_stride * 11] = tvrr + tvri;
        out_i[out_stride * 11] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= CRTM_15_9 * (adr8 + adr10 + adr5);
        tvrr += adr3;
        tvrr += adr6;
        tvri = CRTM_15_10 * (sbi1 - sbi2 + sbi4 - sbi5 + sbi7);

        tvir = CRTM_15_10 * (sbr1 - sbr2 + sbr4 - sbr5 + sbr7);
        tvii = v1i;
        tvii -= CRTM_15_9 * (adi9 + adi8 + adi5);
        tvii += adi3;
        tvii += adi6;

        out_r[out_stride * 5] = tvrr - tvri;
        out_i[out_stride * 5] = tvii + tvir;

        // Output point 11: X(10)
        out_r[out_stride * 10] = tvrr + tvri;
        out_i[out_stride * 10] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = adr12;
        tvrr -= CRTM_15_11 * (adr9);
        tvrr += CRTM_15_5 * (adr11);
        tvri = CRTM_15_12 * (sbi8);
        tvri += CRTM_15_6 * (sbi3 - sbi9);

        tvir = CRTM_15_12 * (sbr8);
        tvir += CRTM_15_6 * (sbr3 - sbr9);
        tvii = adi12;
        tvii -= CRTM_15_11 * (adi10);
        tvii += CRTM_15_5 * (adi11);

        out_r[out_stride * 6] = tvrr - tvri;
        out_i[out_stride * 6] = tvii + tvir;

        // Output point 10: X(9)
        out_r[out_stride * 9] = tvrr + tvri;
        out_i[out_stride * 9] = tvii - tvir;

        // Output point 8: X(7)
        tvrr = (CRTM_15_3 * adr4);
        tvrr -= (CRTM_15_13 * adr1);
        tvrr += (CRTM_15_1 * adr2);
        tvrr += tv25;
        tvrr -= (CRTM_15_7 * adr7);
        tvri = (CRTM_15_14 * sbi1);
        tvri -= (CRTM_15_2 * sbi2);
        tvri += tv26;
        tvri -= (CRTM_15_4 * sbi4);
        tvri += tv5;
        tvri += (CRTM_15_8 * sbi7);

        tvir = (CRTM_15_14 * sbr1);
        tvir -= (CRTM_15_2 * sbr2);
        tvir += tv27;
        tvir -= (CRTM_15_4 * sbr4);
        tvir += tv8;
        tvir += (CRTM_15_8 * sbr7);
        tvii = (CRTM_15_3 * adi4);
        tvii -= (CRTM_15_13 * adi1);
        tvii += (CRTM_15_1 * adi2);
        tvii += tv28;
        tvii -= (CRTM_15_7 * adi7);

        out_r[out_stride * 7] = tvrr - tvri;
        out_i[out_stride * 7] = tvii + tvir;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}

VOID fft15c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    const FLOAT CRTM_15_1 =
        +0.91354545764260089786653411260175763014630651476859;
    const FLOAT CRTM_15_2 =
        +0.40673664307580020244344195165731987476994255841557;
    const FLOAT CRTM_15_3 =
        +0.66913060635885822246624475148920014459869481182487;
    const FLOAT CRTM_15_4 =
        +0.74314482547739422723523183906600387567804039038099;
    const FLOAT CRTM_15_5 =
        +0.30901699437494735775909215694883353691106407759980;
    const FLOAT CRTM_15_6 =
        +0.95105651629515359367265213397317432075121805913368;
    const FLOAT CRTM_15_7 =
        +0.10452846326765344827475689190358280057779765406449;
    const FLOAT CRTM_15_8 =
        +0.99452189536827333935323550615724320908650656433540;
    const FLOAT CRTM_15_9 =
        +0.49999999999999989931390918883503052434117283719579;
    const FLOAT CRTM_15_10 =
        +0.86602540378443870489486480423012865512801223847686;
    const FLOAT CRTM_15_11 =
        +0.80901699437494750610700001978173170344740065252139;
    const FLOAT CRTM_15_12 =
        +0.58778525229247301629891039327884007596190389052978;
    const FLOAT CRTM_15_13 =
        +0.97814760073380564759748190481590945657945761737292;
    const FLOAT CRTM_15_14 =
        +0.20791169081775929161307291104291640870739670391251;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v14r, v14i, v15r, v15i,

            adr1, adr2, adr3, adr4, adr5, adr6, adr7, adi1, adi2, adi3, adi4,
            adi5, adi6, adi7, sbi1, sbi2, sbi3, sbi4, sbi5, sbi6, sbi7, sbr1,
            sbr2, sbr3, sbr4, sbr5, sbr6, sbr7, adr12, adi12,

            tvrr, tvri, tvir, tvii,

            tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13,
            tv14, tv15, tv16, tv17, tv18, tv19, tv20, adr9, adr11, sbi8, adi10,
            adi11, sbr8, adr8, adi9, adr10, adi8, sbr9, sbi9, tv21, tv22, tv23,
            tv24, tv25, tv26, tv27, tv28, tv29, tv30;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];

        // Input point 3: x(2)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        // Input point 4: x(3)
        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        // Input point 6: x(5)
        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        // Input point 7: x(6)
        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        // Input point 8: x(7)
        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        // Input point 10: x(9)
        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        // Input point 11: x(10)
        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // Input point 12: x(11)
        v12r = in_r[in_stride * 11];
        v12i = in_i[in_stride * 11];

        // Input point 13: x(12)
        v13r = in_r[in_stride * 12];
        v13i = in_i[in_stride * 12];

        // Input point 14: x(13)
        v14r = in_r[in_stride * 13];
        v14i = in_i[in_stride * 13];

        // Input point 15: x(14)
        v15r = in_r[in_stride * 14];
        v15i = in_i[in_stride * 14];

        // common calculations
        adr1 = v2r + v15r;
        adr2 = v3r + v14r;
        adr3 = v4r + v13r;
        adr4 = v5r + v12r;
        adr5 = v6r + v11r;
        adr6 = v7r + v10r;
        adr7 = v8r + v9r;
        adr8 = adr1 + adr4;
        adr9 = adr8 + adr6;
        adr10 = adr2 + adr7;
        adr11 = adr10 + adr3;
        adr12 = v1r + adr5;

        sbi1 = v15i - v2i;
        sbi2 = v14i - v3i;
        sbi3 = v13i - v4i;
        sbi4 = v12i - v5i;
        sbi5 = v11i - v6i;
        sbi6 = v10i - v7i;
        sbi7 = v9i - v8i;
        sbi8 = sbi1 - sbi4 + sbi6;
        sbi9 = sbi2 + sbi7;

        sbr1 = v15r - v2r;
        sbr2 = v14r - v3r;
        sbr3 = v13r - v4r;
        sbr4 = v12r - v5r;
        sbr5 = v11r - v6r;
        sbr6 = v10r - v7r;
        sbr7 = v9r - v8r;
        sbr8 = sbr1 - sbr4 + sbr6;
        sbr9 = sbr2 + sbr7;

        adi1 = v2i + v15i;
        adi2 = v3i + v14i;
        adi3 = v4i + v13i;
        adi4 = v5i + v12i;
        adi5 = v6i + v11i;
        adi6 = v7i + v10i;
        adi7 = v8i + v9i;
        adi8 = adi2 + adi7;
        adi9 = adi1 + adi4;
        adi10 = adi9 + adi6;
        adi11 = adi2 + adi3 + adi7;
        adi12 = v1i + adi5;

        tv1 = CRTM_15_5 * adr3;
        tv2 = CRTM_15_9 * adr5;
        tv3 = CRTM_15_11 * adr6;
        tv4 = CRTM_15_6 * sbi3;
        tv5 = CRTM_15_10 * sbi5;
        tv6 = CRTM_15_12 * sbi6;
        tv7 = CRTM_15_6 * sbr3;
        tv8 = CRTM_15_10 * sbr5;
        tv9 = CRTM_15_12 * sbr6;
        tv10 = CRTM_15_5 * adi3;
        tv11 = CRTM_15_9 * adi5;
        tv12 = CRTM_15_11 * adi6;
        tv13 = CRTM_15_11 * adr3;
        tv14 = CRTM_15_5 * adr6;
        tv15 = CRTM_15_12 * sbi3;
        tv16 = CRTM_15_6 * sbi6;
        tv17 = CRTM_15_12 * sbr3;
        tv18 = CRTM_15_6 * sbr6;
        tv19 = CRTM_15_11 * adi3;
        tv20 = CRTM_15_5 * adi6;

        tv21 = tv7 + tv9;
        tv22 = tv4 + tv6;
        tv23 = tv2 + tv3;
        tv24 = tv11 + tv12;
        tv25 = v1r + tv14 - tv13 - tv2;
        tv26 = tv15 - tv16;
        tv27 = tv17 - tv18;
        tv28 = v1i + tv20 - tv19 - tv11;
        tv29 = v1r + tv1 - tv23;
        tv30 = v1i + tv10 - tv24;

        // Output point 1: X(0)
        *out_r = adr12 + adr9 + adr11;
        *out_i = adi12 + adi10 + adi11;

        // Output point 2: X(1)
        tvrr = (CRTM_15_1 * adr1);
        tvrr += (CRTM_15_3 * adr2);
        tvrr += tv29;
        tvrr -= (CRTM_15_7 * adr4);
        tvrr -= (CRTM_15_13 * adr7);
        tvri = (CRTM_15_2 * sbi1);
        tvri += (CRTM_15_4 * sbi2);
        tvri += tv22;
        tvri += (CRTM_15_8 * sbi4);
        tvri += tv5;
        tvri += (CRTM_15_14 * sbi7);

        tvir = (CRTM_15_2 * sbr1);
        tvir += (CRTM_15_4 * sbr2);
        tvir += tv21;
        tvir += (CRTM_15_8 * sbr4);
        tvir += tv8;
        tvir += (CRTM_15_14 * sbr7);
        tvii = (CRTM_15_1 * adi1);
        tvii += (CRTM_15_3 * adi2);
        tvii += tv30;
        tvii -= (CRTM_15_7 * adi4);
        tvii -= (CRTM_15_13 * adi7);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvii + tvir;

        // Output point 15: X(14)
        out_r[out_stride * 14] = tvrr + tvri;
        out_i[out_stride * 14] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = (CRTM_15_3 * adr1);
        tvrr -= (CRTM_15_7 * adr2);
        tvrr -= (CRTM_15_13 * adr4);
        tvrr += tv25;
        tvrr += (CRTM_15_1 * adr7);
        tvri = (CRTM_15_4 * sbi1);
        tvri += (CRTM_15_8 * sbi2);
        tvri += tv26;
        tvri -= (CRTM_15_14 * sbi4);
        tvri -= tv5;
        tvri -= (CRTM_15_2 * sbi7);

        tvir = (CRTM_15_4 * sbr1);
        tvir += (CRTM_15_8 * sbr2);
        tvir += tv27;
        tvir -= (CRTM_15_14 * sbr4);
        tvir -= tv8;
        tvir -= (CRTM_15_2 * sbr7);
        tvii = (CRTM_15_3 * adi1);
        tvii -= (CRTM_15_7 * adi2);
        tvii -= (CRTM_15_13 * adi4);
        tvii += tv28;
        tvii += (CRTM_15_1 * adi7);

        out_r[out_stride * 2] = tvrr - tvri;
        out_i[out_stride * 2] = tvii + tvir;

        // Output point 14: X(13)
        out_r[out_stride * 13] = tvrr + tvri;
        out_i[out_stride * 13] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = adr12;
        tvrr += CRTM_15_5 * (adr9);
        tvrr -= CRTM_15_11 * (adr11);
        tvri = CRTM_15_6 * (sbi8);
        tvri += CRTM_15_12 * (sbi9 - sbi3);

        tvir = CRTM_15_6 * (sbr8);
        tvir += CRTM_15_12 * (sbr9 - sbr3);
        tvii = adi12;
        tvii += CRTM_15_5 * (adi10);
        tvii -= CRTM_15_11 * (adi11);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvii + tvir;

        // Output point 13: X(12)
        out_r[out_stride * 12] = tvrr + tvri;
        out_i[out_stride * 12] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = (CRTM_15_1 * adr4);
        tvrr -= (CRTM_15_7 * adr1);
        tvrr -= (CRTM_15_13 * adr2);
        tvrr += tv29;
        tvrr += (CRTM_15_3 * adr7);
        tvri = (CRTM_15_8 * sbi1);
        tvri -= (CRTM_15_14 * sbi2);
        tvri -= tv22;
        tvri += (CRTM_15_2 * sbi4);
        tvri += tv5;
        tvri -= (CRTM_15_4 * sbi7);

        tvir = (CRTM_15_8 * sbr1);
        tvir -= (CRTM_15_14 * sbr2);
        tvir -= tv21;
        tvir += (CRTM_15_2 * sbr4);
        tvir += tv8;
        tvir -= (CRTM_15_4 * sbr7);
        tvii = (CRTM_15_1 * adi4);
        tvii -= (CRTM_15_7 * adi1);
        tvii -= (CRTM_15_13 * adi2);
        tvii += tv30;
        tvii += (CRTM_15_3 * adi7);

        out_r[out_stride * 4] = tvrr - tvri;
        out_i[out_stride * 4] = tvii + tvir;

        // Output point 12: X(11)
        out_r[out_stride * 11] = tvrr + tvri;
        out_i[out_stride * 11] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= CRTM_15_9 * (adr8 + adr10 + adr5);
        tvrr += adr3;
        tvrr += adr6;
        tvri = CRTM_15_10 * (sbi1 - sbi2 + sbi4 - sbi5 + sbi7);

        tvir = CRTM_15_10 * (sbr1 - sbr2 + sbr4 - sbr5 + sbr7);
        tvii = v1i;
        tvii -= CRTM_15_9 * (adi9 + adi8 + adi5);
        tvii += adi3;
        tvii += adi6;

        out_r[out_stride * 5] = tvrr - tvri;
        out_i[out_stride * 5] = tvii + tvir;

        // Output point 11: X(10)
        out_r[out_stride * 10] = tvrr + tvri;
        out_i[out_stride * 10] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = adr12;
        tvrr -= CRTM_15_11 * (adr9);
        tvrr += CRTM_15_5 * (adr11);
        tvri = CRTM_15_12 * (sbi8);
        tvri += CRTM_15_6 * (sbi3 - sbi9);

        tvir = CRTM_15_12 * (sbr8);
        tvir += CRTM_15_6 * (sbr3 - sbr9);
        tvii = adi12;
        tvii -= CRTM_15_11 * (adi10);
        tvii += CRTM_15_5 * (adi11);

        out_r[out_stride * 6] = tvrr - tvri;
        out_i[out_stride * 6] = tvii + tvir;

        // Output point 10: X(9)
        out_r[out_stride * 9] = tvrr + tvri;
        out_i[out_stride * 9] = tvii - tvir;

        // Output point 8: X(7)
        tvrr = (CRTM_15_3 * adr4);
        tvrr -= (CRTM_15_13 * adr1);
        tvrr += (CRTM_15_1 * adr2);
        tvrr += tv25;
        tvrr -= (CRTM_15_7 * adr7);
        tvri = (CRTM_15_14 * sbi1);
        tvri -= (CRTM_15_2 * sbi2);
        tvri += tv26;
        tvri -= (CRTM_15_4 * sbi4);
        tvri += tv5;
        tvri += (CRTM_15_8 * sbi7);

        tvir = (CRTM_15_14 * sbr1);
        tvir -= (CRTM_15_2 * sbr2);
        tvir += tv27;
        tvir -= (CRTM_15_4 * sbr4);
        tvir += tv8;
        tvir += (CRTM_15_8 * sbr7);
        tvii = (CRTM_15_3 * adi4);
        tvii -= (CRTM_15_13 * adi1);
        tvii += (CRTM_15_1 * adi2);
        tvii += tv28;
        tvii -= (CRTM_15_7 * adi7);

        out_r[out_stride * 7] = tvrr - tvri;
        out_i[out_stride * 7] = tvii + tvir;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "core/kernels/kernel_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 368, 1520, 150, 0, 1681}, {0, 368, 1520, 150, 0, 1681}};
ops_cycles_t get_ops_cnt_fft15c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

const DOUBLE CRTM_15[RADIX_15][2] = {{1.0, 0.0},
                                     {0.913545457642601, -0.4067366430758},
                                     {0.669130606358858, -0.743144825477394},
                                     {0.309016994374947, -0.951056516295154},
                                     {-0.104528463267653, -0.994521895368273},
                                     {-0.5, -0.866025403784439},
                                     {-0.809016994374947, -0.587785252292473},
                                     {-0.978147600733806, -0.207911690817759},
                                     {-0.978147600733806, 0.207911690817759},
                                     {-0.809016994374948, 0.587785252292473},
                                     {-0.5, 0.866025403784438},
                                     {-0.104528463267654, 0.994521895368273},
                                     {0.309016994374947, 0.951056516295154},
                                     {0.669130606358858, 0.743144825477394},
                                     {0.913545457642601, 0.4067366430758}};

VOID fft15c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;
    // temp variable to store power (constant_multiplier)
    DOUBLE pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    DOUBLE temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    DOUBLE cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    DOUBLE cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    DOUBLE *in_dr = (DOUBLE *)in_real;
    DOUBLE *in_di = (DOUBLE *)in_imag;
    DOUBLE *input_r = (DOUBLE *)in_real;
    DOUBLE *input_i = (DOUBLE *)in_imag;
    // buffer to store current output
    DOUBLE *output_r = (DOUBLE *)out_real;
    DOUBLE *output_i = (DOUBLE *)out_imag;
    DOUBLE *out_dr = (DOUBLE *)out_real;
    DOUBLE *out_di = (DOUBLE *)out_imag;
    // local buffer to store input
    DOUBLE local_in[RADIX_15][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[2]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[3]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[4]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[5]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[6]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[7]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[8]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[9]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[10]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[11]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[12]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[13]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[14]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 15i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        CADD(local_in[13], temp_out, temp_out);
        CADD(local_in[14], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_15[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_15[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_15[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_15[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_15[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_15[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_15[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_15[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_15[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_15[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_15[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_15[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_15[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_15[14], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

VOID fft15c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;

    // temp variable to store power (constant_multiplier)
    FLOAT pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    FLOAT temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    FLOAT cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    FLOAT cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    FLOAT *in_fr = (FLOAT *)in_real;
    FLOAT *in_fi = (FLOAT *)in_imag;
    FLOAT *input_r = (FLOAT *)in_real;
    FLOAT *input_i = (FLOAT *)in_imag;
    // buffer to store current output
    FLOAT *output_r = (FLOAT *)out_real;
    FLOAT *output_i = (FLOAT *)out_imag;
    FLOAT *out_fr = (FLOAT *)out_real;
    FLOAT *out_fi = (FLOAT *)out_imag;
    // local buffer to store input
    FLOAT local_in[RADIX_15][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[2]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[3]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[4]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[5]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[6]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[7]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[8]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[9]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[10]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[11]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[12]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[13]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[14]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 15i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        CADD(local_in[13], temp_out, temp_out);
        CADD(local_in[14], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_15[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_15[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_15[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_15[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_15[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_15[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_15[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_15[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_15[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_15[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_15[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_15[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_15[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_15[14], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 15i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_15[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_15[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_15[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_15[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_15[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_15[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_15[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_15[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_15[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_15[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_15[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_15[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_15[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_15[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[14], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}
#endif // USE_OPT_KERNEL_VARIANT