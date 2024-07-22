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

/** @file fft16c.c
 *
 *  @brief Radix-16 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-16 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author S. Biplab Raut
 * @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 24, 144, 64, 0, 0},
                                                     {0, 24, 144, 64, 0, 0}};

ops_cycles_t get_ops_cnt_fft16c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

static VOID fft16c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_16_1 =
        +0.92387953251128675612818318939678828682241662586364;
    const DOUBLE CRTM_16_2 =
        +0.70710678118654752440084436210484903928483593768847;
    const DOUBLE CRTM_16_3 =
        +0.38268343236508977172845998403039886676134456248563;

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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v14r, v14i, v15r, v15i, v16r, v16i,

            ad1, ad2, ad3, ad4, ad5, ad6, ad7, adi1, adi2, adi3, adi4, adi5,
            adi6, adi7, sbr1, sbr2, sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3,
            sb4, sb5, sb6, sb7,

            tvrr, tvri, tvir, tvii,

            tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3, tvi4, tvr5, tvr6, tvi5,
            tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10, tvr11, tvr12, tvi9,
            tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,

            ad17, sb17, adi17, sbr17, sb35, sbr35, sb26, sbr26, ad35, adi35,
            ad62, adi62,

            cv1r, cv1i, cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r,
            cav2i, cav3r, cav3i, cav4r, cav4i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        // Input point 5: x(4)
        v5r = in_r[in_strides[4]];
        v5i = in_i[in_strides[4]];

        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v6i = in_i[in_strides[5]];

        // Input point 7: x(6)
        v7r = in_r[in_strides[6]];
        v7i = in_i[in_strides[6]];

        // Input point 8: x(7)
        v8r = in_r[in_strides[7]];
        v8i = in_i[in_strides[7]];

        // Input point 9: x(8)
        v9r = in_r[in_strides[8]];
        v9i = in_i[in_strides[8]];

        // Input point 10: x(9)
        v10r = in_r[in_strides[9]];
        v10i = in_i[in_strides[9]];

        // Input point 11: x(10)
        v11r = in_r[in_strides[10]];
        v11i = in_i[in_strides[10]];

        // Input point 12: x(11)
        v12r = in_r[in_strides[11]];
        v12i = in_i[in_strides[11]];

        // Input point 13: x(12)
        v13r = in_r[in_strides[12]];
        v13i = in_i[in_strides[12]];

        // Input point 14: x(13)
        v14r = in_r[in_strides[13]];
        v14i = in_i[in_strides[13]];

        // Input point 15: x(14)
        v15r = in_r[in_strides[14]];
        v15i = in_i[in_strides[14]];

        // Input point 16: x(15)
        v16r = in_r[in_strides[15]];
        v16i = in_i[in_strides[15]];

        cv1r = v1r + v9r;
        cv1i = v1i + v9i;
        cv2r = v1r - v9r;
        cv2i = v1i - v9i;

        ad1 = v2r + v16r;
        ad2 = v3r + v15r;
        ad3 = v4r + v14r;
        ad4 = v5r + v13r;
        ad5 = v6r + v12r;
        ad6 = v7r + v11r;
        ad7 = v8r + v10r;

        sb1 = v16i - v2i;
        sb2 = v15i - v3i;
        sb3 = v14i - v4i;
        sb4 = v13i - v5i;
        sb5 = v12i - v6i;
        sb6 = v11i - v7i;
        sb7 = v10i - v8i;

        sbr1 = v16r - v2r;
        sbr2 = v15r - v3r;
        sbr3 = v14r - v4r;
        sbr4 = v13r - v5r;
        sbr5 = v12r - v6r;
        sbr6 = v11r - v7r;
        sbr7 = v10r - v8r;

        adi1 = v2i + v16i;
        adi2 = v3i + v15i;
        adi3 = v4i + v14i;
        adi4 = v5i + v13i;
        adi5 = v6i + v12i;
        adi6 = v7i + v11i;
        adi7 = v8i + v10i;

        ad26 = ad2 + ad6;
        ad49 = cv1r + ad4;
        adi26 = adi2 + adi6;
        adi49 = cv1i + adi4;

        ad17 = ad1 - ad7;
        sb17 = sb1 + sb7;
        adi17 = adi1 - adi7;
        sbr17 = sbr1 + sbr7;
        sb35 = sb3 + sb5;
        sbr35 = sbr3 + sbr5;
        sb26 = sb2 + sb6;
        sbr26 = sbr2 + sbr6;

        ad35 = ad3 - ad5;
        adi35 = adi3 - adi5;
        ad62 = ad2 - ad6;
        adi62 = adi2 - adi6;

        ctv1 = CRTM_16_2 * ad62;
        ctv2 = CRTM_16_2 * sb26;
        ctv3 = CRTM_16_2 * adi62;
        ctv4 = CRTM_16_2 * sbr26;

        cav1r = ad1 + ad7;
        cav1i = adi1 + adi7;
        cav2r = ad3 + ad5;
        cav2i = adi3 + adi5;
        cav3r = sbr1 - sbr7;
        cav3i = sb1 - sb7;
        cav4r = sbr3 - sbr5;
        cav4i = sb3 - sb5;

        // common operations
        tvr1  = (CRTM_16_1 * ad17);
        tvr2  = cv2r + ctv1;
        tvr3  = (CRTM_16_3 * sb17);
        tvr4  = ctv2 + sb4;
        tvr5  = cv1r - ad4;
        tvr6  = CRTM_16_2 * (cav1r - cav2r);
        tvr7  = CRTM_16_2 * (cav3i + cav4i);
        tvr8  = sb2 - sb6;
        tvr9  = cv2r - ctv1;
        tvr10 = (CRTM_16_3 * ad17);
        tvr11 = (CRTM_16_1 * sb17);
        tvr12 = ctv2 - sb4;

        tvi1  = (CRTM_16_1 * adi17);
        tvi2  = cv2i + ctv3;
        tvi3  = (CRTM_16_3 * sbr17);
        tvi4  = ctv4 + sbr4;
        tvi5  = cv1i - adi4;
        tvi6  = CRTM_16_2 * (cav1i - cav2i);
        tvi7  = CRTM_16_2 * (cav3r + cav4r);
        tvi8  = sbr2 - sbr6;
        tvi9  = cv2i - ctv3;
        tvi10 = (CRTM_16_3 * adi17);
        tvi11 = (CRTM_16_1 * sbr17);
        tvi12 = ctv4 - sbr4;

        tvr1 += (CRTM_16_3 * ad35);
        tvr3 += (CRTM_16_1 * sb35);
        tvi1 += (CRTM_16_3 * adi35);
        tvi3 += (CRTM_16_1 * sbr35);

        tvr10 -= (CRTM_16_1 * ad35);
        tvr11 -= (CRTM_16_3 * sb35);
        tvi10 -= (CRTM_16_1 * adi35);
        tvi11 -= (CRTM_16_3 * sbr35);

        // Output point 1: X(0)
        tvrr = cav1r + cav2r;
        tvri = ad26 + ad49;
        tvii = cav1i + cav2i;
        tvir = adi26 + adi49;

        *out_r = tvri + tvrr;
        *out_i = tvir + tvii;

        // Output point 9: X(8)
        out_r[out_strides[8]] = tvri - tvrr;
        out_i[out_strides[8]] = tvir - tvii;

        // Output point 2: X(1)
        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;
        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 16: X(15)
        out_r[out_strides[15]] = tvrr + tvri;
        out_i[out_strides[15]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        out_r[out_strides[2]] = tvrr - tvri;
        out_i[out_strides[2]] = tvii + tvir;

        // Output point 15: X(14)
        out_r[out_strides[14]] = tvrr + tvri;
        out_i[out_strides[14]] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        out_r[out_strides[3]] = tvrr - tvri;
        out_i[out_strides[3]] = tvii + tvir;

        // Output point 14: X(13)
        out_r[out_strides[13]] = tvrr + tvri;
        out_i[out_strides[13]] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        out_r[out_strides[4]] = tvrr - tvri;
        out_i[out_strides[4]] = tvii + tvir;

        // Output point 13: X(12)
        out_r[out_strides[12]] = tvrr + tvri;
        out_i[out_strides[12]] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        out_r[out_strides[5]] = tvrr - tvri;
        out_i[out_strides[5]] = tvii + tvir;

        // Output point 12: X(11)
        out_r[out_strides[11]] = tvrr + tvri;
        out_i[out_strides[11]] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        out_r[out_strides[6]] = tvrr - tvri;
        out_i[out_strides[6]] = tvii + tvir;

        // Output point 11: X(10)
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[10]] = tvii - tvir;

        // Output point 8: X(7)
        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        out_r[out_strides[7]] = tvrr - tvri;
        out_i[out_strides[7]] = tvii + tvir;

        // Output point 10: X(9)
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[9]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft16c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_16_1 =
        +0.92387953251128675612818318939678828682241662586364;
    const FLOAT CRTM_16_2 =
        +0.70710678118654752440084436210484903928483593768847;
    const FLOAT CRTM_16_3 =
        +0.38268343236508977172845998403039886676134456248563;

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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v14r, v14i, v15r, v15i, v16r, v16i,

            ad1, ad2, ad3, ad4, ad5, ad6, ad7, adi1, adi2, adi3, adi4, adi5,
            adi6, adi7, sbr1, sbr2, sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3,
            sb4, sb5, sb6, sb7,

            tvrr, tvri, tvir, tvii,

            tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3, tvi4, tvr5, tvr6, tvi5,
            tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10, tvr11, tvr12, tvi9,
            tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,

            ad17, sb17, adi17, sbr17, sb35, sbr35, sb26, sbr26, ad35, adi35,
            ad62, adi62,

            cv1r, cv1i, cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r,
            cav2i, cav3r, cav3i, cav4r, cav4i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        // Input point 5: x(4)
        v5r = in_r[in_strides[4]];
        v5i = in_i[in_strides[4]];

        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v6i = in_i[in_strides[5]];

        // Input point 7: x(6)
        v7r = in_r[in_strides[6]];
        v7i = in_i[in_strides[6]];

        // Input point 8: x(7)
        v8r = in_r[in_strides[7]];
        v8i = in_i[in_strides[7]];

        // Input point 9: x(8)
        v9r = in_r[in_strides[8]];
        v9i = in_i[in_strides[8]];

        // Input point 10: x(9)
        v10r = in_r[in_strides[9]];
        v10i = in_i[in_strides[9]];

        // Input point 11: x(10)
        v11r = in_r[in_strides[10]];
        v11i = in_i[in_strides[10]];

        // Input point 12: x(11)
        v12r = in_r[in_strides[11]];
        v12i = in_i[in_strides[11]];

        // Input point 13: x(12)
        v13r = in_r[in_strides[12]];
        v13i = in_i[in_strides[12]];

        // Input point 14: x(13)
        v14r = in_r[in_strides[13]];
        v14i = in_i[in_strides[13]];

        // Input point 15: x(14)
        v15r = in_r[in_strides[14]];
        v15i = in_i[in_strides[14]];

        // Input point 16: x(15)
        v16r = in_r[in_strides[15]];
        v16i = in_i[in_strides[15]];

        cv1r = v1r + v9r;
        cv1i = v1i + v9i;
        cv2r = v1r - v9r;
        cv2i = v1i - v9i;

        ad1 = v2r + v16r;
        ad2 = v3r + v15r;
        ad3 = v4r + v14r;
        ad4 = v5r + v13r;
        ad5 = v6r + v12r;
        ad6 = v7r + v11r;
        ad7 = v8r + v10r;

        sb1 = v16i - v2i;
        sb2 = v15i - v3i;
        sb3 = v14i - v4i;
        sb4 = v13i - v5i;
        sb5 = v12i - v6i;
        sb6 = v11i - v7i;
        sb7 = v10i - v8i;

        sbr1 = v16r - v2r;
        sbr2 = v15r - v3r;
        sbr3 = v14r - v4r;
        sbr4 = v13r - v5r;
        sbr5 = v12r - v6r;
        sbr6 = v11r - v7r;
        sbr7 = v10r - v8r;

        adi1 = v2i + v16i;
        adi2 = v3i + v15i;
        adi3 = v4i + v14i;
        adi4 = v5i + v13i;
        adi5 = v6i + v12i;
        adi6 = v7i + v11i;
        adi7 = v8i + v10i;

        ad26 = ad2 + ad6;
        ad49 = cv1r + ad4;
        adi26 = adi2 + adi6;
        adi49 = cv1i + adi4;

        ad17 = ad1 - ad7;
        sb17 = sb1 + sb7;
        adi17 = adi1 - adi7;
        sbr17 = sbr1 + sbr7;
        sb35 = sb3 + sb5;
        sbr35 = sbr3 + sbr5;
        sb26 = sb2 + sb6;
        sbr26 = sbr2 + sbr6;

        ad35 = ad3 - ad5;
        adi35 = adi3 - adi5;
        ad62 = ad2 - ad6;
        adi62 = adi2 - adi6;

        ctv1 = CRTM_16_2 * ad62;
        ctv2 = CRTM_16_2 * sb26;
        ctv3 = CRTM_16_2 * adi62;
        ctv4 = CRTM_16_2 * sbr26;

        cav1r = ad1 + ad7;
        cav1i = adi1 + adi7;
        cav2r = ad3 + ad5;
        cav2i = adi3 + adi5;
        cav3r = sbr1 - sbr7;
        cav3i = sb1 - sb7;
        cav4r = sbr3 - sbr5;
        cav4i = sb3 - sb5;

        // common operations
        tvr1  = (CRTM_16_1 * ad17);
        tvr2  = cv2r + ctv1;
        tvr3  = (CRTM_16_3 * sb17);
        tvr4  = ctv2 + sb4;
        tvr5  = cv1r - ad4;
        tvr6  = CRTM_16_2 * (cav1r - cav2r);
        tvr7  = CRTM_16_2 * (cav3i + cav4i);
        tvr8  = sb2 - sb6;
        tvr9  = cv2r - ctv1;
        tvr10 = (CRTM_16_3 * ad17);
        tvr11 = (CRTM_16_1 * sb17);
        tvr12 = ctv2 - sb4;

        tvi1  = (CRTM_16_1 * adi17);
        tvi2  = cv2i + ctv3;
        tvi3  = (CRTM_16_3 * sbr17);
        tvi4  = ctv4 + sbr4;
        tvi5  = cv1i - adi4;
        tvi6  = CRTM_16_2 * (cav1i - cav2i);
        tvi7  = CRTM_16_2 * (cav3r + cav4r);
        tvi8  = sbr2 - sbr6;
        tvi9  = cv2i - ctv3;
        tvi10 = (CRTM_16_3 * adi17);
        tvi11 = (CRTM_16_1 * sbr17);
        tvi12 = ctv4 - sbr4;

        tvr1 += (CRTM_16_3 * ad35);
        tvr3 += (CRTM_16_1 * sb35);
        tvi1 += (CRTM_16_3 * adi35);
        tvi3 += (CRTM_16_1 * sbr35);

        tvr10 -= (CRTM_16_1 * ad35);
        tvr11 -= (CRTM_16_3 * sb35);
        tvi10 -= (CRTM_16_1 * adi35);
        tvi11 -= (CRTM_16_3 * sbr35);

        // Output point 1: X(0)
        tvrr = cav1r + cav2r;
        tvri = ad26 + ad49;
        tvii = cav1i + cav2i;
        tvir = adi26 + adi49;

        *out_r = tvri + tvrr;
        *out_i = tvir + tvii;

        // Output point 9: X(8)
        out_r[out_strides[8]] = tvri - tvrr;
        out_i[out_strides[8]] = tvir - tvii;

        // Output point 2: X(1)
        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;
        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 16: X(15)
        out_r[out_strides[15]] = tvrr + tvri;
        out_i[out_strides[15]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        out_r[out_strides[2]] = tvrr - tvri;
        out_i[out_strides[2]] = tvii + tvir;

        // Output point 15: X(14)
        out_r[out_strides[14]] = tvrr + tvri;
        out_i[out_strides[14]] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        out_r[out_strides[3]] = tvrr - tvri;
        out_i[out_strides[3]] = tvii + tvir;

        // Output point 14: X(13)
        out_r[out_strides[13]] = tvrr + tvri;
        out_i[out_strides[13]] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        out_r[out_strides[4]] = tvrr - tvri;
        out_i[out_strides[4]] = tvii + tvir;

        // Output point 13: X(12)
        out_r[out_strides[12]] = tvrr + tvri;
        out_i[out_strides[12]] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        out_r[out_strides[5]] = tvrr - tvri;
        out_i[out_strides[5]] = tvii + tvir;

        // Output point 12: X(11)
        out_r[out_strides[11]] = tvrr + tvri;
        out_i[out_strides[11]] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        out_r[out_strides[6]] = tvrr - tvri;
        out_i[out_strides[6]] = tvii + tvir;

        // Output point 11: X(10)
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[10]] = tvii - tvir;

        // Output point 8: X(7)
        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        out_r[out_strides[7]] = tvrr - tvri;
        out_i[out_strides[7]] = tvii + tvir;

        // Output point 10: X(9)
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[9]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 424, 1748, 160, 0, 2041}, {0, 424, 1748, 160, 0, 2041}};

ops_cycles_t get_ops_cnt_fft16c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

const DOUBLE CRTM_16[RADIX_16][2] = {{1.0, 0.0},
                                     {0.923879532511287, -0.38268343236509},
                                     {0.707106781186548, -0.707106781186547},
                                     {0.38268343236509, -0.923879532511287},
                                     {0.0, -1.0},
                                     {-0.38268343236509, -0.923879532511287},
                                     {-0.707106781186547, -0.707106781186548},
                                     {-0.923879532511287, -0.38268343236509},
                                     {-1.0, 0.0},
                                     {-0.923879532511287, 0.38268343236509},
                                     {-0.707106781186548, 0.707106781186547},
                                     {-0.38268343236509, 0.923879532511287},
                                     {-0.0, 1.0},
                                     {0.38268343236509, 0.923879532511287},
                                     {0.707106781186547, 0.707106781186548},
                                     {0.923879532511287, 0.38268343236509}};

static VOID fft16c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
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
    DOUBLE local_in[RADIX_16][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);
        LOAD_INPUT(input_r + in_strides[6], input_i + in_strides[6],
                   local_in[6]);
        LOAD_INPUT(input_r + in_strides[7], input_i + in_strides[7],
                   local_in[7]);
        LOAD_INPUT(input_r + in_strides[8], input_i + in_strides[8],
                   local_in[8]);
        LOAD_INPUT(input_r + in_strides[9], input_i + in_strides[9],
                   local_in[9]);
        LOAD_INPUT(input_r + in_strides[10], input_i + in_strides[10],
                   local_in[10]);
        LOAD_INPUT(input_r + in_strides[11], input_i + in_strides[11],
                   local_in[11]);
        LOAD_INPUT(input_r + in_strides[12], input_i + in_strides[12],
                   local_in[12]);
        LOAD_INPUT(input_r + in_strides[13], input_i + in_strides[13],
                   local_in[13]);
        LOAD_INPUT(input_r + in_strides[14], input_i + in_strides[14],
                   local_in[14]);
        LOAD_INPUT(input_r + in_strides[15], input_i + in_strides[15],
                   local_in[15]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 16i ********************/
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
        CADD(local_in[15], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 16i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_16[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_16[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_16[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_16[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_16[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_16[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_16[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_16[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_16[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_16[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_16[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_16[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_16[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_16[14], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[15], CRTM_16[15], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 16i+2 *********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 16i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 16i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 16i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 16i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 16i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 16i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 16i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 16i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 16i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 16i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

        /******************** Output 16i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[13],
                     output_i + out_strides[13]);

        /******************** Output 16i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[14],
                     output_i + out_strides[14]);

        /******************** Output 16i+15 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 15, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[15],
                     output_i + out_strides[15]);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

VOID fft16c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

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
    FLOAT local_in[RADIX_16][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);
        LOAD_INPUT(input_r + in_strides[6], input_i + in_strides[6],
                   local_in[6]);
        LOAD_INPUT(input_r + in_strides[7], input_i + in_strides[7],
                   local_in[7]);
        LOAD_INPUT(input_r + in_strides[8], input_i + in_strides[8],
                   local_in[8]);
        LOAD_INPUT(input_r + in_strides[9], input_i + in_strides[9],
                   local_in[9]);
        LOAD_INPUT(input_r + in_strides[10], input_i + in_strides[10],
                   local_in[10]);
        LOAD_INPUT(input_r + in_strides[11], input_i + in_strides[11],
                   local_in[11]);
        LOAD_INPUT(input_r + in_strides[12], input_i + in_strides[12],
                   local_in[12]);
        LOAD_INPUT(input_r + in_strides[13], input_i + in_strides[13],
                   local_in[13]);
        LOAD_INPUT(input_r + in_strides[14], input_i + in_strides[14],
                   local_in[14]);
        LOAD_INPUT(input_r + in_strides[15], input_i + in_strides[15],
                   local_in[15]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 16i ********************/
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
        CADD(local_in[15], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 16i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_16[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_16[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_16[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_16[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_16[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_16[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_16[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_16[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_16[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_16[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_16[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_16[12], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[13], CRTM_16[13], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[14], CRTM_16[14], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[15], CRTM_16[15], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 16i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 16i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 16i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 16i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 16i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 16i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 16i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 16i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 16i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 16i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 16i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

        /******************** Output 16i+13 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 13, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 13, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 13, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[13],
                     output_i + out_strides[13]);

        /******************** Output 16i+14 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 14, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 14, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 14, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[14],
                     output_i + out_strides[14]);

        /******************** Output 16i+15 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_16[1], 15, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_16[2], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_16[3], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_16[4], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_16[5], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_16[6], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_16[7], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_16[8], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_16[9], 15, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_16[10], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_16[11], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_16[12], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[12], CRTM_16[13], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[13], CRTM_16[14], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[14], CRTM_16[15], 15, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[15], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[15],
                     output_i + out_strides[15]);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#endif // USE_OPT_KERNEL_VARIANT

kfft_ register_kernel_fft16c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft16c_fp32;
    else if (precision == DT_DOUBLE)
        return fft16c_fp64;
    else
        return NULL;
}
