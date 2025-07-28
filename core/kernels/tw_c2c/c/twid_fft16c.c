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

/** @file twid_fft16c.c
 *
 *  @brief Twiddle Radix-16 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-16 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"

// twiddle cost = 15 * (4 muls, 2 adds, 2 movs [loads])
//              = (60, 30, 30)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 84, 174, 94, 0, 0},
                                                     {0, 84, 174, 94, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft16c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft16c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twid_buf,
                             UINT8 flag)
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
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    DOUBLE *tw = (DOUBLE *)twid_buf;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, ad1, ad2, ad3, ad4, ad5,
            ad6, ad7, adi1, adi2, adi3, adi4, adi5, adi6, adi7, sbr1, sbr2,
            sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3, sb4, sb5, sb6, sb7,
            tvrr, tvri, tvir, tvii, tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3,
            tvi4, tvr5, tvr6, tvi5, tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10,
            tvr11, tvr12, tvi9, tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,
            ad17, sb17, adi17, sbr17, sb35, sbr35, ad35, adi35, cv1r, cv1i,
            cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r, cav2i,
            cav3r, cav3i, cav4r, cav4i;

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (1 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (15 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v1r_temp = in_r[in_strides[1]];
                DOUBLE v1i_temp = in_i[in_strides[1]];
                DOUBLE v2r_temp = in_r[in_strides[15]];
                DOUBLE v2i_temp = in_i[in_strides[15]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad1 = v1r + v2r;
                sbr1 = v2r - v1r;
                adi1 = v1i + v2i;
                sb1 = v2i - v1i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (7 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (9 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v1r_temp = in_r[in_strides[7]];
                DOUBLE v1i_temp = in_i[in_strides[7]];
                DOUBLE v2r_temp = in_r[in_strides[9]];
                DOUBLE v2i_temp = in_i[in_strides[9]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad7 = v1r + v2r;
                sbr7 = v2r - v1r;
                adi7 = v1i + v2i;
                sb7 = v2i - v1i;
            }

            cav1r = ad1 + ad7;
            ad17 = ad1 - ad7;
            sbr17 = sbr1 + sbr7;
            cav3r = sbr1 - sbr7;

            cav1i = adi1 + adi7;
            adi17 = adi1 - adi7;
            sb17 = sb1 + sb7;
            cav3i = sb1 - sb7;

            {
                UINTP twid_addr1 = DATA_STRIDE * (3 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (13 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v1r_temp = in_r[in_strides[3]];
                DOUBLE v1i_temp = in_i[in_strides[3]];
                DOUBLE v2r_temp = in_r[in_strides[13]];
                DOUBLE v2i_temp = in_i[in_strides[13]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad3 = v1r + v2r;
                sbr3 = v2r - v1r;
                adi3 = v1i + v2i;
                sb3 = v2i - v1i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (5 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (11 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v1r_temp = in_r[in_strides[5]];
                DOUBLE v1i_temp = in_i[in_strides[5]];
                DOUBLE v2r_temp = in_r[in_strides[11]];
                DOUBLE v2i_temp = in_i[in_strides[11]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad5 = v1r + v2r;
                sbr5 = v2r - v1r;
                adi5 = v1i + v2i;
                sb5 = v2i - v1i;
            }

            cav2r = ad3 + ad5;
            ad35 = ad3 - ad5;
            sbr35 = sbr3 + sbr5;
            cav4r = sbr3 - sbr5;

            cav2i = adi3 + adi5;
            adi35 = adi3 - adi5;
            sb35 = sb3 + sb5;
            cav4i = sb3 - sb5;
        }

        tvr3 = (CRTM_16_1 * sb35) + (CRTM_16_3 * sb17);
        tvr11 = (CRTM_16_1 * sb17) - (CRTM_16_3 * sb35);
        tvi1 = (CRTM_16_3 * adi35) + (CRTM_16_1 * adi17);
        tvi10 = (CRTM_16_3 * adi17) - (CRTM_16_1 * adi35);
        tvr7 = CRTM_16_2 * (cav3i + cav4i);
        tvi6 = CRTM_16_2 * (cav1i - cav2i);
        tvrr = cav1r + cav2r;
        tvii = cav1i + cav2i;

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (2 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (14 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v3r_temp = in_r[in_strides[2]];
                DOUBLE v3i_temp = in_i[in_strides[2]];
                DOUBLE v4r_temp = in_r[in_strides[14]];
                DOUBLE v4i_temp = in_i[in_strides[14]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad2 = v3r + v4r;
                sbr2 = v4r - v3r;
                adi2 = v3i + v4i;
                sb2 = v4i - v3i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (6 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (10 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v3r_temp = in_r[in_strides[6]];
                DOUBLE v3i_temp = in_i[in_strides[6]];
                DOUBLE v4r_temp = in_r[in_strides[10]];
                DOUBLE v4i_temp = in_i[in_strides[10]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad6 = v3r + v4r;
                sbr6 = v4r - v3r;
                adi6 = v3i + v4i;
                sb6 = v4i - v3i;
            }
        }

        ad26 = ad2 + ad6;
        tvi8 = sbr2 - sbr6;
        ctv4 = CRTM_16_2 * (sbr2 + sbr6);
        ctv1 = CRTM_16_2 * (ad2 - ad6);

        adi26 = adi2 + adi6;
        tvr8 = sb2 - sb6;
        ctv2 = CRTM_16_2 * (sb2 + sb6);
        ctv3 = CRTM_16_2 * (adi2 - adi6);

        tvr1 = (CRTM_16_3 * ad35) + (CRTM_16_1 * ad17);
        tvr10 = (CRTM_16_3 * ad17) - (CRTM_16_1 * ad35);
        tvi3 = (CRTM_16_1 * sbr35) + (CRTM_16_3 * sbr17);
        tvi11 = (CRTM_16_1 * sbr17) - (CRTM_16_3 * sbr35);

        tvr6 = CRTM_16_2 * (cav1r - cav2r);
        tvi7 = CRTM_16_2 * (cav3r + cav4r);

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (4 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (12 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v3r_temp = in_r[in_strides[4]];
                DOUBLE v3i_temp = in_i[in_strides[4]];
                DOUBLE v4r_temp = in_r[in_strides[12]];
                DOUBLE v4i_temp = in_i[in_strides[12]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad4 = v3r + v4r;
                sbr4 = v4r - v3r;
                adi4 = v3i + v4i;
                sb4 = v4i - v3i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (0 * n + cnt);
                DOUBLE twid_r1 = tw[twid_addr1];
                DOUBLE twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (8 * n + cnt);
                DOUBLE twid_r2 = tw[twid_addr2];
                DOUBLE twid_i2 = tw[1 + twid_addr2];

                DOUBLE v1r_temp = *in_r;
                DOUBLE v1i_temp = *in_i;
                DOUBLE v2r_temp = in_r[in_strides[8]];
                DOUBLE v2i_temp = in_i[in_strides[8]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                cv1r = v1r + v2r;
                cv2r = v1r - v2r;
                cv1i = v1i + v2i;
                cv2i = v1i - v2i;
            }
        }

        tvr5 = cv1r - ad4;
        ad49 = cv1r + ad4;
        tvi4 = ctv4 + sbr4;
        tvi12 = ctv4 - sbr4;
        tvr2 = cv2r + ctv1;
        tvr9 = cv2r - ctv1;
        tvri = ad26 + ad49;
        *out_r = tvri + tvrr;
        out_r[out_strides[8]] = tvri - tvrr;

        adi49 = cv1i + adi4;
        tvi5 = cv1i - adi4;
        tvr4 = ctv2 + sb4;
        tvr12 = ctv2 - sb4;
        tvi2 = cv2i + ctv3;
        tvi9 = cv2i - ctv3;
        tvir = adi26 + adi49;
        *out_i = tvir + tvii;
        out_i[out_strides[8]] = tvir - tvii;

        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        out_r[out_strides[4]] = tvrr - tvri;
        out_r[out_strides[12]] = tvrr + tvri;

        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        out_i[out_strides[4]] = tvii + tvir;
        out_i[out_strides[12]] = tvii - tvir;

        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[15]] = tvrr + tvri;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;

        out_i[out_strides[1]] = tvii + tvir;
        out_i[out_strides[15]] = tvii - tvir;

        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        out_r[out_strides[7]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        out_i[out_strides[7]] = tvii + tvir;
        out_i[out_strides[9]] = tvii - tvir;

        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[14]] = tvrr + tvri;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        out_i[out_strides[2]] = tvii + tvir;
        out_i[out_strides[14]] = tvii - tvir;

        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        out_r[out_strides[6]] = tvrr - tvri;
        out_r[out_strides[10]] = tvrr + tvri;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        out_i[out_strides[6]] = tvii + tvir;

        out_i[out_strides[10]] = tvii - tvir;

        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[13]] = tvrr + tvri;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        out_i[out_strides[3]] = tvii + tvir;
        out_i[out_strides[13]] = tvii - tvir;

        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        out_r[out_strides[5]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        out_i[out_strides[5]] = tvii + tvir;
        out_i[out_strides[11]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft16c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twid_buf,
                             UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_16_1 =
        +0.92387953251128675612818318939678828682241662586364f;
    const FLOAT CRTM_16_2 =
        +0.70710678118654752440084436210484903928483593768847f;
    const FLOAT CRTM_16_3 =
        +0.38268343236508977172845998403039886676134456248563f;

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

    FLOAT *tw = (FLOAT *)twid_buf;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, ad1, ad2, ad3, ad4, ad5,
            ad6, ad7, adi1, adi2, adi3, adi4, adi5, adi6, adi7, sbr1, sbr2,
            sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3, sb4, sb5, sb6, sb7,
            tvrr, tvri, tvir, tvii, tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3,
            tvi4, tvr5, tvr6, tvi5, tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10,
            tvr11, tvr12, tvi9, tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,
            ad17, sb17, adi17, sbr17, sb35, sbr35, ad35, adi35, cv1r, cv1i,
            cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r, cav2i,
            cav3r, cav3i, cav4r, cav4i;

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (1 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (15 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v1r_temp = in_r[in_strides[1]];
                FLOAT v1i_temp = in_i[in_strides[1]];
                FLOAT v2r_temp = in_r[in_strides[15]];
                FLOAT v2i_temp = in_i[in_strides[15]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad1 = v1r + v2r;
                sbr1 = v2r - v1r;
                adi1 = v1i + v2i;
                sb1 = v2i - v1i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (7 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (9 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v1r_temp = in_r[in_strides[7]];
                FLOAT v1i_temp = in_i[in_strides[7]];
                FLOAT v2r_temp = in_r[in_strides[9]];
                FLOAT v2i_temp = in_i[in_strides[9]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad7 = v1r + v2r;
                sbr7 = v2r - v1r;
                adi7 = v1i + v2i;
                sb7 = v2i - v1i;
            }

            cav1r = ad1 + ad7;
            ad17 = ad1 - ad7;
            sbr17 = sbr1 + sbr7;
            cav3r = sbr1 - sbr7;

            cav1i = adi1 + adi7;
            adi17 = adi1 - adi7;
            sb17 = sb1 + sb7;
            cav3i = sb1 - sb7;

            {
                UINTP twid_addr1 = DATA_STRIDE * (3 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (13 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v1r_temp = in_r[in_strides[3]];
                FLOAT v1i_temp = in_i[in_strides[3]];
                FLOAT v2r_temp = in_r[in_strides[13]];
                FLOAT v2i_temp = in_i[in_strides[13]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad3 = v1r + v2r;
                sbr3 = v2r - v1r;
                adi3 = v1i + v2i;
                sb3 = v2i - v1i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (5 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (11 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v1r_temp = in_r[in_strides[5]];
                FLOAT v1i_temp = in_i[in_strides[5]];
                FLOAT v2r_temp = in_r[in_strides[11]];
                FLOAT v2i_temp = in_i[in_strides[11]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                ad5 = v1r + v2r;
                sbr5 = v2r - v1r;
                adi5 = v1i + v2i;
                sb5 = v2i - v1i;
            }

            cav2r = ad3 + ad5;
            ad35 = ad3 - ad5;
            sbr35 = sbr3 + sbr5;
            cav4r = sbr3 - sbr5;

            cav2i = adi3 + adi5;
            adi35 = adi3 - adi5;
            sb35 = sb3 + sb5;
            cav4i = sb3 - sb5;
        }

        tvr3 = (CRTM_16_1 * sb35) + (CRTM_16_3 * sb17);
        tvr11 = (CRTM_16_1 * sb17) - (CRTM_16_3 * sb35);
        tvi1 = (CRTM_16_3 * adi35) + (CRTM_16_1 * adi17);
        tvi10 = (CRTM_16_3 * adi17) - (CRTM_16_1 * adi35);
        tvr7 = CRTM_16_2 * (cav3i + cav4i);
        tvi6 = CRTM_16_2 * (cav1i - cav2i);
        tvrr = cav1r + cav2r;
        tvii = cav1i + cav2i;

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (2 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (14 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v3r_temp = in_r[in_strides[2]];
                FLOAT v3i_temp = in_i[in_strides[2]];
                FLOAT v4r_temp = in_r[in_strides[14]];
                FLOAT v4i_temp = in_i[in_strides[14]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad2 = v3r + v4r;
                sbr2 = v4r - v3r;
                adi2 = v3i + v4i;
                sb2 = v4i - v3i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (6 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (10 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v3r_temp = in_r[in_strides[6]];
                FLOAT v3i_temp = in_i[in_strides[6]];
                FLOAT v4r_temp = in_r[in_strides[10]];
                FLOAT v4i_temp = in_i[in_strides[10]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad6 = v3r + v4r;
                sbr6 = v4r - v3r;
                adi6 = v3i + v4i;
                sb6 = v4i - v3i;
            }
        }

        ad26 = ad2 + ad6;
        tvi8 = sbr2 - sbr6;
        ctv4 = CRTM_16_2 * (sbr2 + sbr6);
        ctv1 = CRTM_16_2 * (ad2 - ad6);

        adi26 = adi2 + adi6;
        tvr8 = sb2 - sb6;
        ctv2 = CRTM_16_2 * (sb2 + sb6);
        ctv3 = CRTM_16_2 * (adi2 - adi6);

        tvr1 = (CRTM_16_3 * ad35) + (CRTM_16_1 * ad17);
        tvr10 = (CRTM_16_3 * ad17) - (CRTM_16_1 * ad35);
        tvi3 = (CRTM_16_1 * sbr35) + (CRTM_16_3 * sbr17);
        tvi11 = (CRTM_16_1 * sbr17) - (CRTM_16_3 * sbr35);

        tvr6 = CRTM_16_2 * (cav1r - cav2r);
        tvi7 = CRTM_16_2 * (cav3r + cav4r);

        {
            {
                UINTP twid_addr1 = DATA_STRIDE * (4 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (12 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v3r_temp = in_r[in_strides[4]];
                FLOAT v3i_temp = in_i[in_strides[4]];
                FLOAT v4r_temp = in_r[in_strides[12]];
                FLOAT v4i_temp = in_i[in_strides[12]];

                v3r = (v3r_temp * twid_r1) - (v3i_temp * twid_i1);
                v3i = (v3r_temp * twid_i1) + (v3i_temp * twid_r1);
                v4r = (v4r_temp * twid_r2) - (v4i_temp * twid_i2);
                v4i = (v4r_temp * twid_i2) + (v4i_temp * twid_r2);

                ad4 = v3r + v4r;
                sbr4 = v4r - v3r;
                adi4 = v3i + v4i;
                sb4 = v4i - v3i;
            }

            {
                UINTP twid_addr1 = DATA_STRIDE * (0 * n + cnt);
                FLOAT twid_r1 = tw[twid_addr1];
                FLOAT twid_i1 = tw[1 + twid_addr1];
                UINTP twid_addr2 = DATA_STRIDE * (8 * n + cnt);
                FLOAT twid_r2 = tw[twid_addr2];
                FLOAT twid_i2 = tw[1 + twid_addr2];

                FLOAT v1r_temp = *in_r;
                FLOAT v1i_temp = *in_i;
                FLOAT v2r_temp = in_r[in_strides[8]];
                FLOAT v2i_temp = in_i[in_strides[8]];

                v1r = (v1r_temp * twid_r1) - (v1i_temp * twid_i1);
                v1i = (v1r_temp * twid_i1) + (v1i_temp * twid_r1);
                v2r = (v2r_temp * twid_r2) - (v2i_temp * twid_i2);
                v2i = (v2r_temp * twid_i2) + (v2i_temp * twid_r2);

                cv1r = v1r + v2r;
                cv2r = v1r - v2r;
                cv1i = v1i + v2i;
                cv2i = v1i - v2i;
            }
        }

        tvr5 = cv1r - ad4;
        ad49 = cv1r + ad4;
        tvi4 = ctv4 + sbr4;
        tvi12 = ctv4 - sbr4;
        tvr2 = cv2r + ctv1;
        tvr9 = cv2r - ctv1;
        tvri = ad26 + ad49;
        *out_r = tvri + tvrr;
        out_r[out_strides[8]] = tvri - tvrr;

        adi49 = cv1i + adi4;
        tvi5 = cv1i - adi4;
        tvr4 = ctv2 + sb4;
        tvr12 = ctv2 - sb4;
        tvi2 = cv2i + ctv3;
        tvi9 = cv2i - ctv3;
        tvir = adi26 + adi49;
        *out_i = tvir + tvii;
        out_i[out_strides[8]] = tvir - tvii;

        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        out_r[out_strides[4]] = tvrr - tvri;
        out_r[out_strides[12]] = tvrr + tvri;

        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        out_i[out_strides[4]] = tvii + tvir;
        out_i[out_strides[12]] = tvii - tvir;

        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[15]] = tvrr + tvri;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;

        out_i[out_strides[1]] = tvii + tvir;
        out_i[out_strides[15]] = tvii - tvir;

        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        out_r[out_strides[7]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        out_i[out_strides[7]] = tvii + tvir;
        out_i[out_strides[9]] = tvii - tvir;

        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[14]] = tvrr + tvri;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        out_i[out_strides[2]] = tvii + tvir;
        out_i[out_strides[14]] = tvii - tvir;

        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        out_r[out_strides[6]] = tvrr - tvri;
        out_r[out_strides[10]] = tvrr + tvri;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        out_i[out_strides[6]] = tvii + tvir;

        out_i[out_strides[10]] = tvii - tvir;

        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[13]] = tvrr + tvri;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        out_i[out_strides[3]] = tvii + tvir;
        out_i[out_strides[13]] = tvii - tvir;

        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        out_r[out_strides[5]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        out_i[out_strides[5]] = tvii + tvir;
        out_i[out_strides[11]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft16c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft16c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft16c_fp64;
    }
    else
    {
        return NULL;
    }
}
