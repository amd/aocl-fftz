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

/** @file fft13c.c
 *
 *  @brief Radix-13 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-13 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 *  @author Ashwin K. Godbole
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

#ifdef USE_OPT_KERNEL_VARIANT

/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 68, 176, 52, 0, 0},
                                                     {0, 68, 176, 52, 0, 0}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
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

static VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
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

    const DOUBLE CRTM_13_1 =
        +1.73205080756887719317660412343684583902359008789060;
    const DOUBLE CRTM_13_2 =
        +0.38739058546761714712838513245787843298017865366345;
    const DOUBLE CRTM_13_3 =
        +0.13298312460741867056777367279491668244720145803175;
    const DOUBLE CRTM_13_4 =
        +0.11385447905579067614456922127652058891458054833493;
    const DOUBLE CRTM_13_5 =
        +0.25176851643188325619942540855595308576560112726050;
    const DOUBLE CRTM_13_6 =
        +0.86602540378443864676372317075293618347140262700000;
    const DOUBLE CRTM_13_7 =
        +0.50000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_13_8 =
        +2.00000000000000000000000000000000000000000000000000;

    // DGC - constants(C) of the diagonal(DG) matrix used in rader's algorithm
    const DOUBLE DGC_1 = -0.08333333333333341693177688718872429763576971755296;
    const DOUBLE DGC_2 = +0.25624767158293663769405689230781736686616059890980;
    const DOUBLE DGC_3 = +0.15689139105158457352993666120753768234094124525375;
    const DOUBLE DGC_4 = +0.25826039031174479484752691274654956353060547073611;
    const DOUBLE DGC_5 = +0.26596624921483734113554734558983336489440291606351;
    const DOUBLE DGC_6 = +0.57514072947400312136838554745545338846100160800000;
    const DOUBLE DGC_7 = +0.17413860115213590500566079492926474261696467600000;
    const DOUBLE DGC_8 = +0.07590298603719379320004091411468632874055721777845;
    const DOUBLE DGC_9 = +0.50353703286376651239885081711190617153120225452101;
    const DOUBLE DGC_10 = +0.3002386359663325979287571057309917466781880617299;
    const DOUBLE DGC_11 = +0.0115991056057681999237624507209650002461566663358;
    const DOUBLE DGC_12 = +0.3004626062886657229721584869768537486323029264720;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE vr[13];
        DOUBLE vi[13];

        DOUBLE tvrr_1o, tvii_1o, tvrr_2o, tvii_2o, tvri_2o, tvir_2o, tvri_1o,
            tvir_1o, tvrr_5o, tvri_5o, tvir_5o, tvii_5o, cv1rr, cv1ri, cv1ir,
            cv1ii, cv2rr, cv2ri, cv2ir, cv2ii, tv2, tv3, tv4, tv5, tv7, tv8,
            v711r, v711i;
        DOUBLE cv1, cv13, cv9, cv17, cv7, cv3, cv15, cv11, cv19, cv5, cv4, cv16,
            cv12, cv20, cv6, cv2, cv14, cv10, cv18, cv8, ad6, ad5, ad3, ad1,
            sb6, sb5, ad2, sb2, ad4, sb4, sb1, sb3, v17r, v71r, v17i, v71i;

        DOUBLE adr1_DFT1, adr2_DFT1, adr3_DFT1, adr4_DFT1, adr5_DFT1, adi1_DFT1,
            adi2_DFT1, adi3_DFT1, adi4_DFT1, adi5_DFT1, sbi1_DFT1, sbi2_DFT1,
            sbi3_DFT1, sbi4_DFT1, sbi5_DFT1, sbr1_DFT1, sbr2_DFT1, sbr3_DFT1,
            sbr4_DFT1, sbr5_DFT1, cv1rr_DFT1, cv1ri_DFT1, cv1ir_DFT1,
            cv1ii_DFT1, cv2rr_DFT1, cv2ri_DFT1, cv2ir_DFT1, cv2ii_DFT1,
            adr24_DFT1, adi24_DFT1, adr15_DFT1, adi15_DFT1, sbi15_DFT1,
            sbi51_DFT1, sbr15_DFT1, sbr51_DFT1, sbi24_DFT1, sbr24_DFT1,
            v17r_DFT1, v71r_DFT1, v17i_DFT1, v71i_DFT1, tv1_DFT1, tv2_DFT1,
            tv3_DFT1, tv4_DFT1, tv5_DFT1, tv6_DFT1, tv7_DFT1, tv8_DFT1,
            v711r_DFT1, v711i_DFT1, tvii_DFT1, tvrr_DFT1, tvii_2_DFT1,
            tvrr_2_DFT1;

        DOUBLE tvrr, tvri, tvir, tvii;
        DOUBLE tvrr2, tvri2, tvir2, tvii2;

        {
            vr[2] = in_r[in_strides[2]];
            vr[12] = in_r[in_strides[7]];
            adr1_DFT1 = vr[2] + vr[12];
            sbr1_DFT1 = vr[12] - vr[2];

            vr[6] = in_r[in_strides[6]];
            vr[8] = in_r[in_strides[11]];
            adr5_DFT1 = vr[6] + vr[8];
            sbr5_DFT1 = vr[8] - vr[6];

            adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
            adr1_DFT1 = adr1_DFT1 - adr5_DFT1;
            tv7_DFT1 = CRTM_13_6 * adr1_DFT1;
            sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
            sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;

            vr[3] = in_r[in_strides[4]];
            vr[11] = in_r[in_strides[10]];
            adr2_DFT1 = vr[3] + vr[11];
            sbr2_DFT1 = vr[11] - vr[3];

            vr[5] = in_r[in_strides[3]];
            vr[9] = in_r[in_strides[9]];
            adr4_DFT1 = vr[5] + vr[9];
            sbr4_DFT1 = vr[9] - vr[5];

            adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
            adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
            sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
            tv5_DFT1 = CRTM_13_6 * (sbr2_DFT1 + sbr4_DFT1);
            tv1_DFT1 = CRTM_13_7 * adr2_DFT1;

            cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
            cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;

            vr[4] = in_r[in_strides[8]];
            vr[10] = in_r[in_strides[5]];
            adr3_DFT1 = vr[4] + vr[10];
            sbr3_DFT1 = vr[10] - vr[4];
            tv4_DFT1 = (CRTM_13_7 * sbr15_DFT1) + sbr3_DFT1;

            vr[1] = in_r[in_strides[1]];
            vr[7] = in_r[in_strides[12]];
            v71r_DFT1 = vr[1] - vr[7];
            v17r_DFT1 = vr[1] + vr[7];
            v711r_DFT1 = v71r_DFT1 + tv1_DFT1;
            tvrr = v71r_DFT1 - adr2_DFT1;
            tvir = sbr15_DFT1 - sbr3_DFT1;
            cv10 = DGC_6 * tvrr + DGC_7 * tvir;
            cv11 = DGC_6 * tvir - DGC_7 * tvrr;

            cv1rr_DFT1 = v17r_DFT1 + adr3_DFT1;
            cv2rr_DFT1 = v17r_DFT1 - adr3_DFT1;
            tvrr_2_DFT1 = cv1rr_DFT1 - (CRTM_13_7 * cv1ri_DFT1);
            vr[1] = cv1rr_DFT1 + cv1ri_DFT1;
            tvrr_DFT1 = cv2rr_DFT1 + (CRTM_13_7 * cv2ri_DFT1);
            vr[7] = (cv2rr_DFT1 - cv2ri_DFT1) * DGC_12;
            vr[0] = in_r[0];
            out_r[0] = vr[1] + in_r[0];
            vr[1] = vr[1] * DGC_1 + vr[0];
            v17r = vr[1] + vr[7];
            v71r = vr[1] - vr[7];
        }

        {
            vi[2] = in_i[in_strides[2]];
            vi[12] = in_i[in_strides[7]];
            adi1_DFT1 = vi[2] + vi[12];
            sbi1_DFT1 = vi[12] - vi[2];

            vi[6] = in_i[in_strides[6]];
            vi[8] = in_i[in_strides[11]];
            adi5_DFT1 = vi[6] + vi[8];
            sbi5_DFT1 = vi[8] - vi[6];

            adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
            adi1_DFT1 = adi1_DFT1 - adi5_DFT1;
            sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
            sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
            tv8_DFT1 = CRTM_13_6 * adi1_DFT1;

            vi[3] = in_i[in_strides[4]];
            vi[11] = in_i[in_strides[10]];
            adi2_DFT1 = vi[3] + vi[11];
            sbi2_DFT1 = vi[11] - vi[3];

            vi[5] = in_i[in_strides[3]];
            vi[9] = in_i[in_strides[9]];
            adi4_DFT1 = vi[5] + vi[9];
            sbi4_DFT1 = vi[9] - vi[5];

            adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
            adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
            sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            tv3_DFT1 = CRTM_13_6 * (sbi2_DFT1 + sbi4_DFT1);
            tv6_DFT1 = CRTM_13_7 * adi4_DFT1;

            cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
            cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;

            vi[4] = in_i[in_strides[8]];
            vi[10] = in_i[in_strides[5]];
            adi3_DFT1 = vi[4] + vi[10];
            sbi3_DFT1 = vi[10] - vi[4];
            tv2_DFT1 = (CRTM_13_7 * sbi15_DFT1) + sbi3_DFT1;

            vi[1] = in_i[in_strides[1]];
            vi[7] = in_i[in_strides[12]];
            v71i_DFT1 = vi[1] - vi[7];
            v17i_DFT1 = vi[1] + vi[7];
            v711i_DFT1 = v71i_DFT1 + tv6_DFT1;
            tvri = sbi15_DFT1 - sbi3_DFT1;
            tvii = v71i_DFT1 - adi4_DFT1;
            cv9 =  DGC_7 * tvii - DGC_6 * tvri;
            cv12 = DGC_6 * tvii + DGC_7 * tvri;
            cv1rr = v17r + cv9;
            cv2rr = v17r - cv9;

            cv1ir_DFT1 = v17i_DFT1 + adi3_DFT1;
            cv2ir_DFT1 = v17i_DFT1 - adi3_DFT1;
            tvii_2_DFT1 = cv1ir_DFT1 - (CRTM_13_7 * cv1ii_DFT1);
            vi[1] = cv1ii_DFT1 + cv1ir_DFT1;
            tvii_DFT1 = cv2ir_DFT1 + (CRTM_13_7 * cv2ii_DFT1);
            vi[7] = (cv2ir_DFT1 - cv2ii_DFT1) * DGC_12;
            vi[0] = in_i[0];
            out_i[0] = vi[1] + in_i[0];
            vi[1] = vi[1] * DGC_1 + vi[0];
            v17i = vi[1] + vi[7];
            v71i = vi[1] - vi[7];
            cv1ir = v17i + cv11;
            cv2ir = v17i - cv11;
        }

        {

            tvrr = v711r_DFT1 + tv7_DFT1;
            tvrr2 = v711r_DFT1 - tv7_DFT1;
            tvir = tv4_DFT1 + tv5_DFT1;
            tvir2 = tv4_DFT1 - tv5_DFT1;
            cv2 = -DGC_2 * tvrr - DGC_3 * tvir;
            cv3 = DGC_2 * tvir - DGC_3 * tvrr;
            cv5 = -DGC_10 * tvir2 + DGC_11 * tvrr2;
            cv8 = DGC_10 * tvrr2 + DGC_11 * tvir2;

            tvri = tv2_DFT1 + tv3_DFT1;
            tvri2 = tv2_DFT1 - tv3_DFT1;
            tvii = v711i_DFT1 + tv8_DFT1;
            tvii2 = v711i_DFT1 - tv8_DFT1;
            cv1 = -DGC_2 * tvri + DGC_3 * tvii;
            cv4 = -DGC_2 * tvii - DGC_3 * tvri;
            cv6 = DGC_10 * tvii2 + DGC_11 * tvri2;
            cv7 = DGC_10 * tvri2 - DGC_11 * tvii2;
        }

        ad1 = cv3 + cv5;
        tv8 = CRTM_13_1 * (cv3 - cv5);
        ad2 = cv4 + cv6;
        sb2 = CRTM_13_1 * (cv4 - cv6);
        ad3 = cv1 + cv7;
        tv7 = CRTM_13_1 * (cv1 - cv7);
        ad4 = cv2 + cv8;
        sb4 = CRTM_13_1 * (cv2 - cv8);

        {
            tvrr = tvrr_DFT1;
            tvir = sbr51_DFT1 + sbr24_DFT1;
            tvii = tvii_DFT1;
            tvri = sbi51_DFT1 + sbi24_DFT1;

            cv13 = DGC_4 * tvrr - CRTM_13_3 * tvir;
            cv16 = -CRTM_13_2 * tvir - DGC_5 * tvrr;
            cv14 = CRTM_13_2 * tvri + DGC_5 * tvii;
            cv15 = DGC_4 * tvii - CRTM_13_3 * tvri;
        }

        {
            tvrr = tvrr_2_DFT1;
            tvir = sbr51_DFT1 - sbr24_DFT1;
            tvii = tvii_2_DFT1;
            tvri = sbi51_DFT1 - sbi24_DFT1;

            cv17 = DGC_8 * tvrr - CRTM_13_5 * tvir;
            cv20 = -CRTM_13_4 * tvir - DGC_9 * tvrr;
            cv18 = CRTM_13_4 * tvri + DGC_9 * tvii;
            cv19 = DGC_8 * tvii - CRTM_13_5 * tvri;
        }

        tv3 = cv16 + cv20;
        sb1 = cv16 - cv20;
        tv5 = cv14 + cv18;
        sb3 = cv14 - cv18;
        sb5 = cv15 - cv19;
        ad5 = cv15 + cv19;
        ad6 = cv13 + cv17;
        sb6 = cv13 - cv17;

        cv1ri = ad3 + ad6;
        cv2ri = ad3 - ad6;
        cv1ii = ad1 + ad5;
        cv2ii = ad1 - ad5;

        {
            {
                // Output point 1, post permutation point 1
                out_r[out_strides[1]] = cv1rr + (CRTM_13_8 * cv1ri);
                out_i[out_strides[1]] = cv1ir + (CRTM_13_8 * cv1ii);

                // Output point 7: post permutation point 12
                out_r[out_strides[12]] = cv2rr - (CRTM_13_8 * cv2ri);
                out_i[out_strides[12]] = cv2ir - (CRTM_13_8 * cv2ii);
            }

            {
                tvrr_1o = cv2rr + cv2ri;
                tvii_1o = cv2ir + cv2ii;
                tvrr_2o = cv1rr - cv1ri;
                tvri_2o = sb2 - sb1;
                tvri_1o = sb2 + sb1;
                tvii_2o = cv1ir - cv1ii;
                tvir_2o = sb4 - sb3;
                tvir_1o = sb4 + sb3;
                // Output point 5, post permutation point 3
                out_r[out_strides[3]] = tvrr_2o - tvri_2o;
                out_i[out_strides[3]] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_r[out_strides[9]] = tvrr_2o + tvri_2o;
                out_i[out_strides[9]] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_r[out_strides[4]] = tvrr_1o - tvri_1o;
                out_i[out_strides[4]] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_r[out_strides[10]] = tvrr_1o + tvri_1o;
                out_i[out_strides[10]] = tvii_1o - tvir_1o;
            }

            {
                tv2 = ad2 - cv12;
                tvri_5o = (CRTM_13_8 * ad2) + cv12;
                tv4 = ad4 - cv10;
                tvir_5o = (CRTM_13_8 * ad4) + cv10;
                tvrr_5o = v71r - CRTM_13_8 * sb6;
                v711r = v71r + sb6;
                tvii_5o = v71i - CRTM_13_8 * sb5;
                v711i = v71i + sb5;
                // Output point 4, post permutation point 8
                out_r[out_strides[8]] = tvrr_5o - tvri_5o;
                out_i[out_strides[8]] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_r[out_strides[5]] = tvrr_5o + tvri_5o;
                out_i[out_strides[5]] = tvii_5o - tvir_5o;
            }

            {
                DOUBLE tvri_3o = tv2 + tv3;
                DOUBLE tvri_4o = tv2 - tv3;
                DOUBLE tvir_3o = tv4 + tv5;
                DOUBLE tvir_4o = tv4 - tv5;
                DOUBLE tvrr_3o = v711r + tv7;
                DOUBLE tvrr_4o = v711r - tv7;
                DOUBLE tvii_3o = v711i + tv8;
                DOUBLE tvii_4o = v711i - tv8;

                // Output point 2, post permutation point 2
                out_r[out_strides[2]] = tvrr_3o - tvri_3o;
                out_i[out_strides[2]] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_r[out_strides[7]] = tvrr_3o + tvri_3o;
                out_i[out_strides[7]] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_r[out_strides[6]] = tvrr_4o - tvri_4o;
                out_i[out_strides[6]] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_r[out_strides[11]] = tvrr_4o + tvri_4o;
                out_i[out_strides[11]] = tvii_4o - tvir_4o;
            }
        }

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
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

    const FLOAT CRTM_13_1 =
        +1.73205080756887719317660412343684583902359008789060;
    const FLOAT CRTM_13_2 =
        +0.38739058546761714712838513245787843298017865366345;
    const FLOAT CRTM_13_3 =
        +0.13298312460741867056777367279491668244720145803175;
    const FLOAT CRTM_13_4 =
        +0.11385447905579067614456922127652058891458054833493;
    const FLOAT CRTM_13_5 =
        +0.25176851643188325619942540855595308576560112726050;
    const FLOAT CRTM_13_6 =
        +0.86602540378443864676372317075293618347140262700000;
    const FLOAT CRTM_13_7 =
        +0.50000000000000000000000000000000000000000000000000;
    const FLOAT CRTM_13_8 =
        +2.00000000000000000000000000000000000000000000000000;

    // DGC - constants(C) of the diagonal(DG) matrix used in rader's algorithm
    const FLOAT DGC_1 = -0.08333333333333341693177688718872429763576971755296;
    const FLOAT DGC_2 = +0.25624767158293663769405689230781736686616059890980;
    const FLOAT DGC_3 = +0.15689139105158457352993666120753768234094124525375;
    const FLOAT DGC_4 = +0.25826039031174479484752691274654956353060547073611;
    const FLOAT DGC_5 = +0.26596624921483734113554734558983336489440291606351;
    const FLOAT DGC_6 = +0.57514072947400312136838554745545338846100160800000;
    const FLOAT DGC_7 = +0.17413860115213590500566079492926474261696467600000;
    const FLOAT DGC_8 = +0.07590298603719379320004091411468632874055721777845;
    const FLOAT DGC_9 = +0.50353703286376651239885081711190617153120225452101;
    const FLOAT DGC_10 = +0.3002386359663325979287571057309917466781880617299;
    const FLOAT DGC_11 = +0.0115991056057681999237624507209650002461566663358;
    const FLOAT DGC_12 = +0.3004626062886657229721584869768537486323029264720;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT vr[13];
        FLOAT vi[13];

        FLOAT tvrr_1o, tvii_1o, tvrr_2o, tvii_2o, tvri_2o, tvir_2o, tvri_1o,
            tvir_1o, tvrr_5o, tvri_5o, tvir_5o, tvii_5o, cv1rr, cv1ri, cv1ir,
            cv1ii, cv2rr, cv2ri, cv2ir, cv2ii, tv2, tv3, tv4, tv5, tv7, tv8,
            v711r, v711i;
        FLOAT cv1, cv13, cv9, cv17, cv7, cv3, cv15, cv11, cv19, cv5, cv4, cv16,
            cv12, cv20, cv6, cv2, cv14, cv10, cv18, cv8, ad6, ad5, ad3, ad1,
            sb6, sb5, ad2, sb2, ad4, sb4, sb1, sb3, v17r, v71r, v17i, v71i;

        FLOAT adr1_DFT1, adr2_DFT1, adr3_DFT1, adr4_DFT1, adr5_DFT1, adi1_DFT1,
            adi2_DFT1, adi3_DFT1, adi4_DFT1, adi5_DFT1, sbi1_DFT1, sbi2_DFT1,
            sbi3_DFT1, sbi4_DFT1, sbi5_DFT1, sbr1_DFT1, sbr2_DFT1, sbr3_DFT1,
            sbr4_DFT1, sbr5_DFT1, cv1rr_DFT1, cv1ri_DFT1, cv1ir_DFT1,
            cv1ii_DFT1, cv2rr_DFT1, cv2ri_DFT1, cv2ir_DFT1, cv2ii_DFT1,
            adr24_DFT1, adi24_DFT1, adr15_DFT1, adi15_DFT1, sbi15_DFT1,
            sbi51_DFT1, sbr15_DFT1, sbr51_DFT1, sbi24_DFT1, sbr24_DFT1,
            v17r_DFT1, v71r_DFT1, v17i_DFT1, v71i_DFT1, tv1_DFT1, tv2_DFT1,
            tv3_DFT1, tv4_DFT1, tv5_DFT1, tv6_DFT1, tv7_DFT1, tv8_DFT1,
            v711r_DFT1, v711i_DFT1, tvii_DFT1, tvrr_DFT1, tvii_2_DFT1,
            tvrr_2_DFT1;

        FLOAT tvrr, tvri, tvir, tvii;
        FLOAT tvrr2, tvri2, tvir2, tvii2;

        {
            vr[2] = in_r[in_strides[2]];
            vr[12] = in_r[in_strides[7]];
            adr1_DFT1 = vr[2] + vr[12];
            sbr1_DFT1 = vr[12] - vr[2];

            vr[6] = in_r[in_strides[6]];
            vr[8] = in_r[in_strides[11]];
            adr5_DFT1 = vr[6] + vr[8];
            sbr5_DFT1 = vr[8] - vr[6];

            adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
            adr1_DFT1 = adr1_DFT1 - adr5_DFT1;
            tv7_DFT1 = CRTM_13_6 * adr1_DFT1;
            sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
            sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;

            vr[3] = in_r[in_strides[4]];
            vr[11] = in_r[in_strides[10]];
            adr2_DFT1 = vr[3] + vr[11];
            sbr2_DFT1 = vr[11] - vr[3];

            vr[5] = in_r[in_strides[3]];
            vr[9] = in_r[in_strides[9]];
            adr4_DFT1 = vr[5] + vr[9];
            sbr4_DFT1 = vr[9] - vr[5];

            adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
            adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
            sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
            tv5_DFT1 = CRTM_13_6 * (sbr2_DFT1 + sbr4_DFT1);
            tv1_DFT1 = CRTM_13_7 * adr2_DFT1;

            cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
            cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;

            vr[4] = in_r[in_strides[8]];
            vr[10] = in_r[in_strides[5]];
            adr3_DFT1 = vr[4] + vr[10];
            sbr3_DFT1 = vr[10] - vr[4];
            tv4_DFT1 = (CRTM_13_7 * sbr15_DFT1) + sbr3_DFT1;

            vr[1] = in_r[in_strides[1]];
            vr[7] = in_r[in_strides[12]];
            v71r_DFT1 = vr[1] - vr[7];
            v17r_DFT1 = vr[1] + vr[7];
            v711r_DFT1 = v71r_DFT1 + tv1_DFT1;
            tvrr = v71r_DFT1 - adr2_DFT1;
            tvir = sbr15_DFT1 - sbr3_DFT1;
            cv10 = DGC_6 * tvrr + DGC_7 * tvir;
            cv11 = DGC_6 * tvir - DGC_7 * tvrr;

            cv1rr_DFT1 = v17r_DFT1 + adr3_DFT1;
            cv2rr_DFT1 = v17r_DFT1 - adr3_DFT1;
            tvrr_2_DFT1 = cv1rr_DFT1 - (CRTM_13_7 * cv1ri_DFT1);
            vr[1] = cv1rr_DFT1 + cv1ri_DFT1;
            tvrr_DFT1 = cv2rr_DFT1 + (CRTM_13_7 * cv2ri_DFT1);
            vr[7] = (cv2rr_DFT1 - cv2ri_DFT1) * DGC_12;
            vr[0] = in_r[0];
            out_r[0] = vr[1] + in_r[0];
            vr[1] = vr[1] * DGC_1 + vr[0];
            v17r = vr[1] + vr[7];
            v71r = vr[1] - vr[7];
        }

        {
            vi[2] = in_i[in_strides[2]];
            vi[12] = in_i[in_strides[7]];
            adi1_DFT1 = vi[2] + vi[12];
            sbi1_DFT1 = vi[12] - vi[2];

            vi[6] = in_i[in_strides[6]];
            vi[8] = in_i[in_strides[11]];
            adi5_DFT1 = vi[6] + vi[8];
            sbi5_DFT1 = vi[8] - vi[6];

            adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
            adi1_DFT1 = adi1_DFT1 - adi5_DFT1;
            sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
            sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
            tv8_DFT1 = CRTM_13_6 * adi1_DFT1;

            vi[3] = in_i[in_strides[4]];
            vi[11] = in_i[in_strides[10]];
            adi2_DFT1 = vi[3] + vi[11];
            sbi2_DFT1 = vi[11] - vi[3];

            vi[5] = in_i[in_strides[3]];
            vi[9] = in_i[in_strides[9]];
            adi4_DFT1 = vi[5] + vi[9];
            sbi4_DFT1 = vi[9] - vi[5];

            adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
            adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
            sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            tv3_DFT1 = CRTM_13_6 * (sbi2_DFT1 + sbi4_DFT1);
            tv6_DFT1 = CRTM_13_7 * adi4_DFT1;

            cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
            cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;

            vi[4] = in_i[in_strides[8]];
            vi[10] = in_i[in_strides[5]];
            adi3_DFT1 = vi[4] + vi[10];
            sbi3_DFT1 = vi[10] - vi[4];
            tv2_DFT1 = (CRTM_13_7 * sbi15_DFT1) + sbi3_DFT1;

            vi[1] = in_i[in_strides[1]];
            vi[7] = in_i[in_strides[12]];
            v71i_DFT1 = vi[1] - vi[7];
            v17i_DFT1 = vi[1] + vi[7];
            v711i_DFT1 = v71i_DFT1 + tv6_DFT1;
            tvri = sbi15_DFT1 - sbi3_DFT1;
            tvii = v71i_DFT1 - adi4_DFT1;
            cv9 = DGC_7 * tvii - DGC_6 * tvri;
            cv12 = DGC_6 * tvii + DGC_7 * tvri;
            cv1rr = v17r + cv9;
            cv2rr = v17r - cv9;

            cv1ir_DFT1 = v17i_DFT1 + adi3_DFT1;
            cv2ir_DFT1 = v17i_DFT1 - adi3_DFT1;
            tvii_2_DFT1 = cv1ir_DFT1 - (CRTM_13_7 * cv1ii_DFT1);
            vi[1] = cv1ii_DFT1 + cv1ir_DFT1;
            tvii_DFT1 = cv2ir_DFT1 + (CRTM_13_7 * cv2ii_DFT1);
            vi[7] = (cv2ir_DFT1 - cv2ii_DFT1) * DGC_12;
            vi[0] = in_i[0];
            out_i[0] = vi[1] + in_i[0];
            vi[1] = vi[1] * DGC_1 + vi[0];
            v17i = vi[1] + vi[7];
            v71i = vi[1] - vi[7];
            cv1ir = v17i + cv11;
            cv2ir = v17i - cv11;
        }

        {

            tvrr = v711r_DFT1 + tv7_DFT1;
            tvrr2 = v711r_DFT1 - tv7_DFT1;
            tvir = tv4_DFT1 + tv5_DFT1;
            tvir2 = tv4_DFT1 - tv5_DFT1;
            cv2 = -DGC_2 * tvrr - DGC_3 * tvir;
            cv3 = DGC_2 * tvir - DGC_3 * tvrr;
            cv5 = -DGC_10 * tvir2 + DGC_11 * tvrr2;
            cv8 = DGC_10 * tvrr2 + DGC_11 * tvir2;

            tvri = tv2_DFT1 + tv3_DFT1;
            tvri2 = tv2_DFT1 - tv3_DFT1;
            tvii = v711i_DFT1 + tv8_DFT1;
            tvii2 = v711i_DFT1 - tv8_DFT1;
            cv1 = -DGC_2 * tvri + DGC_3 * tvii;
            cv4 = -DGC_2 * tvii - DGC_3 * tvri;
            cv6 = DGC_10 * tvii2 + DGC_11 * tvri2;
            cv7 = DGC_10 * tvri2 - DGC_11 * tvii2;
        }

        ad1 = cv3 + cv5;
        tv8 = CRTM_13_1 * (cv3 - cv5);
        ad2 = cv4 + cv6;
        sb2 = CRTM_13_1 * (cv4 - cv6);
        ad3 = cv1 + cv7;
        tv7 = CRTM_13_1 * (cv1 - cv7);
        ad4 = cv2 + cv8;
        sb4 = CRTM_13_1 * (cv2 - cv8);

        {
            tvrr = tvrr_DFT1;
            tvir = sbr51_DFT1 + sbr24_DFT1;
            tvii = tvii_DFT1;
            tvri = sbi51_DFT1 + sbi24_DFT1;

            cv13 = DGC_4 * tvrr - CRTM_13_3 * tvir;
            cv16 = -CRTM_13_2 * tvir - DGC_5 * tvrr;
            cv14 = CRTM_13_2 * tvri + DGC_5 * tvii;
            cv15 = DGC_4 * tvii - CRTM_13_3 * tvri;
        }

        {
            tvrr = tvrr_2_DFT1;
            tvir = sbr51_DFT1 - sbr24_DFT1;
            tvii = tvii_2_DFT1;
            tvri = sbi51_DFT1 - sbi24_DFT1;

            cv17 = DGC_8 * tvrr - CRTM_13_5 * tvir;
            cv20 = -CRTM_13_4 * tvir - DGC_9 * tvrr;
            cv18 = CRTM_13_4 * tvri + DGC_9 * tvii;
            cv19 = DGC_8 * tvii - CRTM_13_5 * tvri;
        }

        tv3 = cv16 + cv20;
        sb1 = cv16 - cv20;
        tv5 = cv14 + cv18;
        sb3 = cv14 - cv18;
        sb5 = cv15 - cv19;
        ad5 = cv15 + cv19;
        ad6 = cv13 + cv17;
        sb6 = cv13 - cv17;

        cv1ri = ad3 + ad6;
        cv2ri = ad3 - ad6;
        cv1ii = ad1 + ad5;
        cv2ii = ad1 - ad5;

        {
            {
                // Output point 1, post permutation point 1
                out_r[out_strides[1]] = cv1rr + (CRTM_13_8 * cv1ri);
                out_i[out_strides[1]] = cv1ir + (CRTM_13_8 * cv1ii);

                // Output point 7: post permutation point 12
                out_r[out_strides[12]] = cv2rr - (CRTM_13_8 * cv2ri);
                out_i[out_strides[12]] = cv2ir - (CRTM_13_8 * cv2ii);
            }

            {
                tvrr_1o = cv2rr + cv2ri;
                tvii_1o = cv2ir + cv2ii;
                tvrr_2o = cv1rr - cv1ri;
                tvri_2o = sb2 - sb1;
                tvri_1o = sb2 + sb1;
                tvii_2o = cv1ir - cv1ii;
                tvir_2o = sb4 - sb3;
                tvir_1o = sb4 + sb3;
                // Output point 5, post permutation point 3
                out_r[out_strides[3]] = tvrr_2o - tvri_2o;
                out_i[out_strides[3]] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_r[out_strides[9]] = tvrr_2o + tvri_2o;
                out_i[out_strides[9]] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_r[out_strides[4]] = tvrr_1o - tvri_1o;
                out_i[out_strides[4]] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_r[out_strides[10]] = tvrr_1o + tvri_1o;
                out_i[out_strides[10]] = tvii_1o - tvir_1o;
            }

            {
                tv2 = ad2 - cv12;
                tvri_5o = (CRTM_13_8 * ad2) + cv12;
                tv4 = ad4 - cv10;
                tvir_5o = (CRTM_13_8 * ad4) + cv10;
                tvrr_5o = v71r - CRTM_13_8 * sb6;
                v711r = v71r + sb6;
                tvii_5o = v71i - CRTM_13_8 * sb5;
                v711i = v71i + sb5;
                // Output point 4, post permutation point 8
                out_r[out_strides[8]] = tvrr_5o - tvri_5o;
                out_i[out_strides[8]] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_r[out_strides[5]] = tvrr_5o + tvri_5o;
                out_i[out_strides[5]] = tvii_5o - tvir_5o;
            }

            {
                FLOAT tvri_3o = tv2 + tv3;
                FLOAT tvri_4o = tv2 - tv3;
                FLOAT tvir_3o = tv4 + tv5;
                FLOAT tvir_4o = tv4 - tv5;
                FLOAT tvrr_3o = v711r + tv7;
                FLOAT tvrr_4o = v711r - tv7;
                FLOAT tvii_3o = v711i + tv8;
                FLOAT tvii_4o = v711i - tv8;

                // Output point 2, post permutation point 2
                out_r[out_strides[2]] = tvrr_3o - tvri_3o;
                out_i[out_strides[2]] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_r[out_strides[7]] = tvrr_3o + tvri_3o;
                out_i[out_strides[7]] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_r[out_strides[6]] = tvrr_4o - tvri_4o;
                out_i[out_strides[6]] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_r[out_strides[11]] = tvrr_4o + tvri_4o;
                out_i[out_strides[11]] = tvii_4o - tvir_4o;
            }
        }

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
    {0, 268, 1112, 130, 0, 1093}, {0, 268, 1112, 130, 0, 1093}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
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

const DOUBLE CRTM_13[RADIX_13][2] = {{1.0, 0.0},
                                     {0.88545602565321, -0.464723172043769},
                                     {0.568064746731156, -0.822983865893656},
                                     {0.120536680255323, -0.992708874098054},
                                     {-0.354604887042536, -0.935016242685415},
                                     {-0.748510748171101, -0.663122658240795},
                                     {-0.970941817426052, -0.239315664287558},
                                     {-0.970941817426052, 0.239315664287557},
                                     {-0.748510748171101, 0.663122658240795},
                                     {-0.354604887042536, 0.935016242685415},
                                     {0.120536680255323, 0.992708874098054},
                                     {0.568064746731156, 0.822983865893657},
                                     {0.88545602565321, 0.464723172043768}};

static VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
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
    DOUBLE local_in[RADIX_13][2] = {0};

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

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 13i ********************/
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
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 13i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_13[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_13[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_13[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_13[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_13[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_13[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_13[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_13[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_13[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_13[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_13[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_13[12], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 13i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 13i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 13i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 13i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 13i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 13i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 13i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 13i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 13i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 13i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 13i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

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

static VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        UINT8 flag)
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
    FLOAT local_in[RADIX_13][2] = {0};

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

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 13i ********************/
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
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 13i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_13[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_13[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_13[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_13[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_13[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_13[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_13[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_13[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_13[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_13[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_13[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_13[12], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 13i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 13i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 13i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 13i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        /******************** Output 13i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[6],
                     output_i + out_strides[6]);

        /******************** Output 13i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[7],
                     output_i + out_strides[7]);

        /******************** Output 13i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[8],
                     output_i + out_strides[8]);

        /******************** Output 13i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[9],
                     output_i + out_strides[9]);

        /******************** Output 13i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[10],
                     output_i + out_strides[10]);

        /******************** Output 13i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[11],
                     output_i + out_strides[11]);

        /******************** Output 13i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[12],
                     output_i + out_strides[12]);

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

kfft_ register_kernel_fft13c(INT32 precision, INT32 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft13c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft13c_fp64;
    }
    else
    {
        return NULL;
    }
}
