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
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft13c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft13c_fp32;
    else if (precision == DT_DOUBLE)
        return fft13c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT

// complex multiplication (double) (in-place) in var form: A = A * B
#define CMUL_INPLACE_FP64(c1r, c1i, c2r, c2i)                                  \
    {                                                                          \
        DOUBLE copy_c1r = c1r;                                                 \
        c1r = (copy_c1r * c2r) - (c1i * c2i);                                  \
        c1i = (copy_c1r * c2i) + (c1i * c2r);                                  \
    }

// complex multiplication (float) (in-place) in var form: A = A * B
#define CMUL_INPLACE_FP32(c1r, c1i, c2r, c2i)                                  \
    {                                                                          \
        FLOAT copy_c1r = c1r;                                                  \
        c1r = (copy_c1r * c2r) - (c1i * c2i);                                  \
        c1i = (copy_c1r * c2i) + (c1i * c2r);                                  \
    }

/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 78, 216, 230, 0, 0},
                                                     {0, 78, 216, 230, 0, 0}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    const DOUBLE CRTM_12_1 =
        0.86602540378443864676372317075293618347140262700000;
    const DOUBLE CRTM_12_2 =
        0.50000000000000000000000000000000000000000000000000;

    // DGC - constants(C) of the diagonal(DG) matrix used in rader's algorithm
    const DOUBLE DGC_1 = -0.08333333333333341693177688718872429763576971755296;
    const DOUBLE DGC_2 = +0.25624767158293663769405689230781736686616059890980;
    const DOUBLE DGC_3 = +0.15689139105158457352993666120753768234094124525375;
    const DOUBLE DGC_4 = +0.25826039031174479484752691274654956353060547073611;
    const DOUBLE DGC_5 = +0.15355568557954144041605893849125243204698762637388;
    const DOUBLE DGC_6 = +0.28757036473700157031322574591241948572798864686781;
    const DOUBLE DGC_7 = +0.08706930057606790908049634621971372668862591093985;
    const DOUBLE DGC_8 = +0.07590298603719379320004091411468632874055721777845;
    const DOUBLE DGC_9 = +0.29071724147084106396127254528701896403545620197956;
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

        {
            DOUBLE adr1, adr2, adr3, adr4, adr5, adi1, adi2, adi3, adi4, adi5,
                sbi1, sbi2, sbi3, sbi4, sbi5, sbr1, sbr2, sbr3, sbr4, sbr5,
                adr24, adi24, adr15, adi15, adr51, adi51, adr42, adi42, sbi15,
                sbi51, sbr15, sbr51, sbi24, sbr24, v17r, v71r, v17i, v71i;

            DOUBLE adr1_DFT1, adr2_DFT1, adr3_DFT1, adr4_DFT1, adr5_DFT1,
                adi1_DFT1, adi2_DFT1, adi3_DFT1, adi4_DFT1, adi5_DFT1,
                sbi1_DFT1, sbi2_DFT1, sbi3_DFT1, sbi4_DFT1, sbi5_DFT1,
                sbr1_DFT1, sbr2_DFT1, sbr3_DFT1, sbr4_DFT1, sbr5_DFT1,
                cv1rr_DFT1, cv1ri_DFT1, cv1ir_DFT1, cv1ii_DFT1, cv2rr_DFT1,
                cv2ri_DFT1, cv2ir_DFT1, cv2ii_DFT1, adr24_DFT1, adi24_DFT1,
                adr15_DFT1, adi15_DFT1, sbi15_DFT1, sbi51_DFT1, sbr15_DFT1,
                sbr51_DFT1, sbi24_DFT1, sbr24_DFT1, v17r_DFT1, v71r_DFT1,
                v17i_DFT1, v71i_DFT1, tv1_DFT1, tv2_DFT1, tv3_DFT1, tv4_DFT1,
                tv5_DFT1, tv6_DFT1, tv7_DFT1, tv8_DFT1, v711r_DFT1, v711i_DFT1,
                early_tvii_DFT1, early_tvrr_DFT1, early_tvii_2_DFT1,
                early_tvrr_2_DFT1;

            {
                vr[1] = in_r[in_stride];
                vr[2] = in_r[in_stride * 2];
                vr[3] = in_r[in_stride * 4];
                vr[4] = in_r[in_stride * 8];
                vr[5] = in_r[in_stride * 3];
                vr[6] = in_r[in_stride * 6];
                vr[7] = in_r[in_stride * 12];
                vr[8] = in_r[in_stride * 11];
                vr[9] = in_r[in_stride * 9];
                vr[10] = in_r[in_stride * 5];
                vr[11] = in_r[in_stride * 10];
                vr[12] = in_r[in_stride * 7];

                v71r_DFT1 = vr[1] - vr[7];

                {
                    adr1_DFT1 = vr[2] + vr[12];
                    adr5_DFT1 = vr[6] + vr[8];
                    v17r_DFT1 = vr[1] + vr[7];
                    adr2_DFT1 = vr[3] + vr[11];
                    adr3_DFT1 = vr[4] + vr[10];
                    adr4_DFT1 = vr[5] + vr[9];
                    adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
                    adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
                    cv1rr_DFT1 = v17r_DFT1 + adr3_DFT1;
                    cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
                    early_tvrr_2_DFT1 = cv1rr_DFT1 - (CRTM_12_2 * cv1ri_DFT1);

                    // [[ DFT1 ]] Output point 1: X(0) / V(1)
                    vr[1] = cv1rr_DFT1 + cv1ri_DFT1;

                    // Note: Since out_r[0] can be an alias to in_r[0], we make
                    // a copy of the 0th input point to use for the calculation
                    // of the point vr[1]
                    vr[0] = in_r[0];
                    out_r[0] = vr[1] + in_r[0];
                    vr[1] = (vr[1] - (12 * vr[0])) * DGC_1;

                    cv2rr_DFT1 = v17r_DFT1 - adr3_DFT1;
                    cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;
                    early_tvrr_DFT1 = cv2rr_DFT1 + (CRTM_12_2 * cv2ri_DFT1);

                    // [[ DFT1 ]] Output point 7: X(6) / V(7)
                    vr[7] = (cv2rr_DFT1 - cv2ri_DFT1) * DGC_12;
                }

                v17r = vr[1] + vr[7];
                v71r = vr[1] - vr[7];

                adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
                adr1_DFT1 = adr1_DFT1 - adr5_DFT1;

                sbr1_DFT1 = vr[12] - vr[2];
                sbr2_DFT1 = vr[11] - vr[3];
                sbr3_DFT1 = vr[10] - vr[4];
                sbr4_DFT1 = vr[9] - vr[5];
                sbr5_DFT1 = vr[8] - vr[6];

                sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
                sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
                sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;
            }

            {
                vi[1] = in_i[in_stride];
                vi[2] = in_i[in_stride * 2];
                vi[3] = in_i[in_stride * 4];
                vi[4] = in_i[in_stride * 8];
                vi[5] = in_i[in_stride * 3];
                vi[6] = in_i[in_stride * 6];
                vi[7] = in_i[in_stride * 12];
                vi[8] = in_i[in_stride * 11];
                vi[9] = in_i[in_stride * 9];
                vi[10] = in_i[in_stride * 5];
                vi[11] = in_i[in_stride * 10];
                vi[12] = in_i[in_stride * 7];

                v71i_DFT1 = vi[1] - vi[7];

                {
                    adi1_DFT1 = vi[2] + vi[12];
                    adi5_DFT1 = vi[6] + vi[8];
                    v17i_DFT1 = vi[1] + vi[7];
                    adi2_DFT1 = vi[3] + vi[11];
                    adi3_DFT1 = vi[4] + vi[10];
                    adi4_DFT1 = vi[5] + vi[9];
                    adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
                    adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
                    cv1ir_DFT1 = v17i_DFT1 + adi3_DFT1;
                    cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
                    early_tvii_2_DFT1 = cv1ir_DFT1 - (CRTM_12_2 * cv1ii_DFT1);

                    // [[ DFT1 ]] Output point 1: X(0) / V(1)
                    vi[1] = cv1ii_DFT1 + cv1ir_DFT1;

                    // Note: Since out_i[0] can be an alias to in_i[0], we make
                    // a copy of the 0th input point to use for the calculation
                    // of the point vi[1]
                    vi[0] = in_i[0];
                    out_i[0] = vi[1] + in_i[0];
                    vi[1] = (vi[1] - (12 * vi[0])) * DGC_1;

                    cv2ir_DFT1 = v17i_DFT1 - adi3_DFT1;
                    cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;
                    early_tvii_DFT1 = cv2ir_DFT1 + (CRTM_12_2 * cv2ii_DFT1);

                    // [[ DFT1 ]] Output point 7: X(6) / V(7)
                    vi[7] = (cv2ir_DFT1 - cv2ii_DFT1) * DGC_12;
                }

                v17i = vi[1] + vi[7];
                v71i = vi[1] - vi[7];

                adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
                adi1_DFT1 = adi1_DFT1 - adi5_DFT1;

                sbi1_DFT1 = vi[12] - vi[2];
                sbi2_DFT1 = vi[11] - vi[3];
                sbi3_DFT1 = vi[10] - vi[4];
                sbi4_DFT1 = vi[9] - vi[5];
                sbi5_DFT1 = vi[8] - vi[6];

                sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
                sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
                sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            }

            {
                tv1_DFT1 = CRTM_12_2 * adr2_DFT1;
                tv4_DFT1 = CRTM_12_2 * sbr15_DFT1 + sbr3_DFT1;
                tv2_DFT1 = CRTM_12_2 * sbi15_DFT1 + sbi3_DFT1;
                tv6_DFT1 = CRTM_12_2 * adi4_DFT1;
            }

            {
                tv7_DFT1 = CRTM_12_1 * adr1_DFT1;
                tv5_DFT1 = CRTM_12_1 * (sbr2_DFT1 + sbr4_DFT1);
                tv3_DFT1 = CRTM_12_1 * (sbi2_DFT1 + sbi4_DFT1);
                tv8_DFT1 = CRTM_12_1 * adi1_DFT1;
            }

            v711r_DFT1 = v71r_DFT1 + tv1_DFT1;
            v711i_DFT1 = v71i_DFT1 + tv6_DFT1;

            {

                DOUBLE tvrr, tvri, tvir, tvii;
                DOUBLE tvrr2, tvri2, tvir2, tvii2;

                tvrr = v711r_DFT1 + tv7_DFT1;
                tvrr2 = v711r_DFT1 - tv7_DFT1;
                tvri = tv2_DFT1 + tv3_DFT1;
                tvri2 = tv2_DFT1 - tv3_DFT1;
                tvir = tv4_DFT1 + tv5_DFT1;
                tvir2 = tv4_DFT1 - tv5_DFT1;
                tvii = v711i_DFT1 + tv8_DFT1;
                tvii2 = v711i_DFT1 - tv8_DFT1;

                // [[ DFT1 ]] Output point 2: X(1) / V(2)
                vr[2] = tvrr - tvri;
                vi[2] = tvii + tvir;

                // [[ DFT1 ]] Output point 12: X(11) / V(12)
                vr[12] = tvrr + tvri;
                vi[12] = tvii - tvir;

                // [[ DFT1 ]] Output point 6: X(5) / V(6)
                vr[6] = tvrr2 - tvri2;
                vi[6] = tvii2 + tvir2;

                // [[ DFT1 ]] Output point 8: X(7) / V(8)
                vr[8] = tvrr2 + tvri2;
                vi[8] = tvii2 - tvir2;

                CMUL_INPLACE_FP64(vr[2], vi[2], DGC_2, -DGC_3);
                CMUL_INPLACE_FP64(vr[12], vi[12], -DGC_2, -DGC_3);

                adr1 = vr[2] + vr[12];
                sbr1 = vr[12] - vr[2];
                adi1 = vi[2] + vi[12];
                sbi1 = vi[12] - vi[2];

                CMUL_INPLACE_FP64(vr[8], vi[8], DGC_10, DGC_11);
                CMUL_INPLACE_FP64(vr[6], vi[6], -DGC_10, DGC_11);

                adi5 = vi[6] + vi[8];
                sbi5 = vi[8] - vi[6];
                adr5 = vr[6] + vr[8];
                sbr5 = vr[8] - vr[6];
            }

            DOUBLE tvrr, tvri, tvir, tvii;

            adi51 = adi1 - adi5;
            adi15 = adi1 + adi5;
            sbi15 = sbi1 + sbi5;
            sbi51 = sbi1 - sbi5;
            adr51 = adr1 - adr5;
            adr15 = adr1 + adr5;
            sbr15 = sbr1 + sbr5;
            sbr51 = sbr1 - sbr5;

            tv7 = CRTM_12_1 * adr51;
            tv8 = CRTM_12_1 * adi51;

            {
                tvrr = v71r_DFT1 - adr2_DFT1;
                tvri = sbi15_DFT1 - sbi3_DFT1;
                tvir = sbr15_DFT1 - sbr3_DFT1;
                tvii = v71i_DFT1 - adi4_DFT1;

                // [[ DFT1 ]] Output point 4: X(3) / V(4)
                vr[4] = tvrr - tvri;
                vi[4] = tvii + tvir;

                // [[ DFT1 ]] Output point 10: X(9) / V(10)
                vr[10] = tvrr + tvri;
                vi[10] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[10], vi[10], -DGC_6, -DGC_7);
                CMUL_INPLACE_FP64(vr[4], vi[4], DGC_6, -DGC_7);
            }

            adr3 = vr[4] + vr[10];
            sbr3 = vr[10] - vr[4];
            adi3 = vi[4] + vi[10];
            sbi3 = vi[10] - vi[4];
            tv2 = CRTM_12_2 * sbi15 + sbi3;
            tv4 = CRTM_12_2 * sbr15 + sbr3;
            cv1rr = v17r + adr3;
            cv1ir = v17i + adi3;
            cv2rr = v17r - adr3;
            cv2ir = v17i - adi3;
            tvri_5o = sbi15 - sbi3;
            tvir_5o = sbr15 - sbr3;

            {
                // [[ DFT1 ]] Output point 3: X(2) / V(3)
                tvrr = early_tvrr_DFT1;
                tvii = early_tvii_DFT1;
                tvri = CRTM_12_1 * (sbi51_DFT1 + sbi24_DFT1);
                tvir = CRTM_12_1 * (sbr51_DFT1 + sbr24_DFT1);

                vr[3] = tvrr - tvri;
                vi[3] = tvii + tvir;

                // [[ DFT1 ]] Output point 11: X(10) / V(11)
                vr[11] = tvrr + tvri;
                vi[11] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[11], vi[11], DGC_4, -DGC_5);
                CMUL_INPLACE_FP64(vr[3], vi[3], DGC_4, DGC_5);
            }

            adr2 = vr[3] + vr[11];
            sbr2 = vr[11] - vr[3];
            adi2 = vi[3] + vi[11];
            sbi2 = vi[11] - vi[3];

            {
                tvrr = early_tvrr_2_DFT1;
                tvii = early_tvii_2_DFT1;
                tvri = CRTM_12_1 * (sbi51_DFT1 - sbi24_DFT1);
                tvir = CRTM_12_1 * (sbr51_DFT1 - sbr24_DFT1);

                // [[ DFT1 ]] Output point 5: X(4) / V(5)
                vr[5] = tvrr - tvri;
                vi[5] = tvii + tvir;

                // [[ DFT1 ]] Output point 9: X(8) / V(9)
                vr[9] = tvrr + tvri;
                vi[9] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[9], vi[9], DGC_8, -DGC_9);
                CMUL_INPLACE_FP64(vr[5], vi[5], DGC_8, DGC_9);
            }

            adr4 = vr[5] + vr[9];
            sbr4 = vr[9] - vr[5];
            adi4 = vi[5] + vi[9];
            sbi4 = vi[9] - vi[5];
            adi42 = adi2 - adi4;
            adi24 = adi2 + adi4;
            sbi24 = sbi2 - sbi4;
            adr24 = adr2 + adr4;
            adr42 = adr2 - adr4;
            sbr24 = sbr2 - sbr4;
            tvrr_5o = v71r - adr42;
            tvii_5o = v71i - adi42;
            cv1ri = adr15 + adr24;
            cv1ii = adi15 + adi24;
            cv2ri = adr15 - adr24;
            cv2ii = adi15 - adi24;

            {
                v711r = v71r + (CRTM_12_2 * adr42);
                v711i = v71i + (CRTM_12_2 * adi42);
                tvrr_1o = cv2rr + (CRTM_12_2 * cv2ri);
                tvii_1o = cv2ir + (CRTM_12_2 * cv2ii);
                tvrr_2o = cv1rr - (CRTM_12_2 * cv1ri);
                tvii_2o = cv1ir - (CRTM_12_2 * cv1ii);
            }

            {
                tv3 = CRTM_12_1 * (sbi2 + sbi4);
                tv5 = CRTM_12_1 * (sbr2 + sbr4);
                tvri_2o = CRTM_12_1 * (sbi51 - sbi24);
                tvir_2o = CRTM_12_1 * (sbr51 - sbr24);
                tvri_1o = CRTM_12_1 * (sbi51 + sbi24);
                tvir_1o = CRTM_12_1 * (sbr51 + sbr24);
            }
        }

        {
            {
                // Output point 1, post permutation point 1
                out_r[out_stride] = cv1rr + cv1ri;
                out_i[out_stride] = cv1ii + cv1ir;

                // Output point 7: post permutation point 12
                out_r[out_stride * 12] = cv2rr - cv2ri;
                out_i[out_stride * 12] = cv2ir - cv2ii;
            }

            {
                // Output point 5, post permutation point 3
                out_r[out_stride * 3] = tvrr_2o - tvri_2o;
                out_i[out_stride * 3] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_r[out_stride * 9] = tvrr_2o + tvri_2o;
                out_i[out_stride * 9] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_r[out_stride * 4] = tvrr_1o - tvri_1o;
                out_i[out_stride * 4] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_r[out_stride * 10] = tvrr_1o + tvri_1o;
                out_i[out_stride * 10] = tvii_1o - tvir_1o;
            }

            {
                // Output point 4, post permutation point 8
                out_r[out_stride * 8] = tvrr_5o - tvri_5o;
                out_i[out_stride * 8] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_r[out_stride * 5] = tvrr_5o + tvri_5o;
                out_i[out_stride * 5] = tvii_5o - tvir_5o;
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
                out_r[out_stride * 2] = tvrr_3o - tvri_3o;
                out_i[out_stride * 2] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_r[out_stride * 7] = tvrr_3o + tvri_3o;
                out_i[out_stride * 7] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_r[out_stride * 6] = tvrr_4o - tvri_4o;
                out_i[out_stride * 6] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_r[out_stride * 11] = tvrr_4o + tvri_4o;
                out_i[out_stride * 11] = tvii_4o - tvir_4o;
            }
        }

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}

VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    const FLOAT CRTM_12_1 =
        0.86602540378443864676372317075293618347140262700000;
    const FLOAT CRTM_12_2 =
        0.50000000000000000000000000000000000000000000000000;

    // DGC - constants(C) of the diagonal(DG) matrix used in rader's algorithm
    const FLOAT DGC_1 = -0.08333333333333341693177688718872429763576971755296;
    const FLOAT DGC_2 = +0.25624767158293663769405689230781736686616059890980;
    const FLOAT DGC_3 = +0.15689139105158457352993666120753768234094124525375;
    const FLOAT DGC_4 = +0.25826039031174479484752691274654956353060547073611;
    const FLOAT DGC_5 = +0.15355568557954144041605893849125243204698762637388;
    const FLOAT DGC_6 = +0.28757036473700157031322574591241948572798864686781;
    const FLOAT DGC_7 = +0.08706930057606790908049634621971372668862591093985;
    const FLOAT DGC_8 = +0.07590298603719379320004091411468632874055721777845;
    const FLOAT DGC_9 = +0.29071724147084106396127254528701896403545620197956;
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

        {
            FLOAT adr1, adr2, adr3, adr4, adr5, adi1, adi2, adi3, adi4, adi5,
                sbi1, sbi2, sbi3, sbi4, sbi5, sbr1, sbr2, sbr3, sbr4, sbr5,
                adr24, adi24, adr15, adi15, adr51, adi51, adr42, adi42, sbi15,
                sbi51, sbr15, sbr51, sbi24, sbr24, v17r, v71r, v17i, v71i;

            FLOAT adr1_DFT1, adr2_DFT1, adr3_DFT1, adr4_DFT1, adr5_DFT1,
                adi1_DFT1, adi2_DFT1, adi3_DFT1, adi4_DFT1, adi5_DFT1,
                sbi1_DFT1, sbi2_DFT1, sbi3_DFT1, sbi4_DFT1, sbi5_DFT1,
                sbr1_DFT1, sbr2_DFT1, sbr3_DFT1, sbr4_DFT1, sbr5_DFT1,
                cv1rr_DFT1, cv1ri_DFT1, cv1ir_DFT1, cv1ii_DFT1, cv2rr_DFT1,
                cv2ri_DFT1, cv2ir_DFT1, cv2ii_DFT1, adr24_DFT1, adi24_DFT1,
                adr15_DFT1, adi15_DFT1, sbi15_DFT1, sbi51_DFT1, sbr15_DFT1,
                sbr51_DFT1, sbi24_DFT1, sbr24_DFT1, v17r_DFT1, v71r_DFT1,
                v17i_DFT1, v71i_DFT1, tv1_DFT1, tv2_DFT1, tv3_DFT1, tv4_DFT1,
                tv5_DFT1, tv6_DFT1, tv7_DFT1, tv8_DFT1, v711r_DFT1, v711i_DFT1,
                early_tvii_DFT1, early_tvrr_DFT1, early_tvii_2_DFT1,
                early_tvrr_2_DFT1;

            {
                vr[1] = in_r[in_stride];
                vr[2] = in_r[in_stride * 2];
                vr[3] = in_r[in_stride * 4];
                vr[4] = in_r[in_stride * 8];
                vr[5] = in_r[in_stride * 3];
                vr[6] = in_r[in_stride * 6];
                vr[7] = in_r[in_stride * 12];
                vr[8] = in_r[in_stride * 11];
                vr[9] = in_r[in_stride * 9];
                vr[10] = in_r[in_stride * 5];
                vr[11] = in_r[in_stride * 10];
                vr[12] = in_r[in_stride * 7];

                v71r_DFT1 = vr[1] - vr[7];

                {
                    adr1_DFT1 = vr[2] + vr[12];
                    adr5_DFT1 = vr[6] + vr[8];
                    v17r_DFT1 = vr[1] + vr[7];
                    adr2_DFT1 = vr[3] + vr[11];
                    adr3_DFT1 = vr[4] + vr[10];
                    adr4_DFT1 = vr[5] + vr[9];
                    adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
                    adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
                    cv1rr_DFT1 = v17r_DFT1 + adr3_DFT1;
                    cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
                    early_tvrr_2_DFT1 = cv1rr_DFT1 - (CRTM_12_2 * cv1ri_DFT1);

                    // [[ DFT1 ]] Output point 1: X(0) / V(1)
                    vr[1] = cv1rr_DFT1 + cv1ri_DFT1;

                    // Note: Since out_r[0] can be an alias to in_r[0], we make
                    // a copy of the 0th input point to use for the calculation
                    // of the point vr[1]
                    vr[0] = in_r[0];
                    out_r[0] = vr[1] + in_r[0];
                    vr[1] = (vr[1] - (12 * vr[0])) * DGC_1;

                    cv2rr_DFT1 = v17r_DFT1 - adr3_DFT1;
                    cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;
                    early_tvrr_DFT1 = cv2rr_DFT1 + (CRTM_12_2 * cv2ri_DFT1);

                    // [[ DFT1 ]] Output point 7: X(6) / V(7)
                    vr[7] = (cv2rr_DFT1 - cv2ri_DFT1) * DGC_12;
                }

                v17r = vr[1] + vr[7];
                v71r = vr[1] - vr[7];

                adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
                adr1_DFT1 = adr1_DFT1 - adr5_DFT1;

                sbr1_DFT1 = vr[12] - vr[2];
                sbr2_DFT1 = vr[11] - vr[3];
                sbr3_DFT1 = vr[10] - vr[4];
                sbr4_DFT1 = vr[9] - vr[5];
                sbr5_DFT1 = vr[8] - vr[6];

                sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
                sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
                sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;
            }

            {
                vi[1] = in_i[in_stride];
                vi[2] = in_i[in_stride * 2];
                vi[3] = in_i[in_stride * 4];
                vi[4] = in_i[in_stride * 8];
                vi[5] = in_i[in_stride * 3];
                vi[6] = in_i[in_stride * 6];
                vi[7] = in_i[in_stride * 12];
                vi[8] = in_i[in_stride * 11];
                vi[9] = in_i[in_stride * 9];
                vi[10] = in_i[in_stride * 5];
                vi[11] = in_i[in_stride * 10];
                vi[12] = in_i[in_stride * 7];

                v71i_DFT1 = vi[1] - vi[7];

                {
                    adi1_DFT1 = vi[2] + vi[12];
                    adi5_DFT1 = vi[6] + vi[8];
                    v17i_DFT1 = vi[1] + vi[7];
                    adi2_DFT1 = vi[3] + vi[11];
                    adi3_DFT1 = vi[4] + vi[10];
                    adi4_DFT1 = vi[5] + vi[9];
                    adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
                    adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
                    cv1ir_DFT1 = v17i_DFT1 + adi3_DFT1;
                    cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
                    early_tvii_2_DFT1 = cv1ir_DFT1 - (CRTM_12_2 * cv1ii_DFT1);

                    // [[ DFT1 ]] Output point 1: X(0) / V(1)
                    vi[1] = cv1ii_DFT1 + cv1ir_DFT1;

                    // Note: Since out_i[0] can be an alias to in_i[0], we make
                    // a copy of the 0th input point to use for the calculation
                    // of the point vi[1]
                    vi[0] = in_i[0];
                    out_i[0] = vi[1] + in_i[0];
                    vi[1] = (vi[1] - (12 * vi[0])) * DGC_1;

                    cv2ir_DFT1 = v17i_DFT1 - adi3_DFT1;
                    cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;
                    early_tvii_DFT1 = cv2ir_DFT1 + (CRTM_12_2 * cv2ii_DFT1);

                    // [[ DFT1 ]] Output point 7: X(6) / V(7)
                    vi[7] = (cv2ir_DFT1 - cv2ii_DFT1) * DGC_12;
                }

                v17i = vi[1] + vi[7];
                v71i = vi[1] - vi[7];

                adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
                adi1_DFT1 = adi1_DFT1 - adi5_DFT1;

                sbi1_DFT1 = vi[12] - vi[2];
                sbi2_DFT1 = vi[11] - vi[3];
                sbi3_DFT1 = vi[10] - vi[4];
                sbi4_DFT1 = vi[9] - vi[5];
                sbi5_DFT1 = vi[8] - vi[6];

                sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
                sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
                sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            }

            {
                tv1_DFT1 = CRTM_12_2 * adr2_DFT1;
                tv4_DFT1 = CRTM_12_2 * sbr15_DFT1 + sbr3_DFT1;
                tv2_DFT1 = CRTM_12_2 * sbi15_DFT1 + sbi3_DFT1;
                tv6_DFT1 = CRTM_12_2 * adi4_DFT1;
            }

            {
                tv7_DFT1 = CRTM_12_1 * adr1_DFT1;
                tv5_DFT1 = CRTM_12_1 * (sbr2_DFT1 + sbr4_DFT1);
                tv3_DFT1 = CRTM_12_1 * (sbi2_DFT1 + sbi4_DFT1);
                tv8_DFT1 = CRTM_12_1 * adi1_DFT1;
            }

            v711r_DFT1 = v71r_DFT1 + tv1_DFT1;
            v711i_DFT1 = v71i_DFT1 + tv6_DFT1;

            {

                FLOAT tvrr, tvri, tvir, tvii;
                FLOAT tvrr2, tvri2, tvir2, tvii2;

                tvrr = v711r_DFT1 + tv7_DFT1;
                tvrr2 = v711r_DFT1 - tv7_DFT1;
                tvri = tv2_DFT1 + tv3_DFT1;
                tvri2 = tv2_DFT1 - tv3_DFT1;
                tvir = tv4_DFT1 + tv5_DFT1;
                tvir2 = tv4_DFT1 - tv5_DFT1;
                tvii = v711i_DFT1 + tv8_DFT1;
                tvii2 = v711i_DFT1 - tv8_DFT1;

                // [[ DFT1 ]] Output point 2: X(1) / V(2)
                vr[2] = tvrr - tvri;
                vi[2] = tvii + tvir;

                // [[ DFT1 ]] Output point 12: X(11) / V(12)
                vr[12] = tvrr + tvri;
                vi[12] = tvii - tvir;

                // [[ DFT1 ]] Output point 6: X(5) / V(6)
                vr[6] = tvrr2 - tvri2;
                vi[6] = tvii2 + tvir2;

                // [[ DFT1 ]] Output point 8: X(7) / V(8)
                vr[8] = tvrr2 + tvri2;
                vi[8] = tvii2 - tvir2;

                CMUL_INPLACE_FP64(vr[2], vi[2], DGC_2, -DGC_3);
                CMUL_INPLACE_FP64(vr[12], vi[12], -DGC_2, -DGC_3);

                adr1 = vr[2] + vr[12];
                sbr1 = vr[12] - vr[2];
                adi1 = vi[2] + vi[12];
                sbi1 = vi[12] - vi[2];

                CMUL_INPLACE_FP64(vr[8], vi[8], DGC_10, DGC_11);
                CMUL_INPLACE_FP64(vr[6], vi[6], -DGC_10, DGC_11);

                adi5 = vi[6] + vi[8];
                sbi5 = vi[8] - vi[6];
                adr5 = vr[6] + vr[8];
                sbr5 = vr[8] - vr[6];
            }

            FLOAT tvrr, tvri, tvir, tvii;

            adi51 = adi1 - adi5;
            adi15 = adi1 + adi5;
            sbi15 = sbi1 + sbi5;
            sbi51 = sbi1 - sbi5;
            adr51 = adr1 - adr5;
            adr15 = adr1 + adr5;
            sbr15 = sbr1 + sbr5;
            sbr51 = sbr1 - sbr5;

            tv7 = CRTM_12_1 * adr51;
            tv8 = CRTM_12_1 * adi51;

            {
                tvrr = v71r_DFT1 - adr2_DFT1;
                tvri = sbi15_DFT1 - sbi3_DFT1;
                tvir = sbr15_DFT1 - sbr3_DFT1;
                tvii = v71i_DFT1 - adi4_DFT1;

                // [[ DFT1 ]] Output point 4: X(3) / V(4)
                vr[4] = tvrr - tvri;
                vi[4] = tvii + tvir;

                // [[ DFT1 ]] Output point 10: X(9) / V(10)
                vr[10] = tvrr + tvri;
                vi[10] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[10], vi[10], -DGC_6, -DGC_7);
                CMUL_INPLACE_FP64(vr[4], vi[4], DGC_6, -DGC_7);
            }

            adr3 = vr[4] + vr[10];
            sbr3 = vr[10] - vr[4];
            adi3 = vi[4] + vi[10];
            sbi3 = vi[10] - vi[4];
            tv2 = CRTM_12_2 * sbi15 + sbi3;
            tv4 = CRTM_12_2 * sbr15 + sbr3;
            cv1rr = v17r + adr3;
            cv1ir = v17i + adi3;
            cv2rr = v17r - adr3;
            cv2ir = v17i - adi3;
            tvri_5o = sbi15 - sbi3;
            tvir_5o = sbr15 - sbr3;

            {
                // [[ DFT1 ]] Output point 3: X(2) / V(3)
                tvrr = early_tvrr_DFT1;
                tvii = early_tvii_DFT1;
                tvri = CRTM_12_1 * (sbi51_DFT1 + sbi24_DFT1);
                tvir = CRTM_12_1 * (sbr51_DFT1 + sbr24_DFT1);

                vr[3] = tvrr - tvri;
                vi[3] = tvii + tvir;

                // [[ DFT1 ]] Output point 11: X(10) / V(11)
                vr[11] = tvrr + tvri;
                vi[11] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[11], vi[11], DGC_4, -DGC_5);
                CMUL_INPLACE_FP64(vr[3], vi[3], DGC_4, DGC_5);
            }

            adr2 = vr[3] + vr[11];
            sbr2 = vr[11] - vr[3];
            adi2 = vi[3] + vi[11];
            sbi2 = vi[11] - vi[3];

            {
                tvrr = early_tvrr_2_DFT1;
                tvii = early_tvii_2_DFT1;
                tvri = CRTM_12_1 * (sbi51_DFT1 - sbi24_DFT1);
                tvir = CRTM_12_1 * (sbr51_DFT1 - sbr24_DFT1);

                // [[ DFT1 ]] Output point 5: X(4) / V(5)
                vr[5] = tvrr - tvri;
                vi[5] = tvii + tvir;

                // [[ DFT1 ]] Output point 9: X(8) / V(9)
                vr[9] = tvrr + tvri;
                vi[9] = tvii - tvir;

                CMUL_INPLACE_FP64(vr[9], vi[9], DGC_8, -DGC_9);
                CMUL_INPLACE_FP64(vr[5], vi[5], DGC_8, DGC_9);
            }

            adr4 = vr[5] + vr[9];
            sbr4 = vr[9] - vr[5];
            adi4 = vi[5] + vi[9];
            sbi4 = vi[9] - vi[5];
            adi42 = adi2 - adi4;
            adi24 = adi2 + adi4;
            sbi24 = sbi2 - sbi4;
            adr24 = adr2 + adr4;
            adr42 = adr2 - adr4;
            sbr24 = sbr2 - sbr4;
            tvrr_5o = v71r - adr42;
            tvii_5o = v71i - adi42;
            cv1ri = adr15 + adr24;
            cv1ii = adi15 + adi24;
            cv2ri = adr15 - adr24;
            cv2ii = adi15 - adi24;

            {
                v711r = v71r + (CRTM_12_2 * adr42);
                v711i = v71i + (CRTM_12_2 * adi42);
                tvrr_1o = cv2rr + (CRTM_12_2 * cv2ri);
                tvii_1o = cv2ir + (CRTM_12_2 * cv2ii);
                tvrr_2o = cv1rr - (CRTM_12_2 * cv1ri);
                tvii_2o = cv1ir - (CRTM_12_2 * cv1ii);
            }

            {
                tv3 = CRTM_12_1 * (sbi2 + sbi4);
                tv5 = CRTM_12_1 * (sbr2 + sbr4);
                tvri_2o = CRTM_12_1 * (sbi51 - sbi24);
                tvir_2o = CRTM_12_1 * (sbr51 - sbr24);
                tvri_1o = CRTM_12_1 * (sbi51 + sbi24);
                tvir_1o = CRTM_12_1 * (sbr51 + sbr24);
            }
        }

        {
            {
                // Output point 1, post permutation point 1
                out_r[out_stride] = cv1rr + cv1ri;
                out_i[out_stride] = cv1ii + cv1ir;

                // Output point 7: post permutation point 12
                out_r[out_stride * 12] = cv2rr - cv2ri;
                out_i[out_stride * 12] = cv2ir - cv2ii;
            }

            {
                // Output point 5, post permutation point 3
                out_r[out_stride * 3] = tvrr_2o - tvri_2o;
                out_i[out_stride * 3] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_r[out_stride * 9] = tvrr_2o + tvri_2o;
                out_i[out_stride * 9] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_r[out_stride * 4] = tvrr_1o - tvri_1o;
                out_i[out_stride * 4] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_r[out_stride * 10] = tvrr_1o + tvri_1o;
                out_i[out_stride * 10] = tvii_1o - tvir_1o;
            }

            {
                // Output point 4, post permutation point 8
                out_r[out_stride * 8] = tvrr_5o - tvri_5o;
                out_i[out_stride * 8] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_r[out_stride * 5] = tvrr_5o + tvri_5o;
                out_i[out_stride * 5] = tvii_5o - tvir_5o;
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
                out_r[out_stride * 2] = tvrr_3o - tvri_3o;
                out_i[out_stride * 2] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_r[out_stride * 7] = tvrr_3o + tvri_3o;
                out_i[out_stride * 7] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_r[out_stride * 6] = tvrr_4o - tvri_4o;
                out_i[out_stride * 6] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_r[out_stride * 11] = tvrr_4o + tvri_4o;
                out_i[out_stride * 11] = tvii_4o - tvir_4o;
            }
        }

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 268, 1112, 130, 0, 1093}, {0, 268, 1112, 130, 0, 1093}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
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

VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
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
    DOUBLE local_in[RADIX_13][2] = {0};

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
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
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
    FLOAT local_in[RADIX_13][2] = {0};

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
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

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
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}
#endif // USE_OPT_KERNEL_VARIANT
