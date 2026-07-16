// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fwd_fft13c.c
 *
 *  @brief Forward-only twiddle Radix-13 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-13 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 116, 200, 76, 0, 0},
                                                     {0, 116, 200, 76, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fwd_fft13c(UINT8 precision, UINT8 direction)
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

static VOID twid_fwd_fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    DOUBLE twr, twi;

    in_h1_r = (DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE vr[13], vi[13];

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

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        vr[0] = *in_h1_r;
        vi[0] = *in_h1_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_h1_r[in_strides[1]];
        DOUBLE v2i_t = in_h1_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        vr[1] = v2r_t * twr - v2i_t * twi;
        vi[1] = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_h1_r[in_strides[2]];
        DOUBLE v3i_t = in_h1_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        vr[2] = v3r_t * twr - v3i_t * twi;
        vi[2] = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_h1_r[in_strides[3]];
        DOUBLE v4i_t = in_h1_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        vr[5] = v4r_t * twr - v4i_t * twi;
        vi[5] = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_h1_r[in_strides[4]];
        DOUBLE v5i_t = in_h1_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        vr[3] = v5r_t * twr - v5i_t * twi;
        vi[3] = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_h1_r[in_strides[5]];
        DOUBLE v6i_t = in_h1_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        vr[10] = v6r_t * twr - v6i_t * twi;
        vi[10] = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_h1_r[in_strides[6]];
        DOUBLE v7i_t = in_h1_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        vr[6] = v7r_t * twr - v7i_t * twi;
        vi[6] = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_h2_r[in_strides[7]];
        DOUBLE v8i_t = in_h2_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        vr[12] = v8r_t * twr - v8i_t * twi;
        vi[12] = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        DOUBLE v9r_t = in_h2_r[in_strides[8]];
        DOUBLE v9i_t = in_h2_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        vr[4] = v9r_t * twr - v9i_t * twi;
        vi[4] = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        DOUBLE v10r_t = in_h2_r[in_strides[9]];
        DOUBLE v10i_t = in_h2_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        vr[9] = v10r_t * twr - v10i_t * twi;
        vi[9] = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        DOUBLE v11r_t = in_h2_r[in_strides[10]];
        DOUBLE v11i_t = in_h2_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        vr[11] = v11r_t * twr - v11i_t * twi;
        vi[11] = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        DOUBLE v12r_t = in_h2_r[in_strides[11]];
        DOUBLE v12i_t = in_h2_i[in_strides[11]];
        UINTP twid_addr12 = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr12];
        twi = tw[1 + twid_addr12];
        vr[8] = v12r_t * twr - v12i_t * twi;
        vi[8] = v12r_t * twi + v12i_t * twr;

        // Input point 13: x(12)
        DOUBLE v13r_t = in_h2_r[in_strides[12]];
        DOUBLE v13i_t = in_h2_i[in_strides[12]];
        UINTP twid_addr13 = DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr13];
        twi = tw[1 + twid_addr13];
        vr[7] = v13r_t * twr - v13i_t * twi;
        vi[7] = v13r_t * twi + v13i_t * twr;

        {
            adr1_DFT1 = vr[2] + vr[12];
            sbr1_DFT1 = vr[12] - vr[2];

            adr5_DFT1 = vr[6] + vr[8];
            sbr5_DFT1 = vr[8] - vr[6];

            adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
            adr1_DFT1 = adr1_DFT1 - adr5_DFT1;
            tv7_DFT1 = CRTM_13_6 * adr1_DFT1;
            sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
            sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;

            adr2_DFT1 = vr[3] + vr[11];
            sbr2_DFT1 = vr[11] - vr[3];

            adr4_DFT1 = vr[5] + vr[9];
            sbr4_DFT1 = vr[9] - vr[5];

            adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
            adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
            sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
            tv5_DFT1 = CRTM_13_6 * (sbr2_DFT1 + sbr4_DFT1);
            tv1_DFT1 = CRTM_13_7 * adr2_DFT1;

            cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
            cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;

            adr3_DFT1 = vr[4] + vr[10];
            sbr3_DFT1 = vr[10] - vr[4];
            tv4_DFT1 = (CRTM_13_7 * sbr15_DFT1) + sbr3_DFT1;

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
            out_h1_r[0] = vr[1] + in_h1_r[0];
            vr[1] = vr[1] * DGC_1 + vr[0];
            v17r = vr[1] + vr[7];
            v71r = vr[1] - vr[7];
        }

        {
            adi1_DFT1 = vi[2] + vi[12];
            sbi1_DFT1 = vi[12] - vi[2];

            adi5_DFT1 = vi[6] + vi[8];
            sbi5_DFT1 = vi[8] - vi[6];

            adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
            adi1_DFT1 = adi1_DFT1 - adi5_DFT1;
            sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
            sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
            tv8_DFT1 = CRTM_13_6 * adi1_DFT1;

            adi2_DFT1 = vi[3] + vi[11];
            sbi2_DFT1 = vi[11] - vi[3];

            adi4_DFT1 = vi[5] + vi[9];
            sbi4_DFT1 = vi[9] - vi[5];

            adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
            adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
            sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            tv3_DFT1 = CRTM_13_6 * (sbi2_DFT1 + sbi4_DFT1);
            tv6_DFT1 = CRTM_13_7 * adi4_DFT1;

            cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
            cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;

            adi3_DFT1 = vi[4] + vi[10];
            sbi3_DFT1 = vi[10] - vi[4];
            tv2_DFT1 = (CRTM_13_7 * sbi15_DFT1) + sbi3_DFT1;

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
            out_h1_i[0] = vi[1] + in_h1_i[0];
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
                out_h1_r[out_strides[1]] = cv1rr + (CRTM_13_8 * cv1ri);
                out_h1_i[out_strides[1]] = cv1ir + (CRTM_13_8 * cv1ii);

                // Output point 7: post permutation point 12
                out_h2_r[out_strides[12]] = cv2rr - (CRTM_13_8 * cv2ri);
                out_h2_i[out_strides[12]] = cv2ir - (CRTM_13_8 * cv2ii);
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
                out_h1_r[out_strides[3]] = tvrr_2o - tvri_2o;
                out_h1_i[out_strides[3]] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_h2_r[out_strides[9]] = tvrr_2o + tvri_2o;
                out_h2_i[out_strides[9]] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_h1_r[out_strides[4]] = tvrr_1o - tvri_1o;
                out_h1_i[out_strides[4]] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_h2_r[out_strides[10]] = tvrr_1o + tvri_1o;
                out_h2_i[out_strides[10]] = tvii_1o - tvir_1o;
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
                out_h2_r[out_strides[8]] = tvrr_5o - tvri_5o;
                out_h2_i[out_strides[8]] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_h1_r[out_strides[5]] = tvrr_5o + tvri_5o;
                out_h1_i[out_strides[5]] = tvii_5o - tvir_5o;
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
                out_h1_r[out_strides[2]] = tvrr_3o - tvri_3o;
                out_h1_i[out_strides[2]] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_h2_r[out_strides[7]] = tvrr_3o + tvri_3o;
                out_h2_i[out_strides[7]] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_h1_r[out_strides[6]] = tvrr_4o - tvri_4o;
                out_h1_i[out_strides[6]] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_h2_r[out_strides[11]] = tvrr_4o + tvri_4o;
                out_h2_i[out_strides[11]] = tvii_4o - tvir_4o;
            }
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

static VOID twid_fwd_fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;

    const FLOAT CRTM_13_1 =
        +1.73205080756887719317660412343684583902359008789060f;
    const FLOAT CRTM_13_2 =
        +0.38739058546761714712838513245787843298017865366345f;
    const FLOAT CRTM_13_3 =
        +0.13298312460741867056777367279491668244720145803175f;
    const FLOAT CRTM_13_4 =
        +0.11385447905579067614456922127652058891458054833493f;
    const FLOAT CRTM_13_5 =
        +0.25176851643188325619942540855595308576560112726050f;
    const FLOAT CRTM_13_6 =
        +0.86602540378443864676372317075293618347140262700000f;
    const FLOAT CRTM_13_7 =
        +0.50000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_13_8 =
        +2.00000000000000000000000000000000000000000000000000f;

    // DGC - constants(C) of the diagonal(DG) matrix used in rader's algorithm
    const FLOAT DGC_1 = -0.08333333333333341693177688718872429763576971755296f;
    const FLOAT DGC_2 = +0.25624767158293663769405689230781736686616059890980f;
    const FLOAT DGC_3 = +0.15689139105158457352993666120753768234094124525375f;
    const FLOAT DGC_4 = +0.25826039031174479484752691274654956353060547073611f;
    const FLOAT DGC_5 = +0.26596624921483734113554734558983336489440291606351f;
    const FLOAT DGC_6 = +0.57514072947400312136838554745545338846100160800000f;
    const FLOAT DGC_7 = +0.17413860115213590500566079492926474261696467600000f;
    const FLOAT DGC_8 = +0.07590298603719379320004091411468632874055721777845f;
    const FLOAT DGC_9 = +0.50353703286376651239885081711190617153120225452101f;
    const FLOAT DGC_10 = +0.3002386359663325979287571057309917466781880617299f;
    const FLOAT DGC_11 = +0.0115991056057681999237624507209650002461566663358f;
    const FLOAT DGC_12 = +0.3004626062886657229721584869768537486323029264720f;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    FLOAT twr, twi;

    in_h1_r = (FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT vr[13], vi[13];

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

        // Process input points with twiddle factors
        // Input point 1: x(0) - no twiddle needed
        vr[0] = *in_h1_r;
        vi[0] = *in_h1_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_h1_r[in_strides[1]];
        FLOAT v2i_t = in_h1_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        vr[1] = v2r_t * twr - v2i_t * twi;
        vi[1] = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_h1_r[in_strides[2]];
        FLOAT v3i_t = in_h1_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        vr[2] = v3r_t * twr - v3i_t * twi;
        vi[2] = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_h1_r[in_strides[3]];
        FLOAT v4i_t = in_h1_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        vr[5] = v4r_t * twr - v4i_t * twi;
        vi[5] = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_h1_r[in_strides[4]];
        FLOAT v5i_t = in_h1_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        vr[3] = v5r_t * twr - v5i_t * twi;
        vi[3] = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FLOAT v6r_t = in_h1_r[in_strides[5]];
        FLOAT v6i_t = in_h1_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        vr[10] = v6r_t * twr - v6i_t * twi;
        vi[10] = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FLOAT v7r_t = in_h1_r[in_strides[6]];
        FLOAT v7i_t = in_h1_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        vr[6] = v7r_t * twr - v7i_t * twi;
        vi[6] = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FLOAT v8r_t = in_h2_r[in_strides[7]];
        FLOAT v8i_t = in_h2_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        vr[12] = v8r_t * twr - v8i_t * twi;
        vi[12] = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FLOAT v9r_t = in_h2_r[in_strides[8]];
        FLOAT v9i_t = in_h2_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        vr[4] = v9r_t * twr - v9i_t * twi;
        vi[4] = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FLOAT v10r_t = in_h2_r[in_strides[9]];
        FLOAT v10i_t = in_h2_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        vr[9] = v10r_t * twr - v10i_t * twi;
        vi[9] = v10r_t * twi + v10i_t * twr;

        // Input point 11: x(10)
        FLOAT v11r_t = in_h2_r[in_strides[10]];
        FLOAT v11i_t = in_h2_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        vr[11] = v11r_t * twr - v11i_t * twi;
        vi[11] = v11r_t * twi + v11i_t * twr;

        // Input point 12: x(11)
        FLOAT v12r_t = in_h2_r[in_strides[11]];
        FLOAT v12i_t = in_h2_i[in_strides[11]];
        UINTP twid_addr12 = DATA_STRIDE * (11 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr12];
        twi = tw[1 + twid_addr12];
        vr[8] = v12r_t * twr - v12i_t * twi;
        vi[8] = v12r_t * twi + v12i_t * twr;

        // Input point 13: x(12)
        FLOAT v13r_t = in_h2_r[in_strides[12]];
        FLOAT v13i_t = in_h2_i[in_strides[12]];
        UINTP twid_addr13 = DATA_STRIDE * (12 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr13];
        twi = tw[1 + twid_addr13];
        vr[7] = v13r_t * twr - v13i_t * twi;
        vi[7] = v13r_t * twi + v13i_t * twr;

        {
            adr1_DFT1 = vr[2] + vr[12];
            sbr1_DFT1 = vr[12] - vr[2];

            adr5_DFT1 = vr[6] + vr[8];
            sbr5_DFT1 = vr[8] - vr[6];

            adr15_DFT1 = adr1_DFT1 + adr5_DFT1;
            adr1_DFT1 = adr1_DFT1 - adr5_DFT1;
            tv7_DFT1 = CRTM_13_6 * adr1_DFT1;
            sbr15_DFT1 = sbr1_DFT1 + sbr5_DFT1;
            sbr51_DFT1 = sbr1_DFT1 - sbr5_DFT1;

            adr2_DFT1 = vr[3] + vr[11];
            sbr2_DFT1 = vr[11] - vr[3];

            adr4_DFT1 = vr[5] + vr[9];
            sbr4_DFT1 = vr[9] - vr[5];

            adr24_DFT1 = adr2_DFT1 + adr4_DFT1;
            adr2_DFT1 = adr2_DFT1 - adr4_DFT1;
            sbr24_DFT1 = sbr2_DFT1 - sbr4_DFT1;
            tv5_DFT1 = CRTM_13_6 * (sbr2_DFT1 + sbr4_DFT1);
            tv1_DFT1 = CRTM_13_7 * adr2_DFT1;

            cv1ri_DFT1 = adr15_DFT1 + adr24_DFT1;
            cv2ri_DFT1 = adr15_DFT1 - adr24_DFT1;

            adr3_DFT1 = vr[4] + vr[10];
            sbr3_DFT1 = vr[10] - vr[4];
            tv4_DFT1 = (CRTM_13_7 * sbr15_DFT1) + sbr3_DFT1;

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
            out_h1_r[0] = vr[1] + in_h1_r[0];
            vr[1] = vr[1] * DGC_1 + vr[0];
            v17r = vr[1] + vr[7];
            v71r = vr[1] - vr[7];
        }

        {
            adi1_DFT1 = vi[2] + vi[12];
            sbi1_DFT1 = vi[12] - vi[2];

            adi5_DFT1 = vi[6] + vi[8];
            sbi5_DFT1 = vi[8] - vi[6];

            adi15_DFT1 = adi1_DFT1 + adi5_DFT1;
            adi1_DFT1 = adi1_DFT1 - adi5_DFT1;
            sbi15_DFT1 = sbi1_DFT1 + sbi5_DFT1;
            sbi51_DFT1 = sbi1_DFT1 - sbi5_DFT1;
            tv8_DFT1 = CRTM_13_6 * adi1_DFT1;

            adi2_DFT1 = vi[3] + vi[11];
            sbi2_DFT1 = vi[11] - vi[3];

            adi4_DFT1 = vi[5] + vi[9];
            sbi4_DFT1 = vi[9] - vi[5];

            adi24_DFT1 = adi2_DFT1 + adi4_DFT1;
            adi4_DFT1 = adi2_DFT1 - adi4_DFT1;
            sbi24_DFT1 = sbi2_DFT1 - sbi4_DFT1;
            tv3_DFT1 = CRTM_13_6 * (sbi2_DFT1 + sbi4_DFT1);
            tv6_DFT1 = CRTM_13_7 * adi4_DFT1;

            cv1ii_DFT1 = adi15_DFT1 + adi24_DFT1;
            cv2ii_DFT1 = adi15_DFT1 - adi24_DFT1;

            adi3_DFT1 = vi[4] + vi[10];
            sbi3_DFT1 = vi[10] - vi[4];
            tv2_DFT1 = (CRTM_13_7 * sbi15_DFT1) + sbi3_DFT1;

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
            out_h1_i[0] = vi[1] + in_h1_i[0];
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
                out_h1_r[out_strides[1]] = cv1rr + (CRTM_13_8 * cv1ri);
                out_h1_i[out_strides[1]] = cv1ir + (CRTM_13_8 * cv1ii);

                // Output point 7: post permutation point 12
                out_h2_r[out_strides[12]] = cv2rr - (CRTM_13_8 * cv2ri);
                out_h2_i[out_strides[12]] = cv2ir - (CRTM_13_8 * cv2ii);
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
                out_h1_r[out_strides[3]] = tvrr_2o - tvri_2o;
                out_h1_i[out_strides[3]] = tvii_2o + tvir_2o;

                // Output point 9, post permutation point 9
                out_h2_r[out_strides[9]] = tvrr_2o + tvri_2o;
                out_h2_i[out_strides[9]] = tvii_2o - tvir_2o;
            }

            {
                // Output point 3, post permutation point 4
                out_h1_r[out_strides[4]] = tvrr_1o - tvri_1o;
                out_h1_i[out_strides[4]] = tvii_1o + tvir_1o;

                // Output point 11, post permutation point 10
                out_h2_r[out_strides[10]] = tvrr_1o + tvri_1o;
                out_h2_i[out_strides[10]] = tvii_1o - tvir_1o;
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
                out_h2_r[out_strides[8]] = tvrr_5o - tvri_5o;
                out_h2_i[out_strides[8]] = tvii_5o + tvir_5o;

                // Output point 10, post permutation point 5
                out_h1_r[out_strides[5]] = tvrr_5o + tvri_5o;
                out_h1_i[out_strides[5]] = tvii_5o - tvir_5o;
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
                out_h1_r[out_strides[2]] = tvrr_3o - tvri_3o;
                out_h1_i[out_strides[2]] = tvii_3o + tvir_3o;

                // Output point 12, post permutation point 7
                out_h2_r[out_strides[7]] = tvrr_3o + tvri_3o;
                out_h2_i[out_strides[7]] = tvii_3o - tvir_3o;

                // Output point 6, post permutation point 6
                out_h1_r[out_strides[6]] = tvrr_4o - tvri_4o;
                out_h1_i[out_strides[6]] = tvii_4o + tvir_4o;

                // Output point 8, post permutation point 11
                out_h2_r[out_strides[11]] = tvrr_4o + tvri_4o;
                out_h2_i[out_strides[11]] = tvii_4o - tvir_4o;
            }
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

kfft_ register_kernel_twid_fwd_fft13c(UINT8 precision,
                                      UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fwd_fft13c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fwd_fft13c_fp64;
    }
    else
    {
        return NULL;
    }
}
