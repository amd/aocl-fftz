// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft16c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-16 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-16 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 * @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"

#define RADIX 16

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 84, 174, 94, 0, 0},
                                                     {0, 84, 174, 94, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft16c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_c2r_fft16c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_16_1 =
        +0.92387953251128675612818318939678828682241662586364;
    const FFTZ_DOUBLE CRTM_16_2 =
        +0.70710678118654752440084436210484903928483593768847;
    const FFTZ_DOUBLE CRTM_16_3 =
        +0.38268343236508977172845998403039886676134456248563;

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
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, ad1, ad2, ad3, ad4,
            ad5, ad6, ad7, adi1, adi2, adi3, adi4, adi5, adi6, adi7, sbr1, sbr2,
            sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3, sb4, sb5, sb6, sb7,
            tvrr, tvri, tvir, tvii, tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3,
            tvi4, tvr5, tvr6, tvi5, tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10,
            tvr11, tvr12, tvi9, tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,
            ad17, sb17, adi17, sbr17, sb35, sbr35, ad35, adi35, cv1r, cv1i,
            cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r, cav2i,
            cav3r, cav3i, cav4r, cav4i;

        {
            {

                FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[1]];
                FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[1]];
                FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[15]];
                FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[15]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

                ad1 = v1r + v2r;
                sbr1 = v2r - v1r;
                adi1 = v1i + v2i;
                sb1 = v2i - v1i;
            }

            {

                FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[7]];
                FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[7]];
                FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[9]];
                FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[9]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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

                FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[3]];
                FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[3]];
                FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[13]];
                FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[13]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

                ad3 = v1r + v2r;
                sbr3 = v2r - v1r;
                adi3 = v1i + v2i;
                sb3 = v2i - v1i;
            }

            {

                FFTZ_DOUBLE v1r_temp = in_h1_r[in_strides[5]];
                FFTZ_DOUBLE v1i_temp = in_h1_i[in_strides[5]];
                FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[11]];
                FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[11]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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

                FFTZ_DOUBLE v3r_temp = in_h1_r[in_strides[2]];
                FFTZ_DOUBLE v3i_temp = in_h1_i[in_strides[2]];
                FFTZ_DOUBLE v4r_temp = in_h2_r[in_strides[14]];
                FFTZ_DOUBLE v4i_temp = in_h2_i[in_strides[14]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

                ad2 = v3r + v4r;
                sbr2 = v4r - v3r;
                adi2 = v3i + v4i;
                sb2 = v4i - v3i;
            }

            {

                FFTZ_DOUBLE v3r_temp = in_h1_r[in_strides[6]];
                FFTZ_DOUBLE v3i_temp = in_h1_i[in_strides[6]];
                FFTZ_DOUBLE v4r_temp = in_h2_r[in_strides[10]];
                FFTZ_DOUBLE v4i_temp = in_h2_i[in_strides[10]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

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

                FFTZ_DOUBLE v3r_temp = in_h1_r[in_strides[4]];
                FFTZ_DOUBLE v3i_temp = in_h1_i[in_strides[4]];
                FFTZ_DOUBLE v4r_temp = in_h2_r[in_strides[12]];
                FFTZ_DOUBLE v4i_temp = in_h2_i[in_strides[12]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

                ad4 = v3r + v4r;
                sbr4 = v4r - v3r;
                adi4 = v3i + v4i;
                sb4 = v4i - v3i;
            }

            {

                FFTZ_DOUBLE v1r_temp = *in_h1_r;
                FFTZ_DOUBLE v1i_temp = *in_h1_i;
                FFTZ_DOUBLE v2r_temp = in_h2_r[in_strides[8]];
                FFTZ_DOUBLE v2i_temp = in_h2_i[in_strides[8]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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
        *out_h1_r = tvri + tvrr;
        FFTZ_DOUBLE _or_8 = tvri - tvrr;

        adi49 = cv1i + adi4;
        tvi5 = cv1i - adi4;
        tvr4 = ctv2 + sb4;
        tvr12 = ctv2 - sb4;
        tvi2 = cv2i + ctv3;
        tvi9 = cv2i - ctv3;
        tvir = adi26 + adi49;
        *out_h1_i = tvir + tvii;
        FFTZ_DOUBLE _oi_8 = tvir - tvii;
        {
            FFTZ_DOUBLE _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[7 * DATA_STRIDE + 1];
            out_h2_r[out_strides[8]] = _or_8 * _twr - _oi_8 * _twi;
            out_h2_i[out_strides[8]] = _or_8 * _twi + _oi_8 * _twr;
        }

        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        FFTZ_DOUBLE _or_4 = tvrr + tvri;
        FFTZ_DOUBLE _or_12 = tvrr - tvri;

        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        FFTZ_DOUBLE _oi_4 = tvii - tvir;
        FFTZ_DOUBLE _oi_12 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[3 * DATA_STRIDE + 1];
            out_h1_r[out_strides[4]] = _or_4 * _twr - _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _or_4 * _twi + _oi_4 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[11 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[11 * DATA_STRIDE + 1];
            out_h2_r[out_strides[12]] = _or_12 * _twr - _oi_12 * _twi;
            out_h2_i[out_strides[12]] = _or_12 * _twi + _oi_12 * _twr;
        }

        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        FFTZ_DOUBLE _or_1 = tvrr + tvri;
        FFTZ_DOUBLE _or_15 = tvrr - tvri;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;
        FFTZ_DOUBLE _oi_1 = tvii - tvir;
        FFTZ_DOUBLE _oi_15 = tvii + tvir;

        {
            FFTZ_DOUBLE _twr = tw_ptr[0];
            FFTZ_DOUBLE _twi = tw_ptr[1];
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[14 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[14 * DATA_STRIDE + 1];
            out_h2_r[out_strides[15]] = _or_15 * _twr - _oi_15 * _twi;
            out_h2_i[out_strides[15]] = _or_15 * _twi + _oi_15 * _twr;
        }

        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        FFTZ_DOUBLE _or_7 = tvrr + tvri;
        FFTZ_DOUBLE _or_9 = tvrr - tvri;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        FFTZ_DOUBLE _oi_7 = tvii - tvir;
        FFTZ_DOUBLE _oi_9 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[6 * DATA_STRIDE + 1];
            out_h1_r[out_strides[7]] = _or_7 * _twr - _oi_7 * _twi;
            out_h1_i[out_strides[7]] = _or_7 * _twi + _oi_7 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[8 * DATA_STRIDE + 1];
            out_h2_r[out_strides[9]] = _or_9 * _twr - _oi_9 * _twi;
            out_h2_i[out_strides[9]] = _or_9 * _twi + _oi_9 * _twr;
        }

        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        FFTZ_DOUBLE _or_2 = tvrr + tvri;
        FFTZ_DOUBLE _or_14 = tvrr - tvri;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        FFTZ_DOUBLE _oi_2 = tvii - tvir;
        FFTZ_DOUBLE _oi_14 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[DATA_STRIDE + 1];
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[13 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[13 * DATA_STRIDE + 1];
            out_h2_r[out_strides[14]] = _or_14 * _twr - _oi_14 * _twi;
            out_h2_i[out_strides[14]] = _or_14 * _twi + _oi_14 * _twr;
        }

        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        FFTZ_DOUBLE _or_6 = tvrr + tvri;
        FFTZ_DOUBLE _or_10 = tvrr - tvri;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        FFTZ_DOUBLE _oi_6 = tvii - tvir;
        FFTZ_DOUBLE _oi_10 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[5 * DATA_STRIDE + 1];
            out_h1_r[out_strides[6]] = _or_6 * _twr - _oi_6 * _twi;
            out_h1_i[out_strides[6]] = _or_6 * _twi + _oi_6 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[9 * DATA_STRIDE + 1];
            out_h2_r[out_strides[10]] = _or_10 * _twr - _oi_10 * _twi;
            out_h2_i[out_strides[10]] = _or_10 * _twi + _oi_10 * _twr;
        }

        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        FFTZ_DOUBLE _or_3 = tvrr + tvri;
        FFTZ_DOUBLE _or_13 = tvrr - tvri;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        FFTZ_DOUBLE _oi_3 = tvii - tvir;
        FFTZ_DOUBLE _oi_13 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[2 * DATA_STRIDE + 1];
            out_h1_r[out_strides[3]] = _or_3 * _twr - _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _or_3 * _twi + _oi_3 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[12 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[12 * DATA_STRIDE + 1];
            out_h2_r[out_strides[13]] = _or_13 * _twr - _oi_13 * _twi;
            out_h2_i[out_strides[13]] = _or_13 * _twi + _oi_13 * _twr;
        }

        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        FFTZ_DOUBLE _or_5 = tvrr + tvri;
        FFTZ_DOUBLE _or_11 = tvrr - tvri;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        FFTZ_DOUBLE _oi_5 = tvii - tvir;
        FFTZ_DOUBLE _oi_11 = tvii + tvir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[4 * DATA_STRIDE + 1];
            out_h1_r[out_strides[5]] = _or_5 * _twr - _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _or_5 * _twi + _oi_5 * _twr;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[10 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[10 * DATA_STRIDE + 1];
            out_h2_r[out_strides[11]] = _or_11 * _twr - _oi_11 * _twi;
            out_h2_i[out_strides[11]] = _or_11 * _twi + _oi_11 * _twr;
        }

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

static FFTZ_VOID twid_c2r_fft16c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_16_1 =
        +0.92387953251128675612818318939678828682241662586364f;
    const FFTZ_FLOAT CRTM_16_2 =
        +0.70710678118654752440084436210484903928483593768847f;
    const FFTZ_FLOAT CRTM_16_3 =
        +0.38268343236508977172845998403039886676134456248563f;

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
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, ad1, ad2, ad3, ad4,
            ad5, ad6, ad7, adi1, adi2, adi3, adi4, adi5, adi6, adi7, sbr1, sbr2,
            sbr3, sbr4, sbr5, sbr6, sbr7, sb1, sb2, sb3, sb4, sb5, sb6, sb7,
            tvrr, tvri, tvir, tvii, tvr1, tvr2, tvr3, tvr4, tvi1, tvi2, tvi3,
            tvi4, tvr5, tvr6, tvi5, tvi6, tvr7, tvr8, tvi7, tvi8, tvr9, tvr10,
            tvr11, tvr12, tvi9, tvi10, tvi11, tvi12, ad26, ad49, adi26, adi49,
            ad17, sb17, adi17, sbr17, sb35, sbr35, ad35, adi35, cv1r, cv1i,
            cv2r, cv2i, ctv1, ctv2, ctv3, ctv4, cav1r, cav1i, cav2r, cav2i,
            cav3r, cav3i, cav4r, cav4i;

        {
            {

                FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[1]];
                FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[1]];
                FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[15]];
                FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[15]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

                ad1 = v1r + v2r;
                sbr1 = v2r - v1r;
                adi1 = v1i + v2i;
                sb1 = v2i - v1i;
            }

            {

                FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[7]];
                FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[7]];
                FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[9]];
                FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[9]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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

                FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[3]];
                FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[3]];
                FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[13]];
                FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[13]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

                ad3 = v1r + v2r;
                sbr3 = v2r - v1r;
                adi3 = v1i + v2i;
                sb3 = v2i - v1i;
            }

            {

                FFTZ_FLOAT v1r_temp = in_h1_r[in_strides[5]];
                FFTZ_FLOAT v1i_temp = in_h1_i[in_strides[5]];
                FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[11]];
                FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[11]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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

                FFTZ_FLOAT v3r_temp = in_h1_r[in_strides[2]];
                FFTZ_FLOAT v3i_temp = in_h1_i[in_strides[2]];
                FFTZ_FLOAT v4r_temp = in_h2_r[in_strides[14]];
                FFTZ_FLOAT v4i_temp = in_h2_i[in_strides[14]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

                ad2 = v3r + v4r;
                sbr2 = v4r - v3r;
                adi2 = v3i + v4i;
                sb2 = v4i - v3i;
            }

            {

                FFTZ_FLOAT v3r_temp = in_h1_r[in_strides[6]];
                FFTZ_FLOAT v3i_temp = in_h1_i[in_strides[6]];
                FFTZ_FLOAT v4r_temp = in_h2_r[in_strides[10]];
                FFTZ_FLOAT v4i_temp = in_h2_i[in_strides[10]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

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

                FFTZ_FLOAT v3r_temp = in_h1_r[in_strides[4]];
                FFTZ_FLOAT v3i_temp = in_h1_i[in_strides[4]];
                FFTZ_FLOAT v4r_temp = in_h2_r[in_strides[12]];
                FFTZ_FLOAT v4i_temp = in_h2_i[in_strides[12]];
                v4i_temp = -v4i_temp;

                v3r = v3r_temp;
                v3i = v3i_temp;
                v4r = v4r_temp;
                v4i = v4i_temp;

                ad4 = v3r + v4r;
                sbr4 = v4r - v3r;
                adi4 = v3i + v4i;
                sb4 = v4i - v3i;
            }

            {

                FFTZ_FLOAT v1r_temp = *in_h1_r;
                FFTZ_FLOAT v1i_temp = *in_h1_i;
                FFTZ_FLOAT v2r_temp = in_h2_r[in_strides[8]];
                FFTZ_FLOAT v2i_temp = in_h2_i[in_strides[8]];
                v2i_temp = -v2i_temp;

                v1r = v1r_temp;
                v1i = v1i_temp;
                v2r = v2r_temp;
                v2i = v2i_temp;

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
        *out_h1_r = tvri + tvrr;
        FFTZ_FLOAT _or_8 = tvri - tvrr;

        adi49 = cv1i + adi4;
        tvi5 = cv1i - adi4;
        tvr4 = ctv2 + sb4;
        tvr12 = ctv2 - sb4;
        tvi2 = cv2i + ctv3;
        tvi9 = cv2i - ctv3;
        tvir = adi26 + adi49;
        *out_h1_i = tvir + tvii;
        FFTZ_FLOAT _oi_8 = tvir - tvii;
        {
            FFTZ_FLOAT _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[7 * DATA_STRIDE + 1];
            out_h2_r[out_strides[8]] = _or_8 * _twr - _oi_8 * _twi;
            out_h2_i[out_strides[8]] = _or_8 * _twi + _oi_8 * _twr;
        }

        tvrr = ad49 - ad26;
        tvri = cav3i - cav4i;
        FFTZ_FLOAT _or_4 = tvrr + tvri;
        FFTZ_FLOAT _or_12 = tvrr - tvri;

        tvir = cav3r - cav4r;
        tvii = adi49 - adi26;
        FFTZ_FLOAT _oi_4 = tvii - tvir;
        FFTZ_FLOAT _oi_12 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[3 * DATA_STRIDE + 1];
            out_h1_r[out_strides[4]] = _or_4 * _twr - _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _or_4 * _twi + _oi_4 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[11 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[11 * DATA_STRIDE + 1];
            out_h2_r[out_strides[12]] = _or_12 * _twr - _oi_12 * _twi;
            out_h2_i[out_strides[12]] = _or_12 * _twi + _oi_12 * _twr;
        }

        tvrr = tvr2 + tvr1;
        tvri = tvr3 + tvr4;
        FFTZ_FLOAT _or_1 = tvrr + tvri;
        FFTZ_FLOAT _or_15 = tvrr - tvri;
        tvir = tvi3 + tvi4;
        tvii = tvi2 + tvi1;
        FFTZ_FLOAT _oi_1 = tvii - tvir;
        FFTZ_FLOAT _oi_15 = tvii + tvir;

        {
            FFTZ_FLOAT _twr = tw_ptr[0];
            FFTZ_FLOAT _twi = tw_ptr[1];
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[14 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[14 * DATA_STRIDE + 1];
            out_h2_r[out_strides[15]] = _or_15 * _twr - _oi_15 * _twi;
            out_h2_i[out_strides[15]] = _or_15 * _twi + _oi_15 * _twr;
        }

        tvrr = tvr2 - tvr1;
        tvri = tvr3 - tvr4;
        FFTZ_FLOAT _or_7 = tvrr + tvri;
        FFTZ_FLOAT _or_9 = tvrr - tvri;
        tvir = tvi3 - tvi4;
        tvii = tvi2 - tvi1;
        FFTZ_FLOAT _oi_7 = tvii - tvir;
        FFTZ_FLOAT _oi_9 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[6 * DATA_STRIDE + 1];
            out_h1_r[out_strides[7]] = _or_7 * _twr - _oi_7 * _twi;
            out_h1_i[out_strides[7]] = _or_7 * _twi + _oi_7 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[8 * DATA_STRIDE + 1];
            out_h2_r[out_strides[9]] = _or_9 * _twr - _oi_9 * _twi;
            out_h2_i[out_strides[9]] = _or_9 * _twi + _oi_9 * _twr;
        }

        tvrr = tvr5 + tvr6;
        tvri = tvr7 + tvr8;
        FFTZ_FLOAT _or_2 = tvrr + tvri;
        FFTZ_FLOAT _or_14 = tvrr - tvri;
        tvir = tvi7 + tvi8;
        tvii = tvi5 + tvi6;
        FFTZ_FLOAT _oi_2 = tvii - tvir;
        FFTZ_FLOAT _oi_14 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[DATA_STRIDE + 1];
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[13 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[13 * DATA_STRIDE + 1];
            out_h2_r[out_strides[14]] = _or_14 * _twr - _oi_14 * _twi;
            out_h2_i[out_strides[14]] = _or_14 * _twi + _oi_14 * _twr;
        }

        tvrr = tvr5 - tvr6;
        tvri = tvr7 - tvr8;
        FFTZ_FLOAT _or_6 = tvrr + tvri;
        FFTZ_FLOAT _or_10 = tvrr - tvri;
        tvir = tvi7 - tvi8;
        tvii = tvi5 - tvi6;
        FFTZ_FLOAT _oi_6 = tvii - tvir;
        FFTZ_FLOAT _oi_10 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[5 * DATA_STRIDE + 1];
            out_h1_r[out_strides[6]] = _or_6 * _twr - _oi_6 * _twi;
            out_h1_i[out_strides[6]] = _or_6 * _twi + _oi_6 * _twr;
        }

        {
            FFTZ_FLOAT _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[9 * DATA_STRIDE + 1];
            out_h2_r[out_strides[10]] = _or_10 * _twr - _oi_10 * _twi;
            out_h2_i[out_strides[10]] = _or_10 * _twi + _oi_10 * _twr;
        }

        tvrr = tvr9 + tvr10;
        tvri = tvr11 + tvr12;
        FFTZ_FLOAT _or_3 = tvrr + tvri;
        FFTZ_FLOAT _or_13 = tvrr - tvri;
        tvir = tvi11 + tvi12;
        tvii = tvi9 + tvi10;
        FFTZ_FLOAT _oi_3 = tvii - tvir;
        FFTZ_FLOAT _oi_13 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[2 * DATA_STRIDE + 1];
            out_h1_r[out_strides[3]] = _or_3 * _twr - _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _or_3 * _twi + _oi_3 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[12 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[12 * DATA_STRIDE + 1];
            out_h2_r[out_strides[13]] = _or_13 * _twr - _oi_13 * _twi;
            out_h2_i[out_strides[13]] = _or_13 * _twi + _oi_13 * _twr;
        }

        tvrr = tvr9 - tvr10;
        tvri = tvr11 - tvr12;
        FFTZ_FLOAT _or_5 = tvrr + tvri;
        FFTZ_FLOAT _or_11 = tvrr - tvri;
        tvir = tvi11 - tvi12;
        tvii = tvi9 - tvi10;
        FFTZ_FLOAT _oi_5 = tvii - tvir;
        FFTZ_FLOAT _oi_11 = tvii + tvir;
        {
            FFTZ_FLOAT _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[4 * DATA_STRIDE + 1];
            out_h1_r[out_strides[5]] = _or_5 * _twr - _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _or_5 * _twi + _oi_5 * _twr;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[10 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[10 * DATA_STRIDE + 1];
            out_h2_r[out_strides[11]] = _or_11 * _twr - _oi_11 * _twi;
            out_h2_i[out_strides[11]] = _or_11 * _twi + _oi_11 * _twr;
        }

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

kfft_ register_kernel_twid_c2r_fft16c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft16c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft16c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
