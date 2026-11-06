// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft11c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-11 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-11 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 11

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 140, 160, 64, 0, 0},
                                                     {0, 140, 160, 64, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft11c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_c2r_fft11c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_11_1 =
        +0.84125353283118116029052039464203089547681594330064;
    const FFTZ_DOUBLE CRTM_11_2 =
        +0.54064081745559759544482548159299693174139803024473;
    const FFTZ_DOUBLE CRTM_11_3 =
        +0.41541501300188639668675795488636098054966524290126;
    const FFTZ_DOUBLE CRTM_11_4 =
        +0.90963199535451838458365117807108162835411650732265;
    const FFTZ_DOUBLE CRTM_11_5 =
        +0.14231483827328501490317354898047094957684096668515;
    const FFTZ_DOUBLE CRTM_11_6 =
        +0.98982144188093275042610808187068914262031166769031;
    const FFTZ_DOUBLE CRTM_11_7 =
        +0.65486073394528511198338203198719613618953603946564;
    const FFTZ_DOUBLE CRTM_11_8 =
        +0.75574957435425824224552448923467521721665586591805;
    const FFTZ_DOUBLE CRTM_11_9 =
        +0.95949297361449738989036805706632769906245484800000;
    const FFTZ_DOUBLE CRTM_11_10 =
        +0.28173255684142978898192655345478532989004751779983;

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
    FFTZ_INTP v_in_h2_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_sym_stride;

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

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v211r, v310r, v49r,
            v58r, v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i,
            v58i, v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii,
            tvrr1, tvri1, tvir1, tvii1, tvrr2, tvri2, tvir2, tvii2, tvrr3,
            tvri3, tvir3, tvii3, tvrr4, tvri4, tvir4, tvii4;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        v1r = v2r_t;
        v1i = v2i_t;

        // Input point 11: x(10)
        FFTZ_DOUBLE v11r_t = in_h2_r[in_strides[10]];
        FFTZ_DOUBLE v11i_t = in_h2_i[in_strides[10]];
        v11i_t = -v11i_t;
        v2r = v11r_t;
        v2i = v11i_t;

        v211r = v1r + v2r;
        v112r = v2r - v1r;
        v211i = v1i + v2i;
        v112i = v2i - v1i;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 10: x(9)
        FFTZ_DOUBLE v10r_t = in_h2_r[in_strides[9]];
        FFTZ_DOUBLE v10i_t = in_h2_i[in_strides[9]];
        v10i_t = -v10i_t;
        v4r = v10r_t;
        v4i = v10i_t;

        v310r = v3r + v4r;
        v103r = v4r - v3r;
        v310i = v3i + v4i;
        v103i = v4i - v3i;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        v1r = v4r_t;
        v1i = v4i_t;

        // Input point 9: x(8)
        FFTZ_DOUBLE v9r_t = in_h2_r[in_strides[8]];
        FFTZ_DOUBLE v9i_t = in_h2_i[in_strides[8]];
        v9i_t = -v9i_t;
        v2r = v9r_t;
        v2i = v9i_t;

        v49r = v1r + v2r;
        v94r = v2r - v1r;
        v49i = v1i + v2i;
        v94i = v2i - v1i;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h1_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h1_i[in_strides[4]];
        v3r = v5r_t;
        v3i = v5i_t;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        v8i_t = -v8i_t;
        v4r = v8r_t;
        v4i = v8i_t;

        v58r = v3r + v4r;
        v85r = v4r - v3r;
        v58i = v3i + v4i;
        v85i = v4i - v3i;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h1_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h1_i[in_strides[5]];
        v1r = v6r_t;
        v1i = v6i_t;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        v7i_t = -v7i_t;
        v2r = v7r_t;
        v2i = v7i_t;

        v67r = v1r + v2r;
        v76r = v2r - v1r;
        v67i = v1i + v2i;
        v76i = v2i - v1i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Output point 1: X(0)
        *out_h1_r = v1r + v211r + v310r + v49r + v58r + v67r;

        tvrr = v1r + (CRTM_11_1 * v211r) + (CRTM_11_3 * v310r) -
               ((CRTM_11_5 * v49r) + (CRTM_11_7 * v58r) + (CRTM_11_9 * v67r));
        tvir = (CRTM_11_2 * v112r) + (CRTM_11_4 * v103r) + (CRTM_11_6 * v94r) +
               (CRTM_11_8 * v85r) + (CRTM_11_10 * v76r);
        tvrr1 = v1r + (CRTM_11_1 * v67r) + (CRTM_11_3 * v211r) -
                ((CRTM_11_7 * v310r) + (CRTM_11_9 * v49r) + (CRTM_11_5 * v58r));
        tvir1 = (CRTM_11_4 * v112r) + (CRTM_11_8 * v103r) -
                ((CRTM_11_10 * v94r) + (CRTM_11_6 * v85r) + (CRTM_11_2 * v76r));
        tvrr2 =
            v1r + (CRTM_11_1 * v58r) + (CRTM_11_3 * v49r) -
            ((CRTM_11_5 * v211r) + (CRTM_11_9 * v310r) + (CRTM_11_7 * v67r));
        tvir2 = (CRTM_11_2 * v85r) + (CRTM_11_6 * v112r) + (CRTM_11_8 * v76r) -
                ((CRTM_11_4 * v94r) + (CRTM_11_10 * v103r));

        tvrr3 =
            v1r + (CRTM_11_1 * v49r) + (CRTM_11_3 * v67r) -
            ((CRTM_11_5 * v310r) + (CRTM_11_7 * v211r) + (CRTM_11_9 * v58r));
        tvir3 = (CRTM_11_2 * v94r) + (CRTM_11_8 * v112r) + (CRTM_11_10 * v85r) -
                ((CRTM_11_4 * v76r) + (CRTM_11_6 * v103r));
        tvrr4 = v1r + (CRTM_11_1 * v310r) + (CRTM_11_3 * v58r) -
                ((CRTM_11_5 * v67r) + (CRTM_11_7 * v49r) + (CRTM_11_9 * v211r));
        tvir4 = (CRTM_11_6 * v76r) + (CRTM_11_8 * v94r) + (CRTM_11_10 * v112r) -
                ((CRTM_11_2 * v103r) + (CRTM_11_4 * v85r));

        // Output point 1: X(0)
        *out_h1_i = v1i + v211i + v310i + v49i + v58i + v67i;

        tvii = v1i + (CRTM_11_1 * v211i) + (CRTM_11_3 * v310i) -
               ((CRTM_11_5 * v49i) + (CRTM_11_7 * v58i) + (CRTM_11_9 * v67i));
        tvri = (CRTM_11_2 * v112i) + (CRTM_11_4 * v103i) + (CRTM_11_6 * v94i) +
               (CRTM_11_8 * v85i) + (CRTM_11_10 * v76i);
        tvii1 = v1i + (CRTM_11_1 * v67i) + (CRTM_11_3 * v211i) -
                ((CRTM_11_7 * v310i) + (CRTM_11_9 * v49i) + (CRTM_11_5 * v58i));
        tvri1 = (CRTM_11_4 * v112i) + (CRTM_11_8 * v103i) -
                ((CRTM_11_10 * v94i) + (CRTM_11_6 * v85i) + (CRTM_11_2 * v76i));
        tvii2 =
            v1i + (CRTM_11_1 * v58i) + (CRTM_11_3 * v49i) -
            ((CRTM_11_5 * v211i) + (CRTM_11_7 * v67i) + (CRTM_11_9 * v310i));
        tvri2 = (CRTM_11_2 * v85i) + (CRTM_11_6 * v112i) + (CRTM_11_8 * v76i) -
                ((CRTM_11_10 * v103i) + (CRTM_11_4 * v94i));

        tvii3 =
            v1i + (CRTM_11_1 * v49i) + (CRTM_11_3 * v67i) -
            ((CRTM_11_5 * v310i) + (CRTM_11_7 * v211i) + (CRTM_11_9 * v58i));
        tvri3 = (CRTM_11_2 * v94i) + (CRTM_11_8 * v112i) + (CRTM_11_10 * v85i) -
                ((CRTM_11_4 * v76i) + (CRTM_11_6 * v103i));
        tvii4 = v1i + (CRTM_11_1 * v310i) + (CRTM_11_3 * v58i) -
                ((CRTM_11_5 * v67i) + (CRTM_11_7 * v49i) + (CRTM_11_9 * v211i));
        tvri4 = (CRTM_11_6 * v76i) + (CRTM_11_8 * v94i) + (CRTM_11_10 * v112i) -
                ((CRTM_11_2 * v103i) + (CRTM_11_4 * v85i));

        {
            FFTZ_DOUBLE _twr = tw_ptr[0];
            FFTZ_DOUBLE _twi = tw_ptr[1];
            FFTZ_DOUBLE _or_1 = tvrr + tvri;
            FFTZ_DOUBLE _oi_1 = tvii - tvir;
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }

        // Output point 11: X(10)
        FFTZ_DOUBLE _or_10 = tvrr - tvri;
        {
            FFTZ_DOUBLE _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[9 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tvii + tvir;
            out_h2_r[out_strides[10]] = _or_10 * _twr - _oi * _twi;
            out_h2_i[out_strides[10]] = _or_10 * _twi + _oi * _twr;
        }

        {
            FFTZ_DOUBLE _twr = tw_ptr[DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_2 = tvrr1 + tvri1;
            FFTZ_DOUBLE _oi_2 = tvii1 - tvir1;
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }

        // Output point 10: X(9)
        FFTZ_DOUBLE _or_9 = tvrr1 - tvri1;
        {
            FFTZ_DOUBLE _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[8 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tvii1 + tvir1;
            out_h2_r[out_strides[9]] = _or_9 * _twr - _oi * _twi;
            out_h2_i[out_strides[9]] = _or_9 * _twi + _oi * _twr;
        }

        {
            FFTZ_DOUBLE _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_3 = tvrr2 + tvri2;
            FFTZ_DOUBLE _oi_3 = tvii2 - tvir2;
            out_h1_r[out_strides[3]] = _or_3 * _twr - _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _or_3 * _twi + _oi_3 * _twr;
        }

        // Output point 9: X(8)
        FFTZ_DOUBLE _or_8 = tvrr2 - tvri2;
        {
            FFTZ_DOUBLE _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[7 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tvii2 + tvir2;
            out_h2_r[out_strides[8]] = _or_8 * _twr - _oi * _twi;
            out_h2_i[out_strides[8]] = _or_8 * _twi + _oi * _twr;
        }

        {
            FFTZ_DOUBLE _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_4 = tvrr3 + tvri3;
            FFTZ_DOUBLE _oi_4 = tvii3 - tvir3;
            out_h1_r[out_strides[4]] = _or_4 * _twr - _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _or_4 * _twi + _oi_4 * _twr;
        }

        // Output point 8: X(7)
        FFTZ_DOUBLE _or_7 = tvrr3 - tvri3;
        {
            FFTZ_DOUBLE _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[6 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tvii3 + tvir3;
            out_h2_r[out_strides[7]] = _or_7 * _twr - _oi * _twi;
            out_h2_i[out_strides[7]] = _or_7 * _twi + _oi * _twr;
        }

        {
            FFTZ_DOUBLE _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_5 = tvrr4 + tvri4;
            FFTZ_DOUBLE _oi_5 = tvii4 - tvir4;
            out_h1_r[out_strides[5]] = _or_5 * _twr - _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _or_5 * _twi + _oi_5 * _twr;
        }

        // Output point 7: X(6)
        FFTZ_DOUBLE _or_6 = tvrr4 - tvri4;
        {
            FFTZ_DOUBLE _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[5 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tvii4 + tvir4;
            out_h2_r[out_strides[6]] = _or_6 * _twr - _oi * _twi;
            out_h2_i[out_strides[6]] = _or_6 * _twi + _oi * _twr;
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

static FFTZ_VOID twid_c2r_fft11c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_11_1 =
        +0.84125353283118116029052039464203089547681594330064f;
    const FFTZ_FLOAT CRTM_11_2 =
        +0.54064081745559759544482548159299693174139803024473f;
    const FFTZ_FLOAT CRTM_11_3 =
        +0.41541501300188639668675795488636098054966524290126f;
    const FFTZ_FLOAT CRTM_11_4 =
        +0.90963199535451838458365117807108162835411650732265f;
    const FFTZ_FLOAT CRTM_11_5 =
        +0.14231483827328501490317354898047094957684096668515f;
    const FFTZ_FLOAT CRTM_11_6 =
        +0.98982144188093275042610808187068914262031166769031f;
    const FFTZ_FLOAT CRTM_11_7 =
        +0.65486073394528511198338203198719613618953603946564f;
    const FFTZ_FLOAT CRTM_11_8 =
        +0.75574957435425824224552448923467521721665586591805f;
    const FFTZ_FLOAT CRTM_11_9 =
        +0.95949297361449738989036805706632769906245484800000f;
    const FFTZ_FLOAT CRTM_11_10 =
        +0.28173255684142978898192655345478532989004751779983f;

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
    FFTZ_INTP v_in_h2_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_sym_stride;

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

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v211r, v310r, v49r,
            v58r, v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i,
            v58i, v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii,
            tvrr1, tvri1, tvir1, tvii1, tvrr2, tvri2, tvir2, tvii2, tvrr3,
            tvri3, tvir3, tvii3, tvrr4, tvri4, tvir4, tvii4;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        v1r = v2r_t;
        v1i = v2i_t;

        // Input point 11: x(10)
        FFTZ_FLOAT v11r_t = in_h2_r[in_strides[10]];
        FFTZ_FLOAT v11i_t = in_h2_i[in_strides[10]];
        v11i_t = -v11i_t;
        v2r = v11r_t;
        v2i = v11i_t;

        v211r = v1r + v2r;
        v112r = v2r - v1r;
        v211i = v1i + v2i;
        v112i = v2i - v1i;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 10: x(9)
        FFTZ_FLOAT v10r_t = in_h2_r[in_strides[9]];
        FFTZ_FLOAT v10i_t = in_h2_i[in_strides[9]];
        v10i_t = -v10i_t;
        v4r = v10r_t;
        v4i = v10i_t;

        v310r = v3r + v4r;
        v103r = v4r - v3r;
        v310i = v3i + v4i;
        v103i = v4i - v3i;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        v1r = v4r_t;
        v1i = v4i_t;

        // Input point 9: x(8)
        FFTZ_FLOAT v9r_t = in_h2_r[in_strides[8]];
        FFTZ_FLOAT v9i_t = in_h2_i[in_strides[8]];
        v9i_t = -v9i_t;
        v2r = v9r_t;
        v2i = v9i_t;

        v49r = v1r + v2r;
        v94r = v2r - v1r;
        v49i = v1i + v2i;
        v94i = v2i - v1i;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h1_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h1_i[in_strides[4]];
        v3r = v5r_t;
        v3i = v5i_t;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        v8i_t = -v8i_t;
        v4r = v8r_t;
        v4i = v8i_t;

        v58r = v3r + v4r;
        v85r = v4r - v3r;
        v58i = v3i + v4i;
        v85i = v4i - v3i;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h1_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h1_i[in_strides[5]];
        v1r = v6r_t;
        v1i = v6i_t;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        v7i_t = -v7i_t;
        v2r = v7r_t;
        v2i = v7i_t;

        v67r = v1r + v2r;
        v76r = v2r - v1r;
        v67i = v1i + v2i;
        v76i = v2i - v1i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Output point 1: X(0)
        *out_h1_r = v1r + v211r + v310r + v49r + v58r + v67r;

        tvrr = v1r + (CRTM_11_1 * v211r) + (CRTM_11_3 * v310r) -
               ((CRTM_11_5 * v49r) + (CRTM_11_7 * v58r) + (CRTM_11_9 * v67r));
        tvir = (CRTM_11_2 * v112r) + (CRTM_11_4 * v103r) + (CRTM_11_6 * v94r) +
               (CRTM_11_8 * v85r) + (CRTM_11_10 * v76r);
        tvrr1 = v1r + (CRTM_11_1 * v67r) + (CRTM_11_3 * v211r) -
                ((CRTM_11_7 * v310r) + (CRTM_11_9 * v49r) + (CRTM_11_5 * v58r));
        tvir1 = (CRTM_11_4 * v112r) + (CRTM_11_8 * v103r) -
                ((CRTM_11_10 * v94r) + (CRTM_11_6 * v85r) + (CRTM_11_2 * v76r));
        tvrr2 =
            v1r + (CRTM_11_1 * v58r) + (CRTM_11_3 * v49r) -
            ((CRTM_11_5 * v211r) + (CRTM_11_9 * v310r) + (CRTM_11_7 * v67r));
        tvir2 = (CRTM_11_2 * v85r) + (CRTM_11_6 * v112r) + (CRTM_11_8 * v76r) -
                ((CRTM_11_4 * v94r) + (CRTM_11_10 * v103r));

        tvrr3 =
            v1r + (CRTM_11_1 * v49r) + (CRTM_11_3 * v67r) -
            ((CRTM_11_5 * v310r) + (CRTM_11_7 * v211r) + (CRTM_11_9 * v58r));
        tvir3 = (CRTM_11_2 * v94r) + (CRTM_11_8 * v112r) + (CRTM_11_10 * v85r) -
                ((CRTM_11_4 * v76r) + (CRTM_11_6 * v103r));
        tvrr4 = v1r + (CRTM_11_1 * v310r) + (CRTM_11_3 * v58r) -
                ((CRTM_11_5 * v67r) + (CRTM_11_7 * v49r) + (CRTM_11_9 * v211r));
        tvir4 = (CRTM_11_6 * v76r) + (CRTM_11_8 * v94r) + (CRTM_11_10 * v112r) -
                ((CRTM_11_2 * v103r) + (CRTM_11_4 * v85r));

        // Output point 1: X(0)
        *out_h1_i = v1i + v211i + v310i + v49i + v58i + v67i;

        tvii = v1i + (CRTM_11_1 * v211i) + (CRTM_11_3 * v310i) -
               ((CRTM_11_5 * v49i) + (CRTM_11_7 * v58i) + (CRTM_11_9 * v67i));
        tvri = (CRTM_11_2 * v112i) + (CRTM_11_4 * v103i) + (CRTM_11_6 * v94i) +
               (CRTM_11_8 * v85i) + (CRTM_11_10 * v76i);
        tvii1 = v1i + (CRTM_11_1 * v67i) + (CRTM_11_3 * v211i) -
                ((CRTM_11_7 * v310i) + (CRTM_11_9 * v49i) + (CRTM_11_5 * v58i));
        tvri1 = (CRTM_11_4 * v112i) + (CRTM_11_8 * v103i) -
                ((CRTM_11_10 * v94i) + (CRTM_11_6 * v85i) + (CRTM_11_2 * v76i));
        tvii2 =
            v1i + (CRTM_11_1 * v58i) + (CRTM_11_3 * v49i) -
            ((CRTM_11_5 * v211i) + (CRTM_11_7 * v67i) + (CRTM_11_9 * v310i));
        tvri2 = (CRTM_11_2 * v85i) + (CRTM_11_6 * v112i) + (CRTM_11_8 * v76i) -
                ((CRTM_11_10 * v103i) + (CRTM_11_4 * v94i));

        tvii3 =
            v1i + (CRTM_11_1 * v49i) + (CRTM_11_3 * v67i) -
            ((CRTM_11_5 * v310i) + (CRTM_11_7 * v211i) + (CRTM_11_9 * v58i));
        tvri3 = (CRTM_11_2 * v94i) + (CRTM_11_8 * v112i) + (CRTM_11_10 * v85i) -
                ((CRTM_11_4 * v76i) + (CRTM_11_6 * v103i));
        tvii4 = v1i + (CRTM_11_1 * v310i) + (CRTM_11_3 * v58i) -
                ((CRTM_11_5 * v67i) + (CRTM_11_7 * v49i) + (CRTM_11_9 * v211i));
        tvri4 = (CRTM_11_6 * v76i) + (CRTM_11_8 * v94i) + (CRTM_11_10 * v112i) -
                ((CRTM_11_2 * v103i) + (CRTM_11_4 * v85i));

        {
            FFTZ_FLOAT _twr = tw_ptr[0];
            FFTZ_FLOAT _twi = tw_ptr[1];
            FFTZ_FLOAT _or_1 = tvrr + tvri;
            FFTZ_FLOAT _oi_1 = tvii - tvir;
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }

        // Output point 11: X(10)
        FFTZ_FLOAT _or_10 = tvrr - tvri;
        {
            FFTZ_FLOAT _twr = tw_ptr[9 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[9 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tvii + tvir;
            out_h2_r[out_strides[10]] = _or_10 * _twr - _oi * _twi;
            out_h2_i[out_strides[10]] = _or_10 * _twi + _oi * _twr;
        }

        {
            FFTZ_FLOAT _twr = tw_ptr[DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_FLOAT _or_2 = tvrr1 + tvri1;
            FFTZ_FLOAT _oi_2 = tvii1 - tvir1;
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }

        // Output point 10: X(9)
        FFTZ_FLOAT _or_9 = tvrr1 - tvri1;
        {
            FFTZ_FLOAT _twr = tw_ptr[8 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[8 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tvii1 + tvir1;
            out_h2_r[out_strides[9]] = _or_9 * _twr - _oi * _twi;
            out_h2_i[out_strides[9]] = _or_9 * _twi + _oi * _twr;
        }

        {
            FFTZ_FLOAT _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_FLOAT _or_3 = tvrr2 + tvri2;
            FFTZ_FLOAT _oi_3 = tvii2 - tvir2;
            out_h1_r[out_strides[3]] = _or_3 * _twr - _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _or_3 * _twi + _oi_3 * _twr;
        }

        // Output point 9: X(8)
        FFTZ_FLOAT _or_8 = tvrr2 - tvri2;
        {
            FFTZ_FLOAT _twr = tw_ptr[7 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[7 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tvii2 + tvir2;
            out_h2_r[out_strides[8]] = _or_8 * _twr - _oi * _twi;
            out_h2_i[out_strides[8]] = _or_8 * _twi + _oi * _twr;
        }

        {
            FFTZ_FLOAT _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_FLOAT _or_4 = tvrr3 + tvri3;
            FFTZ_FLOAT _oi_4 = tvii3 - tvir3;
            out_h1_r[out_strides[4]] = _or_4 * _twr - _oi_4 * _twi;
            out_h1_i[out_strides[4]] = _or_4 * _twi + _oi_4 * _twr;
        }

        // Output point 8: X(7)
        FFTZ_FLOAT _or_7 = tvrr3 - tvri3;
        {
            FFTZ_FLOAT _twr = tw_ptr[6 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[6 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tvii3 + tvir3;
            out_h2_r[out_strides[7]] = _or_7 * _twr - _oi * _twi;
            out_h2_i[out_strides[7]] = _or_7 * _twi + _oi * _twr;
        }

        {
            FFTZ_FLOAT _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_FLOAT _or_5 = tvrr4 + tvri4;
            FFTZ_FLOAT _oi_5 = tvii4 - tvir4;
            out_h1_r[out_strides[5]] = _or_5 * _twr - _oi_5 * _twi;
            out_h1_i[out_strides[5]] = _or_5 * _twi + _oi_5 * _twr;
        }

        // Output point 7: X(6)
        FFTZ_FLOAT _or_6 = tvrr4 - tvri4;
        {
            FFTZ_FLOAT _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[5 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tvii4 + tvir4;
            out_h2_r[out_strides[6]] = _or_6 * _twr - _oi * _twi;
            out_h2_i[out_strides[6]] = _or_6 * _twi + _oi * _twr;
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

kfft_ register_kernel_twid_c2r_fft11c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft11c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft11c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
