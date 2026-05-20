// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft7c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-7 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-7 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 7

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 72, 40, 0, 0},
                                                     {0, 60, 72, 40, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft7c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_c2r_fft7c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7_1 =
        +0.222520933956314404288902564496794759466355569;
    const FFTZ_DOUBLE CRTM_7_2 =
        +0.900968867902419126236102319507445051165919162;
    const FFTZ_DOUBLE CRTM_7_3 =
        +0.623489801858733530525004884004239810632274731;
    const FFTZ_DOUBLE CRTM_7_4 =
        +0.433883739117558120475768332848358754609990728;
    const FFTZ_DOUBLE CRTM_7_5 =
        +0.781831482468029808708444526674057750232334519;
    const FFTZ_DOUBLE CRTM_7_6 =
        +0.974927912181823607018131682993931217232785801;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i;
        FFTZ_DOUBLE tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv1ir, tv1ii,
            tv2ir, tv2ii, tv3ir, tv3ii;
        FFTZ_DOUBLE av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av1ir, av1ii,
            av2ir, av2ii, av3ir, av3ii;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h2_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h2_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h2_i[in_strides[5]];
        v6r_t = -v6r_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        v7r = v7r_t;
        v7i = v7i_t;

        av1rr = (v2r + v7r);
        av1ii = (v7r - v2r);

        av1ir = (v2i + v7i);
        av1ri = (v7i - v2i);

        av2rr = (v3r + v6r);
        av2ii = (v6r - v3r);

        av2ir = (v3i + v6i);
        av2ri = (v6i - v3i);

        av3rr = (v4r + v5r);
        av3ii = (v5r - v4r);

        av3ir = (v4i + v5i);
        av3ri = (v5i - v4i);

        *out_h1_r = v1r + av1rr + av2rr + av3rr;
        *out_h1_i = v1i + av1ir + av2ir + av3ir;

        tv1rr = CRTM_7_3 * av1rr;
        tv2rr = CRTM_7_1 * av2rr;
        tv3rr = CRTM_7_2 * av3rr;
        FFTZ_DOUBLE cvrr = v1r + tv1rr - tv2rr - tv3rr;
        tv1ri = CRTM_7_5 * av1ri;
        tv2ri = CRTM_7_6 * av2ri;
        tv3ri = CRTM_7_4 * av3ri;
        FFTZ_DOUBLE cvri = tv1ri + tv2ri + tv3ri;
        FFTZ_DOUBLE _or_6 = cvrr + cvri;

        tv1ii = CRTM_7_3 * av1ir;
        tv2ii = CRTM_7_1 * av2ir;
        tv3ii = CRTM_7_2 * av3ir;
        tv1ir = CRTM_7_5 * av1ii;
        tv2ir = CRTM_7_6 * av2ii;
        tv3ir = CRTM_7_4 * av3ii;

        FFTZ_DOUBLE cvir = tv1ir + tv2ir + tv3ir;
        FFTZ_DOUBLE cvii = v1i + tv1ii - tv2ii - tv3ii;
        {
            FFTZ_DOUBLE _twr = tw_ptr[0];
            FFTZ_DOUBLE _twi = tw_ptr[1];
            FFTZ_DOUBLE _or_1 = cvrr - cvri;
            FFTZ_DOUBLE _oi_1 = cvir + cvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[5 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = cvii - cvir;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tv1rr = CRTM_7_1 * av1rr;
        tv2rr = CRTM_7_2 * av2rr;
        tv3rr = CRTM_7_3 * av3rr;
        cvrr = v1r - tv1rr - tv2rr + tv3rr;

        tv1ri = CRTM_7_6 * av1ri;
        tv2ri = CRTM_7_4 * av2ri;
        tv3ri = CRTM_7_5 * av3ri;
        cvri = tv1ri - tv2ri - tv3ri;
        FFTZ_DOUBLE _or_5 = cvrr + cvri;

        tv1ii = CRTM_7_1 * av1ir;
        tv2ii = CRTM_7_2 * av2ir;
        tv3ii = CRTM_7_3 * av3ir;
        cvii = v1i - tv1ii - tv2ii + tv3ii;

        tv1ir = CRTM_7_6 * av1ii;
        tv2ir = CRTM_7_4 * av2ii;
        tv3ir = CRTM_7_5 * av3ii;
        cvir = tv1ir - tv2ir - tv3ir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_2 = cvrr - cvri;
            FFTZ_DOUBLE _oi_2 = cvii + cvir;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = cvii - cvir;
            out_h2_r[out_strides[5]] = _or_5 * _twr + _oi * _twi;
            out_h2_i[out_strides[5]] = _oi * _twr - _or_5 * _twi;
        }

        tv1rr = CRTM_7_2 * av1rr;
        tv2rr = CRTM_7_3 * av2rr;
        tv3rr = CRTM_7_1 * av3rr;
        cvrr = v1r - tv1rr + tv2rr - tv3rr;

        tv1ri = CRTM_7_4 * av1ri;
        tv2ri = CRTM_7_5 * av2ri;
        tv3ri = CRTM_7_6 * av3ri;
        cvri = tv1ri - tv2ri + tv3ri;
        FFTZ_DOUBLE _or_4 = cvrr + cvri;

        tv1ii = CRTM_7_2 * av1ir;
        tv2ii = CRTM_7_3 * av2ir;
        tv3ii = CRTM_7_1 * av3ir;
        cvii = v1i - tv1ii + tv2ii - tv3ii;

        tv1ir = CRTM_7_4 * av1ii;
        tv2ir = CRTM_7_5 * av2ii;
        tv3ir = CRTM_7_6 * av3ii;
        cvir = tv1ir - tv2ir + tv3ir;
        {
            FFTZ_DOUBLE _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_3 = cvrr - cvri;
            FFTZ_DOUBLE _oi_3 = cvii + cvir;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            FFTZ_DOUBLE _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = cvii - cvir;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID twid_c2r_fft7c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7_1 =
        +0.222520933956314404288902564496794759466355569f;
    const FFTZ_FLOAT CRTM_7_2 =
        +0.900968867902419126236102319507445051165919162f;
    const FFTZ_FLOAT CRTM_7_3 =
        +0.623489801858733530525004884004239810632274731f;
    const FFTZ_FLOAT CRTM_7_4 =
        +0.433883739117558120475768332848358754609990728f;
    const FFTZ_FLOAT CRTM_7_5 =
        +0.781831482468029808708444526674057750232334519f;
    const FFTZ_FLOAT CRTM_7_6 =
        +0.974927912181823607018131682993931217232785801f;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_real;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

    for (FFTZ_INTP cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i;
        FFTZ_FLOAT tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv1ir, tv1ii,
            tv2ir, tv2ii, tv3ir, tv3ii;
        FFTZ_FLOAT av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av1ir, av1ii,
            av2ir, av2ii, av3ir, av3ii;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h2_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h2_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h2_i[in_strides[5]];
        v6r_t = -v6r_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        v7r = v7r_t;
        v7i = v7i_t;

        av1rr = (v2r + v7r);
        av1ii = (v7r - v2r);

        av1ir = (v2i + v7i);
        av1ri = (v7i - v2i);

        av2rr = (v3r + v6r);
        av2ii = (v6r - v3r);

        av2ir = (v3i + v6i);
        av2ri = (v6i - v3i);

        av3rr = (v4r + v5r);
        av3ii = (v5r - v4r);

        av3ir = (v4i + v5i);
        av3ri = (v5i - v4i);

        *out_h1_r = v1r + av1rr + av2rr + av3rr;
        *out_h1_i = v1i + av1ir + av2ir + av3ir;

        tv1rr = CRTM_7_3 * av1rr;
        tv2rr = CRTM_7_1 * av2rr;
        tv3rr = CRTM_7_2 * av3rr;
        FFTZ_FLOAT cvrr = v1r + tv1rr - tv2rr - tv3rr;
        tv1ri = CRTM_7_5 * av1ri;
        tv2ri = CRTM_7_6 * av2ri;
        tv3ri = CRTM_7_4 * av3ri;
        FFTZ_FLOAT cvri = tv1ri + tv2ri + tv3ri;
        FFTZ_FLOAT _or_6 = cvrr + cvri;

        tv1ii = CRTM_7_3 * av1ir;
        tv2ii = CRTM_7_1 * av2ir;
        tv3ii = CRTM_7_2 * av3ir;
        tv1ir = CRTM_7_5 * av1ii;
        tv2ir = CRTM_7_6 * av2ii;
        tv3ir = CRTM_7_4 * av3ii;

        FFTZ_FLOAT cvir = tv1ir + tv2ir + tv3ir;
        FFTZ_FLOAT cvii = v1i + tv1ii - tv2ii - tv3ii;
        {
            FFTZ_FLOAT _twr = tw_ptr[0];
            FFTZ_FLOAT _twi = tw_ptr[1];
            FFTZ_FLOAT _or_1 = cvrr - cvri;
            FFTZ_FLOAT _oi_1 = cvir + cvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[5 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[5 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = cvii - cvir;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tv1rr = CRTM_7_1 * av1rr;
        tv2rr = CRTM_7_2 * av2rr;
        tv3rr = CRTM_7_3 * av3rr;
        cvrr = v1r - tv1rr - tv2rr + tv3rr;

        tv1ri = CRTM_7_6 * av1ri;
        tv2ri = CRTM_7_4 * av2ri;
        tv3ri = CRTM_7_5 * av3ri;
        cvri = tv1ri - tv2ri - tv3ri;
        FFTZ_FLOAT _or_5 = cvrr + cvri;

        tv1ii = CRTM_7_1 * av1ir;
        tv2ii = CRTM_7_2 * av2ir;
        tv3ii = CRTM_7_3 * av3ir;
        cvii = v1i - tv1ii - tv2ii + tv3ii;

        tv1ir = CRTM_7_6 * av1ii;
        tv2ir = CRTM_7_4 * av2ii;
        tv3ir = CRTM_7_5 * av3ii;
        cvir = tv1ir - tv2ir - tv3ir;
        {
            FFTZ_FLOAT _twr = tw_ptr[DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_FLOAT _or_2 = cvrr - cvri;
            FFTZ_FLOAT _oi_2 = cvii + cvir;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = cvii - cvir;
            out_h2_r[out_strides[5]] = _or_5 * _twr + _oi * _twi;
            out_h2_i[out_strides[5]] = _oi * _twr - _or_5 * _twi;
        }

        tv1rr = CRTM_7_2 * av1rr;
        tv2rr = CRTM_7_3 * av2rr;
        tv3rr = CRTM_7_1 * av3rr;
        cvrr = v1r - tv1rr + tv2rr - tv3rr;

        tv1ri = CRTM_7_4 * av1ri;
        tv2ri = CRTM_7_5 * av2ri;
        tv3ri = CRTM_7_6 * av3ri;
        cvri = tv1ri - tv2ri + tv3ri;
        FFTZ_FLOAT _or_4 = cvrr + cvri;

        tv1ii = CRTM_7_2 * av1ir;
        tv2ii = CRTM_7_3 * av2ir;
        tv3ii = CRTM_7_1 * av3ir;
        cvii = v1i - tv1ii + tv2ii - tv3ii;

        tv1ir = CRTM_7_4 * av1ii;
        tv2ir = CRTM_7_5 * av2ii;
        tv3ir = CRTM_7_6 * av3ii;
        cvir = tv1ir - tv2ir + tv3ir;
        {
            FFTZ_FLOAT _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_FLOAT _or_3 = cvrr - cvri;
            FFTZ_FLOAT _oi_3 = cvii + cvir;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            FFTZ_FLOAT _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = cvii - cvir;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_c2r_fft7c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft7c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft7c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
