// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft8c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-8 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-8 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */
#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 32, 66, 46, 0, 0},
                                                     {0, 32, 66, 46, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft8c(UINT8 precision, UINT8 direction)
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

static VOID twid_c2r_fft8c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_8_1 = +0.707106781186547524400844362104849039284835938;

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
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i, tvrr,
            tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_h1_r[in_strides[1]];
        DOUBLE v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_h1_r[in_strides[2]];
        DOUBLE v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_h1_r[in_strides[3]];
        DOUBLE v4i_t = in_h1_i[in_strides[3]];
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_h2_r[in_strides[4]];
        DOUBLE v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_h2_r[in_strides[5]];
        DOUBLE v6i_t = in_h2_i[in_strides[5]];
        v6r_t = -v6r_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_h2_r[in_strides[6]];
        DOUBLE v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        v7r = v7r_t;
        v7i = v7i_t;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_h2_r[in_strides[7]];
        DOUBLE v8i_t = in_h2_i[in_strides[7]];
        v8r_t = -v8r_t;
        v8r = v8r_t;
        v8i = v8i_t;

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_h1_r = tvrr + tvri;
        DOUBLE _or_4 = tvri - tvrr;

        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        DOUBLE _or_2 = tvrr - tvri;
        DOUBLE _or_6 = tvrr + tvri;

        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_h1_i = tvii + tvir;
        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvir - tvii;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi_2 = tvir + tvii;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        DOUBLE _or_1 = tvrr - tvri;
        DOUBLE _or_7 = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        DOUBLE _or_3 = tvrr - tvri;
        DOUBLE _or_5 = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi_1 = tvir + tvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
        }

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi_3 = tvir + tvii;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tvii - tvir;
            out_h2_r[out_strides[5]] = _or_5 * _twr + _oi * _twi;
            out_h2_i[out_strides[5]] = _oi * _twr - _or_5 * _twi;
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

static VOID twid_c2r_fft8c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_8_1 = +0.707106781186547524400844362104849039284835938f;

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
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i, tvrr,
            tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_h1_r[in_strides[1]];
        FLOAT v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        FLOAT v3r_t = in_h1_r[in_strides[2]];
        FLOAT v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        FLOAT v4r_t = in_h1_r[in_strides[3]];
        FLOAT v4i_t = in_h1_i[in_strides[3]];
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FLOAT v5r_t = in_h2_r[in_strides[4]];
        FLOAT v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        FLOAT v6r_t = in_h2_r[in_strides[5]];
        FLOAT v6i_t = in_h2_i[in_strides[5]];
        v6r_t = -v6r_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Input point 7: x(6)
        FLOAT v7r_t = in_h2_r[in_strides[6]];
        FLOAT v7i_t = in_h2_i[in_strides[6]];
        v7r_t = -v7r_t;
        v7r = v7r_t;
        v7i = v7i_t;

        // Input point 8: x(7)
        FLOAT v8r_t = in_h2_r[in_strides[7]];
        FLOAT v8i_t = in_h2_i[in_strides[7]];
        v8r_t = -v8r_t;
        v8r = v8r_t;
        v8i = v8i_t;

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_h1_r = tvrr + tvri;
        FLOAT _or_4 = tvri - tvrr;

        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        FLOAT _or_2 = tvrr - tvri;
        FLOAT _or_6 = tvrr + tvri;

        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_h1_i = tvii + tvir;
        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvir - tvii;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi_2 = tvir + tvii;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (6 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[6]] = _or_6 * _twr + _oi * _twi;
            out_h2_i[out_strides[6]] = _oi * _twr - _or_6 * _twi;
        }

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        FLOAT _or_1 = tvrr - tvri;
        FLOAT _or_7 = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        FLOAT _or_3 = tvrr - tvri;
        FLOAT _or_5 = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi_1 = tvir + tvii;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (7 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[7]] = _or_7 * _twr + _oi * _twi;
            out_h2_i[out_strides[7]] = _oi * _twr - _or_7 * _twi;
        }

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi_3 = tvir + tvii;
            out_h1_r[out_strides[3]] = _or_3 * _twr + _oi_3 * _twi;
            out_h1_i[out_strides[3]] = _oi_3 * _twr - _or_3 * _twi;
        }
        {
            UINTP _twa = DATA_STRIDE * (5 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tvii - tvir;
            out_h2_r[out_strides[5]] = _or_5 * _twr + _oi * _twi;
            out_h2_i[out_strides[5]] = _oi * _twr - _or_5 * _twi;
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

kfft_ register_kernel_twid_c2r_fft8c(UINT8 precision,
                                     UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft8c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft8c_fp64;
    }
    else
    {
        return NULL;
    }
}
