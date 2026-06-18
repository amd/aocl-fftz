// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft5c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-5 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-5 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 40, 28, 0, 0},
                                                     {0, 28, 40, 28, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft5c(UINT8 precision, UINT8 direction)
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

static VOID twid_c2r_fft5c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978;

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
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r, v34r,
            v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr, cv3rr,
            cv1ii, cv2ii, cv3ii;

        // Input point 1: x(0)
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
        DOUBLE v4r_t = in_h2_r[in_strides[3]];
        DOUBLE v4i_t = in_h2_i[in_strides[3]];
        v4r_t = -v4r_t;
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_h2_r[in_strides[4]];
        DOUBLE v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        v25r = v2r + v5r;
        v34r = v3r + v4r;
        v52i = v5i - v2i;
        v43i = v4i - v3i;

        v25i = v5i + v2i;
        v34i = v3i + v4i;
        v52r = v5r - v2r;
        v43r = v4r - v3r;

        // common arithmetic computations
        cv1rr = v25r + v34r;
        cv1ii = v25i + v34i;
        cv2rr = v1r - (CRTM_5_3 * cv1rr);
        cv2ii = v1i - (CRTM_5_3 * cv1ii);

        // Output point 1: X(0)
        *out_h1_r = v1r + cv1rr;
        *out_h1_i = v1i + cv1ii;

        // Output point 2: X(1)
        cv1rr = CRTM_5_1 * (v25r - v34r);
        cv1ii = CRTM_5_1 * (v25i - v34i);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2ii + cv1ii;
        tvri = (CRTM_5_2 * v52i) + (CRTM_5_4 * v43i);
        tvir = (CRTM_5_2 * v52r) + (CRTM_5_4 * v43r);

        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_1 = cv3rr - tvri;
            DOUBLE _oi_1 = cv3ii + tvir;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }

        // Output point 5: X(4)
        DOUBLE _or_4 = cv3rr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = cv3ii - tvir;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        // Output point 3: X(2)
        cv3rr = cv2rr - cv1rr;
        cv3ii = cv2ii - cv1ii;

        tvri = (CRTM_5_4 * v52i) - (CRTM_5_2 * v43i);
        tvir = (CRTM_5_4 * v52r) - (CRTM_5_2 * v43r);

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_2 = cv3rr - tvri;
            DOUBLE _oi_2 = cv3ii + tvir;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }

        // Output point 4: X(3)
        DOUBLE _or_3 = cv3rr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = cv3ii - tvir;
            out_h2_r[out_strides[3]] = _or_3 * _twr + _oi * _twi;
            out_h2_i[out_strides[3]] = _oi * _twr - _or_3 * _twi;
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

static VOID twid_c2r_fft5c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978f;

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
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r, v34r,
            v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr, cv3rr,
            cv1ii, cv2ii, cv3ii;

        // Input point 1: x(0)
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
        FLOAT v4r_t = in_h2_r[in_strides[3]];
        FLOAT v4i_t = in_h2_i[in_strides[3]];
        v4r_t = -v4r_t;
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FLOAT v5r_t = in_h2_r[in_strides[4]];
        FLOAT v5i_t = in_h2_i[in_strides[4]];
        v5r_t = -v5r_t;
        v5r = v5r_t;
        v5i = v5i_t;

        v25r = v2r + v5r;
        v34r = v3r + v4r;
        v52i = v5i - v2i;
        v43i = v4i - v3i;

        v25i = v5i + v2i;
        v34i = v3i + v4i;
        v52r = v5r - v2r;
        v43r = v4r - v3r;

        // common arithmetic computations
        cv1rr = v25r + v34r;
        cv1ii = v25i + v34i;
        cv2rr = v1r - (CRTM_5_3 * cv1rr);
        cv2ii = v1i - (CRTM_5_3 * cv1ii);

        // Output point 1: X(0)
        *out_h1_r = v1r + cv1rr;
        *out_h1_i = v1i + cv1ii;

        // Output point 2: X(1)
        cv1rr = CRTM_5_1 * (v25r - v34r);
        cv1ii = CRTM_5_1 * (v25i - v34i);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2ii + cv1ii;
        tvri = (CRTM_5_2 * v52i) + (CRTM_5_4 * v43i);
        tvir = (CRTM_5_2 * v52r) + (CRTM_5_4 * v43r);

        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_1 = cv3rr - tvri;
            FLOAT _oi_1 = cv3ii + tvir;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }

        // Output point 5: X(4)
        FLOAT _or_4 = cv3rr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (4 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = cv3ii - tvir;
            out_h2_r[out_strides[4]] = _or_4 * _twr + _oi * _twi;
            out_h2_i[out_strides[4]] = _oi * _twr - _or_4 * _twi;
        }

        // Output point 3: X(2)
        cv3rr = cv2rr - cv1rr;
        cv3ii = cv2ii - cv1ii;
        tvri = (CRTM_5_4 * v52i) - (CRTM_5_2 * v43i);
        tvir = (CRTM_5_4 * v52r) - (CRTM_5_2 * v43r);

        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_2 = cv3rr - tvri;
            FLOAT _oi_2 = cv3ii + tvir;
            out_h1_r[out_strides[2]] = _or_2 * _twr + _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _oi_2 * _twr - _or_2 * _twi;
        }

        // Output point 4: X(3)
        FLOAT _or_3 = cv3rr + tvri;
        {
            UINTP _twa = DATA_STRIDE * (3 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = cv3ii - tvir;
            out_h2_r[out_strides[3]] = _or_3 * _twr + _oi * _twi;
            out_h2_i[out_strides[3]] = _oi * _twr - _or_3 * _twi;
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

kfft_ register_kernel_twid_c2r_fft5c(UINT8 precision,
                                     UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft5c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft5c_fp64;
    }
    else
    {
        return NULL;
    }
}
