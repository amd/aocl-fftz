// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft5c.c
 *
 *  @brief Radix-5 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-5 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 12, 32, 20, 0, 0},
                                                     {0, 12, 32, 20, 0, 0}};

ops_cycles_t get_ops_cnt_fft5c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft5c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000;
    const FFTZ_DOUBLE CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978;

    FFTZ_DOUBLE *in_r, *in_i, *out_r, *out_i;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_DOUBLE *)in_imag;
        in_i = (FFTZ_DOUBLE *)in_real;
        out_r = (FFTZ_DOUBLE *)out_imag;
        out_i = (FFTZ_DOUBLE *)out_real;
    }
    else
    {
        in_r = (FFTZ_DOUBLE *)in_real;
        in_i = (FFTZ_DOUBLE *)in_imag;
        out_r = (FFTZ_DOUBLE *)out_real;
        out_i = (FFTZ_DOUBLE *)out_imag;
    }

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r,
            v34r, v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr,
            cv3rr, cv1ii, cv2ii, cv3ii;

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
        *out_r = v1r + cv1rr;
        *out_i = v1i + cv1ii;

        // Output point 2: X(1)
        cv1rr = CRTM_5_1 * (v25r - v34r);
        cv1ii = CRTM_5_1 * (v25i - v34i);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2ii + cv1ii;
        tvri = (CRTM_5_2 * v52i) + (CRTM_5_4 * v43i);
        tvir = (CRTM_5_2 * v52r) + (CRTM_5_4 * v43r);

        out_r[out_strides[1]] = cv3rr - tvri;
        out_i[out_strides[1]] = cv3ii + tvir;

        // Output point 5: X(4)
        out_r[out_strides[4]] = cv3rr + tvri;
        out_i[out_strides[4]] = cv3ii - tvir;

        // Output point 3: X(2)
        cv3rr = cv2rr - cv1rr;
        cv3ii = cv2ii - cv1ii;

        tvri = (CRTM_5_4 * v52i) - (CRTM_5_2 * v43i);
        tvir = (CRTM_5_4 * v52r) - (CRTM_5_2 * v43r);

        out_r[out_strides[2]] = cv3rr - tvri;
        out_i[out_strides[2]] = cv3ii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = cv3rr + tvri;
        out_i[out_strides[3]] = cv3ii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft5c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FFTZ_FLOAT CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000;
    const FFTZ_FLOAT CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const FFTZ_FLOAT CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978;

    FFTZ_FLOAT *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_FLOAT *)in_imag;
        in_i = (FFTZ_FLOAT *)in_real;
        out_r = (FFTZ_FLOAT *)out_imag;
        out_i = (FFTZ_FLOAT *)out_real;
    }
    else
    {
        in_r = (FFTZ_FLOAT *)in_real;
        in_i = (FFTZ_FLOAT *)in_imag;
        out_r = (FFTZ_FLOAT *)out_real;
        out_i = (FFTZ_FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r, v34r,
              v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr,
              cv3rr, cv1ii, cv2ii, cv3ii;

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
        *out_r = v1r + cv1rr;
        *out_i = v1i + cv1ii;

        // Output point 2: X(1)
        cv1rr = CRTM_5_1 * (v25r - v34r);
        cv1ii = CRTM_5_1 * (v25i - v34i);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2ii + cv1ii;
        tvri = (CRTM_5_2 * v52i) + (CRTM_5_4 * v43i);
        tvir = (CRTM_5_2 * v52r) + (CRTM_5_4 * v43r);

        out_r[out_strides[1]] = cv3rr - tvri;
        out_i[out_strides[1]] = cv3ii + tvir;

        // Output point 5: X(4)
        out_r[out_strides[4]] = cv3rr + tvri;
        out_i[out_strides[4]] = cv3ii - tvir;

        // Output point 3: X(2)
        cv3rr = cv2rr - cv1rr;
        cv3ii = cv2ii - cv1ii;
        tvri = (CRTM_5_4 * v52i) - (CRTM_5_2 * v43i);
        tvir = (CRTM_5_4 * v52r) - (CRTM_5_2 * v43r);

        out_r[out_strides[2]] = cv3rr - tvri;
        out_i[out_strides[2]] = cv3ii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = cv3rr + tvri;
        out_i[out_strides[3]] = cv3ii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft5c(FFTZ_UINT8 precision,
                            FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft5c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft5c_fp64;
    }
    else
    {
        return NULL;
    }
}
