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

/** @file twid_fft5c.c
 *
 *  @brief Twiddle Radix-5 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-5 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 4 * (4 muls, 2 adds, 2 movs [loads])
//              = (16, 8, 8)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 40, 28, 0, 0},
                                                     {0, 28, 40, 28, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft5c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft5c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978;

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

    DOUBLE *tw = (DOUBLE *)twd;
    DOUBLE twr, twi;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r, v34r,
            v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr, cv3rr,
            cv1ii, cv2ii, cv3ii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

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
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft5c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_5_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_5_2 =
        +0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_5_3 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_5_4 =
        +0.58778525229247301629891039327884007596190389052978f;

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

    FLOAT *tw = (FLOAT *)twd;
    FLOAT twr, twi;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v25r, v34r,
            v52i, v43i, v25i, v34i, v52r, v43r, tvri, tvir, cv1rr, cv2rr, cv3rr,
            cv1ii, cv2ii, cv3ii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

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
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft5c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft5c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft5c_fp64;
    }
    else
    {
        return NULL;
    }
}
