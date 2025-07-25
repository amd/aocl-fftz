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

/** @file fft10c.c
 *
 *  @brief Radix-10 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-10 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 24, 84, 40, 0, 0},
                                                     {0, 24, 84, 40, 0, 0}};

ops_cycles_t get_ops_cnt_fft10c(UINT8 precision, UINT8 direction)
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

static VOID fft10c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_10_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_10_2 =
        +0.58778525229247315738615484497912915412138427663885;
    const DOUBLE CRTM_10_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_10_4 =
        +0.95105651629515357211643933337938214340569863400000;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r, v102i,
            v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i, v57i,
            tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir, cv3rr1,
            cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3, cv1rr1,
            cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1, cv2ii1,
            cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii,
            cv4rr, cv4ii, v16r, v16i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        // Input point 10: x(9)
        v10r = in_r[in_strides[9]];
        v210r = v2r + v10r;
        v102r = v10r - v2r;

        // Input point 5: x(4)
        v5r = in_r[in_strides[4]];
        // Input point 7: x(6)
        v7r = in_r[in_strides[6]];
        v57r = v5r + v7r;
        v75r = v7r - v5r;

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        // Input point 9: x(8)
        v9r = in_r[in_strides[8]];

        v39r = v3r + v9r;
        v93r = v9r - v3r;

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        // Input point 8: x(7)
        v8r = in_r[in_strides[7]];

        v48r = v4r + v8r;
        v84r = v8r - v4r;

        tv1ir = v39r - v48r;
        tv2ir2 = v39r + v48r;
        tv2ir = v210r - v57r;
        tv1ir2 = v210r + v57r;

        // Input point 1: x(0)
        v1r = *in_r;
        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v16r = v1r - v6r;
        v16r1 = v1r + v6r;

        cv1rr = CRTM_10_1 * (tv1ir + tv2ir);
        tv4rr = tv1ir - tv2ir;

        out_r[out_strides[5]] = v16r + tv4rr;
        cv2rr = v16r - (CRTM_10_3 * tv4rr);

        tv3rr1 = tv1ir2 + tv2ir2;
        cv1rr1 = CRTM_10_1 * (tv1ir2 - tv2ir2);

        *out_r = v16r1 + tv3rr1;
        cv2rr1 = v16r1 - (CRTM_10_3 * tv3rr1);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2rr - cv1rr;
        cv3rr1 = cv2rr1 + cv1rr1;
        cv3ii1 = cv2rr1 - cv1rr1;

        //---------------------------------------------------

        v2i = in_i[in_strides[1]];
        v10i = in_i[in_strides[9]];

        v102i = v10i - v2i;
        v210i = v2i + v10i;

        v5i = in_i[in_strides[4]];
        v7i = in_i[in_strides[6]];
        v75i = v7i - v5i;
        v57i = v5i + v7i;

        v3i = in_i[in_strides[2]];
        v9i = in_i[in_strides[8]];

        v93i = v9i - v3i;
        v39i = v3i + v9i;

        v4i = in_i[in_strides[3]];
        v8i = in_i[in_strides[7]];

        v84i = v8i - v4i;
        v48i = v4i + v8i;

        tv2ir3 = v39i + v48i;
        tv1ir1 = v39i - v48i;
        tv1ir3 = v210i + v57i;
        tv2ir1 = v210i - v57i;

        v1i = *in_i;
        v6i = in_i[in_strides[5]];
        v16i = v1i - v6i;
        v16i1 = v1i + v6i;

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        tv1ii = v102i + v75i;
        tv2ii = v93i + v84i;
        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_r[out_strides[1]] = cv3rr - cv4rr;
        out_r[out_strides[9]] = cv3rr + cv4rr;
        out_r[out_strides[3]] = cv3ii + cv4ii;
        out_r[out_strides[7]] = cv3ii - cv4ii;

        tv1rr = v102r + v75r;
        tv2rr = v93r + v84r;
        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_i[out_strides[1]] = cv3rr + cv4rr;
        out_i[out_strides[9]] = cv3rr - cv4rr;
        out_i[out_strides[3]] = cv3ii + cv4ii;
        out_i[out_strides[7]] = cv3ii - cv4ii;

        tv2ii1 = v75i - v102i;
        tv1ii1 = v84i - v93i;
        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_r[out_strides[8]] = cv3rr1 - cv4rr;

        tv2rr1 = v102r - v75r;
        tv1rr1 = v93r - v84r;
        cv4rr = (CRTM_10_2 * tv1rr1) + (CRTM_10_4 * tv2rr1);
        cv4ii = (CRTM_10_2 * tv2rr1) + (CRTM_10_4 * -tv1rr1);
        cv3rr = cv2ii1 + cv1ii1;
        cv3ii = cv2ii1 - cv1ii1;

        out_i[out_strides[2]] = cv3rr + cv4rr;
        out_i[out_strides[4]] = cv3ii + cv4ii;
        out_i[out_strides[6]] = cv3ii - cv4ii;
        out_i[out_strides[8]] = cv3rr - cv4rr;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft10c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const FLOAT CRTM_10_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FLOAT CRTM_10_2 =
        +0.58778525229247315738615484497912915412138427663885;
    const FLOAT CRTM_10_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const FLOAT CRTM_10_4 =
        +0.95105651629515357211643933337938214340569863400000;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r, v102i,
            v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i, v57i,
            tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir, cv3rr1,
            cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3, cv1rr1,
            cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1, cv2ii1,
            cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii,
            cv4rr, cv4ii, v16r, v16i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        // Input point 10: x(9)
        v10r = in_r[in_strides[9]];
        v210r = v2r + v10r;
        v102r = v10r - v2r;

        // Input point 5: x(4)
        v5r = in_r[in_strides[4]];
        // Input point 7: x(6)
        v7r = in_r[in_strides[6]];
        v57r = v5r + v7r;
        v75r = v7r - v5r;

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        // Input point 9: x(8)
        v9r = in_r[in_strides[8]];

        v39r = v3r + v9r;
        v93r = v9r - v3r;

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        // Input point 8: x(7)
        v8r = in_r[in_strides[7]];

        v48r = v4r + v8r;
        v84r = v8r - v4r;

        tv1ir = v39r - v48r;
        tv2ir2 = v39r + v48r;
        tv2ir = v210r - v57r;
        tv1ir2 = v210r + v57r;

        // Input point 1: x(0)
        v1r = *in_r;
        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v16r = v1r - v6r;
        v16r1 = v1r + v6r;

        cv1rr = CRTM_10_1 * (tv1ir + tv2ir);
        tv4rr = tv1ir - tv2ir;

        out_r[out_strides[5]] = v16r + tv4rr;
        cv2rr = v16r - (CRTM_10_3 * tv4rr);

        tv3rr1 = tv1ir2 + tv2ir2;
        cv1rr1 = CRTM_10_1 * (tv1ir2 - tv2ir2);

        *out_r = v16r1 + tv3rr1;
        cv2rr1 = v16r1 - (CRTM_10_3 * tv3rr1);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2rr - cv1rr;
        cv3rr1 = cv2rr1 + cv1rr1;
        cv3ii1 = cv2rr1 - cv1rr1;

        //---------------------------------------------------

        v2i = in_i[in_strides[1]];
        v10i = in_i[in_strides[9]];

        v102i = v10i - v2i;
        v210i = v2i + v10i;

        v5i = in_i[in_strides[4]];
        v7i = in_i[in_strides[6]];
        v75i = v7i - v5i;
        v57i = v5i + v7i;

        v3i = in_i[in_strides[2]];
        v9i = in_i[in_strides[8]];

        v93i = v9i - v3i;
        v39i = v3i + v9i;

        v4i = in_i[in_strides[3]];
        v8i = in_i[in_strides[7]];

        v84i = v8i - v4i;
        v48i = v4i + v8i;

        tv2ir3 = v39i + v48i;
        tv1ir1 = v39i - v48i;
        tv1ir3 = v210i + v57i;
        tv2ir1 = v210i - v57i;

        v1i = *in_i;
        v6i = in_i[in_strides[5]];
        v16i = v1i - v6i;
        v16i1 = v1i + v6i;

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        tv1ii = v102i + v75i;
        tv2ii = v93i + v84i;
        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_r[out_strides[1]] = cv3rr - cv4rr;
        out_r[out_strides[9]] = cv3rr + cv4rr;
        out_r[out_strides[3]] = cv3ii + cv4ii;
        out_r[out_strides[7]] = cv3ii - cv4ii;

        tv1rr = v102r + v75r;
        tv2rr = v93r + v84r;
        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_i[out_strides[1]] = cv3rr + cv4rr;
        out_i[out_strides[9]] = cv3rr - cv4rr;
        out_i[out_strides[3]] = cv3ii + cv4ii;
        out_i[out_strides[7]] = cv3ii - cv4ii;

        tv2ii1 = v75i - v102i;
        tv1ii1 = v84i - v93i;
        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_r[out_strides[8]] = cv3rr1 - cv4rr;

        tv2rr1 = v102r - v75r;
        tv1rr1 = v93r - v84r;
        cv4rr = (CRTM_10_2 * tv1rr1) + (CRTM_10_4 * tv2rr1);
        cv4ii = (CRTM_10_2 * tv2rr1) + (CRTM_10_4 * -tv1rr1);
        cv3rr = cv2ii1 + cv1ii1;
        cv3ii = cv2ii1 - cv1ii1;

        out_i[out_strides[2]] = cv3rr + cv4rr;
        out_i[out_strides[4]] = cv3ii + cv4ii;
        out_i[out_strides[6]] = cv3ii - cv4ii;
        out_i[out_strides[8]] = cv3rr - cv4rr;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft10c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft10c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft10c_fp64;
    }
    else
    {
        return NULL;
    }
}
