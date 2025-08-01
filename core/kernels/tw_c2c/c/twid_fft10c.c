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

/** @file twid_fft10c.c
 *
 *  @brief Twiddle Radix-10 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-10 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 9 * (4 muls, 2 adds, 2 movs [loads])
//              = (36, 18, 18)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 102, 58, 0, 0},
                                                     {0, 60, 102, 58, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft10c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft10c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
                             UINT8 flag)
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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    DOUBLE twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r, v102i,
            v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i, v57i,
            tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir, cv3rr1,
            cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3, cv1rr1,
            cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1, cv2ii1,
            cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii,
            cv4rr, cv4ii, v16r, v16i;

        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_r[in_strides[5]];
        DOUBLE v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_r[in_strides[6]];
        DOUBLE v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_r[in_strides[7]];
        DOUBLE v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        DOUBLE v9r_t = in_r[in_strides[8]];
        DOUBLE v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        DOUBLE v10r_t = in_r[in_strides[9]];
        DOUBLE v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * cols + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        v10r = v10r_t * twr - v10i_t * twi;
        v10i = v10r_t * twi + v10i_t * twr;

        v210r = v2r + v10r;
        v102r = v10r - v2r;
        v102i = v10i - v2i;
        v210i = v2i + v10i;

        v57r = v5r + v7r;
        v75r = v7r - v5r;
        v75i = v7i - v5i;
        v57i = v5i + v7i;

        tv2ir = v210r - v57r;
        tv1ir2 = v210r + v57r;
        tv1ir3 = v210i + v57i;
        tv2ir1 = v210i - v57i;
        tv1ii = v102i + v75i;
        tv1rr = v102r + v75r;
        tv2ii1 = v75i - v102i;
        tv2rr1 = v102r - v75r;

        v39r = v3r + v9r;
        v93r = v9r - v3r;
        v93i = v9i - v3i;
        v39i = v3i + v9i;

        v48r = v4r + v8r;
        v84r = v8r - v4r;
        v84i = v8i - v4i;
        v48i = v4i + v8i;

        tv1ir = v39r - v48r;
        tv2ir2 = v39r + v48r;
        tv2ir3 = v39i + v48i;
        tv1ir1 = v39i - v48i;
        tv2ii = v93i + v84i;
        tv2rr = v93r + v84r;
        tv1ii1 = v84i - v93i;
        tv1rr1 = v93r - v84r;

        v16r = v1r - v6r;
        v16r1 = v1r + v6r;
        v16i = v1i - v6i;
        v16i1 = v1i + v6i;

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

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_r[out_strides[1]] = cv3rr - cv4rr;
        out_r[out_strides[9]] = cv3rr + cv4rr;
        out_r[out_strides[3]] = cv3ii + cv4ii;
        out_r[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_i[out_strides[1]] = cv3rr + cv4rr;
        out_i[out_strides[9]] = cv3rr - cv4rr;
        out_i[out_strides[3]] = cv3ii + cv4ii;
        out_i[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_r[out_strides[8]] = cv3rr1 - cv4rr;

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

static VOID twid_fft10c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
                             UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const FLOAT CRTM_10_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_10_2 =
        +0.58778525229247315738615484497912915412138427663885f;
    const FLOAT CRTM_10_3 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_10_4 =
        +0.95105651629515357211643933337938214340569863400000f;

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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    FLOAT twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r, v102i,
            v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i, v57i,
            tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir, cv3rr1,
            cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3, cv1rr1,
            cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1, cv2ii1,
            cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr, cv3ii,
            cv4rr, cv4ii, v16r, v16i;

        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * cols + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * cols + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * cols + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FLOAT v6r_t = in_r[in_strides[5]];
        FLOAT v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * cols + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FLOAT v7r_t = in_r[in_strides[6]];
        FLOAT v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * cols + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FLOAT v8r_t = in_r[in_strides[7]];
        FLOAT v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * cols + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FLOAT v9r_t = in_r[in_strides[8]];
        FLOAT v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * cols + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FLOAT v10r_t = in_r[in_strides[9]];
        FLOAT v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * cols + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        v10r = v10r_t * twr - v10i_t * twi;
        v10i = v10r_t * twi + v10i_t * twr;

        v210r = v2r + v10r;
        v102r = v10r - v2r;
        v102i = v10i - v2i;
        v210i = v2i + v10i;

        v57r = v5r + v7r;
        v75r = v7r - v5r;
        v75i = v7i - v5i;
        v57i = v5i + v7i;

        tv2ir = v210r - v57r;
        tv1ir2 = v210r + v57r;
        tv1ir3 = v210i + v57i;
        tv2ir1 = v210i - v57i;
        tv1ii = v102i + v75i;
        tv1rr = v102r + v75r;
        tv2ii1 = v75i - v102i;
        tv2rr1 = v102r - v75r;

        v39r = v3r + v9r;
        v93r = v9r - v3r;
        v93i = v9i - v3i;
        v39i = v3i + v9i;

        v48r = v4r + v8r;
        v84r = v8r - v4r;
        v84i = v8i - v4i;
        v48i = v4i + v8i;

        tv1ir = v39r - v48r;
        tv2ir2 = v39r + v48r;
        tv2ir3 = v39i + v48i;
        tv1ir1 = v39i - v48i;
        tv2ii = v93i + v84i;
        tv2rr = v93r + v84r;
        tv1ii1 = v84i - v93i;
        tv1rr1 = v93r - v84r;

        v16r = v1r - v6r;
        v16r1 = v1r + v6r;
        v16i = v1i - v6i;
        v16i1 = v1i + v6i;

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

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_r[out_strides[1]] = cv3rr - cv4rr;
        out_r[out_strides[9]] = cv3rr + cv4rr;
        out_r[out_strides[3]] = cv3ii + cv4ii;
        out_r[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_i[out_strides[1]] = cv3rr + cv4rr;
        out_i[out_strides[9]] = cv3rr - cv4rr;
        out_i[out_strides[3]] = cv3ii + cv4ii;
        out_i[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_r[out_strides[8]] = cv3rr1 - cv4rr;

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

kfft_ register_kernel_twid_fft10c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft10c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft10c_fp64;
    }
    else
    {
        return NULL;
    }
}
