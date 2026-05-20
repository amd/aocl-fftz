// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_bwd_fft10c.c
 *
 *  @brief Backward-only twiddle Radix-10 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-10 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 10

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 60, 102, 58, 0, 0},
                                                     {0, 60, 102, 58, 0, 0}};

ops_cycles_t get_ops_cnt_twid_bwd_fft10c(FFTZ_UINT8 precision,
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

static FFTZ_VOID twid_bwd_fft10c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_10_1 =
        +0.55901699437494742410229341718281905886015458990288;
    const FFTZ_DOUBLE CRTM_10_2 =
        +0.58778525229247315738615484497912915412138427663885;
    const FFTZ_DOUBLE CRTM_10_3 =
        +0.25000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_10_4 =
        +0.95105651629515357211643933337938214340569863400000;

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
    FFTZ_DOUBLE twr, twi;

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
            v7r, v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r,
            v102i, v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i,
            v57i, tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir,
            cv3rr1, cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3,
            cv1rr1, cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1,
            cv2ii1, cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr,
            cv3ii, cv4rr, cv4ii, v16r, v16i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h1_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h2_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_DOUBLE v9r_t = in_h2_r[in_strides[8]];
        FFTZ_DOUBLE v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FFTZ_DOUBLE v10r_t = in_h2_r[in_strides[9]];
        FFTZ_DOUBLE v10i_t = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
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

        out_h2_r[out_strides[5]] = v16r + tv4rr;
        cv2rr = v16r - (CRTM_10_3 * tv4rr);

        tv3rr1 = tv1ir2 + tv2ir2;
        cv1rr1 = CRTM_10_1 * (tv1ir2 - tv2ir2);

        *out_h1_r = v16r1 + tv3rr1;
        cv2rr1 = v16r1 - (CRTM_10_3 * tv3rr1);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2rr - cv1rr;
        cv3rr1 = cv2rr1 + cv1rr1;
        cv3ii1 = cv2rr1 - cv1rr1;

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_h2_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_h1_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_h1_r[out_strides[1]] = cv3rr - cv4rr;
        out_h2_r[out_strides[9]] = cv3rr + cv4rr;
        out_h1_r[out_strides[3]] = cv3ii + cv4ii;
        out_h2_r[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_h1_i[out_strides[1]] = cv3rr + cv4rr;
        out_h2_i[out_strides[9]] = cv3rr - cv4rr;
        out_h1_i[out_strides[3]] = cv3ii + cv4ii;
        out_h2_i[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_h1_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_h1_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_h2_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_h2_r[out_strides[8]] = cv3rr1 - cv4rr;

        cv4rr = (CRTM_10_2 * tv1rr1) + (CRTM_10_4 * tv2rr1);
        cv4ii = (CRTM_10_2 * tv2rr1) + (CRTM_10_4 * -tv1rr1);
        cv3rr = cv2ii1 + cv1ii1;
        cv3ii = cv2ii1 - cv1ii1;

        out_h1_i[out_strides[2]] = cv3rr + cv4rr;
        out_h1_i[out_strides[4]] = cv3ii + cv4ii;
        out_h2_i[out_strides[6]] = cv3ii - cv4ii;
        out_h2_i[out_strides[8]] = cv3rr - cv4rr;

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

static FFTZ_VOID twid_bwd_fft10c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_10_1 =
        +0.55901699437494742410229341718281905886015458990288f;
    const FFTZ_FLOAT CRTM_10_2 =
        +0.58778525229247315738615484497912915412138427663885f;
    const FFTZ_FLOAT CRTM_10_3 =
        +0.25000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_10_4 =
        +0.95105651629515357211643933337938214340569863400000f;

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
    FFTZ_FLOAT twr, twi;

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
            v7r, v7i, v8r, v8i, v9r, v9i, v10r, v10i, v210r, v39r, v48r, v57r,
            v102i, v93i, v84i, v75i, v102r, v93r, v84r, v75r, v210i, v39i, v48i,
            v57i, tv1rr, tv1ii, tv2rr, tv2ii, tv4rr, tv4ii, tv1ir, tv2ir,
            cv3rr1, cv3ii1, tv1ir1, tv2ir1, tv1ir2, tv2ir2, tv1ir3, tv2ir3,
            cv1rr1, cv1ii1, tv3rr1, tv3ii1, tv2rr1, tv2ii1, tv1rr1, tv1ii1,
            cv2ii1, cv2rr1, v16r1, v16i1, cv1rr, cv1ii, cv2rr, cv2ii, cv3rr,
            cv3ii, cv4rr, cv4ii, v16r, v16i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h1_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h1_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h2_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        // Input point 9: x(8)
        FFTZ_FLOAT v9r_t = in_h2_r[in_strides[8]];
        FFTZ_FLOAT v9i_t = in_h2_i[in_strides[8]];
        twr = tw_ptr[7 * DATA_STRIDE];
        twi = tw_ptr[7 * DATA_STRIDE + 1];
        v9r = v9r_t * twr - v9i_t * twi;
        v9i = v9r_t * twi + v9i_t * twr;

        // Input point 10: x(9)
        FFTZ_FLOAT v10r_t = in_h2_r[in_strides[9]];
        FFTZ_FLOAT v10i_t = in_h2_i[in_strides[9]];
        twr = tw_ptr[8 * DATA_STRIDE];
        twi = tw_ptr[8 * DATA_STRIDE + 1];
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

        out_h2_r[out_strides[5]] = v16r + tv4rr;
        cv2rr = v16r - (CRTM_10_3 * tv4rr);

        tv3rr1 = tv1ir2 + tv2ir2;
        cv1rr1 = CRTM_10_1 * (tv1ir2 - tv2ir2);

        *out_h1_r = v16r1 + tv3rr1;
        cv2rr1 = v16r1 - (CRTM_10_3 * tv3rr1);
        cv3rr = cv2rr + cv1rr;
        cv3ii = cv2rr - cv1rr;
        cv3rr1 = cv2rr1 + cv1rr1;
        cv3ii1 = cv2rr1 - cv1rr1;

        cv1ii = CRTM_10_1 * (tv1ir1 + tv2ir1);
        tv4ii = tv1ir1 - tv2ir1;

        out_h2_i[out_strides[5]] = v16i + tv4ii;
        cv2ii = v16i - (CRTM_10_3 * tv4ii);
        tv3ii1 = tv1ir3 + tv2ir3;
        cv1ii1 = CRTM_10_1 * (tv1ir3 - tv2ir3);
        *out_h1_i = v16i1 + tv3ii1;
        cv2ii1 = v16i1 - (CRTM_10_3 * tv3ii1);

        cv4rr = (CRTM_10_4 * tv2ii) + (CRTM_10_2 * tv1ii);
        cv4ii = (CRTM_10_2 * tv2ii) - (CRTM_10_4 * tv1ii);

        out_h1_r[out_strides[1]] = cv3rr - cv4rr;
        out_h2_r[out_strides[9]] = cv3rr + cv4rr;
        out_h1_r[out_strides[3]] = cv3ii + cv4ii;
        out_h2_r[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1rr) + (CRTM_10_4 * tv2rr);
        cv4ii = (CRTM_10_2 * -tv2rr) + (CRTM_10_4 * tv1rr);
        cv3rr = cv2ii + cv1ii;
        cv3ii = cv2ii - cv1ii;

        out_h1_i[out_strides[1]] = cv3rr + cv4rr;
        out_h2_i[out_strides[9]] = cv3rr - cv4rr;
        out_h1_i[out_strides[3]] = cv3ii + cv4ii;
        out_h2_i[out_strides[7]] = cv3ii - cv4ii;

        cv4rr = (CRTM_10_2 * tv1ii1) + (CRTM_10_4 * tv2ii1);
        cv4ii = (CRTM_10_2 * tv2ii1) + (CRTM_10_4 * -tv1ii1);

        out_h1_r[out_strides[2]] = cv3rr1 + cv4rr;
        out_h1_r[out_strides[4]] = cv3ii1 + cv4ii;
        out_h2_r[out_strides[6]] = cv3ii1 - cv4ii;
        out_h2_r[out_strides[8]] = cv3rr1 - cv4rr;

        cv4rr = (CRTM_10_2 * tv1rr1) + (CRTM_10_4 * tv2rr1);
        cv4ii = (CRTM_10_2 * tv2rr1) + (CRTM_10_4 * -tv1rr1);
        cv3rr = cv2ii1 + cv1ii1;
        cv3ii = cv2ii1 - cv1ii1;

        out_h1_i[out_strides[2]] = cv3rr + cv4rr;
        out_h1_i[out_strides[4]] = cv3ii + cv4ii;
        out_h2_i[out_strides[6]] = cv3ii - cv4ii;
        out_h2_i[out_strides[8]] = cv3rr - cv4rr;

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

kfft_ register_kernel_twid_bwd_fft10c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_bwd_fft10c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_bwd_fft10c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
