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

/** @file fft9c.c
 *
 *  @brief Radix-9 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-9 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author Prasandh Sankarankutty
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft9c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft9c_fp32;
    else if (precision == DT_DOUBLE)
        return fft9c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT

// TODO : to be updated
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 44, 84, 36, 0, 0},
                                                     {0, 44, 84, 36, 0, 0}};
ops_cycles_t get_ops_cnt_fft9c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

VOID fft9c_fp64(VOID* in_real, VOID* in_imag, VOID* out_real, VOID* out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    const DOUBLE CRTM_9_1 = +0.939692620785908384054109277324731469936208134;
    const DOUBLE CRTM_9_2 = +0.342020143325668733044099614682259580763083368;
    const DOUBLE CRTM_9_3 = +0.984807753012208059366743024589523013670643252;
    const DOUBLE CRTM_9_4 = +0.173648177666930348851716626769314796000375677;
    const DOUBLE CRTM_9_5 = +0.642787609686539326322643409907263432907559884;
    const DOUBLE CRTM_9_6 = +0.766044443118978035202392650555416673935832457;
    const DOUBLE CRTM_9_7 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_9_8 = +0.866025403784438646763723170752936183471402627;

    DOUBLE *in_r  = (DOUBLE *)in_real;
    DOUBLE *in_i  = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i,
               v6r, v6i, v7r, v7i, v8r, v8i, v9r, v9i,
               tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv4rr, tv4ri,
               tv1ir, tv1ii, tv2ir, tv2ii, tv3ir, tv3ii, tv4ir, tv4ii,
               av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av4rr, av4ri,
               av1ir, av1ii, av2ir, av2ii, av3ir, av3ii, av4ir, av4ii,
               cv47rr, cv47ir, cv47r, cv47i, mvrr, mvii, cvrr, cvri, cvir, cvii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Input point 3: x(2)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];
        // Input point 4: x(3)
        v4r = in_r[(in_stride * 3)];
        v4i = in_i[(in_stride * 3)];
        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];
        // Input point 6: x(5)
        v6r = in_r[(in_stride * 5)];
        v6i = in_i[(in_stride * 5)];
        // Input point 7: x(6)
        v7r = in_r[(in_stride * 6)];
        v7i = in_i[(in_stride * 6)];
        // Input point 8: x(7)
        v8r = in_r[(in_stride * 7)];
        v8i = in_i[(in_stride * 7)];
        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        av1rr = (v2r + v9r);
        av2rr = (v3r + v8r);
        av3rr = (v4r + v7r);
        av4rr = (v5r + v6r);
        av1ri = (v9i - v2i);
        av2ri = (v8i - v3i);
        av3ri = (v7i - v4i);
        av4ri = (v6i - v5i);

        av1ir = (v2i + v9i);
        av2ir = (v3i + v8i);
        av3ir = (v4i + v7i);
        av4ir = (v5i + v6i);
        av1ii = (v9r - v2r);
        av2ii = (v8r - v3r);
        av3ii = (v7r - v4r);
        av4ii = (v6r - v5r);

        cv47rr = av1rr + av2rr + av4rr;
        cv47ir = av1ir + av2ir + av4ir;
        cv47r  = v1r + av3rr;
        cv47i  = v1i + av3ir;
        // output point 1 : X(0)
        *out_r = cv47r + cv47rr;
        *out_i = cv47i + cv47ir;

        tv1rr = CRTM_9_6 * av1rr;
        tv2rr = CRTM_9_4 * av2rr;
        tv3rr = CRTM_9_7 * av3rr;
        tv4rr = CRTM_9_1 * av4rr;
        tv1ri = CRTM_9_5 * av1ri;
        tv2ri = CRTM_9_3 * av2ri;
        tv3ri = CRTM_9_8 * av3ri;
        tv4ri = CRTM_9_2 * av4ri;

        tv1ii = CRTM_9_6 * av1ir;
        tv2ii = CRTM_9_4 * av2ir;
        tv3ii = CRTM_9_7 * av3ir;
        tv4ii = CRTM_9_1 * av4ir;
        tv1ir = CRTM_9_5 * av1ii;
        tv2ir = CRTM_9_3 * av2ii;
        tv3ir = CRTM_9_8 * av3ii;
        tv4ir = CRTM_9_2 * av4ii;

        mvrr = v1r - tv3rr;
        mvii = v1i - tv3ii;

        cvrr = mvrr + tv1rr + tv2rr - tv4rr;
        cvri = tv1ri + tv2ri + tv3ri + tv4ri;
        cvir = tv1ir + tv2ir + tv3ir + tv4ir;
        cvii = mvii + tv1ii + tv2ii - tv4ii;

        // output point 2 : X(1)
        out_r[out_stride] = cvrr - cvri;
        out_i[out_stride] = cvii + cvir;

        // output point 9 : X(8)
        out_r[out_stride << 3] = cvrr + cvri;
        out_i[out_stride << 3] = cvii - cvir;

        tv1rr = CRTM_9_4 * av1rr;
        tv2rr = CRTM_9_1 * av2rr;
        tv4rr = CRTM_9_6 * av4rr;
        tv1ri = CRTM_9_3 * av1ri;
        tv2ri = CRTM_9_2 * av2ri;
        tv4ri = CRTM_9_5 * av4ri;

        tv1ii = CRTM_9_4 * av1ir;
        tv2ii = CRTM_9_1 * av2ir;
        tv4ii = CRTM_9_6 * av4ir;
        tv1ir = CRTM_9_3 * av1ii;
        tv2ir = CRTM_9_2 * av2ii;
        tv4ir = CRTM_9_5 * av4ii;

        cvrr = mvrr + tv1rr - tv2rr + tv4rr;
        cvri = tv1ri + tv2ri - tv3ri - tv4ri;
        cvir = tv1ir + tv2ir - tv3ir - tv4ir;
        cvii = mvii + tv1ii - tv2ii + tv4ii;

        // output point 3 : X(2)
        out_r[out_stride << 1] = cvrr - cvri;
        out_i[out_stride << 1] = cvii + cvir;

        // output point 8 : X(7)
        out_r[out_stride * 7] = cvrr + cvri;
        out_i[out_stride * 7] = cvii - cvir;

        tv1rr = CRTM_9_7 * (cv47rr);
        tv1ri = CRTM_9_8 * (av1ri + av4ri - av2ri);
        tv1ii = CRTM_9_7 * (cv47ir);
        tv1ir = CRTM_9_8 * (av1ii + av4ii - av2ii);

        cvrr = cv47r - tv1rr;
        cvii = cv47i - tv1ii;
        cvri = tv1ri;
        cvir = tv1ir;

        // output point 4 : X(3)
        out_r[out_stride * 3] = cvrr - cvri;
        out_i[out_stride * 3] = cvii + cvir;

        // output point 7 : X(6)
        out_r[out_stride * 6] = cvrr + cvri;
        out_i[out_stride * 6] = cvii - cvir;

        tv1rr = CRTM_9_1 * av1rr;
        tv2rr = CRTM_9_6 * av2rr;
        tv4rr = CRTM_9_4 * av4rr;
        tv1ri = CRTM_9_2 * av1ri;
        tv2ri = CRTM_9_5 * av2ri;
        tv4ri = CRTM_9_3 * av4ri;

        tv1ii = CRTM_9_1 * av1ir;
        tv2ii = CRTM_9_6 * av2ir;
        tv4ii = CRTM_9_4 * av4ir;
        tv1ir = CRTM_9_2 * av1ii;
        tv2ir = CRTM_9_5 * av2ii;
        tv4ir = CRTM_9_3 * av4ii;

        cvrr = mvrr - tv1rr + tv2rr + tv4rr;
        cvri = tv1ri - tv2ri + tv3ri - tv4ri;
        cvir = tv1ir - tv2ir + tv3ir - tv4ir;
        cvii = mvii - tv1ii + tv2ii + tv4ii;

        // output point 5 : X(4)
        out_r[out_stride << 2] = cvrr - cvri;
        out_i[out_stride << 2] = cvii + cvir;

        // output point 6 : X(5)
        out_r[out_stride * 5] = cvrr + cvri;
        out_i[out_stride * 5] = cvii - cvir;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}

VOID fft9c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    const FLOAT CRTM_9_1 = +0.939692620785908384054109277324731469936208134;
    const FLOAT CRTM_9_2 = +0.342020143325668733044099614682259580763083368;
    const FLOAT CRTM_9_3 = +0.984807753012208059366743024589523013670643252;
    const FLOAT CRTM_9_4 = +0.173648177666930348851716626769314796000375677;
    const FLOAT CRTM_9_5 = +0.642787609686539326322643409907263432907559884;
    const FLOAT CRTM_9_6 = +0.766044443118978035202392650555416673935832457;
    const FLOAT CRTM_9_7 = +0.500000000000000000000000000000000000000000000;
    const FLOAT CRTM_9_8 = +0.866025403784438646763723170752936183471402627;

    FLOAT *in_r  = (FLOAT *)in_real;
    FLOAT *in_i  = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i,
              v6r, v6i, v7r, v7i, v8r, v8i, v9r, v9i,
              tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv4rr, tv4ri,
              tv1ir, tv1ii, tv2ir, tv2ii, tv3ir, tv3ii, tv4ir, tv4ii,
              av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av4rr, av4ri,
              av1ir, av1ii, av2ir, av2ii, av3ir, av3ii, av4ir, av4ii,
              cv47rr, cv47ir, cv47r, cv47i, mvrr, mvii, cvrr, cvri, cvir, cvii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Input point 3: x(2)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];
        // Input point 4: x(3)
        v4r = in_r[(in_stride * 3)];
        v4i = in_i[(in_stride * 3)];
        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];
        // Input point 6: x(5)
        v6r = in_r[(in_stride * 5)];
        v6i = in_i[(in_stride * 5)];
        // Input point 7: x(6)
        v7r = in_r[(in_stride * 6)];
        v7i = in_i[(in_stride * 6)];
        // Input point 8: x(7)
        v8r = in_r[(in_stride * 7)];
        v8i = in_i[(in_stride * 7)];
        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        av1rr = (v2r + v9r);
        av2rr = (v3r + v8r);
        av3rr = (v4r + v7r);
        av4rr = (v5r + v6r);
        av1ri = (v9i - v2i);
        av2ri = (v8i - v3i);
        av3ri = (v7i - v4i);
        av4ri = (v6i - v5i);

        av1ir = (v2i + v9i);
        av2ir = (v3i + v8i);
        av3ir = (v4i + v7i);
        av4ir = (v5i + v6i);
        av1ii = (v9r - v2r);
        av2ii = (v8r - v3r);
        av3ii = (v7r - v4r);
        av4ii = (v6r - v5r);

        cv47rr = av1rr + av2rr + av4rr;
        cv47ir = av1ir + av2ir + av4ir;
        cv47r  = v1r + av3rr;
        cv47i  = v1i + av3ir;
        // output point 1 : X(0)
        *out_r = cv47r + cv47rr;
        *out_i = cv47i + cv47ir;

        tv1rr = CRTM_9_6 * av1rr;
        tv2rr = CRTM_9_4 * av2rr;
        tv3rr = CRTM_9_7 * av3rr;
        tv4rr = CRTM_9_1 * av4rr;
        tv1ri = CRTM_9_5 * av1ri;
        tv2ri = CRTM_9_3 * av2ri;
        tv3ri = CRTM_9_8 * av3ri;
        tv4ri = CRTM_9_2 * av4ri;

        tv1ii = CRTM_9_6 * av1ir;
        tv2ii = CRTM_9_4 * av2ir;
        tv3ii = CRTM_9_7 * av3ir;
        tv4ii = CRTM_9_1 * av4ir;
        tv1ir = CRTM_9_5 * av1ii;
        tv2ir = CRTM_9_3 * av2ii;
        tv3ir = CRTM_9_8 * av3ii;
        tv4ir = CRTM_9_2 * av4ii;

        mvrr = v1r - tv3rr;
        mvii = v1i - tv3ii;

        cvrr = mvrr + tv1rr + tv2rr - tv4rr;
        cvri = tv1ri + tv2ri + tv3ri + tv4ri;
        cvir = tv1ir + tv2ir + tv3ir + tv4ir;
        cvii = mvii + tv1ii + tv2ii - tv4ii;

        // output point 2 : X(1)
        out_r[out_stride] = cvrr - cvri;
        out_i[out_stride] = cvii + cvir;

        // output point 9 : X(8)
        out_r[out_stride << 3] = cvrr + cvri;
        out_i[out_stride << 3] = cvii - cvir;

        tv1rr = CRTM_9_4 * av1rr;
        tv2rr = CRTM_9_1 * av2rr;
        tv4rr = CRTM_9_6 * av4rr;
        tv1ri = CRTM_9_3 * av1ri;
        tv2ri = CRTM_9_2 * av2ri;
        tv4ri = CRTM_9_5 * av4ri;

        tv1ii = CRTM_9_4 * av1ir;
        tv2ii = CRTM_9_1 * av2ir;
        tv4ii = CRTM_9_6 * av4ir;
        tv1ir = CRTM_9_3 * av1ii;
        tv2ir = CRTM_9_2 * av2ii;
        tv4ir = CRTM_9_5 * av4ii;

        cvrr = mvrr + tv1rr - tv2rr + tv4rr;
        cvri = tv1ri + tv2ri - tv3ri - tv4ri;
        cvir = tv1ir + tv2ir - tv3ir - tv4ir;
        cvii = mvii + tv1ii - tv2ii + tv4ii;

        // output point 3 : X(2)
        out_r[out_stride << 1] = cvrr - cvri;
        out_i[out_stride << 1] = cvii + cvir;

        // output point 8 : X(7)
        out_r[out_stride * 7] = cvrr + cvri;
        out_i[out_stride * 7] = cvii - cvir;

        tv1rr = CRTM_9_7 * (cv47rr);
        tv1ri = CRTM_9_8 * (av1ri + av4ri - av2ri);
        tv1ii = CRTM_9_7 * (cv47ir);
        tv1ir = CRTM_9_8 * (av1ii + av4ii - av2ii);

        cvrr = cv47r - tv1rr;
        cvii = cv47i - tv1ii;
        cvri = tv1ri;
        cvir = tv1ir;

        // output point 4 : X(3)
        out_r[out_stride * 3] = cvrr - cvri;
        out_i[out_stride * 3] = cvii + cvir;

        // output point 7 : X(6)
        out_r[out_stride * 6] = cvrr + cvri;
        out_i[out_stride * 6] = cvii - cvir;

        tv1rr = CRTM_9_1 * av1rr;
        tv2rr = CRTM_9_6 * av2rr;
        tv4rr = CRTM_9_4 * av4rr;
        tv1ri = CRTM_9_2 * av1ri;
        tv2ri = CRTM_9_5 * av2ri;
        tv4ri = CRTM_9_3 * av4ri;

        tv1ii = CRTM_9_1 * av1ir;
        tv2ii = CRTM_9_6 * av2ir;
        tv4ii = CRTM_9_4 * av4ir;
        tv1ir = CRTM_9_2 * av1ii;
        tv2ir = CRTM_9_5 * av2ii;
        tv4ir = CRTM_9_3 * av4ii;

        cvrr = mvrr - tv1rr + tv2rr + tv4rr;
        cvri = tv1ri - tv2ri + tv3ri - tv4ri;
        cvir = tv1ir - tv2ir + tv3ir - tv4ir;
        cvii = mvii - tv1ii + tv2ii + tv4ii;

        // output point 5 : X(4)
        out_r[out_stride << 2] = cvrr - cvri;
        out_i[out_stride << 2] = cvii + cvir;

        // output point 6 : X(5)
        out_r[out_stride * 5] = cvrr + cvri;
        out_i[out_stride * 5] = cvii - cvir;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}

#else // Basic version

#include "core/kernels/kernel_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 116, 488, 90, 0, 361},
                                                     {0, 116, 488, 90, 0, 361}};
ops_cycles_t get_ops_cnt_fft9c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

const DOUBLE CRTM_9[RADIX_9][2] = {{1.0, 0.0},
                                   {0.766044443118978, -0.642787609686539},
                                   {0.17364817766693, -0.984807753012208},
                                   {-0.5, -0.866025403784439},
                                   {-0.939692620785908, -0.342020143325669},
                                   {-0.939692620785908, 0.342020143325669},
                                   {-0.5, 0.866025403784438},
                                   {0.17364817766693, 0.984807753012208},
                                   {0.766044443118978, 0.64278760968654}};

VOID fft9c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are multiplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;
    // temp variable to store power (constant_multiplier)
    DOUBLE pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    DOUBLE temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    DOUBLE cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    DOUBLE cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    DOUBLE *in_dr = (DOUBLE *)in_real;
    DOUBLE *in_di = (DOUBLE *)in_imag;
    DOUBLE *input_r = (DOUBLE *)in_real;
    DOUBLE *input_i = (DOUBLE *)in_imag;
    // buffer to store current output
    DOUBLE *output_r = (DOUBLE *)out_real;
    DOUBLE *output_i = (DOUBLE *)out_imag;
    DOUBLE *out_dr = (DOUBLE *)out_real;
    DOUBLE *out_di = (DOUBLE *)out_imag;
    // local buffer to store input
    DOUBLE local_in[RADIX_9][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[2]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[3]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[4]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[5]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[6]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[7]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[8]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 9i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_9[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_9[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_9[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_9[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_9[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_9[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_9[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_9[8], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;
        ;

        /******************** Output 9i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

VOID fft9c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are multiplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;

    // temp variable to store power (constant_multiplier)
    FLOAT pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    FLOAT temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    FLOAT cmul_temp[2] = {0.0, 0.0};
    // buffer to store intermediate CPOW results
    FLOAT cpow_temp[2] = {0.0, 0.0};
    // buffer to store current input
    FLOAT *in_fr = (FLOAT *)in_real;
    FLOAT *in_fi = (FLOAT *)in_imag;
    FLOAT *input_r = (FLOAT *)in_real;
    FLOAT *input_i = (FLOAT *)in_imag;
    // buffer to store current output
    FLOAT *output_r = (FLOAT *)out_real;
    FLOAT *output_i = (FLOAT *)out_imag;
    FLOAT *out_fr = (FLOAT *)out_real;
    FLOAT *out_fi = (FLOAT *)out_imag;
    // local buffer to store input
    FLOAT local_in[RADIX_9][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[2]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[3]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[4]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[5]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[6]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[7]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[8]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 9i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        CADD(local_in[6], temp_out, temp_out);
        CADD(local_in[7], temp_out, temp_out);
        CADD(local_in[8], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_9[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_9[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_9[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_9[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_9[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_9[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_9[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_9[8], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;
        ;

        /******************** Output 9i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 9i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_9[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_9[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_9[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_9[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_9[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_9[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_9[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_9[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[8], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}
#endif // USE_OPT_KERNEL_VARIANT