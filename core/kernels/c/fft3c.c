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

/** @file fft3c.c
 *
 *  @brief Radix-3 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-3 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft3c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft3c_fp32;
    else if (precision == DT_DOUBLE)
        return fft3c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 12, 12, 0, 0},
                                                     {0, 4, 12, 12, 0, 0}};

ops_cycles_t get_ops_cnt_fft3c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

VOID fft3c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    const FLOAT CRTM_3_1 = +0.500000000000000000000000000000000000000000000;
    const FLOAT CRTM_3_2 = +0.866025403784438646763723170752936183471402627;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, tv1r, tv1i, tv2r, tv2i, tv3r, tv3i,
              avrr, avri, avir, avii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Input point 2: x(1)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        avrr = v2r + v3r;
        avri = v3i - v2i;
        avir = v3r - v2r;
        avii = v2i + v3i;

        tv1r = CRTM_3_1 * avrr;
        tv1i = CRTM_3_2 * avri;
        tv2r = CRTM_3_2 * avir;
        tv2i = CRTM_3_1 * avii;

        // Output point 1: X(0)
        *out_r = v1r + avrr;
        *out_i = v1i + avii;

        // Output point 2: X(1)
        tv3r = v1r - tv1r;
        tv3i = v1i - tv2i;
        out_r[out_stride] = tv3r - tv1i;
        out_i[out_stride] = tv3i + tv2r;

        // Output point 3: X(2)
        out_r[(out_stride << 1)] = tv3r + tv1i;
        out_i[(out_stride << 1)] = tv3i - tv2r;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}

VOID fft3c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    const DOUBLE CRTM_3_1 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = +0.866025403784438646763723170752936183471402627;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, tv1r, tv1i, tv2r, tv2i, tv3r, tv3i,
               avrr, avri, avir, avii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Input point 2: x(1)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        avrr = v2r + v3r;
        avri = v3i - v2i;
        avir = v3r - v2r;
        avii = v2i + v3i;

        tv1r = CRTM_3_1 * avrr;
        tv1i = CRTM_3_2 * avri;
        tv2r = CRTM_3_2 * avir;
        tv2i = CRTM_3_1 * avii;

        // Output point 1: X(0)
        *out_r = v1r + avrr;
        *out_i = v1i + avii;

        // Output point 2: X(1)
        tv3r = v1r - tv1r;
        tv3i = v1i - tv2i;
        out_r[out_stride] = tv3r - tv1i;
        out_i[out_stride] = tv3i + tv2r;

        // Output point 3: X(2)
        out_r[(out_stride << 1)] = tv3r + tv1i;
        out_i[(out_stride << 1)] = tv3i - tv2r;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "core/kernels/kernel_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 8, 32, 30, 0, 13},
                                                     {0, 8, 32, 30, 0, 13}};

ops_cycles_t get_ops_cnt_fft3c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

const DOUBLE CRTM_3[RADIX_3][2] = {
    {1.0, 0.0}, {-0.5, -0.866025403784439}, {-0.5, 0.866025403784438}};

VOID fft3c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
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
    DOUBLE local_in[RADIX_3][2] = {0};

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

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 3i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 3i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_3[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_3[2], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 3i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_3[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_3[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[2], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        // Adding vector output stride to output pointer
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}

VOID fft3c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
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
    DOUBLE local_in[RADIX_3][2] = {0};

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

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 3i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 3i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_3[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_3[2], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 3i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_3[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_3[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[2], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        // Adding vector output stride to output pointer
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}
#endif
