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

/** @file fft6c.c
 *
 *  @brief Radix-6 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-6 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 *  @author Murugan Vairavel
 */

#include "core/kernels/kernel.h"

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 8, 36, 24, 0, 0},
                                                     {0, 8, 36, 24, 0, 0}};

ops_cycles_t get_ops_cnt_fft6c(UINT8 precision, UINT8 direction)
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

static VOID fft6c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_6_1 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_6_2 = +0.866025403784438646763723170752936183471402627;

    DOUBLE *in_r  = (DOUBLE *)in_real;
    DOUBLE *in_i  = (DOUBLE *)in_imag;
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
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
               av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
               tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
               tv2ii, tv1ii;
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
        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v6i = in_i[in_strides[5]];

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_r = av1rr + av8rr;
        *out_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_r[out_strides[3]] = av4rr + av7rr;
        out_i[out_strides[3]] = av4ii + av7ii;

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = tv1rr + tv1ii;
        out_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_r[out_strides[5]] = tv1rr - tv1ii;
        out_i[out_strides[5]] = tv2ii + tv2rr;

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_r[out_strides[2]] = tv1rr + tv1ii;
        out_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_r[out_strides[4]] = tv1rr - tv1ii;
        out_i[out_strides[4]] = tv2ii - tv2rr;

        in_r  += v_in_stride;
        in_i  += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft6c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_6_1 = +0.500000000000000000000000000000000000000000000;
    const FLOAT CRTM_6_2 = +0.866025403784438646763723170752936183471402627;

    FLOAT *in_r  = (FLOAT *)in_real;
    FLOAT *in_i  = (FLOAT *)in_imag;
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
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, av1rr,
              av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr, tv2rr,
              av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii, tv2ii,
              tv1ii;
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
        // Input point 6: x(5)
        v6r = in_r[in_strides[5]];
        v6i = in_i[in_strides[5]];

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_r = av1rr + av8rr;
        *out_i = av1ii + av8ii;
        // Output point 4: X(3)
        out_r[out_strides[3]] = av4rr + av7rr;
        out_i[out_strides[3]] = av4ii + av7ii;

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = tv1rr + tv1ii;
        out_i[out_strides[1]] = tv2ii - tv2rr;
        // Output point 6: X(5)
        out_r[out_strides[5]] = tv1rr - tv1ii;
        out_i[out_strides[5]] = tv2ii + tv2rr;

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        out_r[out_strides[2]] = tv1rr + tv1ii;
        out_i[out_strides[2]] = tv2ii + tv2rr;
        // Output point 5: X(4)
        out_r[out_strides[4]] = tv1rr - tv1ii;
        out_i[out_strides[4]] = tv2ii - tv2rr;

        in_r  += v_in_stride;
        in_i  += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 44, 188, 60, 0, 106},
                                                     {0, 44, 188, 60, 0, 106}};

ops_cycles_t get_ops_cnt_fft6c(INT32 precision)
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

const DOUBLE CRTM_6[RADIX_6][2] = {
    {1.0, 0.0},  {0.5, -0.866025403784439}, {-0.5, -0.866025403784439},
    {-1.0, 0.0}, {-0.5, 0.866025403784439}, {0.5, 0.866025403784439}};

static VOID fft6c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
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
    DOUBLE local_in[RADIX_6][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 6i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 6i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_6[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_6[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_6[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_6[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_6[5], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 6i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 6i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 6i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 6i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        // Adding vector output stride to output pointer
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft6c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

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
    FLOAT local_in[RADIX_6][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        LOAD_INPUT(input_r + in_strides[1], input_i + in_strides[1],
                   local_in[1]);
        LOAD_INPUT(input_r + in_strides[2], input_i + in_strides[2],
                   local_in[2]);
        LOAD_INPUT(input_r + in_strides[3], input_i + in_strides[3],
                   local_in[3]);
        LOAD_INPUT(input_r + in_strides[4], input_i + in_strides[4],
                   local_in[4]);
        LOAD_INPUT(input_r + in_strides[5], input_i + in_strides[5],
                   local_in[5]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 6i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], temp_out, temp_out);
        CADD(local_in[1], temp_out, temp_out);
        CADD(local_in[2], temp_out, temp_out);
        CADD(local_in[3], temp_out, temp_out);
        CADD(local_in[4], temp_out, temp_out);
        CADD(local_in[5], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);

        /******************** Output 6i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_6[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_6[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_6[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_6[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_6[5], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[1],
                     output_i + out_strides[1]);

        /******************** Output 6i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[2],
                     output_i + out_strides[2]);

        /******************** Output 6i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[3],
                     output_i + out_strides[3]);

        /******************** Output 6i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[4],
                     output_i + out_strides[4]);

        /******************** Output 6i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_6[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_6[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_6[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_6[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_6[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[5], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r + out_strides[5],
                     output_i + out_strides[5]);

        // Adding vector output stride to output pointer
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}
#endif // USE_OPT_KERNEL_VARIANT

kfft_ register_kernel_fft6c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft6c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft6c_fp64;
    }
    else
    {
        return NULL;
    }
}
