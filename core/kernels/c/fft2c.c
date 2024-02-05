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

/** @file fft2c.c
 *
 *  @brief Radix-2 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-2 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft2c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft2c_fp32;
    else if (precision == DT_DOUBLE)
        return fft2c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 4, 8, 0, 0},
                                                     {0, 0, 4, 8, 0, 0}};

ops_cycles_t get_ops_cnt_fft2c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

VOID fft2c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
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
        FLOAT v1r, v1i, v2r, v2i;
        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Output point 1: X(0)
        *out_r = v1r + v2r;
        *out_i = v1i + v2i;
        // Output point 2: X(0)
        out_r[out_stride] = v1r - v2r;
        out_i[out_stride] = v1i - v2i;
        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}

VOID fft2c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
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
        DOUBLE v1r, v1i, v2r, v2i;
        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        // Output point 1: X(0)
        *out_r = v1r + v2r;
        *out_i = v1i + v2i;
        // Output point 2: X(0)
        out_r[out_stride] = v1r - v2r;
        out_i[out_stride] = v1i - v2i;
        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 5, 12, 16, 0, 0},
                                                     {0, 5, 12, 16, 0, 0}};

ops_cycles_t get_ops_cnt_fft2c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

// Array to store local constant radix multipliers
const FLOAT64 CRTM_2[RADIX_2][2] = {{1.0, 0.0}, {-1.0, 0.0}};

VOID fft2c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;

    // temp variable to store power (constant_multiplier)
    FLOAT32 pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    FLOAT32 temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    FLOAT32 cmul_temp[2] = {0.0, 0.0};
    // buffer to store current input
    FLOAT32 *in_fr = (FLOAT32 *)in_real;
    FLOAT32 *in_fi = (FLOAT32 *)in_imag;
    FLOAT32 *input_r = (FLOAT32 *)in_real;
    FLOAT32 *input_i = (FLOAT32 *)in_imag;
    // buffer to store current output
    FLOAT32 *output_r = (FLOAT32 *)out_real;
    FLOAT32 *output_i = (FLOAT32 *)out_imag;
    FLOAT32 *out_fr = (FLOAT32 *)out_real;
    FLOAT32 *out_fi = (FLOAT32 *)out_imag;
    // local buffer to store input
    FLOAT32 local_in[RADIX_2][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_fr;
        input_i = in_fi;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 2i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], local_in[1], temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 2i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_ADD since (constant_multiplier)^1 = constant_multiplier
        CMUL_CADD(local_in[1], CRTM_2[1], pow_cm, local_in[0], cmul_temp);
        STORE_OUTPUT(local_in[0], output_r, output_i);

        // Adding vector output stride to output pointer
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}

VOID fft2c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    // All strides values are mutliplied with DATA_STRIDE for complex data
    INTP in_stride = strides->in_stride * DATA_STRIDE;
    INTP out_stride = strides->out_stride * DATA_STRIDE;
    INTP v_in_stride = strides->v_in_stride * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * DATA_STRIDE;
    // temp variable to store power (constant_multiplier)
    FLOAT64 pow_cm[2] = {0.0, 0.0};
    // temp variable to store pow_cm * input
    FLOAT64 temp_out[2] = {0.0, 0.0};
    // buffer to store intermediate CMUL results
    FLOAT64 cmul_temp[2] = {0.0, 0.0};
    // buffer to store current input
    FLOAT64 *in_dr = (FLOAT64 *)in_real;
    FLOAT64 *in_di = (FLOAT64 *)in_imag;
    FLOAT64 *input_r = (FLOAT64 *)in_real;
    FLOAT64 *input_i = (FLOAT64 *)in_imag;
    // buffer to store current output
    FLOAT64 *output_r = (FLOAT64 *)out_real;
    FLOAT64 *output_i = (FLOAT64 *)out_imag;
    FLOAT64 *out_dr = (FLOAT64 *)out_real;
    FLOAT64 *out_di = (FLOAT64 *)out_imag;
    // local buffer to store input
    FLOAT64 local_in[RADIX_2][2] = {0};

    for (INTP i = 0; i < n; i++)
    {
        /******************** load input **********************/
        input_r = in_dr;
        input_i = in_di;
        LOAD_INPUT(input_r, input_i, local_in[0]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[1]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 2i ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CADD since (constant_multiplier)^0 = 1
        CADD(local_in[0], local_in[1], temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 2i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_ADD since (constant_multiplier)^1 = constant_multiplier
        CMUL_CADD(local_in[1], CRTM_2[1], pow_cm, local_in[0], cmul_temp);
        STORE_OUTPUT(local_in[0], output_r, output_i);

        // Adding vector output stride to output pointer
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}
#endif
