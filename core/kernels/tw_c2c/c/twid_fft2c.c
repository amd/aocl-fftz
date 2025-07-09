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

/** @file twid_fft2c.c
 *
 *  @brief Twiddle Radix-2 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-2 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 1 * (4 muls, 2 adds, 2 movs [loads])
//              = (4, 2, 2)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 6, 10, 0, 0},
                                                     {0, 4, 6, 10, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft2c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft2c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    FLOAT *in_r, *in_i, *out_r, *out_i;
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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    FLOAT twr, twi;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FLOAT *)in_imag;
        in_i = (FLOAT *)in_real;
        out_r = (FLOAT *)out_imag;
        out_i = (FLOAT *)out_real;
    }
    else
    {
        in_r = (FLOAT *)in_real;
        in_i = (FLOAT *)in_imag;
        out_r = (FLOAT *)out_real;
        out_i = (FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        // Input point 1: x(0)
        FLOAT v1r = *in_r;
        FLOAT v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];

        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];

        FLOAT v2r = v2r_t * twr - v2i_t * twi;
        FLOAT v2i = v2r_t * twi + v2i_t * twr;

        // Output point 1: X(0)
        *out_r = v1r + v2r;
        *out_i = v1i + v2i;

        // Output point 2: X(1)
        out_r[out_strides[1]] = v1r - v2r;
        out_i[out_strides[1]] = v1i - v2i;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft2c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_r, *in_i, *out_r, *out_i;
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

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    DOUBLE twr, twi;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (DOUBLE *)in_imag;
        in_i = (DOUBLE *)in_real;
        out_r = (DOUBLE *)out_imag;
        out_i = (DOUBLE *)out_real;
    }
    else
    {
        in_r = (DOUBLE *)in_real;
        in_i = (DOUBLE *)in_imag;
        out_r = (DOUBLE *)out_real;
        out_i = (DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        // Input point 1: x(0)
        DOUBLE v1r = *in_r;
        DOUBLE v1i = *in_i;
        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        DOUBLE v2r = v2r_t * twr - v2i_t * twi;
        DOUBLE v2i = v2r_t * twi + v2i_t * twr;

        // Output point 1: X(0)
        *out_r = v1r + v2r;
        *out_i = v1i + v2i;
        // Output point 2: X(0)
        out_r[out_strides[1]] = v1r - v2r;
        out_i[out_strides[1]] = v1i - v2i;
        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft2c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft2c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft2c_fp64;
    }
    else
    {
        return NULL;
    }
}
