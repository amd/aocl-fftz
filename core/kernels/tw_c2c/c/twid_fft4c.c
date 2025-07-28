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

/** @file twid_fft4c.c
 *
 *  @brief Twiddle Radix-4 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-4 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 3 * (4 muls, 2 adds, 2 movs [loads])
//              = (12, 6, 6)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 12, 22, 22, 0, 0},
                                                     {0, 12, 22, 22, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft4c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft4c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
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
        DOUBLE tvri, tvir, tvii, tvrr, v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        DOUBLE v1r = *in_r;
        DOUBLE v1i = *in_i;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        DOUBLE v2r = v2r_t * twr - v2i_t * twi;
        DOUBLE v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        DOUBLE v3r = v3r_t * twr - v3i_t * twi;
        DOUBLE v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        DOUBLE v4r = v4r_t * twr - v4i_t * twi;
        DOUBLE v4i = v4r_t * twi + v4i_t * twr;

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft4c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
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
        FLOAT tvri, tvir, tvii, tvrr, v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        FLOAT v1r = *in_r;
        FLOAT v1i = *in_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        FLOAT v2r = v2r_t * twr - v2i_t * twi;
        FLOAT v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        FLOAT v3r = v3r_t * twr - v3i_t * twi;
        FLOAT v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        FLOAT v4r = v4r_t * twr - v4i_t * twi;
        FLOAT v4i = v4r_t * twi + v4i_t * twr;

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft4c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft4c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft4c_fp64;
    }
    else
    {
        return NULL;
    }
}
