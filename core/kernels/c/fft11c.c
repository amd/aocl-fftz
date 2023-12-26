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

/** @file fft11c.c
 *
 *  @brief Radix-11 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-11 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft11c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft11c_fp32;
    else if (precision == DT_DOUBLE)
        return fft11c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 100, 140, 44, 0, 0},
                                                     {0, 100, 140, 44, 0, 0}};

ops_cycles_t get_ops_cnt_fft11c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

VOID fft11c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    const DOUBLE CRTM_11_1 =
        +0.84125353283118116029052039464203089547681594330064;
    const DOUBLE CRTM_11_2 =
        +0.54064081745559759544482548159299693174139803024473;
    const DOUBLE CRTM_11_3 =
        +0.41541501300188639668675795488636098054966524290126;
    const DOUBLE CRTM_11_4 =
        +0.90963199535451838458365117807108162835411650732265;
    const DOUBLE CRTM_11_5 =
        +0.14231483827328501490317354898047094957684096668515;
    const DOUBLE CRTM_11_6 =
        +0.98982144188093275042610808187068914262031166769031;
    const DOUBLE CRTM_11_7 =
        +0.65486073394528511198338203198719613618953603946564;
    const DOUBLE CRTM_11_8 =
        +0.75574957435425824224552448923467521721665586591805;
    const DOUBLE CRTM_11_9 =
        +0.95949297361449738989036805706632769906245484800000;
    const DOUBLE CRTM_11_10 =
        +0.28173255684142978898192655345478532989004751779983;

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
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v211r, v310r, v49r,
            v58r, v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i,
            v58i, v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];

        // Input point 2: x(1)

        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        v9r = in_r[in_stride << 3];
        v9i = in_i[in_stride << 3];

        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // common operations
        v211r = v2r + v11r;
        v310r = v3r + v10r;
        v49r = v4r + v9r;
        v58r = v5r + v8r;
        v67r = v6r + v7r;

        v112i = v11i - v2i;
        v103i = v10i - v3i;
        v94i = v9i - v4i;
        v85i = v8i - v5i;
        v76i = v7i - v6i;

        v211i = v2i + v11i;
        v310i = v3i + v10i;
        v49i = v4i + v9i;
        v58i = v5i + v8i;
        v67i = v6i + v7i;

        v112r = v11r - v2r;
        v103r = v10r - v3r;
        v94r = v9r - v4r;
        v85r = v8r - v5r;
        v76r = v7r - v6r;

        // Output point 1: X(0)
        *out_r = v1r + v211r + v310r + v49r + v58r + v67r;
        *out_i = v1i + v211i + v310i + v49i + v58i + v67i;

        // Output point 2: X(1)
        tvrr = v1r;
        tvrr += (CRTM_11_1 * v211r);
        tvrr += (CRTM_11_3 * v310r);
        tvrr -= (CRTM_11_5 * v49r);
        tvrr -= (CRTM_11_7 * v58r);
        tvrr -= (CRTM_11_9 * v67r);
        tvri = (CRTM_11_2 * v112i);
        tvri += (CRTM_11_4 * v103i);
        tvri += (CRTM_11_6 * v94i);
        tvri += (CRTM_11_8 * v85i);
        tvri += (CRTM_11_10 * v76i);

        tvir = (CRTM_11_2 * v112r);
        tvir += (CRTM_11_4 * v103r);
        tvir += (CRTM_11_6 * v94r);
        tvir += (CRTM_11_8 * v85r);
        tvir += (CRTM_11_10 * v76r);
        tvii = v1i;
        tvii += (CRTM_11_1 * v211i);
        tvii += (CRTM_11_3 * v310i);
        tvii -= (CRTM_11_5 * v49i);
        tvii -= (CRTM_11_7 * v58i);
        tvii -= (CRTM_11_9 * v67i);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvir + tvii;

        // Output point 11: X(10)
        out_r[(out_stride * 10)] = tvrr + tvri;
        out_i[(out_stride * 10)] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v1r;
        tvrr += (CRTM_11_3 * v211r);
        tvrr -= (CRTM_11_7 * v310r);
        tvrr -= (CRTM_11_9 * v49r);
        tvrr -= (CRTM_11_5 * v58r);
        tvrr += (CRTM_11_1 * v67r);
        tvri = (CRTM_11_4 * v112i);
        tvri += (CRTM_11_8 * v103i);
        tvri -= (CRTM_11_10 * v94i);
        tvri -= (CRTM_11_6 * v85i);
        tvri -= (CRTM_11_2 * v76i);

        tvii = v1i;
        tvii += (CRTM_11_3 * v211i);
        tvii -= (CRTM_11_7 * v310i);
        tvii -= (CRTM_11_9 * v49i);
        tvii -= (CRTM_11_5 * v58i);
        tvii += (CRTM_11_1 * v67i);
        tvir = (CRTM_11_4 * v112r);
        tvir += (CRTM_11_8 * v103r);
        tvir -= (CRTM_11_10 * v94r);
        tvir -= (CRTM_11_6 * v85r);
        tvir -= (CRTM_11_2 * v76r);

        out_r[(out_stride << 1)] = tvrr - tvri;
        out_i[(out_stride << 1)] = tvir + tvii;

        // Output point 10: X(9)
        out_r[(out_stride * 9)] = tvrr + tvri;
        out_i[(out_stride * 9)] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = v1r;
        tvrr -= (CRTM_11_5 * v211r);
        tvrr -= (CRTM_11_9 * v310r);
        tvrr += (CRTM_11_3 * v49r);
        tvrr += (CRTM_11_1 * v58r);
        tvrr -= (CRTM_11_7 * v67r);
        tvri = (CRTM_11_6 * v112i);
        tvri -= (CRTM_11_10 * v103i);
        tvri -= (CRTM_11_4 * v94i);
        tvri += (CRTM_11_2 * v85i);
        tvri += (CRTM_11_8 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_5 * v211i);
        tvii -= (CRTM_11_9 * v310i);
        tvii += (CRTM_11_3 * v49i);
        tvii += (CRTM_11_1 * v58i);
        tvii -= (CRTM_11_7 * v67i);
        tvir = (CRTM_11_6 * v112r);
        tvir -= (CRTM_11_10 * v103r);
        tvir -= (CRTM_11_4 * v94r);
        tvir += (CRTM_11_2 * v85r);
        tvir += (CRTM_11_8 * v76r);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvir + tvii;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = v1r;
        tvrr -= (CRTM_11_7 * v211r);
        tvrr -= (CRTM_11_5 * v310r);
        tvrr += (CRTM_11_1 * v49r);
        tvrr -= (CRTM_11_9 * v58r);
        tvrr += (CRTM_11_3 * v67r);
        tvri = (CRTM_11_8 * v112i);
        tvri -= (CRTM_11_6 * v103i);
        tvri += (CRTM_11_2 * v94i);
        tvri += (CRTM_11_10 * v85i);
        tvri -= (CRTM_11_4 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_7 * v211i);
        tvii -= (CRTM_11_5 * v310i);
        tvii += (CRTM_11_1 * v49i);
        tvii -= (CRTM_11_9 * v58i);
        tvii += (CRTM_11_3 * v67i);
        tvir = (CRTM_11_8 * v112r);
        tvir -= (CRTM_11_6 * v103r);
        tvir += (CRTM_11_2 * v94r);
        tvir += (CRTM_11_10 * v85r);
        tvir -= (CRTM_11_4 * v76r);

        out_r[(out_stride << 2)] = tvrr - tvri;
        out_i[(out_stride << 2)] = tvir + tvii;

        // Output point 8: X(7)
        out_r[(out_stride * 7)] = tvrr + tvri;
        out_i[(out_stride * 7)] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= (CRTM_11_9 * v211r);
        tvrr += (CRTM_11_1 * v310r);
        tvrr -= (CRTM_11_7 * v49r);
        tvrr += (CRTM_11_3 * v58r);
        tvrr -= (CRTM_11_5 * v67r);
        tvri = (CRTM_11_10 * v112i);
        tvri -= (CRTM_11_2 * v103i);
        tvri += (CRTM_11_8 * v94i);
        tvri -= (CRTM_11_4 * v85i);
        tvri += (CRTM_11_6 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_9 * v211i);
        tvii += (CRTM_11_1 * v310i);
        tvii -= (CRTM_11_7 * v49i);
        tvii += (CRTM_11_3 * v58i);
        tvii -= (CRTM_11_5 * v67i);
        tvir = (CRTM_11_10 * v112r);
        tvir -= (CRTM_11_2 * v103r);
        tvir += (CRTM_11_8 * v94r);
        tvir -= (CRTM_11_4 * v85r);
        tvir += (CRTM_11_6 * v76r);

        out_r[(out_stride * 5)] = tvrr - tvri;
        out_i[(out_stride * 5)] = tvir + tvii;

        // Output point 7: X(6)
        out_r[(out_stride * 6)] = tvrr + tvri;
        out_i[(out_stride * 6)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}

VOID fft11c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
{
    const FLOAT CRTM_11_1 =
        +0.84125353283118116029052039464203089547681594330064;
    const FLOAT CRTM_11_2 =
        +0.54064081745559759544482548159299693174139803024473;
    const FLOAT CRTM_11_3 =
        +0.41541501300188639668675795488636098054966524290126;
    const FLOAT CRTM_11_4 =
        +0.90963199535451838458365117807108162835411650732265;
    const FLOAT CRTM_11_5 =
        +0.14231483827328501490317354898047094957684096668515;
    const FLOAT CRTM_11_6 =
        +0.98982144188093275042610808187068914262031166769031;
    const FLOAT CRTM_11_7 =
        +0.65486073394528511198338203198719613618953603946564;
    const FLOAT CRTM_11_8 =
        +0.75574957435425824224552448923467521721665586591805;
    const FLOAT CRTM_11_9 =
        +0.95949297361449738989036805706632769906245484800000;
    const FLOAT CRTM_11_10 =
        +0.28173255684142978898192655345478532989004751779983;

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
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v211r, v310r, v49r,
            v58r, v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i,
            v58i, v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];

        // Input point 2: x(1)

        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        v9r = in_r[in_stride << 3];
        v9i = in_i[in_stride << 3];

        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // common operations
        v211r = v2r + v11r;
        v310r = v3r + v10r;
        v49r = v4r + v9r;
        v58r = v5r + v8r;
        v67r = v6r + v7r;

        v112i = v11i - v2i;
        v103i = v10i - v3i;
        v94i = v9i - v4i;
        v85i = v8i - v5i;
        v76i = v7i - v6i;

        v211i = v2i + v11i;
        v310i = v3i + v10i;
        v49i = v4i + v9i;
        v58i = v5i + v8i;
        v67i = v6i + v7i;

        v112r = v11r - v2r;
        v103r = v10r - v3r;
        v94r = v9r - v4r;
        v85r = v8r - v5r;
        v76r = v7r - v6r;

        // Output point 1: X(0)
        *out_r = v1r + v211r + v310r + v49r + v58r + v67r;
        *out_i = v1i + v211i + v310i + v49i + v58i + v67i;

        // Output point 2: X(1)
        tvrr = v1r;
        tvrr += (CRTM_11_1 * v211r);
        tvrr += (CRTM_11_3 * v310r);
        tvrr -= (CRTM_11_5 * v49r);
        tvrr -= (CRTM_11_7 * v58r);
        tvrr -= (CRTM_11_9 * v67r);
        tvri = (CRTM_11_2 * v112i);
        tvri += (CRTM_11_4 * v103i);
        tvri += (CRTM_11_6 * v94i);
        tvri += (CRTM_11_8 * v85i);
        tvri += (CRTM_11_10 * v76i);

        tvir = (CRTM_11_2 * v112r);
        tvir += (CRTM_11_4 * v103r);
        tvir += (CRTM_11_6 * v94r);
        tvir += (CRTM_11_8 * v85r);
        tvir += (CRTM_11_10 * v76r);
        tvii = v1i;
        tvii += (CRTM_11_1 * v211i);
        tvii += (CRTM_11_3 * v310i);
        tvii -= (CRTM_11_5 * v49i);
        tvii -= (CRTM_11_7 * v58i);
        tvii -= (CRTM_11_9 * v67i);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvir + tvii;

        // Output point 11: X(10)
        out_r[(out_stride * 10)] = tvrr + tvri;
        out_i[(out_stride * 10)] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v1r;
        tvrr += (CRTM_11_3 * v211r);
        tvrr -= (CRTM_11_7 * v310r);
        tvrr -= (CRTM_11_9 * v49r);
        tvrr -= (CRTM_11_5 * v58r);
        tvrr += (CRTM_11_1 * v67r);
        tvri = (CRTM_11_4 * v112i);
        tvri += (CRTM_11_8 * v103i);
        tvri -= (CRTM_11_10 * v94i);
        tvri -= (CRTM_11_6 * v85i);
        tvri -= (CRTM_11_2 * v76i);

        tvii = v1i;
        tvii += (CRTM_11_3 * v211i);
        tvii -= (CRTM_11_7 * v310i);
        tvii -= (CRTM_11_9 * v49i);
        tvii -= (CRTM_11_5 * v58i);
        tvii += (CRTM_11_1 * v67i);
        tvir = (CRTM_11_4 * v112r);
        tvir += (CRTM_11_8 * v103r);
        tvir -= (CRTM_11_10 * v94r);
        tvir -= (CRTM_11_6 * v85r);
        tvir -= (CRTM_11_2 * v76r);

        out_r[(out_stride << 1)] = tvrr - tvri;
        out_i[(out_stride << 1)] = tvir + tvii;

        // Output point 10: X(9)
        out_r[(out_stride * 9)] = tvrr + tvri;
        out_i[(out_stride * 9)] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = v1r;
        tvrr -= (CRTM_11_5 * v211r);
        tvrr -= (CRTM_11_9 * v310r);
        tvrr += (CRTM_11_3 * v49r);
        tvrr += (CRTM_11_1 * v58r);
        tvrr -= (CRTM_11_7 * v67r);
        tvri = (CRTM_11_6 * v112i);
        tvri -= (CRTM_11_10 * v103i);
        tvri -= (CRTM_11_4 * v94i);
        tvri += (CRTM_11_2 * v85i);
        tvri += (CRTM_11_8 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_5 * v211i);
        tvii -= (CRTM_11_9 * v310i);
        tvii += (CRTM_11_3 * v49i);
        tvii += (CRTM_11_1 * v58i);
        tvii -= (CRTM_11_7 * v67i);
        tvir = (CRTM_11_6 * v112r);
        tvir -= (CRTM_11_10 * v103r);
        tvir -= (CRTM_11_4 * v94r);
        tvir += (CRTM_11_2 * v85r);
        tvir += (CRTM_11_8 * v76r);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvir + tvii;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = v1r;
        tvrr -= (CRTM_11_7 * v211r);
        tvrr -= (CRTM_11_5 * v310r);
        tvrr += (CRTM_11_1 * v49r);
        tvrr -= (CRTM_11_9 * v58r);
        tvrr += (CRTM_11_3 * v67r);
        tvri = (CRTM_11_8 * v112i);
        tvri -= (CRTM_11_6 * v103i);
        tvri += (CRTM_11_2 * v94i);
        tvri += (CRTM_11_10 * v85i);
        tvri -= (CRTM_11_4 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_7 * v211i);
        tvii -= (CRTM_11_5 * v310i);
        tvii += (CRTM_11_1 * v49i);
        tvii -= (CRTM_11_9 * v58i);
        tvii += (CRTM_11_3 * v67i);
        tvir = (CRTM_11_8 * v112r);
        tvir -= (CRTM_11_6 * v103r);
        tvir += (CRTM_11_2 * v94r);
        tvir += (CRTM_11_10 * v85r);
        tvir -= (CRTM_11_4 * v76r);

        out_r[(out_stride << 2)] = tvrr - tvri;
        out_i[(out_stride << 2)] = tvir + tvii;

        // Output point 8: X(7)
        out_r[(out_stride * 7)] = tvrr + tvri;
        out_i[(out_stride * 7)] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= (CRTM_11_9 * v211r);
        tvrr += (CRTM_11_1 * v310r);
        tvrr -= (CRTM_11_7 * v49r);
        tvrr += (CRTM_11_3 * v58r);
        tvrr -= (CRTM_11_5 * v67r);
        tvri = (CRTM_11_10 * v112i);
        tvri -= (CRTM_11_2 * v103i);
        tvri += (CRTM_11_8 * v94i);
        tvri -= (CRTM_11_4 * v85i);
        tvri += (CRTM_11_6 * v76i);

        tvii = v1i;
        tvii -= (CRTM_11_9 * v211i);
        tvii += (CRTM_11_1 * v310i);
        tvii -= (CRTM_11_7 * v49i);
        tvii += (CRTM_11_3 * v58i);
        tvii -= (CRTM_11_5 * v67i);
        tvir = (CRTM_11_10 * v112r);
        tvir -= (CRTM_11_2 * v103r);
        tvir += (CRTM_11_8 * v94r);
        tvir -= (CRTM_11_4 * v85r);
        tvir += (CRTM_11_6 * v76r);

        out_r[(out_stride * 5)] = tvrr - tvri;
        out_i[(out_stride * 5)] = tvir + tvii;

        // Output point 7: X(6)
        out_r[(out_stride * 6)] = tvrr + tvri;
        out_i[(out_stride * 6)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "utils/complex_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 184, 768, 110, 0, 661}, {0, 184, 768, 110, 0, 661}};

ops_cycles_t get_ops_cnt_fft11c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

const DOUBLE CRTM_11[RADIX_11][2] = {{1.0, 0.0},
                                     {0.841253532831181, -0.540640817455598},
                                     {0.415415013001886, -0.909631995354518},
                                     {-0.142314838273285, -0.989821441880933},
                                     {-0.654860733945285, -0.755749574354258},
                                     {-0.959492973614497, -0.28173255684143},
                                     {-0.959492973614497, 0.281732556841429},
                                     {-0.654860733945285, 0.755749574354258},
                                     {-0.142314838273285, 0.989821441880933},
                                     {0.415415013001887, 0.909631995354518},
                                     {0.841253532831181, 0.540640817455598}};

VOID fft11c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
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
    DOUBLE local_in[RADIX_11][2] = {0};

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
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[9]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[10]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 11i ********************/
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
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(CRTM_11, 1) = CRTM_11
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_11[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_11[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_11[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_11[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_11[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_11[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_11[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_11[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_11[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_11[10], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

VOID fft11c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides, UINT8 flag)
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
    FLOAT local_in[RADIX_11][2] = {0};

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
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[9]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[10]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 11i ********************/
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
        CADD(local_in[9], temp_out, temp_out);
        CADD(local_in[10], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_11[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_11[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_11[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_11[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_11[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_11[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_11[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_11[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_11[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_11[10], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 11i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_11[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_11[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_11[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_11[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_11[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_11[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_11[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_11[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_11[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_11[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[10], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}
#endif // USE_OPT_KERNEL_VARIANT
