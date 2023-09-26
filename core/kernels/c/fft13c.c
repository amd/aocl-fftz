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

/** @file fft13c.c
 *
 *  @brief Radix-13 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-13 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 */

#include "core/kernels/kernel.h"

kfft_ register_kernel_fft13c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft13c_fp32;
    else if (precision == DT_DOUBLE)
        return fft13c_fp64;
    else
        return NULL;
}

#ifdef USE_OPT_KERNEL_VARIANT
/* --------------- optimized C kernel variant --------------- */
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 144, 192, 52, 0, 0},
                                                     {0, 144, 192, 52, 0, 0}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    const DOUBLE CRTM_13_1 =
        +0.88545602565320988343196542919889831911670203027585;
    const DOUBLE CRTM_13_2 =
        +0.46472317204376856941258593703890334798403126280993;
    const DOUBLE CRTM_13_3 =
        +0.56806474673115575835089217112990548833844284643239;
    const DOUBLE CRTM_13_4 =
        +0.82298386589365642506169642271400044432574194139137;
    const DOUBLE CRTM_13_5 =
        +0.12053668025532297344665207990901285783403477592419;
    const DOUBLE CRTM_13_6 =
        +0.99270887409805400250266141680975546593643207268598;
    const DOUBLE CRTM_13_7 =
        +0.35460488704253572631467475379547036937080832292513;
    const DOUBLE CRTM_13_8 =
        +0.93501624268541478538393183812800043375341122426674;
    const DOUBLE CRTM_13_9 =
        +0.74851074817110118759173175330956360908768299853347;
    const DOUBLE CRTM_13_10 =
        +0.66312265824079510196497553240307933438635557099851;
    const DOUBLE CRTM_13_11 =
        +0.97094181742605206568166996315141194600248673801847;
    const DOUBLE CRTM_13_12 =
        +0.23931566428755761084795164627980653908250713845138;

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
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v213r, v312r, v411r, v510r, v69r, v78r, v132i, v123i, v114i,
            v510i, v96i, v87i, v213i, v312i, v411i, v105i, v69i, v78i, v132r,
            v123r, v114r, v105r, v96r, v87r, tvrr, tvri, tvii, tvir;

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
        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        // Input point 6: x(5)
        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        // Input point 7: x(6)
        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        // Input point 8: x(7)
        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        // Input point 10: x(9)
        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        // Input point 11: x(10)
        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // Input point 12: x(11)
        v12r = in_r[in_stride * 11];
        v12i = in_i[in_stride * 11];

        // Input point 13: x(12)
        v13r = in_r[in_stride * 12];
        v13i = in_i[in_stride * 12];

        v213r = v2r + v13r;
        v312r = v3r + v12r;
        v411r = v4r + v11r;
        v510r = v5r + v10r;
        v69r = v6r + v9r;
        v78r = v7r + v8r;

        v132i = v13i - v2i;
        v123i = v12i - v3i;
        v114i = v11i - v4i;
        v510i = v10i - v5i;
        v96i = v9i - v6i;
        v87i = v8i - v7i;

        v213i = v2i + v13i;
        v312i = v3i + v12i;
        v411i = v4i + v11i;
        v105i = v5i + v10i;
        v69i = v6i + v9i;
        v78i = v7i + v8i;

        v132r = v13r - v2r;
        v123r = v12r - v3r;
        v114r = v11r - v4r;
        v105r = v10r - v5r;
        v96r = v9r - v6r;
        v87r = v8r - v7r;

        // Output point 1: X(0)
        *out_r = v1r + v213r + v312r + v411r + v510r + v69r + v78r;
        *out_i = v1i + v213i + v312i + v411i + v105i + v69i + v78i;

        // Output point 2: X(1)
        tvrr = v1r;
        tvrr += (CRTM_13_1 * v213r);
        tvrr += (CRTM_13_3 * v312r);
        tvrr += (CRTM_13_5 * v411r);
        tvrr -= (CRTM_13_7 * v510r);
        tvrr -= (CRTM_13_9 * v69r);
        tvrr -= (CRTM_13_11 * v78r);
        tvri = (CRTM_13_2 * v132i);
        tvri += (CRTM_13_4 * v123i);
        tvri += (CRTM_13_6 * v114i);
        tvri += (CRTM_13_8 * v510i);
        tvri += (CRTM_13_10 * v96i);
        tvri += (CRTM_13_12 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_1 * v213i);
        tvii += (CRTM_13_3 * v312i);
        tvii += (CRTM_13_5 * v411i);
        tvii -= (CRTM_13_7 * v105i);
        tvii -= (CRTM_13_9 * v69i);
        tvii -= (CRTM_13_11 * v78i);
        tvir = (CRTM_13_2 * v132r);
        tvir += (CRTM_13_4 * v123r);
        tvir += (CRTM_13_6 * v114r);
        tvir += (CRTM_13_8 * v105r);
        tvir += (CRTM_13_10 * v96r);
        tvir += (CRTM_13_12 * v87r);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvir + tvii;

        // Output point 13: X(12)
        out_r[out_stride * 12] = tvrr + tvri;
        out_i[out_stride * 12] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v1r;
        tvrr += (CRTM_13_3 * v213r);
        tvrr -= (CRTM_13_7 * v312r);
        tvrr -= (CRTM_13_11 * v411r);
        tvrr -= (CRTM_13_9 * v510r);
        tvrr += (CRTM_13_5 * v69r);
        tvrr += (CRTM_13_1 * v78r);
        tvri = (CRTM_13_4 * v132i);
        tvri += (CRTM_13_8 * v123i);
        tvri += (CRTM_13_12 * v114i);
        tvri -= (CRTM_13_10 * v510i);
        tvri -= (CRTM_13_6 * v96i);
        tvri -= (CRTM_13_2 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_3 * v213i);
        tvii -= (CRTM_13_7 * v312i);
        tvii -= (CRTM_13_11 * v411i);
        tvii -= (CRTM_13_9 * v105i);
        tvii += (CRTM_13_5 * v69i);
        tvii += (CRTM_13_1 * v78i);
        tvir = (CRTM_13_4 * v132r);
        tvir += (CRTM_13_8 * v123r);
        tvir += (CRTM_13_12 * v114r);
        tvir -= (CRTM_13_10 * v105r);
        tvir -= (CRTM_13_6 * v96r);
        tvir -= (CRTM_13_2 * v87r);

        out_r[(out_stride << 1)] = tvrr - tvri;
        out_i[(out_stride << 1)] = tvir + tvii;

        // Output point 12: X(11)
        out_r[(out_stride * 11)] = tvrr + tvri;
        out_i[(out_stride * 11)] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = v1r;
        tvrr += (CRTM_13_5 * v213r);
        tvrr -= (CRTM_13_11 * v312r);
        tvrr -= (CRTM_13_7 * v411r);
        tvrr += (CRTM_13_1 * v510r);
        tvrr += (CRTM_13_3 * v69r);
        tvrr -= (CRTM_13_9 * v78r);
        tvri = (CRTM_13_6 * v132i);
        tvri += (CRTM_13_12 * v123i);
        tvri -= (CRTM_13_8 * v114i);
        tvri -= (CRTM_13_2 * v510i);
        tvri += (CRTM_13_4 * v96i);
        tvri += (CRTM_13_10 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_5 * v213i);
        tvii -= (CRTM_13_11 * v312i);
        tvii -= (CRTM_13_7 * v411i);
        tvii += (CRTM_13_1 * v105i);
        tvii += (CRTM_13_3 * v69i);
        tvii -= (CRTM_13_9 * v78i);
        tvir = (CRTM_13_6 * v132r);
        tvir += (CRTM_13_12 * v123r);
        tvir -= (CRTM_13_8 * v114r);
        tvir -= (CRTM_13_2 * v105r);
        tvir += (CRTM_13_4 * v96r);
        tvir += (CRTM_13_10 * v87r);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvir + tvii;

        // Output point 11: X(10)
        out_r[(out_stride * 10)] = tvrr + tvri;
        out_i[(out_stride * 10)] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = v1r;
        tvrr -= (CRTM_13_7 * v213r);
        tvrr -= (CRTM_13_9 * v312r);
        tvrr += (CRTM_13_1 * v411r);
        tvrr += (CRTM_13_5 * v510r);
        tvrr -= (CRTM_13_11 * v69r);
        tvrr += (CRTM_13_3 * v78r);
        tvri = (CRTM_13_8 * v132i);
        tvri -= (CRTM_13_10 * v123i);
        tvri -= (CRTM_13_2 * v114i);
        tvri += (CRTM_13_6 * v510i);
        tvri -= (CRTM_13_12 * v96i);
        tvri -= (CRTM_13_4 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_7 * v213i);
        tvii -= (CRTM_13_9 * v312i);
        tvii += (CRTM_13_1 * v411i);
        tvii += (CRTM_13_5 * v105i);
        tvii -= (CRTM_13_11 * v69i);
        tvii += (CRTM_13_3 * v78i);
        tvir = (CRTM_13_8 * v132r);
        tvir -= (CRTM_13_10 * v123r);
        tvir -= (CRTM_13_2 * v114r);
        tvir += (CRTM_13_6 * v105r);
        tvir -= (CRTM_13_12 * v96r);
        tvir -= (CRTM_13_4 * v87r);

        out_r[(out_stride << 2)] = tvrr - tvri;
        out_i[(out_stride << 2)] = tvir + tvii;

        // Output point 10: X(9)
        out_r[(out_stride * 9)] = tvrr + tvri;
        out_i[(out_stride * 9)] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= (CRTM_13_9 * v213r);
        tvrr += (CRTM_13_5 * v312r);
        tvrr += (CRTM_13_3 * v411r);
        tvrr -= (CRTM_13_11 * v510r);
        tvrr += (CRTM_13_1 * v69r);
        tvrr -= (CRTM_13_7 * v78r);
        tvri = (CRTM_13_10 * v132i);
        tvri -= (CRTM_13_6 * v123i);
        tvri += (CRTM_13_4 * v114i);
        tvri -= (CRTM_13_12 * v510i);
        tvri -= (CRTM_13_2 * v96i);
        tvri += (CRTM_13_8 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_9 * v213i);
        tvii += (CRTM_13_5 * v312i);
        tvii += (CRTM_13_3 * v411i);
        tvii -= (CRTM_13_11 * v105i);
        tvii += (CRTM_13_1 * v69i);
        tvii -= (CRTM_13_7 * v78i);
        tvir = (CRTM_13_10 * v132r);
        tvir -= (CRTM_13_6 * v123r);
        tvir += (CRTM_13_4 * v114r);
        tvir -= (CRTM_13_12 * v105r);
        tvir -= (CRTM_13_2 * v96r);
        tvir += (CRTM_13_8 * v87r);

        out_r[(out_stride * 5)] = tvrr - tvri;
        out_i[(out_stride * 5)] = tvir + tvii;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = v1r;
        tvrr -= (CRTM_13_11 * v213r);
        tvrr += (CRTM_13_1 * v312r);
        tvrr -= (CRTM_13_9 * v411r);
        tvrr += (CRTM_13_3 * v510r);
        tvrr -= (CRTM_13_7 * v69r);
        tvrr += (CRTM_13_5 * v78r);
        tvri = (CRTM_13_12 * v132i);
        tvri -= (CRTM_13_2 * v123i);
        tvri += (CRTM_13_10 * v114i);
        tvri -= (CRTM_13_4 * v510i);
        tvri += (CRTM_13_8 * v96i);
        tvri -= (CRTM_13_6 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_11 * v213i);
        tvii += (CRTM_13_1 * v312i);
        tvii -= (CRTM_13_9 * v411i);
        tvii += (CRTM_13_3 * v105i);
        tvii -= (CRTM_13_7 * v69i);
        tvii += (CRTM_13_5 * v78i);
        tvir = (CRTM_13_12 * v132r);
        tvir -= (CRTM_13_2 * v123r);
        tvir += (CRTM_13_10 * v114r);
        tvir -= (CRTM_13_4 * v105r);
        tvir += (CRTM_13_8 * v96r);
        tvir -= (CRTM_13_6 * v87r);

        out_r[(out_stride * 6)] = tvrr - tvri;
        out_i[(out_stride * 6)] = tvir + tvii;

        // Output point 8: X(7)
        out_r[(out_stride * 7)] = tvrr + tvri;
        out_i[(out_stride * 7)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}

VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
                 INTP n, aoclfftz_strides_t *strides)
{
    const FLOAT CRTM_13_1 =
        +0.88545602565320988343196542919889831911670203027585;
    const FLOAT CRTM_13_2 =
        +0.46472317204376856941258593703890334798403126280993;
    const FLOAT CRTM_13_3 =
        +0.56806474673115575835089217112990548833844284643239;
    const FLOAT CRTM_13_4 =
        +0.82298386589365642506169642271400044432574194139137;
    const FLOAT CRTM_13_5 =
        +0.12053668025532297344665207990901285783403477592419;
    const FLOAT CRTM_13_6 =
        +0.99270887409805400250266141680975546593643207268598;
    const FLOAT CRTM_13_7 =
        +0.35460488704253572631467475379547036937080832292513;
    const FLOAT CRTM_13_8 =
        +0.93501624268541478538393183812800043375341122426674;
    const FLOAT CRTM_13_9 =
        +0.74851074817110118759173175330956360908768299853347;
    const FLOAT CRTM_13_10 =
        +0.66312265824079510196497553240307933438635557099851;
    const FLOAT CRTM_13_11 =
        +0.97094181742605206568166996315141194600248673801847;
    const FLOAT CRTM_13_12 =
        +0.23931566428755761084795164627980653908250713845138;

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
            v7i, v8r, v8i, v9r, v9i, v10r, v10i, v11r, v11i, v12r, v12i, v13r,
            v13i, v213r, v312r, v411r, v510r, v69r, v78r, v132i, v123i, v114i,
            v510i, v96i, v87i, v213i, v312i, v411i, v105i, v69i, v78i, v132r,
            v123r, v114r, v105r, v96r, v87r, tvrr, tvri, tvii, tvir;

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
        v4r = in_r[in_stride * 3];
        v4i = in_i[in_stride * 3];

        // Input point 5: x(4)
        v5r = in_r[(in_stride << 2)];
        v5i = in_i[(in_stride << 2)];

        // Input point 6: x(5)
        v6r = in_r[in_stride * 5];
        v6i = in_i[in_stride * 5];

        // Input point 7: x(6)
        v7r = in_r[in_stride * 6];
        v7i = in_i[in_stride * 6];

        // Input point 8: x(7)
        v8r = in_r[in_stride * 7];
        v8i = in_i[in_stride * 7];

        // Input point 9: x(8)
        v9r = in_r[(in_stride << 3)];
        v9i = in_i[(in_stride << 3)];

        // Input point 10: x(9)
        v10r = in_r[in_stride * 9];
        v10i = in_i[in_stride * 9];

        // Input point 11: x(10)
        v11r = in_r[in_stride * 10];
        v11i = in_i[in_stride * 10];

        // Input point 12: x(11)
        v12r = in_r[in_stride * 11];
        v12i = in_i[in_stride * 11];

        // Input point 13: x(12)
        v13r = in_r[in_stride * 12];
        v13i = in_i[in_stride * 12];

        v213r = v2r + v13r;
        v312r = v3r + v12r;
        v411r = v4r + v11r;
        v510r = v5r + v10r;
        v69r = v6r + v9r;
        v78r = v7r + v8r;

        v132i = v13i - v2i;
        v123i = v12i - v3i;
        v114i = v11i - v4i;
        v510i = v10i - v5i;
        v96i = v9i - v6i;
        v87i = v8i - v7i;

        v213i = v2i + v13i;
        v312i = v3i + v12i;
        v411i = v4i + v11i;
        v105i = v5i + v10i;
        v69i = v6i + v9i;
        v78i = v7i + v8i;

        v132r = v13r - v2r;
        v123r = v12r - v3r;
        v114r = v11r - v4r;
        v105r = v10r - v5r;
        v96r = v9r - v6r;
        v87r = v8r - v7r;

        // Output point 1: X(0)
        *out_r = v1r + v213r + v312r + v411r + v510r + v69r + v78r;
        *out_i = v1i + v213i + v312i + v411i + v105i + v69i + v78i;

        // Output point 2: X(1)
        tvrr = v1r;
        tvrr += (CRTM_13_1 * v213r);
        tvrr += (CRTM_13_3 * v312r);
        tvrr += (CRTM_13_5 * v411r);
        tvrr -= (CRTM_13_7 * v510r);
        tvrr -= (CRTM_13_9 * v69r);
        tvrr -= (CRTM_13_11 * v78r);
        tvri = (CRTM_13_2 * v132i);
        tvri += (CRTM_13_4 * v123i);
        tvri += (CRTM_13_6 * v114i);
        tvri += (CRTM_13_8 * v510i);
        tvri += (CRTM_13_10 * v96i);
        tvri += (CRTM_13_12 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_1 * v213i);
        tvii += (CRTM_13_3 * v312i);
        tvii += (CRTM_13_5 * v411i);
        tvii -= (CRTM_13_7 * v105i);
        tvii -= (CRTM_13_9 * v69i);
        tvii -= (CRTM_13_11 * v78i);
        tvir = (CRTM_13_2 * v132r);
        tvir += (CRTM_13_4 * v123r);
        tvir += (CRTM_13_6 * v114r);
        tvir += (CRTM_13_8 * v105r);
        tvir += (CRTM_13_10 * v96r);
        tvir += (CRTM_13_12 * v87r);

        out_r[out_stride] = tvrr - tvri;
        out_i[out_stride] = tvir + tvii;

        // Output point 13: X(12)
        out_r[out_stride * 12] = tvrr + tvri;
        out_i[out_stride * 12] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v1r;
        tvrr += (CRTM_13_3 * v213r);
        tvrr -= (CRTM_13_7 * v312r);
        tvrr -= (CRTM_13_11 * v411r);
        tvrr -= (CRTM_13_9 * v510r);
        tvrr += (CRTM_13_5 * v69r);
        tvrr += (CRTM_13_1 * v78r);
        tvri = (CRTM_13_4 * v132i);
        tvri += (CRTM_13_8 * v123i);
        tvri += (CRTM_13_12 * v114i);
        tvri -= (CRTM_13_10 * v510i);
        tvri -= (CRTM_13_6 * v96i);
        tvri -= (CRTM_13_2 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_3 * v213i);
        tvii -= (CRTM_13_7 * v312i);
        tvii -= (CRTM_13_11 * v411i);
        tvii -= (CRTM_13_9 * v105i);
        tvii += (CRTM_13_5 * v69i);
        tvii += (CRTM_13_1 * v78i);
        tvir = (CRTM_13_4 * v132r);
        tvir += (CRTM_13_8 * v123r);
        tvir += (CRTM_13_12 * v114r);
        tvir -= (CRTM_13_10 * v105r);
        tvir -= (CRTM_13_6 * v96r);
        tvir -= (CRTM_13_2 * v87r);

        out_r[(out_stride << 1)] = tvrr - tvri;
        out_i[(out_stride << 1)] = tvir + tvii;

        // Output point 12: X(11)
        out_r[(out_stride * 11)] = tvrr + tvri;
        out_i[(out_stride * 11)] = tvii - tvir;

        // Output point 4: X(3)
        tvrr = v1r;
        tvrr += (CRTM_13_5 * v213r);
        tvrr -= (CRTM_13_11 * v312r);
        tvrr -= (CRTM_13_7 * v411r);
        tvrr += (CRTM_13_1 * v510r);
        tvrr += (CRTM_13_3 * v69r);
        tvrr -= (CRTM_13_9 * v78r);
        tvri = (CRTM_13_6 * v132i);
        tvri += (CRTM_13_12 * v123i);
        tvri -= (CRTM_13_8 * v114i);
        tvri -= (CRTM_13_2 * v510i);
        tvri += (CRTM_13_4 * v96i);
        tvri += (CRTM_13_10 * v87i);

        tvii = v1i;
        tvii += (CRTM_13_5 * v213i);
        tvii -= (CRTM_13_11 * v312i);
        tvii -= (CRTM_13_7 * v411i);
        tvii += (CRTM_13_1 * v105i);
        tvii += (CRTM_13_3 * v69i);
        tvii -= (CRTM_13_9 * v78i);
        tvir = (CRTM_13_6 * v132r);
        tvir += (CRTM_13_12 * v123r);
        tvir -= (CRTM_13_8 * v114r);
        tvir -= (CRTM_13_2 * v105r);
        tvir += (CRTM_13_4 * v96r);
        tvir += (CRTM_13_10 * v87r);

        out_r[out_stride * 3] = tvrr - tvri;
        out_i[out_stride * 3] = tvir + tvii;

        // Output point 11: X(10)
        out_r[(out_stride * 10)] = tvrr + tvri;
        out_i[(out_stride * 10)] = tvii - tvir;

        // Output point 5: X(4)
        tvrr = v1r;
        tvrr -= (CRTM_13_7 * v213r);
        tvrr -= (CRTM_13_9 * v312r);
        tvrr += (CRTM_13_1 * v411r);
        tvrr += (CRTM_13_5 * v510r);
        tvrr -= (CRTM_13_11 * v69r);
        tvrr += (CRTM_13_3 * v78r);
        tvri = (CRTM_13_8 * v132i);
        tvri -= (CRTM_13_10 * v123i);
        tvri -= (CRTM_13_2 * v114i);
        tvri += (CRTM_13_6 * v510i);
        tvri -= (CRTM_13_12 * v96i);
        tvri -= (CRTM_13_4 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_7 * v213i);
        tvii -= (CRTM_13_9 * v312i);
        tvii += (CRTM_13_1 * v411i);
        tvii += (CRTM_13_5 * v105i);
        tvii -= (CRTM_13_11 * v69i);
        tvii += (CRTM_13_3 * v78i);
        tvir = (CRTM_13_8 * v132r);
        tvir -= (CRTM_13_10 * v123r);
        tvir -= (CRTM_13_2 * v114r);
        tvir += (CRTM_13_6 * v105r);
        tvir -= (CRTM_13_12 * v96r);
        tvir -= (CRTM_13_4 * v87r);

        out_r[(out_stride << 2)] = tvrr - tvri;
        out_i[(out_stride << 2)] = tvir + tvii;

        // Output point 10: X(9)
        out_r[(out_stride * 9)] = tvrr + tvri;
        out_i[(out_stride * 9)] = tvii - tvir;

        // Output point 6: X(5)
        tvrr = v1r;
        tvrr -= (CRTM_13_9 * v213r);
        tvrr += (CRTM_13_5 * v312r);
        tvrr += (CRTM_13_3 * v411r);
        tvrr -= (CRTM_13_11 * v510r);
        tvrr += (CRTM_13_1 * v69r);
        tvrr -= (CRTM_13_7 * v78r);
        tvri = (CRTM_13_10 * v132i);
        tvri -= (CRTM_13_6 * v123i);
        tvri += (CRTM_13_4 * v114i);
        tvri -= (CRTM_13_12 * v510i);
        tvri -= (CRTM_13_2 * v96i);
        tvri += (CRTM_13_8 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_9 * v213i);
        tvii += (CRTM_13_5 * v312i);
        tvii += (CRTM_13_3 * v411i);
        tvii -= (CRTM_13_11 * v105i);
        tvii += (CRTM_13_1 * v69i);
        tvii -= (CRTM_13_7 * v78i);
        tvir = (CRTM_13_10 * v132r);
        tvir -= (CRTM_13_6 * v123r);
        tvir += (CRTM_13_4 * v114r);
        tvir -= (CRTM_13_12 * v105r);
        tvir -= (CRTM_13_2 * v96r);
        tvir += (CRTM_13_8 * v87r);

        out_r[(out_stride * 5)] = tvrr - tvri;
        out_i[(out_stride * 5)] = tvir + tvii;

        // Output point 9: X(8)
        out_r[(out_stride << 3)] = tvrr + tvri;
        out_i[(out_stride << 3)] = tvii - tvir;

        // Output point 7: X(6)
        tvrr = v1r;
        tvrr -= (CRTM_13_11 * v213r);
        tvrr += (CRTM_13_1 * v312r);
        tvrr -= (CRTM_13_9 * v411r);
        tvrr += (CRTM_13_3 * v510r);
        tvrr -= (CRTM_13_7 * v69r);
        tvrr += (CRTM_13_5 * v78r);
        tvri = (CRTM_13_12 * v132i);
        tvri -= (CRTM_13_2 * v123i);
        tvri += (CRTM_13_10 * v114i);
        tvri -= (CRTM_13_4 * v510i);
        tvri += (CRTM_13_8 * v96i);
        tvri -= (CRTM_13_6 * v87i);

        tvii = v1i;
        tvii -= (CRTM_13_11 * v213i);
        tvii += (CRTM_13_1 * v312i);
        tvii -= (CRTM_13_9 * v411i);
        tvii += (CRTM_13_3 * v105i);
        tvii -= (CRTM_13_7 * v69i);
        tvii += (CRTM_13_5 * v78i);
        tvir = (CRTM_13_12 * v132r);
        tvir -= (CRTM_13_2 * v123r);
        tvir += (CRTM_13_10 * v114r);
        tvir -= (CRTM_13_4 * v105r);
        tvir += (CRTM_13_8 * v96r);
        tvir -= (CRTM_13_6 * v87r);

        out_r[(out_stride * 6)] = tvrr - tvri;
        out_i[(out_stride * 6)] = tvir + tvii;

        // Output point 8: X(7)
        out_r[(out_stride * 7)] = tvrr + tvri;
        out_i[(out_stride * 7)] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
}
#else
/* --------------- non-optimized C kernel variant --------------- */
#include "core/kernels/kernel_utils.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {
    {0, 268, 1112, 130, 0, 1093}, {0, 268, 1112, 130, 0, 1093}};

ops_cycles_t get_ops_cnt_fft13c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

const DOUBLE CRTM_13[RADIX_13][2] = {{1.0, 0.0},
                                     {0.88545602565321, -0.464723172043769},
                                     {0.568064746731156, -0.822983865893656},
                                     {0.120536680255323, -0.992708874098054},
                                     {-0.354604887042536, -0.935016242685415},
                                     {-0.748510748171101, -0.663122658240795},
                                     {-0.970941817426052, -0.239315664287558},
                                     {-0.970941817426052, 0.239315664287557},
                                     {-0.748510748171101, 0.663122658240795},
                                     {-0.354604887042536, 0.935016242685415},
                                     {0.120536680255323, 0.992708874098054},
                                     {0.568064746731156, 0.822983865893657},
                                     {0.88545602565321, 0.464723172043768}};

VOID fft13c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
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
    DOUBLE local_in[RADIX_13][2] = {0};

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
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[11]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[12]);

        output_r = out_dr;
        output_i = out_di;
        /******************** Output 13i ********************/
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
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_13[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_13[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_13[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_13[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_13[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_13[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_13[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_13[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_13[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_13[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_13[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_13[12], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_dr += v_in_stride;
        in_di += v_in_stride;
        out_dr += v_out_stride;
        out_di += v_out_stride;
    }
}

VOID fft13c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real, VOID *out_imag,
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
    FLOAT local_in[RADIX_13][2] = {0};

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
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[11]);
        input_r += in_stride;
        input_i += in_stride;
        LOAD_INPUT(input_r, input_i, local_in[12]);

        output_r = out_fr;
        output_i = out_fi;
        /******************** Output 13i ********************/
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
        CADD(local_in[11], temp_out, temp_out);
        CADD(local_in[12], temp_out, temp_out);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+1 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        // Using CMUL_CADD since pow(constant_multiplier, 1) =
        // constant_multiplier
        CADD(local_in[0], temp_out, temp_out);
        CMUL_CADD(local_in[1], CRTM_13[1], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[2], CRTM_13[2], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[3], CRTM_13[3], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[4], CRTM_13[4], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[5], CRTM_13[5], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[6], CRTM_13[6], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[7], CRTM_13[7], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[8], CRTM_13[8], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[9], CRTM_13[9], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[10], CRTM_13[10], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[11], CRTM_13[11], pow_cm, temp_out, cmul_temp);
        CMUL_CADD(local_in[12], CRTM_13[12], pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+2 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 2, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 2, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 2, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+3 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 3, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 3, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 3, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+4 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 4, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 4, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 4, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+5 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 5, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 5, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 5, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+6 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 6, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 6, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 6, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+7 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 7, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 7, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 7, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+8 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 8, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 8, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 8, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+9 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 9, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 9, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 9, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+10 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 10, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 10, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 10, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+11 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 11, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 11, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 11, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);
        output_r += out_stride;
        output_i += out_stride;

        /******************** Output 13i+12 ********************/
        pow_cm[0] = pow_cm[1] = temp_out[0] = temp_out[1] = 0.0;
        CADD(local_in[0], temp_out, temp_out);
        CPOW(CRTM_13[1], 12, pow_cm, cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[1], CRTM_13[2], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[2], CRTM_13[3], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[3], CRTM_13[4], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[4], CRTM_13[5], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[5], CRTM_13[6], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[6], CRTM_13[7], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[7], CRTM_13[8], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[8], CRTM_13[9], 12, pow_cm, temp_out, cmul_temp,
                       cpow_temp);
        CMUL_CADD_CPOW(local_in[9], CRTM_13[10], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[10], CRTM_13[11], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD_CPOW(local_in[11], CRTM_13[12], 12, pow_cm, temp_out,
                       cmul_temp, cpow_temp);
        CMUL_CADD(local_in[12], pow_cm, pow_cm, temp_out, cmul_temp);
        STORE_OUTPUT(temp_out, output_r, output_i);

        // next set
        in_fr += v_in_stride;
        in_fi += v_in_stride;
        out_fr += v_out_stride;
        out_fi += v_out_stride;
    }
}
#endif // USE_OPT_KERNEL_VARIANT