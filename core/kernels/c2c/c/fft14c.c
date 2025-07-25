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

/** @file fft14c.c
 *
 *  @brief Radix-14 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-14 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 72, 148, 56, 0, 0},
                                                     {0, 72, 148, 56, 0, 0}};

ops_cycles_t get_ops_cnt_fft14c(UINT8 precision, UINT8 direction)
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

static VOID fft14c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000;
    const DOUBLE CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829;
    const DOUBLE CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372;
    const DOUBLE CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327;
    const DOUBLE CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720;
    const DOUBLE CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r, v313r,
               v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i,
               v97i, v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r,
               v124r, v115r, v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i,
               cv2r, cv2i, cv3r, cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii,
               tv1ri, tv2ri, tv1ir, tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir,
               tv4ir, tv5ir, tv6ir;

        v1r = in_r[in_strides[1]];
        v2r = in_r[in_strides[13]];

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        v1r = in_r[in_strides[3]];
        v2r = in_r[in_strides[11]];

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        v1r = in_r[in_strides[5]];
        v2r = in_r[in_strides[9]];

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        v8r = in_r[in_strides[7]];

        v3r = in_r[in_strides[2]];
        v4r = in_r[in_strides[12]];

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        v3r = in_r[in_strides[4]];
        v4r = in_r[in_strides[10]];

        v511r = v3r + v4r;
        v115r = v3r - v4r;

        tv1rr = (CRTM_14_1 * v214r) + (CRTM_14_5 * v412r) -
                (CRTM_14_3 * v610r) - v8r;
        tv3rr = (CRTM_14_3 * v214r) - (CRTM_14_1 * v412r) -
                (CRTM_14_5 * v610r) + v8r;
        tv5rr = (CRTM_14_5 * v214r) - (CRTM_14_3 * v412r) +
                (CRTM_14_1 * v610r) - v8r;

        tv1ir = (CRTM_14_2 * v142r) + (CRTM_14_6 * v124r) + (CRTM_14_4 * v106r);
        tv3ir = (CRTM_14_4 * v142r) + (CRTM_14_2 * v124r) - (CRTM_14_6 * v106r);
        tv5ir = (CRTM_14_6 * v142r) - (CRTM_14_4 * v124r) + (CRTM_14_2 * v106r);

        v3r = in_r[in_strides[6]];
        v4r = in_r[in_strides[8]];

        v79r = v3r + v4r;
        v97r = v4r - v3r;

        v1r = *in_r;

        tv2ir = (CRTM_14_4 * v133r) - (CRTM_14_6 * v115r) + (CRTM_14_2 * v97r);
        tv4ir = (CRTM_14_2 * v115r) + (CRTM_14_6 * v133r) - (CRTM_14_4 * v97r);
        tv6ir = (CRTM_14_4 * v115r) + (CRTM_14_2 * v133r) + (CRTM_14_6 * v97r);

        tvrr = v1r + v313r + v511r + v79r;
        tvri = v214r + v412r + v610r + v8r;

        *out_r = tvrr + tvri;

        out_r[out_strides[7]] = tvrr - tvri;

        cv1r = v1r + (CRTM_14_3 * v313r) - (CRTM_14_5 * v511r) -
               (CRTM_14_1 * v79r);
        cv2r = v1r + (CRTM_14_3 * v79r) -
               ((CRTM_14_1 * v511r) + (CRTM_14_5 * v313r));
        cv3r = v1r + (CRTM_14_3 * v511r) - (CRTM_14_1 * v313r) -
               (CRTM_14_5 * v79r);

        //-------------------------------------------

        v1i = in_i[in_strides[1]];
        v2i = in_i[in_strides[13]];

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v1i = in_i[in_strides[3]];
        v2i = in_i[in_strides[11]];

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v1i = in_i[in_strides[5]];
        v2i = in_i[in_strides[9]];

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v8i = in_i[in_strides[7]];

        v3i = in_i[in_strides[2]];
        v4i = in_i[in_strides[12]];

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v3i = in_i[in_strides[4]];
        v4i = in_i[in_strides[10]];

        v115i = v3i + v4i;
        v511i = v3i - v4i;

        tv1ii = (CRTM_14_1 * v214i) + (CRTM_14_5 * v412i) -
                (CRTM_14_3 * v610i) - v8i;
        tv3ii = (CRTM_14_3 * v214i) - (CRTM_14_1 * v412i) -
                (CRTM_14_5 * v610i) + v8i;
        tv5ii = (CRTM_14_5 * v214i) - (CRTM_14_3 * v412i) +
                (CRTM_14_1 * v610i) - v8i;

        v3i = in_i[in_strides[6]];
        v4i = in_i[in_strides[8]];

        v79i = v3i + v4i;
        v97i = v4i - v3i;

        v1i = *in_i;

        tv1ri = (CRTM_14_2 * v142i) + (CRTM_14_6 * v124i) + (CRTM_14_4 * v106i);
        tv3ri = (CRTM_14_4 * v142i) + (CRTM_14_2 * v124i) - (CRTM_14_6 * v106i);
        tv5ri = (CRTM_14_6 * v142i) - (CRTM_14_4 * v124i) + (CRTM_14_2 * v106i);

        tvir = v214i + v412i + v610i + v8i;
        tvii = v1i + v313i + v115i + v79i;

        *out_i = tvir + tvii;

        out_i[out_strides[7]] = tvii - tvir;

        tv2ri = (CRTM_14_4 * v133i) - (CRTM_14_6 * v511i) + (CRTM_14_2 * v97i);
        tv4ri = (CRTM_14_6 * v133i) + (CRTM_14_2 * v511i) - (CRTM_14_4 * v97i);
        tv6ri = (CRTM_14_2 * v133i) + (CRTM_14_4 * v511i) + (CRTM_14_6 * v97i);

        cv1i = v1i + (CRTM_14_3 * v313i) - (CRTM_14_5 * v115i) -
               (CRTM_14_1 * v79i);
        cv2i = v1i + (CRTM_14_3 * v79i) -
               ((CRTM_14_5 * v313i) + (CRTM_14_1 * v115i));
        cv3i = v1i + (CRTM_14_3 * v115i) - (CRTM_14_1 * v313i) -
               (CRTM_14_5 * v79i);

        //------------------------------------------

        tvrr = cv1r + tv1rr;
        tvri = tv1ri + tv2ri;

        tvir = tv1ir + tv2ir;
        tvii = cv1i + tv1ii;

        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[13]] = tvrr + tvri;
        out_i[out_strides[1]] = tvir + tvii;

        out_i[out_strides[13]] = tvii - tvir;

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        out_r[out_strides[6]] = tvrr - tvri;
        out_r[out_strides[8]] = tvrr + tvri;
        out_i[out_strides[6]] = tvir + tvii;

        out_i[out_strides[8]] = tvii - tvir;

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[12]] = tvrr + tvri;
        out_i[out_strides[2]] = tvir + tvii;

        out_i[out_strides[12]] = tvii - tvir;

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        out_r[out_strides[5]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[5]] = tvir + tvii;

        out_i[out_strides[9]] = tvii - tvir;

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;
        out_i[out_strides[3]] = tvir + tvii;

        out_i[out_strides[11]] = tvii - tvir;

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        out_r[out_strides[4]] = tvrr - tvri;
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[4]] = tvir + tvii;

        out_i[out_strides[10]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft14c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                        VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                        VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14_1 =
        +0.90096886790241912623610231950744505116591916200000;
    const FLOAT CRTM_14_2 =
        +0.43388373911755809802961881825301518357930603231829;
    const FLOAT CRTM_14_3 =
        +0.62348980185873356948108200474179836074227404291372;
    const FLOAT CRTM_14_4 =
        +0.78183148246802977764200968763519351412805665195327;
    const FLOAT CRTM_14_5 =
        +0.22252093395631447715505298010340457043006139348720;
    const FLOAT CRTM_14_6 =
        +0.97492791218182360701813168299393121723278580100000;

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

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v8r, v8i, v214r, v313r,
              v412r, v511r, v610r, v79r, v142i, v133i, v124i, v511i, v106i,
              v97i, v214i, v313i, v412i, v115i, v610i, v79i, v142r, v133r,
              v124r, v115r, v106r, v97r, tvrr, tvri, tvii, tvir, cv1r, cv1i,
              cv2r, cv2i, cv3r, cv3i, tv1rr, tv3rr, tv5rr, tv1ii, tv3ii, tv5ii,
              tv1ri, tv2ri, tv1ir, tv2ir, tv3ri, tv4ri, tv5ri, tv6ri, tv3ir,
              tv4ir, tv5ir, tv6ir;

        v1r = in_r[in_strides[1]];
        v2r = in_r[in_strides[13]];

        v214r = v1r + v2r;
        v142r = v2r - v1r;

        v1r = in_r[in_strides[3]];
        v2r = in_r[in_strides[11]];

        v412r = v1r + v2r;
        v124r = v2r - v1r;

        v1r = in_r[in_strides[5]];
        v2r = in_r[in_strides[9]];

        v610r = v1r + v2r;
        v106r = v2r - v1r;

        v8r = in_r[in_strides[7]];

        v3r = in_r[in_strides[2]];
        v4r = in_r[in_strides[12]];

        v313r = v3r + v4r;
        v133r = v4r - v3r;

        v3r = in_r[in_strides[4]];
        v4r = in_r[in_strides[10]];

        v511r = v3r + v4r;
        v115r = v3r - v4r;

        tv1rr = (CRTM_14_1 * v214r) + (CRTM_14_5 * v412r) -
                (CRTM_14_3 * v610r) - v8r;
        tv3rr = (CRTM_14_3 * v214r) - (CRTM_14_1 * v412r) -
                (CRTM_14_5 * v610r) + v8r;
        tv5rr = (CRTM_14_5 * v214r) - (CRTM_14_3 * v412r) +
                (CRTM_14_1 * v610r) - v8r;

        tv1ir = (CRTM_14_2 * v142r) + (CRTM_14_6 * v124r) + (CRTM_14_4 * v106r);
        tv3ir = (CRTM_14_4 * v142r) + (CRTM_14_2 * v124r) - (CRTM_14_6 * v106r);
        tv5ir = (CRTM_14_6 * v142r) - (CRTM_14_4 * v124r) + (CRTM_14_2 * v106r);

        v3r = in_r[in_strides[6]];
        v4r = in_r[in_strides[8]];

        v79r = v3r + v4r;
        v97r = v4r - v3r;

        v1r = *in_r;

        tv2ir = (CRTM_14_4 * v133r) - (CRTM_14_6 * v115r) + (CRTM_14_2 * v97r);
        tv4ir = (CRTM_14_2 * v115r) + (CRTM_14_6 * v133r) - (CRTM_14_4 * v97r);
        tv6ir = (CRTM_14_4 * v115r) + (CRTM_14_2 * v133r) + (CRTM_14_6 * v97r);

        tvrr = v1r + v313r + v511r + v79r;
        tvri = v214r + v412r + v610r + v8r;

        *out_r = tvrr + tvri;

        out_r[out_strides[7]] = tvrr - tvri;

        cv1r = v1r + (CRTM_14_3 * v313r) - (CRTM_14_5 * v511r) -
               (CRTM_14_1 * v79r);
        cv2r = v1r + (CRTM_14_3 * v79r) -
               ((CRTM_14_1 * v511r) + (CRTM_14_5 * v313r));
        cv3r = v1r + (CRTM_14_3 * v511r) - (CRTM_14_1 * v313r) -
               (CRTM_14_5 * v79r);

        //-------------------------------------------

        v1i = in_i[in_strides[1]];
        v2i = in_i[in_strides[13]];

        v214i = v1i + v2i;
        v142i = v2i - v1i;

        v1i = in_i[in_strides[3]];
        v2i = in_i[in_strides[11]];

        v412i = v1i + v2i;
        v124i = v2i - v1i;

        v1i = in_i[in_strides[5]];
        v2i = in_i[in_strides[9]];

        v610i = v1i + v2i;
        v106i = v2i - v1i;

        v8i = in_i[in_strides[7]];

        v3i = in_i[in_strides[2]];
        v4i = in_i[in_strides[12]];

        v313i = v3i + v4i;
        v133i = v4i - v3i;

        v3i = in_i[in_strides[4]];
        v4i = in_i[in_strides[10]];

        v115i = v3i + v4i;
        v511i = v3i - v4i;

        tv1ii = (CRTM_14_1 * v214i) + (CRTM_14_5 * v412i) -
                (CRTM_14_3 * v610i) - v8i;
        tv3ii = (CRTM_14_3 * v214i) - (CRTM_14_1 * v412i) -
                (CRTM_14_5 * v610i) + v8i;
        tv5ii = (CRTM_14_5 * v214i) - (CRTM_14_3 * v412i) +
                (CRTM_14_1 * v610i) - v8i;

        v3i = in_i[in_strides[6]];
        v4i = in_i[in_strides[8]];

        v79i = v3i + v4i;
        v97i = v4i - v3i;

        v1i = *in_i;

        tv1ri = (CRTM_14_2 * v142i) + (CRTM_14_6 * v124i) + (CRTM_14_4 * v106i);
        tv3ri = (CRTM_14_4 * v142i) + (CRTM_14_2 * v124i) - (CRTM_14_6 * v106i);
        tv5ri = (CRTM_14_6 * v142i) - (CRTM_14_4 * v124i) + (CRTM_14_2 * v106i);

        tvir = v214i + v412i + v610i + v8i;
        tvii = v1i + v313i + v115i + v79i;

        *out_i = tvir + tvii;

        out_i[out_strides[7]] = tvii - tvir;

        tv2ri = (CRTM_14_4 * v133i) - (CRTM_14_6 * v511i) + (CRTM_14_2 * v97i);
        tv4ri = (CRTM_14_6 * v133i) + (CRTM_14_2 * v511i) - (CRTM_14_4 * v97i);
        tv6ri = (CRTM_14_2 * v133i) + (CRTM_14_4 * v511i) + (CRTM_14_6 * v97i);

        cv1i = v1i + (CRTM_14_3 * v313i) - (CRTM_14_5 * v115i) -
               (CRTM_14_1 * v79i);
        cv2i = v1i + (CRTM_14_3 * v79i) -
               ((CRTM_14_5 * v313i) + (CRTM_14_1 * v115i));
        cv3i = v1i + (CRTM_14_3 * v115i) - (CRTM_14_1 * v313i) -
               (CRTM_14_5 * v79i);

        //------------------------------------------

        tvrr = cv1r + tv1rr;
        tvri = tv1ri + tv2ri;

        tvir = tv1ir + tv2ir;
        tvii = cv1i + tv1ii;

        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[13]] = tvrr + tvri;
        out_i[out_strides[1]] = tvir + tvii;

        out_i[out_strides[13]] = tvii - tvir;

        tvrr = cv1r - tv1rr;
        tvri = tv1ri - tv2ri;

        tvir = tv1ir - tv2ir;
        tvii = cv1i - tv1ii;

        out_r[out_strides[6]] = tvrr - tvri;
        out_r[out_strides[8]] = tvrr + tvri;
        out_i[out_strides[6]] = tvir + tvii;

        out_i[out_strides[8]] = tvii - tvir;

        tvrr = cv2r + tv3rr;
        tvri = tv3ri + tv4ri;

        tvir = tv3ir + tv4ir;
        tvii = cv2i + tv3ii;

        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[12]] = tvrr + tvri;
        out_i[out_strides[2]] = tvir + tvii;

        out_i[out_strides[12]] = tvii - tvir;

        tvrr = cv2r - tv3rr;
        tvri = tv3ri - tv4ri;

        tvir = tv3ir - tv4ir;
        tvii = cv2i - tv3ii;

        out_r[out_strides[5]] = tvrr - tvri;
        out_r[out_strides[9]] = tvrr + tvri;
        out_i[out_strides[5]] = tvir + tvii;

        out_i[out_strides[9]] = tvii - tvir;

        tvrr = cv3r + tv5rr;
        tvri = tv5ri + tv6ri;

        tvir = tv5ir + tv6ir;
        tvii = cv3i + tv5ii;

        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[11]] = tvrr + tvri;
        out_i[out_strides[3]] = tvir + tvii;

        out_i[out_strides[11]] = tvii - tvir;

        tvrr = cv3r - tv5rr;
        tvri = tv5ri - tv6ri;

        tvir = tv5ir - tv6ir;
        tvii = cv3i - tv5ii;

        out_r[out_strides[4]] = tvrr - tvri;
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[4]] = tvir + tvii;

        out_i[out_strides[10]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft14c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft14c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft14c_fp64;
    }
    else
    {
        return NULL;
    }
}
