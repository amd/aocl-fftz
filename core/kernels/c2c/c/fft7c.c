/**
 * Copyright (C) 2023-2024, Advanced Micro Devices. All rights reserved.
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

/** @file fft7c.c
 *
 *  @brief Radix-7 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-7 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 *  @author Prasandh Sankarankutty
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 36, 60, 28, 0, 0},
                                                     {0, 36, 60, 28, 0, 0}};

ops_cycles_t get_ops_cnt_fft7c(UINT8 precision, UINT8 direction)
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

static VOID fft7c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_7_1 = +0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_7_2 = +0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_7_3 = +0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_7_4 = +0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_7_5 = +0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_7_6 = +0.974927912181823607018131682993931217232785801;

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

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
               v7i;
        DOUBLE tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv1ir, tv1ii, tv2ir,
               tv2ii, tv3ir, tv3ii;
        DOUBLE av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av1ir, av1ii, av2ir,
               av2ii, av3ir, av3ii;

        v1r = *in_r;
        v1i = *in_i;

        v2r = in_r[in_strides[1]];
        v7r = in_r[in_strides[6]];
        av1rr = (v2r + v7r);
        av1ii = (v7r - v2r);

        v2i = in_i[in_strides[1]];
        v7i = in_i[in_strides[6]];
        av1ir = (v2i + v7i);
        av1ri = (v7i - v2i);

        v3r = in_r[in_strides[2]];
        v6r = in_r[in_strides[5]];
        av2rr = (v3r + v6r);
        av2ii = (v6r - v3r);

        v3i = in_i[in_strides[2]];
        v6i = in_i[in_strides[5]];
        av2ir = (v3i + v6i);
        av2ri = (v6i - v3i);

        v4r = in_r[in_strides[3]];
        v5r = in_r[in_strides[4]];
        av3rr = (v4r + v5r);
        av3ii = (v5r - v4r);

        v4i = in_i[in_strides[3]];
        v5i = in_i[in_strides[4]];
        av3ir = (v4i + v5i);
        av3ri = (v5i - v4i);

        *out_r = v1r + av1rr + av2rr + av3rr;
        *out_i = v1i + av1ir + av2ir + av3ir;

        tv1rr = CRTM_7_3 * av1rr;
        tv2rr = CRTM_7_1 * av2rr;
        tv3rr = CRTM_7_2 * av3rr;
        DOUBLE cvrr = v1r + tv1rr - tv2rr - tv3rr;
        tv1ri = CRTM_7_5 * av1ri;
        tv2ri = CRTM_7_6 * av2ri;
        tv3ri = CRTM_7_4 * av3ri;
        DOUBLE cvri = tv1ri + tv2ri + tv3ri;
        out_r[out_strides[1]] = cvrr - cvri;
        out_r[out_strides[6]] = cvrr + cvri;

        tv1ii = CRTM_7_3 * av1ir;
        tv2ii = CRTM_7_1 * av2ir;
        tv3ii = CRTM_7_2 * av3ir;
        tv1ir = CRTM_7_5 * av1ii;
        tv2ir = CRTM_7_6 * av2ii;
        tv3ir = CRTM_7_4 * av3ii;

        DOUBLE cvir = tv1ir + tv2ir + tv3ir;
        DOUBLE cvii = v1i + tv1ii - tv2ii - tv3ii;
        out_i[out_strides[1]] = cvir + cvii;
        out_i[out_strides[6]] = cvii - cvir;

        tv1rr = CRTM_7_1 * av1rr;
        tv2rr = CRTM_7_2 * av2rr;
        tv3rr = CRTM_7_3 * av3rr;
        cvrr = v1r - tv1rr - tv2rr + tv3rr;

        tv1ri = CRTM_7_6 * av1ri;
        tv2ri = CRTM_7_4 * av2ri;
        tv3ri = CRTM_7_5 * av3ri;
        cvri = tv1ri - tv2ri - tv3ri;
        out_r[out_strides[2]] = cvrr - cvri;
        out_r[out_strides[5]] = cvrr + cvri;

        tv1ii = CRTM_7_1 * av1ir;
        tv2ii = CRTM_7_2 * av2ir;
        tv3ii = CRTM_7_3 * av3ir;
        cvii = v1i - tv1ii - tv2ii + tv3ii;

        tv1ir = CRTM_7_6 * av1ii;
        tv2ir = CRTM_7_4 * av2ii;
        tv3ir = CRTM_7_5 * av3ii;
        cvir = tv1ir - tv2ir - tv3ir;
        out_i[out_strides[2]] = cvii + cvir;
        out_i[out_strides[5]] = cvii - cvir;

        tv1rr = CRTM_7_2 * av1rr;
        tv2rr = CRTM_7_3 * av2rr;
        tv3rr = CRTM_7_1 * av3rr;
        cvrr = v1r - tv1rr + tv2rr - tv3rr;

        tv1ri = CRTM_7_4 * av1ri;
        tv2ri = CRTM_7_5 * av2ri;
        tv3ri = CRTM_7_6 * av3ri;
        cvri = tv1ri - tv2ri + tv3ri;
        out_r[out_strides[3]] = cvrr - cvri;
        out_r[out_strides[4]] = cvrr + cvri;

        tv1ii = CRTM_7_2 * av1ir;
        tv2ii = CRTM_7_3 * av2ir;
        tv3ii = CRTM_7_1 * av3ir;
        cvii = v1i - tv1ii + tv2ii - tv3ii;

        tv1ir = CRTM_7_4 * av1ii;
        tv2ir = CRTM_7_5 * av2ii;
        tv3ir = CRTM_7_6 * av3ii;
        cvir = tv1ir - tv2ir + tv3ir;
        out_i[out_strides[3]] = cvii + cvir;
        out_i[out_strides[4]] = cvii - cvir;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft7c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_7_1 = +0.222520933956314404288902564496794759466355569;
    const FLOAT CRTM_7_2 = +0.900968867902419126236102319507445051165919162;
    const FLOAT CRTM_7_3 = +0.623489801858733530525004884004239810632274731;
    const FLOAT CRTM_7_4 = +0.433883739117558120475768332848358754609990728;
    const FLOAT CRTM_7_5 = +0.781831482468029808708444526674057750232334519;
    const FLOAT CRTM_7_6 = +0.974927912181823607018131682993931217232785801;

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

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i, v7r,
              v7i;
        FLOAT tv1rr, tv1ri, tv2rr, tv2ri, tv3rr, tv3ri, tv1ir, tv1ii, tv2ir,
              tv2ii, tv3ir, tv3ii;
        FLOAT av1rr, av1ri, av2rr, av2ri, av3rr, av3ri, av1ir, av1ii, av2ir,
              av2ii, av3ir, av3ii;

        v1r = *in_r;
        v1i = *in_i;

        v2r = in_r[in_strides[1]];
        v7r = in_r[in_strides[6]];
        av1rr = (v2r + v7r);
        av1ii = (v7r - v2r);

        v2i = in_i[in_strides[1]];
        v7i = in_i[in_strides[6]];
        av1ir = (v2i + v7i);
        av1ri = (v7i - v2i);

        v3r = in_r[in_strides[2]];
        v6r = in_r[in_strides[5]];
        av2rr = (v3r + v6r);
        av2ii = (v6r - v3r);

        v3i = in_i[in_strides[2]];
        v6i = in_i[in_strides[5]];
        av2ir = (v3i + v6i);
        av2ri = (v6i - v3i);

        v4r = in_r[in_strides[3]];
        v5r = in_r[in_strides[4]];
        av3rr = (v4r + v5r);
        av3ii = (v5r - v4r);

        v4i = in_i[in_strides[3]];
        v5i = in_i[in_strides[4]];
        av3ir = (v4i + v5i);
        av3ri = (v5i - v4i);

        *out_r = v1r + av1rr + av2rr + av3rr;
        *out_i = v1i + av1ir + av2ir + av3ir;

        tv1rr = CRTM_7_3 * av1rr;
        tv2rr = CRTM_7_1 * av2rr;
        tv3rr = CRTM_7_2 * av3rr;
        FLOAT cvrr = v1r + tv1rr - tv2rr - tv3rr;
        tv1ri = CRTM_7_5 * av1ri;
        tv2ri = CRTM_7_6 * av2ri;
        tv3ri = CRTM_7_4 * av3ri;
        FLOAT cvri = tv1ri + tv2ri + tv3ri;
        out_r[out_strides[1]] = cvrr - cvri;
        out_r[out_strides[6]] = cvrr + cvri;

        tv1ii = CRTM_7_3 * av1ir;
        tv2ii = CRTM_7_1 * av2ir;
        tv3ii = CRTM_7_2 * av3ir;
        tv1ir = CRTM_7_5 * av1ii;
        tv2ir = CRTM_7_6 * av2ii;
        tv3ir = CRTM_7_4 * av3ii;

        FLOAT cvir = tv1ir + tv2ir + tv3ir;
        FLOAT cvii = v1i + tv1ii - tv2ii - tv3ii;
        out_i[out_strides[1]] = cvir + cvii;
        out_i[out_strides[6]] = cvii - cvir;

        tv1rr = CRTM_7_1 * av1rr;
        tv2rr = CRTM_7_2 * av2rr;
        tv3rr = CRTM_7_3 * av3rr;
        cvrr = v1r - tv1rr - tv2rr + tv3rr;

        tv1ri = CRTM_7_6 * av1ri;
        tv2ri = CRTM_7_4 * av2ri;
        tv3ri = CRTM_7_5 * av3ri;
        cvri = tv1ri - tv2ri - tv3ri;
        out_r[out_strides[2]] = cvrr - cvri;
        out_r[out_strides[5]] = cvrr + cvri;

        tv1ii = CRTM_7_1 * av1ir;
        tv2ii = CRTM_7_2 * av2ir;
        tv3ii = CRTM_7_3 * av3ir;
        cvii = v1i - tv1ii - tv2ii + tv3ii;

        tv1ir = CRTM_7_6 * av1ii;
        tv2ir = CRTM_7_4 * av2ii;
        tv3ir = CRTM_7_5 * av3ii;
        cvir = tv1ir - tv2ir - tv3ir;
        out_i[out_strides[2]] = cvii + cvir;
        out_i[out_strides[5]] = cvii - cvir;

        tv1rr = CRTM_7_2 * av1rr;
        tv2rr = CRTM_7_3 * av2rr;
        tv3rr = CRTM_7_1 * av3rr;
        cvrr = v1r - tv1rr + tv2rr - tv3rr;

        tv1ri = CRTM_7_4 * av1ri;
        tv2ri = CRTM_7_5 * av2ri;
        tv3ri = CRTM_7_6 * av3ri;
        cvri = tv1ri - tv2ri + tv3ri;
        out_r[out_strides[3]] = cvrr - cvri;
        out_r[out_strides[4]] = cvrr + cvri;

        tv1ii = CRTM_7_2 * av1ir;
        tv2ii = CRTM_7_3 * av2ir;
        tv3ii = CRTM_7_1 * av3ir;
        cvii = v1i - tv1ii + tv2ii - tv3ii;

        tv1ir = CRTM_7_4 * av1ii;
        tv2ir = CRTM_7_5 * av2ii;
        tv3ir = CRTM_7_6 * av3ii;
        cvir = tv1ir - tv2ir + tv3ir;
        out_i[out_strides[3]] = cvii + cvir;
        out_i[out_strides[4]] = cvii - cvir;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft7c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft7c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft7c_fp64;
    }
    else
    {
        return NULL;
    }
}
