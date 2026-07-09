// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft8c.c
 *
 *  @brief Radix-8 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-8 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 */
#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 52, 32, 0, 0},
                                                     {0, 4, 52, 32, 0, 0}};

ops_cycles_t get_ops_cnt_fft8c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft8c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8_1 =
        +0.707106781186547524400844362104849039284835938;

    FFTZ_DOUBLE *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_DOUBLE *)in_imag;
        in_i = (FFTZ_DOUBLE *)in_real;
        out_r = (FFTZ_DOUBLE *)out_imag;
        out_i = (FFTZ_DOUBLE *)out_real;
    }
    else
    {
        in_r = (FFTZ_DOUBLE *)in_real;
        in_i = (FFTZ_DOUBLE *)in_imag;
        out_r = (FFTZ_DOUBLE *)out_real;
        out_i = (FFTZ_DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i,
            tvrr, tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];
        v7r = in_r[in_strides[6]];
        v7i = in_i[in_strides[6]];

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v1r = *in_r;
        v5r = in_r[in_strides[4]];
        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v2r = in_r[in_strides[1]];
        v8r = in_r[in_strides[7]];
        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v4r = in_r[in_strides[3]];
        v6r = in_r[in_strides[5]];
        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v2i = in_i[in_strides[1]];
        v8i = in_i[in_strides[7]];
        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_r = tvrr + tvri;
        out_r[out_strides[4]] = tvri - tvrr;

        v4i = in_i[in_strides[3]];
        v6i = in_i[in_strides[5]];
        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[6]] = tvrr + tvri;

        v1i = *in_i;
        v5i = in_i[in_strides[4]];
        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_i = tvii + tvir;
        out_i[out_strides[4]] = tvir - tvii;

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        out_i[out_strides[2]] = tvir + tvii;
        out_i[out_strides[6]] = tvii - tvir;

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[7]] = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[5]] = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        out_i[out_strides[1]] = tvir + tvii;
        out_i[out_strides[7]] = tvii - tvir;

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        out_i[out_strides[3]] = tvir + tvii;
        out_i[out_strides[5]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft8c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8_1 =
        +0.707106781186547524400844362104849039284835938;

    FFTZ_FLOAT *in_r, *in_i, *out_r, *out_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_FLOAT *)in_imag;
        in_i = (FFTZ_FLOAT *)in_real;
        out_r = (FFTZ_FLOAT *)out_imag;
        out_i = (FFTZ_FLOAT *)out_real;
    }
    else
    {
        in_r = (FFTZ_FLOAT *)in_real;
        in_i = (FFTZ_FLOAT *)in_imag;
        out_r = (FFTZ_FLOAT *)out_real;
        out_i = (FFTZ_FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i,
            tvrr, tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];
        v7r = in_r[in_strides[6]];
        v7i = in_i[in_strides[6]];

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v1r = *in_r;
        v5r = in_r[in_strides[4]];
        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v2r = in_r[in_strides[1]];
        v8r = in_r[in_strides[7]];
        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v4r = in_r[in_strides[3]];
        v6r = in_r[in_strides[5]];
        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v2i = in_i[in_strides[1]];
        v8i = in_i[in_strides[7]];
        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_r = tvrr + tvri;
        out_r[out_strides[4]] = tvri - tvrr;

        v4i = in_i[in_strides[3]];
        v6i = in_i[in_strides[5]];
        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        out_r[out_strides[2]] = tvrr - tvri;
        out_r[out_strides[6]] = tvrr + tvri;

        v1i = *in_i;
        v5i = in_i[in_strides[4]];
        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_i = tvii + tvir;
        out_i[out_strides[4]] = tvir - tvii;

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        out_i[out_strides[2]] = tvir + tvii;
        out_i[out_strides[6]] = tvii - tvir;

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        out_r[out_strides[1]] = tvrr - tvri;
        out_r[out_strides[7]] = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        out_r[out_strides[3]] = tvrr - tvri;
        out_r[out_strides[5]] = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        out_i[out_strides[1]] = tvir + tvii;
        out_i[out_strides[7]] = tvii - tvir;

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        out_i[out_strides[3]] = tvir + tvii;
        out_i[out_strides[5]] = tvii - tvir;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft8c(FFTZ_UINT8 precision,
                            FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft8c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft8c_fp64;
    }
    else
    {
        return NULL;
    }
}
