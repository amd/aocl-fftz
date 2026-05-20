// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_r2c_fft8c.c
 *
 *  @brief R2C fused twiddle (forward twiddle + conjugate output) Radix-8 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-8 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */
#include "core/kernels/kernel.h"

#define RADIX 8

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 32, 66, 46, 0, 0},
                                                     {0, 32, 66, 46, 0, 0}};

ops_cycles_t get_ops_cnt_twid_r2c_fft8c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
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

static FFTZ_VOID twid_r2c_fft8c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8_1 =
        +0.707106781186547524400844362104849039284835938;

    FFTZ_DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_DOUBLE twr, twi;

    in_h1_r = (FFTZ_DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i,
            tvrr, tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h1_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h2_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h2_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h2_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_DOUBLE v7r_t = in_h2_r[in_strides[6]];
        FFTZ_DOUBLE v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_DOUBLE v8r_t = in_h2_r[in_strides[7]];
        FFTZ_DOUBLE v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_h1_r = tvrr + tvri;
        out_h2_r[out_strides[4]] = tvri - tvrr;

        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        out_h1_r[out_strides[2]] = tvrr - tvri;
        out_h2_r[out_strides[6]] = tvrr + tvri;

        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_h1_i = tvii + tvir;
        out_h2_i[out_strides[4]] = -(tvir - tvii);

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        out_h1_i[out_strides[2]] = tvir + tvii;
        out_h2_i[out_strides[6]] = -(tvii - tvir);

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[7]] = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[5]] = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        out_h1_i[out_strides[1]] = tvir + tvii;
        out_h2_i[out_strides[7]] = -(tvii - tvir);

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        out_h1_i[out_strides[3]] = tvir + tvii;
        out_h2_i[out_strides[5]] = -(tvii - tvir);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID twid_r2c_fft8c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8_1 =
        +0.707106781186547524400844362104849039284835938f;

    FFTZ_FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;
    FFTZ_FLOAT twr, twi;

    in_h1_r = (FFTZ_FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            v7r, v7i, v8r, v8i, v28r, v46r, v28i, v82i, v64i, v82r, v64r, v46i,
            tvrr, tvri, tvir, tvii, tv1rr, tv1ii, v37r, v73r, v37i, v73i, tv1ri,
            tv1ir, v15r, v51r, v15i, v51i;

        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        twr = tw_ptr[0];
        twi = tw_ptr[1];
        v2r = v2r_t * twr - v2i_t * twi;
        v2i = v2r_t * twi + v2i_t * twr;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        twr = tw_ptr[DATA_STRIDE];
        twi = tw_ptr[DATA_STRIDE + 1];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h1_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h1_i[in_strides[3]];
        twr = tw_ptr[2 * DATA_STRIDE];
        twi = tw_ptr[2 * DATA_STRIDE + 1];
        v4r = v4r_t * twr - v4i_t * twi;
        v4i = v4r_t * twi + v4i_t * twr;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h2_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h2_i[in_strides[4]];
        twr = tw_ptr[3 * DATA_STRIDE];
        twi = tw_ptr[3 * DATA_STRIDE + 1];
        v5r = v5r_t * twr - v5i_t * twi;
        v5i = v5r_t * twi + v5i_t * twr;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h2_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h2_i[in_strides[5]];
        twr = tw_ptr[4 * DATA_STRIDE];
        twi = tw_ptr[4 * DATA_STRIDE + 1];
        v6r = v6r_t * twr - v6i_t * twi;
        v6i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FFTZ_FLOAT v7r_t = in_h2_r[in_strides[6]];
        FFTZ_FLOAT v7i_t = in_h2_i[in_strides[6]];
        twr = tw_ptr[5 * DATA_STRIDE];
        twi = tw_ptr[5 * DATA_STRIDE + 1];
        v7r = v7r_t * twr - v7i_t * twi;
        v7i = v7r_t * twi + v7i_t * twr;

        // Input point 8: x(7)
        FFTZ_FLOAT v8r_t = in_h2_r[in_strides[7]];
        FFTZ_FLOAT v8i_t = in_h2_i[in_strides[7]];
        twr = tw_ptr[6 * DATA_STRIDE];
        twi = tw_ptr[6 * DATA_STRIDE + 1];
        v8r = v8r_t * twr - v8i_t * twi;
        v8i = v8r_t * twi + v8i_t * twr;

        v37r = v3r + v7r;
        v37i = v7i + v3i;
        v73r = v7r - v3r;
        v73i = v3i - v7i;

        v15r = v1r + v5r;
        v51r = v1r - v5r;

        v28r = v2r + v8r;
        v82r = v8r - v2r;

        v46r = v4r + v6r;
        v64r = v6r - v4r;

        v82i = v8i - v2i;
        v28i = v2i + v8i;

        tvrr = v28r + v46r;
        tvri = v15r + v37r;
        *out_h1_r = tvrr + tvri;
        out_h2_r[out_strides[4]] = tvri - tvrr;

        v46i = v4i + v6i;
        v64i = v6i - v4i;

        tvrr = v15r - v37r;
        tvri = v82i - v64i;
        out_h1_r[out_strides[2]] = tvrr - tvri;
        out_h2_r[out_strides[6]] = tvrr + tvri;

        v15i = v1i + v5i;
        v51i = v1i - v5i;

        tvii = v28i + v46i;
        tvir = v15i + v37i;
        *out_h1_i = tvii + tvir;
        out_h2_i[out_strides[4]] = -(tvir - tvii);

        tvir = v82r - v64r;
        tvii = v15i - v37i;

        out_h1_i[out_strides[2]] = tvir + tvii;
        out_h2_i[out_strides[6]] = -(tvii - tvir);

        tv1rr = CRTM_8_1 * (v28r - v46r);
        tv1ri = CRTM_8_1 * (v82i + v64i);

        tvrr = v51r + tv1rr;
        tvri = tv1ri - v73i;
        out_h1_r[out_strides[1]] = tvrr - tvri;
        out_h2_r[out_strides[7]] = tvrr + tvri;

        tvrr = v51r - tv1rr;
        tvri = tv1ri + v73i;
        out_h1_r[out_strides[3]] = tvrr - tvri;
        out_h2_r[out_strides[5]] = tvrr + tvri;

        tv1ir = CRTM_8_1 * (v82r + v64r);
        tvir = tv1ir + v73r;
        tv1ii = CRTM_8_1 * (v28i - v46i);
        tvii = v51i + tv1ii;
        out_h1_i[out_strides[1]] = tvir + tvii;
        out_h2_i[out_strides[7]] = -(tvii - tvir);

        tvir = tv1ir - v73r;
        tvii = v51i - tv1ii;

        out_h1_i[out_strides[3]] = tvir + tvii;
        out_h2_i[out_strides[5]] = -(tvii - tvir);

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_r2c_fft8c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_r2c_fft8c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_r2c_fft8c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
