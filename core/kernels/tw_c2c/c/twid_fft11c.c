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

/** @file twid_fft11c.c
 *
 *  @brief Twiddle Radix-11 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-11 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

// twiddle cost = 10 * (4 muls, 2 adds, 2 movs [loads])
//              = (40, 20, 20)
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 140, 160, 64, 0, 0},
                                                     {0, 140, 160, 64, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fft11c(UINT8 precision, UINT8 direction)
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

static VOID twid_fft11c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
                             UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

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
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    DOUBLE *tw = (DOUBLE *)twd;
    DOUBLE twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v211r, v310r, v49r, v58r,
            v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i, v58i,
            v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii, tvrr1,
            tvri1, tvir1, tvii1, tvrr2, tvri2, tvir2, tvii2, tvrr3, tvri3,
            tvir3, tvii3, tvrr4, tvri4, tvir4, tvii4;

        // Input point 2: x(1)
        DOUBLE v2r_t = in_r[in_strides[1]];
        DOUBLE v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v1r = v2r_t * twr - v2i_t * twi;
        v1i = v2r_t * twi + v2i_t * twr;

        // Input point 11: x(10)
        DOUBLE v11r_t = in_r[in_strides[10]];
        DOUBLE v11i_t = in_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * n + cnt);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        v2r = v11r_t * twr - v11i_t * twi;
        v2i = v11r_t * twi + v11i_t * twr;

        v211r = v1r + v2r;
        v112r = v2r - v1r;
        v211i = v1i + v2i;
        v112i = v2i - v1i;

        // Input point 3: x(2)
        DOUBLE v3r_t = in_r[in_strides[2]];
        DOUBLE v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 10: x(9)
        DOUBLE v10r_t = in_r[in_strides[9]];
        DOUBLE v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * n + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        v4r = v10r_t * twr - v10i_t * twi;
        v4i = v10r_t * twi + v10i_t * twr;

        v310r = v3r + v4r;
        v103r = v4r - v3r;
        v310i = v3i + v4i;
        v103i = v4i - v3i;

        // Input point 4: x(3)
        DOUBLE v4r_t = in_r[in_strides[3]];
        DOUBLE v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v1r = v4r_t * twr - v4i_t * twi;
        v1i = v4r_t * twi + v4i_t * twr;

        // Input point 9: x(8)
        DOUBLE v9r_t = in_r[in_strides[8]];
        DOUBLE v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * n + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v2r = v9r_t * twr - v9i_t * twi;
        v2i = v9r_t * twi + v9i_t * twr;

        v49r = v1r + v2r;
        v94r = v2r - v1r;
        v49i = v1i + v2i;
        v94i = v2i - v1i;

        // Input point 5: x(4)
        DOUBLE v5r_t = in_r[in_strides[4]];
        DOUBLE v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v3r = v5r_t * twr - v5i_t * twi;
        v3i = v5r_t * twi + v5i_t * twr;

        // Input point 8: x(7)
        DOUBLE v8r_t = in_r[in_strides[7]];
        DOUBLE v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * n + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v4r = v8r_t * twr - v8i_t * twi;
        v4i = v8r_t * twi + v8i_t * twr;

        v58r = v3r + v4r;
        v85r = v4r - v3r;
        v58i = v3i + v4i;
        v85i = v4i - v3i;

        // Input point 6: x(5)
        DOUBLE v6r_t = in_r[in_strides[5]];
        DOUBLE v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * n + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v1r = v6r_t * twr - v6i_t * twi;
        v1i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        DOUBLE v7r_t = in_r[in_strides[6]];
        DOUBLE v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * n + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v2r = v7r_t * twr - v7i_t * twi;
        v2i = v7r_t * twi + v7i_t * twr;

        v67r = v1r + v2r;
        v76r = v2r - v1r;
        v67i = v1i + v2i;
        v76i = v2i - v1i;

        v1r = *in_r;
        v1i = *in_i;

        // Output point 1: X(0)
        *out_r = v1r + v211r + v310r + v49r + v58r + v67r;

        tvrr = v1r + (CRTM_11_1 * v211r) + (CRTM_11_3 * v310r) -
               ((CRTM_11_5 * v49r) + (CRTM_11_7 * v58r) + (CRTM_11_9 * v67r));
        tvir = (CRTM_11_2 * v112r) + (CRTM_11_4 * v103r) + (CRTM_11_6 * v94r) +
               (CRTM_11_8 * v85r) + (CRTM_11_10 * v76r);
        tvrr1 = v1r + (CRTM_11_1 * v67r) + (CRTM_11_3 * v211r) -
                ((CRTM_11_7 * v310r) + (CRTM_11_9 * v49r) + (CRTM_11_5 * v58r));
        tvir1 = (CRTM_11_4 * v112r) + (CRTM_11_8 * v103r) -
                ((CRTM_11_10 * v94r) + (CRTM_11_6 * v85r) + (CRTM_11_2 * v76r));
        tvrr2 =
            v1r + (CRTM_11_1 * v58r) + (CRTM_11_3 * v49r) -
            ((CRTM_11_5 * v211r) + (CRTM_11_9 * v310r) + (CRTM_11_7 * v67r));
        tvir2 = (CRTM_11_2 * v85r) + (CRTM_11_6 * v112r) + (CRTM_11_8 * v76r) -
                ((CRTM_11_4 * v94r) + (CRTM_11_10 * v103r));

        tvrr3 =
            v1r + (CRTM_11_1 * v49r) + (CRTM_11_3 * v67r) -
            ((CRTM_11_5 * v310r) + (CRTM_11_7 * v211r) + (CRTM_11_9 * v58r));
        tvir3 = (CRTM_11_2 * v94r) + (CRTM_11_8 * v112r) + (CRTM_11_10 * v85r) -
                ((CRTM_11_4 * v76r) + (CRTM_11_6 * v103r));
        tvrr4 = v1r + (CRTM_11_1 * v310r) + (CRTM_11_3 * v58r) -
                ((CRTM_11_5 * v67r) + (CRTM_11_7 * v49r) + (CRTM_11_9 * v211r));
        tvir4 = (CRTM_11_6 * v76r) + (CRTM_11_8 * v94r) + (CRTM_11_10 * v112r) -
                ((CRTM_11_2 * v103r) + (CRTM_11_4 * v85r));

        // Output point 1: X(0)
        *out_i = v1i + v211i + v310i + v49i + v58i + v67i;

        tvii = v1i + (CRTM_11_1 * v211i) + (CRTM_11_3 * v310i) -
               ((CRTM_11_5 * v49i) + (CRTM_11_7 * v58i) + (CRTM_11_9 * v67i));
        tvri = (CRTM_11_2 * v112i) + (CRTM_11_4 * v103i) + (CRTM_11_6 * v94i) +
               (CRTM_11_8 * v85i) + (CRTM_11_10 * v76i);
        tvii1 = v1i + (CRTM_11_1 * v67i) + (CRTM_11_3 * v211i) -
                ((CRTM_11_7 * v310i) + (CRTM_11_9 * v49i) + (CRTM_11_5 * v58i));
        tvri1 = (CRTM_11_4 * v112i) + (CRTM_11_8 * v103i) -
                ((CRTM_11_10 * v94i) + (CRTM_11_6 * v85i) + (CRTM_11_2 * v76i));
        tvii2 =
            v1i + (CRTM_11_1 * v58i) + (CRTM_11_3 * v49i) -
            ((CRTM_11_5 * v211i) + (CRTM_11_7 * v67i) + (CRTM_11_9 * v310i));
        tvri2 = (CRTM_11_2 * v85i) + (CRTM_11_6 * v112i) + (CRTM_11_8 * v76i) -
                ((CRTM_11_10 * v103i) + (CRTM_11_4 * v94i));

        tvii3 =
            v1i + (CRTM_11_1 * v49i) + (CRTM_11_3 * v67i) -
            ((CRTM_11_5 * v310i) + (CRTM_11_7 * v211i) + (CRTM_11_9 * v58i));
        tvri3 = (CRTM_11_2 * v94i) + (CRTM_11_8 * v112i) + (CRTM_11_10 * v85i) -
                ((CRTM_11_4 * v76i) + (CRTM_11_6 * v103i));
        tvii4 = v1i + (CRTM_11_1 * v310i) + (CRTM_11_3 * v58i) -
                ((CRTM_11_5 * v67i) + (CRTM_11_7 * v49i) + (CRTM_11_9 * v211i));
        tvri4 = (CRTM_11_6 * v76i) + (CRTM_11_8 * v94i) + (CRTM_11_10 * v112i) -
                ((CRTM_11_2 * v103i) + (CRTM_11_4 * v85i));

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvir + tvii;

        // Output point 11: X(10)
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[10]] = tvii - tvir;

        out_r[out_strides[2]] = tvrr1 - tvri1;
        out_i[out_strides[2]] = tvir1 + tvii1;

        // Output point 10: X(9)
        out_r[out_strides[9]] = tvrr1 + tvri1;
        out_i[out_strides[9]] = tvii1 - tvir1;

        out_r[out_strides[3]] = tvrr2 - tvri2;
        out_i[out_strides[3]] = tvir2 + tvii2;

        // Output point 9: X(8)
        out_r[out_strides[8]] = tvrr2 + tvri2;
        out_i[out_strides[8]] = tvii2 - tvir2;

        out_r[out_strides[4]] = tvrr3 - tvri3;
        out_i[out_strides[4]] = tvir3 + tvii3;

        // Output point 8: X(7)
        out_r[out_strides[7]] = tvrr3 + tvri3;
        out_i[out_strides[7]] = tvii3 - tvir3;

        out_r[out_strides[5]] = tvrr4 - tvri4;
        out_i[out_strides[5]] = tvir4 + tvii4;

        // Output point 7: X(6)
        out_r[out_strides[6]] = tvrr4 + tvri4;
        out_i[out_strides[6]] = tvii4 - tvir4;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID twid_fft11c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                             VOID *out_imag, INTP n,
                             aoclfftz_strides_t *strides, VOID *twd,
                             UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

    const FLOAT CRTM_11_1 =
        +0.84125353283118116029052039464203089547681594330064f;
    const FLOAT CRTM_11_2 =
        +0.54064081745559759544482548159299693174139803024473f;
    const FLOAT CRTM_11_3 =
        +0.41541501300188639668675795488636098054966524290126f;
    const FLOAT CRTM_11_4 =
        +0.90963199535451838458365117807108162835411650732265f;
    const FLOAT CRTM_11_5 =
        +0.14231483827328501490317354898047094957684096668515f;
    const FLOAT CRTM_11_6 =
        +0.98982144188093275042610808187068914262031166769031f;
    const FLOAT CRTM_11_7 =
        +0.65486073394528511198338203198719613618953603946564f;
    const FLOAT CRTM_11_8 =
        +0.75574957435425824224552448923467521721665586591805f;
    const FLOAT CRTM_11_9 =
        +0.95949297361449738989036805706632769906245484800000f;
    const FLOAT CRTM_11_10 =
        +0.28173255684142978898192655345478532989004751779983f;

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

    FLOAT *tw = (FLOAT *)twd;
    FLOAT twr, twi;

    for (INTP cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v211r, v310r, v49r, v58r,
            v67r, v112i, v103i, v94i, v85i, v76i, v211i, v310i, v49i, v58i,
            v67i, v112r, v103r, v94r, v85r, v76r, tvrr, tvri, tvir, tvii, tvrr1,
            tvri1, tvir1, tvii1, tvrr2, tvri2, tvir2, tvii2, tvrr3, tvri3,
            tvir3, tvii3, tvrr4, tvri4, tvir4, tvii4;

        // Input point 2: x(1)
        FLOAT v2r_t = in_r[in_strides[1]];
        FLOAT v2i_t = in_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * n + cnt);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        v1r = v2r_t * twr - v2i_t * twi;
        v1i = v2r_t * twi + v2i_t * twr;

        // Input point 11: x(10)
        FLOAT v11r_t = in_r[in_strides[10]];
        FLOAT v11i_t = in_i[in_strides[10]];
        UINTP twid_addr11 = DATA_STRIDE * (10 * n + cnt);
        twr = tw[twid_addr11];
        twi = tw[1 + twid_addr11];
        v2r = v11r_t * twr - v11i_t * twi;
        v2i = v11r_t * twi + v11i_t * twr;

        v211r = v1r + v2r;
        v112r = v2r - v1r;
        v211i = v1i + v2i;
        v112i = v2i - v1i;

        // Input point 3: x(2)
        FLOAT v3r_t = in_r[in_strides[2]];
        FLOAT v3i_t = in_i[in_strides[2]];
        UINTP twid_addr3 = DATA_STRIDE * (2 * n + cnt);
        twr = tw[twid_addr3];
        twi = tw[1 + twid_addr3];
        v3r = v3r_t * twr - v3i_t * twi;
        v3i = v3r_t * twi + v3i_t * twr;

        // Input point 10: x(9)
        FLOAT v10r_t = in_r[in_strides[9]];
        FLOAT v10i_t = in_i[in_strides[9]];
        UINTP twid_addr10 = DATA_STRIDE * (9 * n + cnt);
        twr = tw[twid_addr10];
        twi = tw[1 + twid_addr10];
        v4r = v10r_t * twr - v10i_t * twi;
        v4i = v10r_t * twi + v10i_t * twr;

        v310r = v3r + v4r;
        v103r = v4r - v3r;
        v310i = v3i + v4i;
        v103i = v4i - v3i;

        // Input point 4: x(3)
        FLOAT v4r_t = in_r[in_strides[3]];
        FLOAT v4i_t = in_i[in_strides[3]];
        UINTP twid_addr4 = DATA_STRIDE * (3 * n + cnt);
        twr = tw[twid_addr4];
        twi = tw[1 + twid_addr4];
        v1r = v4r_t * twr - v4i_t * twi;
        v1i = v4r_t * twi + v4i_t * twr;

        // Input point 9: x(8)
        FLOAT v9r_t = in_r[in_strides[8]];
        FLOAT v9i_t = in_i[in_strides[8]];
        UINTP twid_addr9 = DATA_STRIDE * (8 * n + cnt);
        twr = tw[twid_addr9];
        twi = tw[1 + twid_addr9];
        v2r = v9r_t * twr - v9i_t * twi;
        v2i = v9r_t * twi + v9i_t * twr;

        v49r = v1r + v2r;
        v94r = v2r - v1r;
        v49i = v1i + v2i;
        v94i = v2i - v1i;

        // Input point 5: x(4)
        FLOAT v5r_t = in_r[in_strides[4]];
        FLOAT v5i_t = in_i[in_strides[4]];
        UINTP twid_addr5 = DATA_STRIDE * (4 * n + cnt);
        twr = tw[twid_addr5];
        twi = tw[1 + twid_addr5];
        v3r = v5r_t * twr - v5i_t * twi;
        v3i = v5r_t * twi + v5i_t * twr;

        // Input point 8: x(7)
        FLOAT v8r_t = in_r[in_strides[7]];
        FLOAT v8i_t = in_i[in_strides[7]];
        UINTP twid_addr8 = DATA_STRIDE * (7 * n + cnt);
        twr = tw[twid_addr8];
        twi = tw[1 + twid_addr8];
        v4r = v8r_t * twr - v8i_t * twi;
        v4i = v8r_t * twi + v8i_t * twr;

        v58r = v3r + v4r;
        v85r = v4r - v3r;
        v58i = v3i + v4i;
        v85i = v4i - v3i;

        // Input point 6: x(5)
        FLOAT v6r_t = in_r[in_strides[5]];
        FLOAT v6i_t = in_i[in_strides[5]];
        UINTP twid_addr6 = DATA_STRIDE * (5 * n + cnt);
        twr = tw[twid_addr6];
        twi = tw[1 + twid_addr6];
        v1r = v6r_t * twr - v6i_t * twi;
        v1i = v6r_t * twi + v6i_t * twr;

        // Input point 7: x(6)
        FLOAT v7r_t = in_r[in_strides[6]];
        FLOAT v7i_t = in_i[in_strides[6]];
        UINTP twid_addr7 = DATA_STRIDE * (6 * n + cnt);
        twr = tw[twid_addr7];
        twi = tw[1 + twid_addr7];
        v2r = v7r_t * twr - v7i_t * twi;
        v2i = v7r_t * twi + v7i_t * twr;

        v67r = v1r + v2r;
        v76r = v2r - v1r;
        v67i = v1i + v2i;
        v76i = v2i - v1i;

        v1r = *in_r;
        v1i = *in_i;

        // Output point 1: X(0)
        *out_r = v1r + v211r + v310r + v49r + v58r + v67r;

        tvrr = v1r + (CRTM_11_1 * v211r) + (CRTM_11_3 * v310r) -
               ((CRTM_11_5 * v49r) + (CRTM_11_7 * v58r) + (CRTM_11_9 * v67r));
        tvir = (CRTM_11_2 * v112r) + (CRTM_11_4 * v103r) + (CRTM_11_6 * v94r) +
               (CRTM_11_8 * v85r) + (CRTM_11_10 * v76r);
        tvrr1 = v1r + (CRTM_11_1 * v67r) + (CRTM_11_3 * v211r) -
                ((CRTM_11_7 * v310r) + (CRTM_11_9 * v49r) + (CRTM_11_5 * v58r));
        tvir1 = (CRTM_11_4 * v112r) + (CRTM_11_8 * v103r) -
                ((CRTM_11_10 * v94r) + (CRTM_11_6 * v85r) + (CRTM_11_2 * v76r));
        tvrr2 =
            v1r + (CRTM_11_1 * v58r) + (CRTM_11_3 * v49r) -
            ((CRTM_11_5 * v211r) + (CRTM_11_9 * v310r) + (CRTM_11_7 * v67r));
        tvir2 = (CRTM_11_2 * v85r) + (CRTM_11_6 * v112r) + (CRTM_11_8 * v76r) -
                ((CRTM_11_4 * v94r) + (CRTM_11_10 * v103r));

        tvrr3 =
            v1r + (CRTM_11_1 * v49r) + (CRTM_11_3 * v67r) -
            ((CRTM_11_5 * v310r) + (CRTM_11_7 * v211r) + (CRTM_11_9 * v58r));
        tvir3 = (CRTM_11_2 * v94r) + (CRTM_11_8 * v112r) + (CRTM_11_10 * v85r) -
                ((CRTM_11_4 * v76r) + (CRTM_11_6 * v103r));
        tvrr4 = v1r + (CRTM_11_1 * v310r) + (CRTM_11_3 * v58r) -
                ((CRTM_11_5 * v67r) + (CRTM_11_7 * v49r) + (CRTM_11_9 * v211r));
        tvir4 = (CRTM_11_6 * v76r) + (CRTM_11_8 * v94r) + (CRTM_11_10 * v112r) -
                ((CRTM_11_2 * v103r) + (CRTM_11_4 * v85r));

        // Output point 1: X(0)
        *out_i = v1i + v211i + v310i + v49i + v58i + v67i;

        tvii = v1i + (CRTM_11_1 * v211i) + (CRTM_11_3 * v310i) -
               ((CRTM_11_5 * v49i) + (CRTM_11_7 * v58i) + (CRTM_11_9 * v67i));
        tvri = (CRTM_11_2 * v112i) + (CRTM_11_4 * v103i) + (CRTM_11_6 * v94i) +
               (CRTM_11_8 * v85i) + (CRTM_11_10 * v76i);
        tvii1 = v1i + (CRTM_11_1 * v67i) + (CRTM_11_3 * v211i) -
                ((CRTM_11_7 * v310i) + (CRTM_11_9 * v49i) + (CRTM_11_5 * v58i));
        tvri1 = (CRTM_11_4 * v112i) + (CRTM_11_8 * v103i) -
                ((CRTM_11_10 * v94i) + (CRTM_11_6 * v85i) + (CRTM_11_2 * v76i));
        tvii2 =
            v1i + (CRTM_11_1 * v58i) + (CRTM_11_3 * v49i) -
            ((CRTM_11_5 * v211i) + (CRTM_11_7 * v67i) + (CRTM_11_9 * v310i));
        tvri2 = (CRTM_11_2 * v85i) + (CRTM_11_6 * v112i) + (CRTM_11_8 * v76i) -
                ((CRTM_11_10 * v103i) + (CRTM_11_4 * v94i));

        tvii3 =
            v1i + (CRTM_11_1 * v49i) + (CRTM_11_3 * v67i) -
            ((CRTM_11_5 * v310i) + (CRTM_11_7 * v211i) + (CRTM_11_9 * v58i));
        tvri3 = (CRTM_11_2 * v94i) + (CRTM_11_8 * v112i) + (CRTM_11_10 * v85i) -
                ((CRTM_11_4 * v76i) + (CRTM_11_6 * v103i));
        tvii4 = v1i + (CRTM_11_1 * v310i) + (CRTM_11_3 * v58i) -
                ((CRTM_11_5 * v67i) + (CRTM_11_7 * v49i) + (CRTM_11_9 * v211i));
        tvri4 = (CRTM_11_6 * v76i) + (CRTM_11_8 * v94i) + (CRTM_11_10 * v112i) -
                ((CRTM_11_2 * v103i) + (CRTM_11_4 * v85i));

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvir + tvii;

        // Output point 11: X(10)
        out_r[out_strides[10]] = tvrr + tvri;
        out_i[out_strides[10]] = tvii - tvir;

        out_r[out_strides[2]] = tvrr1 - tvri1;
        out_i[out_strides[2]] = tvir1 + tvii1;

        // Output point 10: X(9)
        out_r[out_strides[9]] = tvrr1 + tvri1;
        out_i[out_strides[9]] = tvii1 - tvir1;

        out_r[out_strides[3]] = tvrr2 - tvri2;
        out_i[out_strides[3]] = tvir2 + tvii2;

        // Output point 9: X(8)
        out_r[out_strides[8]] = tvrr2 + tvri2;
        out_i[out_strides[8]] = tvii2 - tvir2;

        out_r[out_strides[4]] = tvrr3 - tvri3;
        out_i[out_strides[4]] = tvir3 + tvii3;

        // Output point 8: X(7)
        out_r[out_strides[7]] = tvrr3 + tvri3;
        out_i[out_strides[7]] = tvii3 - tvir3;

        out_r[out_strides[5]] = tvrr4 - tvri4;
        out_i[out_strides[5]] = tvir4 + tvii4;

        // Output point 7: X(6)
        out_r[out_strides[6]] = tvrr4 + tvri4;
        out_i[out_strides[6]] = tvii4 - tvir4;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_twid_fft11c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fft11c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fft11c_fp64;
    }
    else
    {
        return NULL;
    }
}
