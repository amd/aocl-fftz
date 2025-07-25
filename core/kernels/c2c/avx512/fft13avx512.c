/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

/** @file fft13avx512.c
 *
 *  @brief Radix-13 FFT kernel with avx512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-13 FFT implementations using avx512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Murugan Vairavel
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 38, 90, 208, 45, 87},
                                                     {0, 38, 90, 104,  6, 87}};

ops_cycles_t get_ops_cnt_fft13avx512(UINT8 precision, UINT8 direction)
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

static VOID fft13avx512fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                     VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                     VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_13[10] =
        {0.866025403784438646763723170752936183471402627f,
         0.500000000000000000000000000000000000000000000f,
         0.866025403784438646763723170752936183471402627f,
         0.500000000000000000000000000000000000000000000f,
         0.447320117602511140667282045633987571324387014f,  // 2 * DGC[3] * CRTM_13[3]
         0.265966249214837287587521063842185948798330267f,  // 2 * DGC[4] * CRTM_13[3]
         0.131467828262610852858973617628781241523254441f,  // 2 * DGC[7] * CRTM_13[3]
         0.503537032863766627246873853868466977093348562f,  // 2 * DGC[8] * CRTM_13[3]
         0.575140729474003121368385547455453388461001608f,  // 2 * DGC[5] * CRTM_13[2]
         0.174138601152135905005660794929264742616964676f}; // 2 * DGC[6] * CRTM_13[2]

    const FLOAT DGC[12] =
        {0.383795939621999107759935105622541328854274714f,   // DGC[0] + DGC[11]
         0.512495343165873201917369308123450118288250350f,   // 2 * DGC[1]
         0.313782782103169222093665453512006539320425272f,   // 2 * DGC[2]
         0.516520780623489722840901288569017135705033622f,   // 2 * DGC[3]
         0.307111371159082800241759918978675468603229334f,   // 2 * DGC[4]
         0.575140729474003121368385547455453388461001608f,   // 2 * DGC[5]
         0.174138601152135905005660794929264742616964676f,   // 2 * DGC[6]
         0.151805972074387731966205794490207080712856746f,   // 2 * DGC[7]
         0.581434482941682194819330939539157246603987480f,   // 2 * DGC[8]
         0.600477271932665282925769253334763009352012848f,   // 2 * DGC[9]
         0.023198211211536581443310913308166504379654082f,   // 2 * DGC[10]
         -0.217129272955332441093268438955874662187608048f}; // DGC[0] - DGC[11]

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_512_S;
    INTP remaining_sets = n % NUM_SETS_512_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    // Registers to hold input data points
    __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
           v_in9, v_in10, v_in11, v_in12;
    // Registers to hold output data points
    __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8, v_out9, v_out10, v_out11, v_out12;
    // Registers to hold intrim outputs after multiplying diagonal constants
    __m512 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
           v_t10, v_t11, v_t12;

    __m512 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
           v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
           v_av18, v_av19, v_av20, v_av21, v_av22;
    __m512 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
           v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
           v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
           v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
           v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
    __m512 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
           v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
           v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
           v_tv26, v_tv27, v_tv28, v_tv29;

    __m512 v_C1  = _mm512_set1_ps(CRTM_13[0]);
    __m512 v_C2  = _mm512_set1_ps(CRTM_13[1]);
    __m512 v_C3  = _mm512_set1_ps(CRTM_13[2]);
    __m512 v_C4  = _mm512_set1_ps(CRTM_13[3]);
    __m512 v_C5  = _mm512_set1_ps(CRTM_13[4]);
    __m512 v_C6  = _mm512_set1_ps(CRTM_13[5]);
    __m512 v_C7  = _mm512_set1_ps(CRTM_13[6]);
    __m512 v_C8  = _mm512_set1_ps(CRTM_13[7]);
    __m512 v_C9  = _mm512_set1_ps(CRTM_13[8]);
    __m512 v_C10 = _mm512_set1_ps(CRTM_13[9]);

    __m512 v_DG1  = _mm512_set1_ps(DGC[0]);
    __m512 v_DG2  = _mm512_set1_ps(DGC[1]);
    __m512 v_DG3  = _mm512_set1_ps(DGC[2]);
    __m512 v_DG4  = _mm512_set1_ps(DGC[3]);
    __m512 v_DG5  = _mm512_set1_ps(DGC[4]);
    __m512 v_DG6  = _mm512_set1_ps(DGC[5]);
    __m512 v_DG7  = _mm512_set1_ps(DGC[6]);
    __m512 v_DG8  = _mm512_set1_ps(DGC[7]);
    __m512 v_DG9  = _mm512_set1_ps(DGC[8]);
    __m512 v_DG10 = _mm512_set1_ps(DGC[9]);
    __m512 v_DG11 = _mm512_set1_ps(DGC[10]);
    __m512 v_DG12 = _mm512_set1_ps(DGC[11]);

    __m512 v_ZERO  = flag ? _neg_zero_512_f.s : _mm512_setzero_ps();

    if (flag)
    {
        in_r  = in_imag;
        out_r = out_imag;
        v_C3  = -v_C3;
        v_C4  = -v_C4;
        v_C5  = -v_C5;
        v_C7  = -v_C7;
        v_C9  = -v_C9;
        v_DG3  = -v_DG3;
        v_DG5  = -v_DG5;
        v_DG7  = -v_DG7;
        v_DG9  = -v_DG9;
        v_DG11 = -v_DG11;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER8_512_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER8_512_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER8_512_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER8_512_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER8_512_S(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER8_512_S(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER8_512_S(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER8_512_S(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER8_512_S(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER8_512_S(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER8_512_S(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER8_512_S(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER8_512_S(curr_in, v_in_stride, v_in12);

        v_av1 = _mm512_add_ps(v_in1, v_in12);
        v_av2 = _mm512_add_ps(v_in4, v_in3);
        v_av3 = _mm512_add_ps(v_in9, v_in10);
        v_av4 = _mm512_add_ps(v_in2, v_in6);
        v_av5 = _mm512_add_ps(v_in11, v_in7);
        v_av6 = _mm512_add_ps(v_in8, v_in5);

        v_av7  = _mm512_sub_ps(v_in1, v_in12);
        v_av8  = _mm512_sub_ps(v_in4, v_in3);
        v_av9  = _mm512_sub_ps(v_in9, v_in10);
        v_av10 = _mm512_sub_ps(v_in2, v_in6);
        v_av11 = _mm512_sub_ps(v_in11, v_in7);
        v_av12 = _mm512_sub_ps(v_in8, v_in5);

        v_cv1 = _mm512_add_ps(v_av2, v_av3);
        v_cv2 = _mm512_add_ps(v_av4, v_av5);
        v_cv3 = _mm512_add_ps(v_av1, v_cv1);
        v_cv4 = _mm512_add_ps(v_av6, v_cv2);

        v_tv1 = _mm512_mul_ps(v_cv3, v_DG12);
        v_tv2 = _mm512_mul_ps(v_cv4, v_DG1);
        v_cv5 = _mm512_add_ps(v_tv1, v_tv2);
        v_t1  = _mm512_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm512_mul_ps(v_cv3, v_DG1);
        v_tv2 = _mm512_mul_ps(v_cv4, v_DG12);
        v_cv5 = _mm512_add_ps(v_tv1, v_tv2);
        v_t7  = _mm512_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm512_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm512_mul_ps(v_C2, v_cv5);
        v_cv6 = _mm512_add_ps(v_av10, v_av11);
        v_cv7 = _mm512_add_ps(v_av8, v_av9);
        v_cv8 = _mm512_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm512_mul_ps(v_C6, v_cv8);

        v_cv9  = _mm512_sub_ps(v_av1, v_av6);
        v_cv10 = _mm512_sub_ps(v_cv9, v_tv3);
        v_tv5  = _mm512_mul_ps(v_DG4, v_cv10);
        v_t2   = _mm512_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm512_mul_ps(v_C5, v_cv8);
        v_tv5 = _mm512_mul_ps(v_DG5, v_cv10);
        v_t8  = _mm512_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm512_add_ps(v_cv1, v_cv2);
        v_tv6  = _mm512_mul_ps(v_C2, v_cv11);
        v_cv12 = _mm512_add_ps(v_av1, v_av6);
        v_cv13 = _mm512_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm512_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm512_mul_ps(v_C8, v_cv14);
        v_tv8 = _mm512_mul_ps(v_DG8, v_cv13);
        v_t3  = _mm512_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm512_mul_ps(v_C7, v_cv14);
        v_tv8 = _mm512_mul_ps(v_DG9, v_cv13);
        v_t9  = _mm512_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm512_sub_ps(v_av2, v_av3);
        v_cv16 = _mm512_sub_ps(v_av4, v_av5);
        v_cv17 = _mm512_sub_ps(v_av8, v_av9);
        v_cv18 = _mm512_sub_ps(v_av10, v_av11);
        v_tv9  = _mm512_mul_ps(v_C4, v_cv16);
        v_tv10 = _mm512_mul_ps(v_C3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm512_xor_ps(v_ZERO, v_av12);
        v_cv19 = _mm512_add_ps(v_tv9, v_tv11);
        v_cv20 = _mm512_add_ps(v_cv19, v_tv10);

        v_tv12 = _mm512_mul_ps(v_C2, v_cv17);
        v_tv13 = _mm512_mul_ps(v_C1, v_cv18);
        v_cv21 = _mm512_add_ps(v_av7, v_tv12);
        v_cv22 = _mm512_add_ps(v_cv21, v_tv13);

        v_tv14 = _mm512_mul_ps(v_DG2, v_cv20);
        v_tv15 = _mm512_mul_ps(v_DG3, v_cv22);
        v_t4   = _mm512_add_ps(v_tv14, v_tv15);
        v_t4   = CONJ_512_S(SWAP_RI_512_S(v_t4));

        v_tv14 = _mm512_mul_ps(v_DG3, v_cv20);
        v_tv15 = _mm512_mul_ps(v_DG2, v_cv22);
        v_t10  = _mm512_sub_ps(v_tv15, v_tv14);
        v_t10  = SWAP_RI_512_S(v_t10);

        v_cv23 = _mm512_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm512_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm512_mul_ps(v_DG10, v_cv23);
        v_tv17 = _mm512_mul_ps(v_DG11, v_cv24);
        v_t5   = _mm512_add_ps(v_tv16, v_tv17);
        v_t5   = SWAP_RI_512_S(CONJ_512_S(v_t5));

        v_tv16 = _mm512_mul_ps(v_DG11, v_cv23);
        v_tv17 = _mm512_mul_ps(v_DG10, v_cv24);
        v_t11  = _mm512_sub_ps(v_tv16, v_tv17);
        v_t11  = SWAP_RI_512_S(v_t11);

        v_cv25 = _mm512_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm512_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm512_mul_ps(v_C9, v_cv25);
        v_tv19 = _mm512_mul_ps(v_DG7, v_cv26);
        v_t6   = _mm512_add_ps(v_tv18, v_tv19);
        v_t6   = CONJ_512_S(SWAP_RI_512_S(v_t6));

        v_tv18 = _mm512_mul_ps(v_C10, v_cv25);
        v_tv19 = _mm512_mul_ps(v_DG6, v_cv26);
        v_t12  = _mm512_sub_ps(v_tv19, v_tv18);
        v_t12  = SWAP_RI_512_S(v_t12);

        v_cv27 = _mm512_add_ps(v_cv11, v_cv12);
        v_out0 = _mm512_add_ps(v_in0, v_cv27);

        v_av13 = _mm512_add_ps(v_t2, v_t3);
        v_av14 = _mm512_add_ps(v_t4, v_t5);
        v_av15 = _mm512_add_ps(v_t1, v_t6);

        v_cv28 = _mm512_add_ps(v_av13, v_av14);
        v_out1 = _mm512_add_ps(v_cv28, v_av15);

        v_av16  = _mm512_sub_ps(v_t1, v_t6);
        v_cv29  = _mm512_sub_ps(v_av13, v_av14);
        v_out12 = _mm512_add_ps(v_cv29, v_av16);

        v_av17 = _mm512_sub_ps(v_t2, v_t3);
        v_cv30 = _mm512_sub_ps(v_t7, v_av17);
        v_av18 = _mm512_add_ps(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm512_xor_ps(v_ZERO, v_av18);
        v_tv21 = _mm512_xor_ps(v_ZERO, v_t12);
        v_cv31 = _mm512_sub_ps(v_tv20, v_tv21);
        v_cv31 = CONJ_512_S(v_cv31);

        v_out8 = _mm512_add_ps(v_cv30, v_cv31);
        v_out5 = _mm512_sub_ps(v_cv30, v_cv31);

        v_tv22 = _mm512_mul_ps(v_C2, v_cv28);
        v_cv32 = _mm512_sub_ps(v_av15, v_tv22);

        v_av19 = _mm512_sub_ps(v_t10, v_t11);
        v_av20 = _mm512_sub_ps(v_t8, v_t9);
        v_tv23 = _mm512_mul_ps(v_C3, v_av19);
        v_tv23 = CONJ_512_S(v_tv23);
        v_tv24 = _mm512_mul_ps(v_C3, v_av20);
        v_cv33 = _mm512_add_ps(v_tv23, v_tv24);

        v_out3 = _mm512_add_ps(v_cv32, v_cv33);
        v_out9 = _mm512_sub_ps(v_cv32, v_cv33);

        v_tv25 = _mm512_mul_ps(v_C2, v_cv29);
        v_cv34 = _mm512_sub_ps(v_av16, v_tv25);
        v_cv35 = _mm512_sub_ps(v_tv23, v_tv24);

        v_out4  = _mm512_add_ps(v_cv34, v_cv35);
        v_out10 = _mm512_sub_ps(v_cv34, v_cv35);

        v_tv26 = _mm512_mul_ps(v_C2, v_av17);
        v_cv36 = _mm512_add_ps(v_tv26, v_t7);
        v_av21 = _mm512_sub_ps(v_t4, v_t5);
        v_tv27 = _mm512_mul_ps(v_C1, v_av21);
        v_tv28 = _mm512_mul_ps(v_C4, v_av18);
        v_av22 = _mm512_add_ps(v_t8, v_t9);
        v_tv29 = _mm512_mul_ps(v_C3, v_av22);

        v_cv37 = _mm512_add_ps(v_cv36, v_tv27);
        v_cv38 = _mm512_add_ps(v_tv21, v_tv28);
        v_cv38 = CONJ_512_S(v_cv38);
        v_cv39 = _mm512_sub_ps(v_cv38, v_tv29);

        v_out2 = _mm512_add_ps(v_cv37, v_cv39);
        v_out7 = _mm512_sub_ps(v_cv37, v_cv39);

        v_cv40 = _mm512_sub_ps(v_cv36, v_tv27);
        v_cv41 = _mm512_add_ps(v_cv38, v_tv29);

        v_out6  = _mm512_add_ps(v_cv40, v_cv41);
        v_out11 = _mm512_sub_ps(v_cv40, v_cv41);

        curr_out = out_r;
        SCATTER8_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER8_512_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER8_512_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER8_512_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER8_512_S(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER8_512_S(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER8_512_S(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER8_512_S(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER8_512_S(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER8_512_S(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER8_512_S(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER8_512_S(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER8_512_S(curr_out, v_out_stride, v_out12);

        in_r  += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }

    // tail cases
    if (remaining_sets >= NUM_SETS_256_S)
    {
        // Registers to hold input data points
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m256 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11, v_t12;

        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m256 v_K1  = _mm512_castps512_ps256(v_C1);
        __m256 v_K2  = _mm512_castps512_ps256(v_C2);
        __m256 v_K3  = _mm512_castps512_ps256(v_C3);
        __m256 v_K4  = _mm512_castps512_ps256(v_C4);
        __m256 v_K5  = _mm512_castps512_ps256(v_C5);
        __m256 v_K6  = _mm512_castps512_ps256(v_C6);
        __m256 v_K7  = _mm512_castps512_ps256(v_C7);
        __m256 v_K8  = _mm512_castps512_ps256(v_C8);
        __m256 v_K9  = _mm512_castps512_ps256(v_C9);
        __m256 v_K10 = _mm512_castps512_ps256(v_C10);

        __m256 v_D1  = _mm512_castps512_ps256(v_DG1);
        __m256 v_D2  = _mm512_castps512_ps256(v_DG2);
        __m256 v_D3  = _mm512_castps512_ps256(v_DG3);
        __m256 v_D4  = _mm512_castps512_ps256(v_DG4);
        __m256 v_D5  = _mm512_castps512_ps256(v_DG5);
        __m256 v_D6  = _mm512_castps512_ps256(v_DG6);
        __m256 v_D7  = _mm512_castps512_ps256(v_DG7);
        __m256 v_D8  = _mm512_castps512_ps256(v_DG8);
        __m256 v_D9  = _mm512_castps512_ps256(v_DG9);
        __m256 v_D10 = _mm512_castps512_ps256(v_DG10);
        __m256 v_D11 = _mm512_castps512_ps256(v_DG11);
        __m256 v_D12 = _mm512_castps512_ps256(v_DG12);

        __m256 v_ZERO_256 = _mm512_castps512_ps256(v_ZERO);

        curr_in = in_r;
        curr_out = out_r;

        GATHER4_256_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER4_256_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER4_256_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER4_256_S(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER4_256_S(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER4_256_S(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER4_256_S(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER4_256_S(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER4_256_S(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER4_256_S(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER4_256_S(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER4_256_S(curr_in, v_in_stride, v_in12);

        v_av1 = _mm256_add_ps(v_in1, v_in12);
        v_av2 = _mm256_add_ps(v_in4, v_in3);
        v_av3 = _mm256_add_ps(v_in9, v_in10);
        v_av4 = _mm256_add_ps(v_in2, v_in6);
        v_av5 = _mm256_add_ps(v_in11, v_in7);
        v_av6 = _mm256_add_ps(v_in8, v_in5);

        v_av7  = _mm256_sub_ps(v_in1, v_in12);
        v_av8  = _mm256_sub_ps(v_in4, v_in3);
        v_av9  = _mm256_sub_ps(v_in9, v_in10);
        v_av10 = _mm256_sub_ps(v_in2, v_in6);
        v_av11 = _mm256_sub_ps(v_in11, v_in7);
        v_av12 = _mm256_sub_ps(v_in8, v_in5);

        v_cv1 = _mm256_add_ps(v_av2, v_av3);
        v_cv2 = _mm256_add_ps(v_av4, v_av5);
        v_cv3 = _mm256_add_ps(v_av1, v_cv1);
        v_cv4 = _mm256_add_ps(v_av6, v_cv2);

        v_tv1 = _mm256_mul_ps(v_cv3, v_D12);
        v_tv2 = _mm256_mul_ps(v_cv4, v_D1);
        v_cv5 = _mm256_add_ps(v_tv1, v_tv2);
        v_t1  = _mm256_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm256_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm256_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm256_add_ps(v_tv1, v_tv2);
        v_t7  = _mm256_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm256_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm256_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm256_add_ps(v_av10, v_av11);
        v_cv7 = _mm256_add_ps(v_av8, v_av9);
        v_cv8 = _mm256_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm256_mul_ps(v_K6, v_cv8);

        v_cv9  = _mm256_sub_ps(v_av1, v_av6);
        v_cv10 = _mm256_sub_ps(v_cv9, v_tv3);
        v_tv5  = _mm256_mul_ps(v_D4, v_cv10);
        v_t2   = _mm256_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm256_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm256_mul_ps(v_D5, v_cv10);
        v_t8  = _mm256_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm256_add_ps(v_cv1, v_cv2);
        v_tv6  = _mm256_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm256_add_ps(v_av1, v_av6);
        v_cv13 = _mm256_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm256_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm256_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm256_mul_ps(v_D8, v_cv13);
        v_t3  = _mm256_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm256_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm256_mul_ps(v_D9, v_cv13);
        v_t9  = _mm256_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm256_sub_ps(v_av2, v_av3);
        v_cv16 = _mm256_sub_ps(v_av4, v_av5);
        v_cv17 = _mm256_sub_ps(v_av8, v_av9);
        v_cv18 = _mm256_sub_ps(v_av10, v_av11);
        v_tv9  = _mm256_mul_ps(v_K4, v_cv16);
        v_tv10 = _mm256_mul_ps(v_K3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm256_xor_ps(v_ZERO_256, v_av12);
        v_cv19 = _mm256_add_ps(v_tv9, v_tv11);
        v_cv20 = _mm256_add_ps(v_cv19, v_tv10);

        v_tv12 = _mm256_mul_ps(v_K2, v_cv17);
        v_tv13 = _mm256_mul_ps(v_K1, v_cv18);
        v_cv21 = _mm256_add_ps(v_av7, v_tv12);
        v_cv22 = _mm256_add_ps(v_cv21, v_tv13);

        v_tv14 = _mm256_mul_ps(v_D2, v_cv20);
        v_tv15 = _mm256_mul_ps(v_D3, v_cv22);
        v_t4   = _mm256_add_ps(v_tv14, v_tv15);
        v_t4   = CONJ_256_S(SWAP_RI_256_S(v_t4));

        v_tv14 = _mm256_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm256_mul_ps(v_D2, v_cv22);
        v_t10  = _mm256_sub_ps(v_tv15, v_tv14);
        v_t10  = SWAP_RI_256_S(v_t10);

        v_cv23 = _mm256_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm256_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm256_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm256_mul_ps(v_D11, v_cv24);
        v_t5   = _mm256_add_ps(v_tv16, v_tv17);
        v_t5   = SWAP_RI_256_S(CONJ_256_S(v_t5));

        v_tv16 = _mm256_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm256_mul_ps(v_D10, v_cv24);
        v_t11  = _mm256_sub_ps(v_tv16, v_tv17);
        v_t11  = SWAP_RI_256_S(v_t11);

        v_cv25 = _mm256_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm256_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm256_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm256_mul_ps(v_D7, v_cv26);
        v_t6   = _mm256_add_ps(v_tv18, v_tv19);
        v_t6   = CONJ_256_S(SWAP_RI_256_S(v_t6));

        v_tv18 = _mm256_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm256_mul_ps(v_D6, v_cv26);
        v_t12  = _mm256_sub_ps(v_tv19, v_tv18);
        v_t12  = SWAP_RI_256_S(v_t12);

        v_cv27 = _mm256_add_ps(v_cv11, v_cv12);
        v_out0 = _mm256_add_ps(v_in0, v_cv27);

        v_av13 = _mm256_add_ps(v_t2, v_t3);
        v_av14 = _mm256_add_ps(v_t4, v_t5);
        v_av15 = _mm256_add_ps(v_t1, v_t6);

        v_cv28 = _mm256_add_ps(v_av13, v_av14);
        v_out1 = _mm256_add_ps(v_cv28, v_av15);

        v_av16  = _mm256_sub_ps(v_t1, v_t6);
        v_cv29  = _mm256_sub_ps(v_av13, v_av14);
        v_out12 = _mm256_add_ps(v_cv29, v_av16);

        v_av17 = _mm256_sub_ps(v_t2, v_t3);
        v_cv30 = _mm256_sub_ps(v_t7, v_av17);
        v_av18 = _mm256_add_ps(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm256_xor_ps(v_ZERO_256, v_av18);
        v_tv21 = _mm256_xor_ps(v_ZERO_256, v_t12);
        v_cv31 = _mm256_sub_ps(v_tv20, v_tv21);
        v_cv31 = CONJ_256_S(v_cv31);

        v_out8 = _mm256_add_ps(v_cv30, v_cv31);
        v_out5 = _mm256_sub_ps(v_cv30, v_cv31);

        v_tv22 = _mm256_mul_ps(v_K2, v_cv28);
        v_cv32 = _mm256_sub_ps(v_av15, v_tv22);

        v_av19 = _mm256_sub_ps(v_t10, v_t11);
        v_av20 = _mm256_sub_ps(v_t8, v_t9);
        v_tv23 = _mm256_mul_ps(v_K3, v_av19);
        v_tv23 = CONJ_256_S(v_tv23);
        v_tv24 = _mm256_mul_ps(v_K3, v_av20);
        v_cv33 = _mm256_add_ps(v_tv23, v_tv24);

        v_out3 = _mm256_add_ps(v_cv32, v_cv33);
        v_out9 = _mm256_sub_ps(v_cv32, v_cv33);

        v_tv25 = _mm256_mul_ps(v_K2, v_cv29);
        v_cv34 = _mm256_sub_ps(v_av16, v_tv25);
        v_cv35 = _mm256_sub_ps(v_tv23, v_tv24);

        v_out4  = _mm256_add_ps(v_cv34, v_cv35);
        v_out10 = _mm256_sub_ps(v_cv34, v_cv35);

        v_tv26 = _mm256_mul_ps(v_K2, v_av17);
        v_cv36 = _mm256_add_ps(v_tv26, v_t7);
        v_av21 = _mm256_sub_ps(v_t4, v_t5);
        v_tv27 = _mm256_mul_ps(v_K1, v_av21);
        v_tv28 = _mm256_mul_ps(v_K4, v_av18);
        v_av22 = _mm256_add_ps(v_t8, v_t9);
        v_tv29 = _mm256_mul_ps(v_K3, v_av22);

        v_cv37 = _mm256_add_ps(v_cv36, v_tv27);
        v_cv38 = _mm256_add_ps(v_tv21, v_tv28);
        v_cv38 = CONJ_256_S(v_cv38);
        v_cv39 = _mm256_sub_ps(v_cv38, v_tv29);

        v_out2 = _mm256_add_ps(v_cv37, v_cv39);
        v_out7 = _mm256_sub_ps(v_cv37, v_cv39);

        v_cv40 = _mm256_sub_ps(v_cv36, v_tv27);
        v_cv41 = _mm256_add_ps(v_cv38, v_tv29);

        v_out6  = _mm256_add_ps(v_cv40, v_cv41);
        v_out11 = _mm256_sub_ps(v_cv40, v_cv41);

        curr_out = out_r;
        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER4_256_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER4_256_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER4_256_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER4_256_S(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER4_256_S(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER4_256_S(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER4_256_S(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER4_256_S(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER4_256_S(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER4_256_S(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER4_256_S(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER4_256_S(curr_out, v_out_stride, v_out12);

        in_r  += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets % NUM_SETS_256_S;
    }
    if (remaining_sets >= NUM_SETS_128_S)
    {
          // Registers to hold input data points
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11, v_t12;

        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m128 v_K1  = _mm512_castps512_ps128(v_C1);
        __m128 v_K2  = _mm512_castps512_ps128(v_C2);
        __m128 v_K3  = _mm512_castps512_ps128(v_C3);
        __m128 v_K4  = _mm512_castps512_ps128(v_C4);
        __m128 v_K5  = _mm512_castps512_ps128(v_C5);
        __m128 v_K6  = _mm512_castps512_ps128(v_C6);
        __m128 v_K7  = _mm512_castps512_ps128(v_C7);
        __m128 v_K8  = _mm512_castps512_ps128(v_C8);
        __m128 v_K9  = _mm512_castps512_ps128(v_C9);
        __m128 v_K10 = _mm512_castps512_ps128(v_C10);

        __m128 v_D1  = _mm512_castps512_ps128(v_DG1);
        __m128 v_D2  = _mm512_castps512_ps128(v_DG2);
        __m128 v_D3  = _mm512_castps512_ps128(v_DG3);
        __m128 v_D4  = _mm512_castps512_ps128(v_DG4);
        __m128 v_D5  = _mm512_castps512_ps128(v_DG5);
        __m128 v_D6  = _mm512_castps512_ps128(v_DG6);
        __m128 v_D7  = _mm512_castps512_ps128(v_DG7);
        __m128 v_D8  = _mm512_castps512_ps128(v_DG8);
        __m128 v_D9  = _mm512_castps512_ps128(v_DG9);
        __m128 v_D10 = _mm512_castps512_ps128(v_DG10);
        __m128 v_D11 = _mm512_castps512_ps128(v_DG11);
        __m128 v_D12 = _mm512_castps512_ps128(v_DG12);

        __m128 v_ZERO_128 = _mm512_castps512_ps128(v_ZERO);

        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER2_128_S(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER2_128_S(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER2_128_S(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER2_128_S(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER2_128_S(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER2_128_S(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER2_128_S(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER2_128_S(curr_in, v_in_stride, v_in12);

        v_av1 = _mm_add_ps(v_in1, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in3);
        v_av3 = _mm_add_ps(v_in9, v_in10);
        v_av4 = _mm_add_ps(v_in2, v_in6);
        v_av5 = _mm_add_ps(v_in11, v_in7);
        v_av6 = _mm_add_ps(v_in8, v_in5);

        v_av7  = _mm_sub_ps(v_in1, v_in12);
        v_av8  = _mm_sub_ps(v_in4, v_in3);
        v_av9  = _mm_sub_ps(v_in9, v_in10);
        v_av10 = _mm_sub_ps(v_in2, v_in6);
        v_av11 = _mm_sub_ps(v_in11, v_in7);
        v_av12 = _mm_sub_ps(v_in8, v_in5);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_cv1);
        v_cv4 = _mm_add_ps(v_av6, v_cv2);

        v_tv1 = _mm_mul_ps(v_cv3, v_D12);
        v_tv2 = _mm_mul_ps(v_cv4, v_D1);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t1  = _mm_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t7  = _mm_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm_add_ps(v_av10, v_av11);
        v_cv7 = _mm_add_ps(v_av8, v_av9);
        v_cv8 = _mm_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm_mul_ps(v_K6, v_cv8);

        v_cv9  = _mm_sub_ps(v_av1, v_av6);
        v_cv10 = _mm_sub_ps(v_cv9, v_tv3);
        v_tv5  = _mm_mul_ps(v_D4, v_cv10);
        v_t2   = _mm_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm_mul_ps(v_D5, v_cv10);
        v_t8  = _mm_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm_add_ps(v_cv1, v_cv2);
        v_tv6  = _mm_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm_add_ps(v_av1, v_av6);
        v_cv13 = _mm_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm_mul_ps(v_D8, v_cv13);
        v_t3  = _mm_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm_mul_ps(v_D9, v_cv13);
        v_t9  = _mm_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm_sub_ps(v_av2, v_av3);
        v_cv16 = _mm_sub_ps(v_av4, v_av5);
        v_cv17 = _mm_sub_ps(v_av8, v_av9);
        v_cv18 = _mm_sub_ps(v_av10, v_av11);
        v_tv9  = _mm_mul_ps(v_K4, v_cv16);
        v_tv10 = _mm_mul_ps(v_K3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm_xor_ps(v_ZERO_128, v_av12);
        v_cv19 = _mm_add_ps(v_tv9, v_tv11);
        v_cv20 = _mm_add_ps(v_cv19, v_tv10);

        v_tv12 = _mm_mul_ps(v_K2, v_cv17);
        v_tv13 = _mm_mul_ps(v_K1, v_cv18);
        v_cv21 = _mm_add_ps(v_av7, v_tv12);
        v_cv22 = _mm_add_ps(v_cv21, v_tv13);

        v_tv14 = _mm_mul_ps(v_D2, v_cv20);
        v_tv15 = _mm_mul_ps(v_D3, v_cv22);
        v_t4   = _mm_add_ps(v_tv14, v_tv15);
        v_t4   = CONJ_128_S(SWAP_RI_128_S(v_t4));

        v_tv14 = _mm_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm_mul_ps(v_D2, v_cv22);
        v_t10  = _mm_sub_ps(v_tv15, v_tv14);
        v_t10  = SWAP_RI_128_S(v_t10);

        v_cv23 = _mm_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm_mul_ps(v_D11, v_cv24);
        v_t5   = _mm_add_ps(v_tv16, v_tv17);
        v_t5   = SWAP_RI_128_S(CONJ_128_S(v_t5));

        v_tv16 = _mm_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm_mul_ps(v_D10, v_cv24);
        v_t11  = _mm_sub_ps(v_tv16, v_tv17);
        v_t11  = SWAP_RI_128_S(v_t11);

        v_cv25 = _mm_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm_mul_ps(v_D7, v_cv26);
        v_t6   = _mm_add_ps(v_tv18, v_tv19);
        v_t6   = CONJ_128_S(SWAP_RI_128_S(v_t6));

        v_tv18 = _mm_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm_mul_ps(v_D6, v_cv26);
        v_t12  = _mm_sub_ps(v_tv19, v_tv18);
        v_t12  = SWAP_RI_128_S(v_t12);

        v_cv27 = _mm_add_ps(v_cv11, v_cv12);
        v_out0 = _mm_add_ps(v_in0, v_cv27);

        v_av13 = _mm_add_ps(v_t2, v_t3);
        v_av14 = _mm_add_ps(v_t4, v_t5);
        v_av15 = _mm_add_ps(v_t1, v_t6);

        v_cv28 = _mm_add_ps(v_av13, v_av14);
        v_out1 = _mm_add_ps(v_cv28, v_av15);

        v_av16  = _mm_sub_ps(v_t1, v_t6);
        v_cv29  = _mm_sub_ps(v_av13, v_av14);
        v_out12 = _mm_add_ps(v_cv29, v_av16);

        v_av17 = _mm_sub_ps(v_t2, v_t3);
        v_cv30 = _mm_sub_ps(v_t7, v_av17);
        v_av18 = _mm_add_ps(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm_xor_ps(v_ZERO_128, v_av18);
        v_tv21 = _mm_xor_ps(v_ZERO_128, v_t12);
        v_cv31 = _mm_sub_ps(v_tv20, v_tv21);
        v_cv31 = CONJ_128_S(v_cv31);

        v_out8 = _mm_add_ps(v_cv30, v_cv31);
        v_out5 = _mm_sub_ps(v_cv30, v_cv31);

        v_tv22 = _mm_mul_ps(v_K2, v_cv28);
        v_cv32 = _mm_sub_ps(v_av15, v_tv22);

        v_av19 = _mm_sub_ps(v_t10, v_t11);
        v_av20 = _mm_sub_ps(v_t8, v_t9);
        v_tv23 = _mm_mul_ps(v_K3, v_av19);
        v_tv23 = CONJ_128_S(v_tv23);
        v_tv24 = _mm_mul_ps(v_K3, v_av20);
        v_cv33 = _mm_add_ps(v_tv23, v_tv24);

        v_out3 = _mm_add_ps(v_cv32, v_cv33);
        v_out9 = _mm_sub_ps(v_cv32, v_cv33);

        v_tv25 = _mm_mul_ps(v_K2, v_cv29);
        v_cv34 = _mm_sub_ps(v_av16, v_tv25);
        v_cv35 = _mm_sub_ps(v_tv23, v_tv24);

        v_out4  = _mm_add_ps(v_cv34, v_cv35);
        v_out10 = _mm_sub_ps(v_cv34, v_cv35);

        v_tv26 = _mm_mul_ps(v_K2, v_av17);
        v_cv36 = _mm_add_ps(v_tv26, v_t7);
        v_av21 = _mm_sub_ps(v_t4, v_t5);
        v_tv27 = _mm_mul_ps(v_K1, v_av21);
        v_tv28 = _mm_mul_ps(v_K4, v_av18);
        v_av22 = _mm_add_ps(v_t8, v_t9);
        v_tv29 = _mm_mul_ps(v_K3, v_av22);

        v_cv37 = _mm_add_ps(v_cv36, v_tv27);
        v_cv38 = _mm_add_ps(v_tv21, v_tv28);
        v_cv38 = CONJ_128_S(v_cv38);
        v_cv39 = _mm_sub_ps(v_cv38, v_tv29);

        v_out2 = _mm_add_ps(v_cv37, v_cv39);
        v_out7 = _mm_sub_ps(v_cv37, v_cv39);

        v_cv40 = _mm_sub_ps(v_cv36, v_tv27);
        v_cv41 = _mm_add_ps(v_cv38, v_tv29);

        v_out6  = _mm_add_ps(v_cv40, v_cv41);
        v_out11 = _mm_sub_ps(v_cv40, v_cv41);

        curr_out = out_r;
        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER2_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER2_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER2_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER2_128_S(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER2_128_S(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER2_128_S(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER2_128_S(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER2_128_S(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER2_128_S(curr_out, v_out_stride, v_out12);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }

    if (remaining_sets & 1)
    {
        // Registers to hold input data points
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11, v_t12;

        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m128 v_K1  = _mm512_castps512_ps128(v_C1);
        __m128 v_K2  = _mm512_castps512_ps128(v_C2);
        __m128 v_K3  = _mm512_castps512_ps128(v_C3);
        __m128 v_K4  = _mm512_castps512_ps128(v_C4);
        __m128 v_K5  = _mm512_castps512_ps128(v_C5);
        __m128 v_K6  = _mm512_castps512_ps128(v_C6);
        __m128 v_K7  = _mm512_castps512_ps128(v_C7);
        __m128 v_K8  = _mm512_castps512_ps128(v_C8);
        __m128 v_K9  = _mm512_castps512_ps128(v_C9);
        __m128 v_K10 = _mm512_castps512_ps128(v_C10);

        __m128 v_D1  = _mm512_castps512_ps128(v_DG1);
        __m128 v_D2  = _mm512_castps512_ps128(v_DG2);
        __m128 v_D3  = _mm512_castps512_ps128(v_DG3);
        __m128 v_D4  = _mm512_castps512_ps128(v_DG4);
        __m128 v_D5  = _mm512_castps512_ps128(v_DG5);
        __m128 v_D6  = _mm512_castps512_ps128(v_DG6);
        __m128 v_D7  = _mm512_castps512_ps128(v_DG7);
        __m128 v_D8  = _mm512_castps512_ps128(v_DG8);
        __m128 v_D9  = _mm512_castps512_ps128(v_DG9);
        __m128 v_D10 = _mm512_castps512_ps128(v_DG10);
        __m128 v_D11 = _mm512_castps512_ps128(v_DG11);
        __m128 v_D12 = _mm512_castps512_ps128(v_DG12);

        __m128 v_ZERO_128 = _mm512_castps512_ps128(v_ZERO);

        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_LOW_128_S(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_LOW_128_S(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_LOW_128_S(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_LOW_128_S(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_LOW_128_S(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_LOW_128_S(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_LOW_128_S(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_LOW_128_S(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_LOW_128_S(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_LOW_128_S(curr_in, v_in12);

        v_av1 = _mm_add_ps(v_in1, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in3);
        v_av3 = _mm_add_ps(v_in9, v_in10);
        v_av4 = _mm_add_ps(v_in2, v_in6);
        v_av5 = _mm_add_ps(v_in11, v_in7);
        v_av6 = _mm_add_ps(v_in8, v_in5);

        v_av7  = _mm_sub_ps(v_in1, v_in12);
        v_av8  = _mm_sub_ps(v_in4, v_in3);
        v_av9  = _mm_sub_ps(v_in9, v_in10);
        v_av10 = _mm_sub_ps(v_in2, v_in6);
        v_av11 = _mm_sub_ps(v_in11, v_in7);
        v_av12 = _mm_sub_ps(v_in8, v_in5);

        v_cv1 = _mm_add_ps(v_av2, v_av3);
        v_cv2 = _mm_add_ps(v_av4, v_av5);
        v_cv3 = _mm_add_ps(v_av1, v_cv1);
        v_cv4 = _mm_add_ps(v_av6, v_cv2);

        v_tv1 = _mm_mul_ps(v_cv3, v_D12);
        v_tv2 = _mm_mul_ps(v_cv4, v_D1);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t1  = _mm_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t7  = _mm_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm_add_ps(v_av10, v_av11);
        v_cv7 = _mm_add_ps(v_av8, v_av9);
        v_cv8 = _mm_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm_mul_ps(v_K6, v_cv8);

        v_cv9  = _mm_sub_ps(v_av1, v_av6);
        v_cv10 = _mm_sub_ps(v_cv9, v_tv3);
        v_tv5  = _mm_mul_ps(v_D4, v_cv10);
        v_t2   = _mm_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm_mul_ps(v_D5, v_cv10);
        v_t8  = _mm_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm_add_ps(v_cv1, v_cv2);
        v_tv6  = _mm_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm_add_ps(v_av1, v_av6);
        v_cv13 = _mm_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm_mul_ps(v_D8, v_cv13);
        v_t3  = _mm_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm_mul_ps(v_D9, v_cv13);
        v_t9  = _mm_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm_sub_ps(v_av2, v_av3);
        v_cv16 = _mm_sub_ps(v_av4, v_av5);
        v_cv17 = _mm_sub_ps(v_av8, v_av9);
        v_cv18 = _mm_sub_ps(v_av10, v_av11);
        v_tv9  = _mm_mul_ps(v_K4, v_cv16);
        v_tv10 = _mm_mul_ps(v_K3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm_xor_ps(v_ZERO_128, v_av12);
        v_cv19 = _mm_add_ps(v_tv9, v_tv11);
        v_cv20 = _mm_add_ps(v_cv19, v_tv10);

        v_tv12 = _mm_mul_ps(v_K2, v_cv17);
        v_tv13 = _mm_mul_ps(v_K1, v_cv18);
        v_cv21 = _mm_add_ps(v_av7, v_tv12);
        v_cv22 = _mm_add_ps(v_cv21, v_tv13);

        v_tv14 = _mm_mul_ps(v_D2, v_cv20);
        v_tv15 = _mm_mul_ps(v_D3, v_cv22);
        v_t4   = _mm_add_ps(v_tv14, v_tv15);
        v_t4   = CONJ_128_S(SWAP_RI_128_S(v_t4));

        v_tv14 = _mm_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm_mul_ps(v_D2, v_cv22);
        v_t10  = _mm_sub_ps(v_tv15, v_tv14);
        v_t10  = SWAP_RI_128_S(v_t10);

        v_cv23 = _mm_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm_mul_ps(v_D11, v_cv24);
        v_t5   = _mm_add_ps(v_tv16, v_tv17);
        v_t5   = SWAP_RI_128_S(CONJ_128_S(v_t5));

        v_tv16 = _mm_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm_mul_ps(v_D10, v_cv24);
        v_t11  = _mm_sub_ps(v_tv16, v_tv17);
        v_t11  = SWAP_RI_128_S(v_t11);

        v_cv25 = _mm_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm_mul_ps(v_D7, v_cv26);
        v_t6   = _mm_add_ps(v_tv18, v_tv19);
        v_t6   = CONJ_128_S(SWAP_RI_128_S(v_t6));

        v_tv18 = _mm_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm_mul_ps(v_D6, v_cv26);
        v_t12  = _mm_sub_ps(v_tv19, v_tv18);
        v_t12  = SWAP_RI_128_S(v_t12);

        v_cv27 = _mm_add_ps(v_cv11, v_cv12);
        v_out0 = _mm_add_ps(v_in0, v_cv27);

        v_av13 = _mm_add_ps(v_t2, v_t3);
        v_av14 = _mm_add_ps(v_t4, v_t5);
        v_av15 = _mm_add_ps(v_t1, v_t6);

        v_cv28 = _mm_add_ps(v_av13, v_av14);
        v_out1 = _mm_add_ps(v_cv28, v_av15);

        v_av16  = _mm_sub_ps(v_t1, v_t6);
        v_cv29  = _mm_sub_ps(v_av13, v_av14);
        v_out12 = _mm_add_ps(v_cv29, v_av16);

        v_av17 = _mm_sub_ps(v_t2, v_t3);
        v_cv30 = _mm_sub_ps(v_t7, v_av17);
        v_av18 = _mm_add_ps(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm_xor_ps(v_ZERO_128, v_av18);
        v_tv21 = _mm_xor_ps(v_ZERO_128, v_t12);
        v_cv31 = _mm_sub_ps(v_tv20, v_tv21);
        v_cv31 = CONJ_128_S(v_cv31);

        v_out8 = _mm_add_ps(v_cv30, v_cv31);
        v_out5 = _mm_sub_ps(v_cv30, v_cv31);

        v_tv22 = _mm_mul_ps(v_K2, v_cv28);
        v_cv32 = _mm_sub_ps(v_av15, v_tv22);

        v_av19 = _mm_sub_ps(v_t10, v_t11);
        v_av20 = _mm_sub_ps(v_t8, v_t9);
        v_tv23 = _mm_mul_ps(v_K3, v_av19);
        v_tv23 = CONJ_128_S(v_tv23);
        v_tv24 = _mm_mul_ps(v_K3, v_av20);
        v_cv33 = _mm_add_ps(v_tv23, v_tv24);

        v_out3 = _mm_add_ps(v_cv32, v_cv33);
        v_out9 = _mm_sub_ps(v_cv32, v_cv33);

        v_tv25 = _mm_mul_ps(v_K2, v_cv29);
        v_cv34 = _mm_sub_ps(v_av16, v_tv25);
        v_cv35 = _mm_sub_ps(v_tv23, v_tv24);

        v_out4  = _mm_add_ps(v_cv34, v_cv35);
        v_out10 = _mm_sub_ps(v_cv34, v_cv35);

        v_tv26 = _mm_mul_ps(v_K2, v_av17);
        v_cv36 = _mm_add_ps(v_tv26, v_t7);
        v_av21 = _mm_sub_ps(v_t4, v_t5);
        v_tv27 = _mm_mul_ps(v_K1, v_av21);
        v_tv28 = _mm_mul_ps(v_K4, v_av18);
        v_av22 = _mm_add_ps(v_t8, v_t9);
        v_tv29 = _mm_mul_ps(v_K3, v_av22);

        v_cv37 = _mm_add_ps(v_cv36, v_tv27);
        v_cv38 = _mm_add_ps(v_tv21, v_tv28);
        v_cv38 = CONJ_128_S(v_cv38);
        v_cv39 = _mm_sub_ps(v_cv38, v_tv29);

        v_out2 = _mm_add_ps(v_cv37, v_cv39);
        v_out7 = _mm_sub_ps(v_cv37, v_cv39);

        v_cv40 = _mm_sub_ps(v_cv36, v_tv27);
        v_cv41 = _mm_add_ps(v_cv38, v_tv29);

        v_out6  = _mm_add_ps(v_cv40, v_cv41);
        v_out11 = _mm_sub_ps(v_cv40, v_cv41);

        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_LOW_128_S(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_LOW_128_S(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_LOW_128_S(curr_out, v_out5);
        curr_out = out_r + out_strides[6];
        ST_LOW_128_S(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_LOW_128_S(curr_out, v_out7);
        curr_out = out_r + out_strides[8];
        ST_LOW_128_S(curr_out, v_out8);
        curr_out = out_r + out_strides[9];
        ST_LOW_128_S(curr_out, v_out9);
        curr_out = out_r + out_strides[10];
        ST_LOW_128_S(curr_out, v_out10);
        curr_out = out_r + out_strides[11];
        ST_LOW_128_S(curr_out, v_out11);
        curr_out = out_r + out_strides[12];
        ST_LOW_128_S(curr_out, v_out12);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft13avx512fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                     VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                     VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_13[10] =
        {0.866025403784438646763723170752936183471402627,
         0.500000000000000000000000000000000000000000000,
         0.866025403784438646763723170752936183471402627,
         0.500000000000000000000000000000000000000000000,
         0.447320117602511140667282045633987571324387014,  // 2 * DGC[3] * CRTM_13[3]
         0.265966249214837287587521063842185948798330267,  // 2 * DGC[4] * CRTM_13[3]
         0.131467828262610852858973617628781241523254441,  // 2 * DGC[7] * CRTM_13[3]
         0.503537032863766627246873853868466977093348562,  // 2 * DGC[8] * CRTM_13[3]
         0.575140729474003121368385547455453388461001608,  // 2 * DGC[5] * CRTM_13[2]
         0.174138601152135905005660794929264742616964676}; // 2 * DGC[6] * CRTM_13[2]

    const DOUBLE DGC[12] =
        {0.383795939621999107759935105622541328854274714,   // DGC[0] + DGC[11]
         0.512495343165873201917369308123450118288250350,   // 2 * DGC[1]
         0.313782782103169222093665453512006539320425272,   // 2 * DGC[2]
         0.516520780623489722840901288569017135705033622,   // 2 * DGC[3]
         0.307111371159082800241759918978675468603229334,   // 2 * DGC[4]
         0.575140729474003121368385547455453388461001608,   // 2 * DGC[5]
         0.174138601152135905005660794929264742616964676,   // 2 * DGC[6]
         0.151805972074387731966205794490207080712856746,   // 2 * DGC[7]
         0.581434482941682194819330939539157246603987480,   // 2 * DGC[8]
         0.600477271932665282925769253334763009352012848,   // 2 * DGC[9]
         0.023198211211536581443310913308166504379654082,   // 2 * DGC[10]
         -0.217129272955332441093268438955874662187608048}; // DGC[0] - DGC[11]

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
    #ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
    #else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
    #endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_512_D;
    INTP remaining_sets = n % NUM_SETS_512_D;
    INTP count;
    DOUBLE *curr_in, *curr_out;

    // Registers to hold input data points
    __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
           v_in9, v_in10, v_in11, v_in12;
    // Registers to hold output data points
    __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
           v_out8, v_out9, v_out10, v_out11, v_out12;
    // Registers to hold intrim outputs after multiplying diagonal constants
    __m512d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
           v_t10, v_t11, v_t12;

    __m512d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
           v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
           v_av18, v_av19, v_av20, v_av21, v_av22;
    __m512d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
           v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
           v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
           v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
           v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
    __m512d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
           v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
           v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
           v_tv26, v_tv27, v_tv28, v_tv29;

    __m512d v_C1  = _mm512_set1_pd(CRTM_13[0]);
    __m512d v_C2  = _mm512_set1_pd(CRTM_13[1]);
    __m512d v_C3  = _mm512_set1_pd(CRTM_13[2]);
    __m512d v_C4  = _mm512_set1_pd(CRTM_13[3]);
    __m512d v_C5  = _mm512_set1_pd(CRTM_13[4]);
    __m512d v_C6  = _mm512_set1_pd(CRTM_13[5]);
    __m512d v_C7  = _mm512_set1_pd(CRTM_13[6]);
    __m512d v_C8  = _mm512_set1_pd(CRTM_13[7]);
    __m512d v_C9  = _mm512_set1_pd(CRTM_13[8]);
    __m512d v_C10 = _mm512_set1_pd(CRTM_13[9]);

    __m512d v_DG1  = _mm512_set1_pd(DGC[0]);
    __m512d v_DG2  = _mm512_set1_pd(DGC[1]);
    __m512d v_DG3  = _mm512_set1_pd(DGC[2]);
    __m512d v_DG4  = _mm512_set1_pd(DGC[3]);
    __m512d v_DG5  = _mm512_set1_pd(DGC[4]);
    __m512d v_DG6  = _mm512_set1_pd(DGC[5]);
    __m512d v_DG7  = _mm512_set1_pd(DGC[6]);
    __m512d v_DG8  = _mm512_set1_pd(DGC[7]);
    __m512d v_DG9  = _mm512_set1_pd(DGC[8]);
    __m512d v_DG10 = _mm512_set1_pd(DGC[9]);
    __m512d v_DG11 = _mm512_set1_pd(DGC[10]);
    __m512d v_DG12 = _mm512_set1_pd(DGC[11]);

    __m512d v_ZERO  = flag ? _neg_zero_512_d.d : _mm512_setzero_pd();

    if (flag)
    {
        in_r  = in_imag;
        out_r = out_imag;
        v_C3  = -v_C3;
        v_C4  = -v_C4;
        v_C5  = -v_C5;
        v_C7  = -v_C7;
        v_C9  = -v_C9;
        v_DG3  = -v_DG3;
        v_DG5  = -v_DG5;
        v_DG7  = -v_DG7;
        v_DG9  = -v_DG9;
        v_DG11 = -v_DG11;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        GATHER4_512_D(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER4_512_D(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER4_512_D(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER4_512_D(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER4_512_D(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER4_512_D(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER4_512_D(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER4_512_D(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER4_512_D(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER4_512_D(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER4_512_D(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER4_512_D(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER4_512_D(curr_in, v_in_stride, v_in12);

        v_av1 = _mm512_add_pd(v_in1, v_in12);
        v_av2 = _mm512_add_pd(v_in4, v_in3);
        v_av3 = _mm512_add_pd(v_in9, v_in10);
        v_av4 = _mm512_add_pd(v_in2, v_in6);
        v_av5 = _mm512_add_pd(v_in11, v_in7);
        v_av6 = _mm512_add_pd(v_in8, v_in5);

        v_av7  = _mm512_sub_pd(v_in1, v_in12);
        v_av8  = _mm512_sub_pd(v_in4, v_in3);
        v_av9  = _mm512_sub_pd(v_in9, v_in10);
        v_av10 = _mm512_sub_pd(v_in2, v_in6);
        v_av11 = _mm512_sub_pd(v_in11, v_in7);
        v_av12 = _mm512_sub_pd(v_in8, v_in5);

        v_cv1 = _mm512_add_pd(v_av2, v_av3);
        v_cv2 = _mm512_add_pd(v_av4, v_av5);

        v_cv3 = _mm512_add_pd(v_av1, v_cv1);
        v_cv4 = _mm512_add_pd(v_av6, v_cv2);

        v_tv1 = _mm512_mul_pd(v_cv3, v_DG12);
        v_tv2 = _mm512_mul_pd(v_cv4, v_DG1);
        v_cv5 = _mm512_add_pd(v_tv1, v_tv2);
        v_t1  = _mm512_sub_pd(v_in0, v_cv5);

        v_tv1 = _mm512_mul_pd(v_cv3, v_DG1);
        v_tv2 = _mm512_mul_pd(v_cv4, v_DG12);
        v_cv5 = _mm512_add_pd(v_tv1, v_tv2);
        v_t7  = _mm512_sub_pd(v_in0, v_cv5);

        v_cv5 = _mm512_sub_pd(v_cv1, v_cv2);
        v_tv3 = _mm512_mul_pd(v_C2, v_cv5);
        v_cv6 = _mm512_add_pd(v_av10, v_av11);
        v_cv7 = _mm512_add_pd(v_av8, v_av9);
        v_cv8 = _mm512_add_pd(v_cv6, v_cv7);
        v_tv4 = _mm512_mul_pd(v_C6, v_cv8);

        v_cv9  = _mm512_sub_pd(v_av1, v_av6);
        v_cv10 = _mm512_sub_pd(v_cv9, v_tv3);
        v_tv5  = _mm512_mul_pd(v_DG4, v_cv10);
        v_t2   = _mm512_add_pd(v_tv4, v_tv5);

        v_tv4 = _mm512_mul_pd(v_C5, v_cv8);
        v_tv5 = _mm512_mul_pd(v_DG5, v_cv10);
        v_t8  = _mm512_sub_pd(v_tv4, v_tv5);

        v_cv11 = _mm512_add_pd(v_cv1, v_cv2);
        v_tv6  = _mm512_mul_pd(v_C2, v_cv11);
        v_cv12 = _mm512_add_pd(v_av1, v_av6);
        v_cv13 = _mm512_sub_pd(v_cv12, v_tv6);
        v_cv14 = _mm512_sub_pd(v_cv6, v_cv7);

        v_tv7 = _mm512_mul_pd(v_C8, v_cv14);
        v_tv8 = _mm512_mul_pd(v_DG8, v_cv13);
        v_t3  = _mm512_add_pd(v_tv7, v_tv8);

        v_tv7 = _mm512_mul_pd(v_C7, v_cv14);
        v_tv8 = _mm512_mul_pd(v_DG9, v_cv13);
        v_t9  = _mm512_sub_pd(v_tv7, v_tv8);

        v_cv15 = _mm512_sub_pd(v_av2, v_av3);
        v_cv16 = _mm512_sub_pd(v_av4, v_av5);
        v_cv17 = _mm512_sub_pd(v_av8, v_av9);
        v_cv18 = _mm512_sub_pd(v_av10, v_av11);
        v_tv9  = _mm512_mul_pd(v_C4, v_cv16);
        v_tv10 = _mm512_mul_pd(v_C3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm512_xor_pd(v_ZERO, v_av12);
        v_cv19 = _mm512_add_pd(v_tv9, v_tv11);
        v_cv20 = _mm512_add_pd(v_cv19, v_tv10);

        v_tv12 = _mm512_mul_pd(v_C2, v_cv17);
        v_tv13 = _mm512_mul_pd(v_C1, v_cv18);
        v_cv21 = _mm512_add_pd(v_av7, v_tv12);
        v_cv22 = _mm512_add_pd(v_cv21, v_tv13);

        v_tv14 = _mm512_mul_pd(v_DG2, v_cv20);
        v_tv15 = _mm512_mul_pd(v_DG3, v_cv22);
        v_t4   = _mm512_add_pd(v_tv14, v_tv15);
        v_t4   = CONJ_512_D(SWAP_RI_512_D(v_t4));

        v_tv14 = _mm512_mul_pd(v_DG3, v_cv20);
        v_tv15 = _mm512_mul_pd(v_DG2, v_cv22);
        v_t10  = _mm512_sub_pd(v_tv15, v_tv14);
        v_t10  = SWAP_RI_512_D(v_t10);

        v_cv23 = _mm512_sub_pd(v_cv19, v_tv10);
        v_cv24 = _mm512_sub_pd(v_cv21, v_tv13);

        v_tv16 = _mm512_mul_pd(v_DG10, v_cv23);
        v_tv17 = _mm512_mul_pd(v_DG11, v_cv24);
        v_t5   = _mm512_add_pd(v_tv16, v_tv17);
        v_t5   = SWAP_RI_512_D(CONJ_512_D(v_t5));

        v_tv16 = _mm512_mul_pd(v_DG11, v_cv23);
        v_tv17 = _mm512_mul_pd(v_DG10, v_cv24);
        v_t11  = _mm512_sub_pd(v_tv16, v_tv17);
        v_t11  = SWAP_RI_512_D(v_t11);

        v_cv25 = _mm512_sub_pd(v_cv16, v_av12);
        v_cv26 = _mm512_sub_pd(v_av7, v_cv17);

        v_tv18 = _mm512_mul_pd(v_C9, v_cv25);
        v_tv19 = _mm512_mul_pd(v_DG7, v_cv26);
        v_t6   = _mm512_add_pd(v_tv18, v_tv19);
        v_t6   = CONJ_512_D(SWAP_RI_512_D(v_t6));

        v_tv18 = _mm512_mul_pd(v_C10, v_cv25);
        v_tv19 = _mm512_mul_pd(v_DG6, v_cv26);
        v_t12  = _mm512_sub_pd(v_tv19, v_tv18);
        v_t12  = SWAP_RI_512_D(v_t12);

        v_cv27 = _mm512_add_pd(v_cv11, v_cv12);
        v_out0 = _mm512_add_pd(v_in0, v_cv27);

        v_av13 = _mm512_add_pd(v_t2, v_t3);
        v_av14 = _mm512_add_pd(v_t4, v_t5);
        v_av15 = _mm512_add_pd(v_t1, v_t6);

        v_cv28 = _mm512_add_pd(v_av13, v_av14);
        v_out1 = _mm512_add_pd(v_cv28, v_av15);

        v_av16  = _mm512_sub_pd(v_t1, v_t6);
        v_cv29  = _mm512_sub_pd(v_av13, v_av14);
        v_out12 = _mm512_add_pd(v_cv29, v_av16);

        v_av17 = _mm512_sub_pd(v_t2, v_t3);
        v_cv30 = _mm512_sub_pd(v_t7, v_av17);
        v_av18 = _mm512_add_pd(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm512_xor_pd(v_ZERO, v_av18);
        v_tv21 = _mm512_xor_pd(v_ZERO, v_t12);
        v_cv31 = _mm512_sub_pd(v_tv20, v_tv21);
        v_cv31 = CONJ_512_D(v_cv31);

        v_out8 = _mm512_add_pd(v_cv30, v_cv31);
        v_out5 = _mm512_sub_pd(v_cv30, v_cv31);

        v_tv22 = _mm512_mul_pd(v_C2, v_cv28);
        v_cv32 = _mm512_sub_pd(v_av15, v_tv22);

        v_av19 = _mm512_sub_pd(v_t10, v_t11);
        v_av20 = _mm512_sub_pd(v_t8, v_t9);
        v_tv23 = _mm512_mul_pd(v_C3, v_av19);
        v_tv23 = CONJ_512_D(v_tv23);
        v_tv24 = _mm512_mul_pd(v_C3, v_av20);
        v_cv33 = _mm512_add_pd(v_tv23, v_tv24);

        v_out3 = _mm512_add_pd(v_cv32, v_cv33);
        v_out9 = _mm512_sub_pd(v_cv32, v_cv33);

        v_tv25 = _mm512_mul_pd(v_C2, v_cv29);
        v_cv34 = _mm512_sub_pd(v_av16, v_tv25);
        v_cv35 = _mm512_sub_pd(v_tv23, v_tv24);

        v_out4  = _mm512_add_pd(v_cv34, v_cv35);
        v_out10 = _mm512_sub_pd(v_cv34, v_cv35);

        v_tv26 = _mm512_mul_pd(v_C2, v_av17);
        v_cv36 = _mm512_add_pd(v_tv26, v_t7);
        v_av21 = _mm512_sub_pd(v_t4, v_t5);
        v_tv27 = _mm512_mul_pd(v_C1, v_av21);
        v_tv28 = _mm512_mul_pd(v_C4, v_av18);
        v_av22 = _mm512_add_pd(v_t8, v_t9);
        v_tv29 = _mm512_mul_pd(v_C3, v_av22);

        v_cv37 = _mm512_add_pd(v_cv36, v_tv27);

        v_cv38 = _mm512_add_pd(v_tv21, v_tv28);
        v_cv38 = CONJ_512_D(v_cv38);
        v_cv39 = _mm512_sub_pd(v_cv38, v_tv29);

        v_out2 = _mm512_add_pd(v_cv37, v_cv39);
        v_out7 = _mm512_sub_pd(v_cv37, v_cv39);

        v_cv40 = _mm512_sub_pd(v_cv36, v_tv27);
        v_cv41 = _mm512_add_pd(v_cv38, v_tv29);

        v_out6  = _mm512_add_pd(v_cv40, v_cv41);
        v_out11 = _mm512_sub_pd(v_cv40, v_cv41);

        curr_out = out_r;
        SCATTER4_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER4_512_D(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER4_512_D(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER4_512_D(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER4_512_D(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER4_512_D(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER4_512_D(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER4_512_D(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER4_512_D(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER4_512_D(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER4_512_D(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER4_512_D(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER4_512_D(curr_out, v_out_stride, v_out12);

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_D)
    {
        // Registers to hold input data points
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m256d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11, v_t12;

        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m256d v_K1  = _mm512_castpd512_pd256(v_C1);
        __m256d v_K2  = _mm512_castpd512_pd256(v_C2);
        __m256d v_K3  = _mm512_castpd512_pd256(v_C3);
        __m256d v_K4  = _mm512_castpd512_pd256(v_C4);
        __m256d v_K5  = _mm512_castpd512_pd256(v_C5);
        __m256d v_K6  = _mm512_castpd512_pd256(v_C6);
        __m256d v_K7  = _mm512_castpd512_pd256(v_C7);
        __m256d v_K8  = _mm512_castpd512_pd256(v_C8);
        __m256d v_K9  = _mm512_castpd512_pd256(v_C9);
        __m256d v_K10 = _mm512_castpd512_pd256(v_C10);

        __m256d v_D1  = _mm512_castpd512_pd256(v_DG1);
        __m256d v_D2  = _mm512_castpd512_pd256(v_DG2);
        __m256d v_D3  = _mm512_castpd512_pd256(v_DG3);
        __m256d v_D4  = _mm512_castpd512_pd256(v_DG4);
        __m256d v_D5  = _mm512_castpd512_pd256(v_DG5);
        __m256d v_D6  = _mm512_castpd512_pd256(v_DG6);
        __m256d v_D7  = _mm512_castpd512_pd256(v_DG7);
        __m256d v_D8  = _mm512_castpd512_pd256(v_DG8);
        __m256d v_D9  = _mm512_castpd512_pd256(v_DG9);
        __m256d v_D10 = _mm512_castpd512_pd256(v_DG10);
        __m256d v_D11 = _mm512_castpd512_pd256(v_DG11);
        __m256d v_D12 = _mm512_castpd512_pd256(v_DG12);

        __m256d v_ZERO_256 = _mm512_castpd512_pd256(v_ZERO);

        curr_in = in_r;
        GATHER2_256_D(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_256_D(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER2_256_D(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER2_256_D(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER2_256_D(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER2_256_D(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER2_256_D(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER2_256_D(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER2_256_D(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER2_256_D(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER2_256_D(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER2_256_D(curr_in, v_in_stride, v_in12);

        v_av1 = _mm256_add_pd(v_in1, v_in12);
        v_av2 = _mm256_add_pd(v_in4, v_in3);
        v_av3 = _mm256_add_pd(v_in9, v_in10);
        v_av4 = _mm256_add_pd(v_in2, v_in6);
        v_av5 = _mm256_add_pd(v_in11, v_in7);
        v_av6 = _mm256_add_pd(v_in8, v_in5);

        v_av7  = _mm256_sub_pd(v_in1, v_in12);
        v_av8  = _mm256_sub_pd(v_in4, v_in3);
        v_av9  = _mm256_sub_pd(v_in9, v_in10);
        v_av10 = _mm256_sub_pd(v_in2, v_in6);
        v_av11 = _mm256_sub_pd(v_in11, v_in7);
        v_av12 = _mm256_sub_pd(v_in8, v_in5);

        v_cv1 = _mm256_add_pd(v_av2, v_av3);
        v_cv2 = _mm256_add_pd(v_av4, v_av5);

        v_cv3 = _mm256_add_pd(v_av1, v_cv1);
        v_cv4 = _mm256_add_pd(v_av6, v_cv2);

        v_tv1 = _mm256_mul_pd(v_cv3, v_D12);
        v_tv2 = _mm256_mul_pd(v_cv4, v_D1);
        v_cv5 = _mm256_add_pd(v_tv1, v_tv2);
        v_t1  = _mm256_sub_pd(v_in0, v_cv5);

        v_tv1 = _mm256_mul_pd(v_cv3, v_D1);
        v_tv2 = _mm256_mul_pd(v_cv4, v_D12);
        v_cv5 = _mm256_add_pd(v_tv1, v_tv2);
        v_t7  = _mm256_sub_pd(v_in0, v_cv5);

        v_cv5 = _mm256_sub_pd(v_cv1, v_cv2);
        v_tv3 = _mm256_mul_pd(v_K2, v_cv5);
        v_cv6 = _mm256_add_pd(v_av10, v_av11);
        v_cv7 = _mm256_add_pd(v_av8, v_av9);
        v_cv8 = _mm256_add_pd(v_cv6, v_cv7);
        v_tv4 = _mm256_mul_pd(v_K6, v_cv8);

        v_cv9  = _mm256_sub_pd(v_av1, v_av6);
        v_cv10 = _mm256_sub_pd(v_cv9, v_tv3);
        v_tv5  = _mm256_mul_pd(v_D4, v_cv10);
        v_t2   = _mm256_add_pd(v_tv4, v_tv5);

        v_tv4 = _mm256_mul_pd(v_K5, v_cv8);
        v_tv5 = _mm256_mul_pd(v_D5, v_cv10);
        v_t8  = _mm256_sub_pd(v_tv4, v_tv5);

        v_cv11 = _mm256_add_pd(v_cv1, v_cv2);
        v_tv6  = _mm256_mul_pd(v_K2, v_cv11);
        v_cv12 = _mm256_add_pd(v_av1, v_av6);
        v_cv13 = _mm256_sub_pd(v_cv12, v_tv6);
        v_cv14 = _mm256_sub_pd(v_cv6, v_cv7);

        v_tv7 = _mm256_mul_pd(v_K8, v_cv14);
        v_tv8 = _mm256_mul_pd(v_D8, v_cv13);
        v_t3  = _mm256_add_pd(v_tv7, v_tv8);

        v_tv7 = _mm256_mul_pd(v_K7, v_cv14);
        v_tv8 = _mm256_mul_pd(v_D9, v_cv13);
        v_t9  = _mm256_sub_pd(v_tv7, v_tv8);

        v_cv15 = _mm256_sub_pd(v_av2, v_av3);
        v_cv16 = _mm256_sub_pd(v_av4, v_av5);
        v_cv17 = _mm256_sub_pd(v_av8, v_av9);
        v_cv18 = _mm256_sub_pd(v_av10, v_av11);
        v_tv9  = _mm256_mul_pd(v_K4, v_cv16);
        v_tv10 = _mm256_mul_pd(v_K3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm256_xor_pd(v_ZERO_256, v_av12);
        v_cv19 = _mm256_add_pd(v_tv9, v_tv11);
        v_cv20 = _mm256_add_pd(v_cv19, v_tv10);

        v_tv12 = _mm256_mul_pd(v_K2, v_cv17);
        v_tv13 = _mm256_mul_pd(v_K1, v_cv18);
        v_cv21 = _mm256_add_pd(v_av7, v_tv12);
        v_cv22 = _mm256_add_pd(v_cv21, v_tv13);

        v_tv14 = _mm256_mul_pd(v_D2, v_cv20);
        v_tv15 = _mm256_mul_pd(v_D3, v_cv22);
        v_t4   = _mm256_add_pd(v_tv14, v_tv15);
        v_t4   = CONJ_256_D(SWAP_RI_256_D(v_t4));

        v_tv14 = _mm256_mul_pd(v_D3, v_cv20);
        v_tv15 = _mm256_mul_pd(v_D2, v_cv22);
        v_t10  = _mm256_sub_pd(v_tv15, v_tv14);
        v_t10  = SWAP_RI_256_D(v_t10);

        v_cv23 = _mm256_sub_pd(v_cv19, v_tv10);
        v_cv24 = _mm256_sub_pd(v_cv21, v_tv13);

        v_tv16 = _mm256_mul_pd(v_D10, v_cv23);
        v_tv17 = _mm256_mul_pd(v_D11, v_cv24);
        v_t5   = _mm256_add_pd(v_tv16, v_tv17);
        v_t5   = SWAP_RI_256_D(CONJ_256_D(v_t5));

        v_tv16 = _mm256_mul_pd(v_D11, v_cv23);
        v_tv17 = _mm256_mul_pd(v_D10, v_cv24);
        v_t11  = _mm256_sub_pd(v_tv16, v_tv17);
        v_t11  = SWAP_RI_256_D(v_t11);

        v_cv25 = _mm256_sub_pd(v_cv16, v_av12);
        v_cv26 = _mm256_sub_pd(v_av7, v_cv17);

        v_tv18 = _mm256_mul_pd(v_K9, v_cv25);
        v_tv19 = _mm256_mul_pd(v_D7, v_cv26);
        v_t6   = _mm256_add_pd(v_tv18, v_tv19);
        v_t6   = CONJ_256_D(SWAP_RI_256_D(v_t6));

        v_tv18 = _mm256_mul_pd(v_K10, v_cv25);
        v_tv19 = _mm256_mul_pd(v_D6, v_cv26);
        v_t12  = _mm256_sub_pd(v_tv19, v_tv18);
        v_t12  = SWAP_RI_256_D(v_t12);

        v_cv27 = _mm256_add_pd(v_cv11, v_cv12);
        v_out0 = _mm256_add_pd(v_in0, v_cv27);

        v_av13 = _mm256_add_pd(v_t2, v_t3);
        v_av14 = _mm256_add_pd(v_t4, v_t5);
        v_av15 = _mm256_add_pd(v_t1, v_t6);

        v_cv28 = _mm256_add_pd(v_av13, v_av14);
        v_out1 = _mm256_add_pd(v_cv28, v_av15);

        v_av16  = _mm256_sub_pd(v_t1, v_t6);
        v_cv29  = _mm256_sub_pd(v_av13, v_av14);
        v_out12 = _mm256_add_pd(v_cv29, v_av16);

        v_av17 = _mm256_sub_pd(v_t2, v_t3);
        v_cv30 = _mm256_sub_pd(v_t7, v_av17);
        v_av18 = _mm256_add_pd(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm256_xor_pd(v_ZERO_256, v_av18);
        v_tv21 = _mm256_xor_pd(v_ZERO_256, v_t12);
        v_cv31 = _mm256_sub_pd(v_tv20, v_tv21);
        v_cv31 = CONJ_256_D(v_cv31);

        v_out8 = _mm256_add_pd(v_cv30, v_cv31);
        v_out5 = _mm256_sub_pd(v_cv30, v_cv31);

        v_tv22 = _mm256_mul_pd(v_K2, v_cv28);
        v_cv32 = _mm256_sub_pd(v_av15, v_tv22);

        v_av19 = _mm256_sub_pd(v_t10, v_t11);
        v_av20 = _mm256_sub_pd(v_t8, v_t9);
        v_tv23 = _mm256_mul_pd(v_K3, v_av19);
        v_tv23 = CONJ_256_D(v_tv23);
        v_tv24 = _mm256_mul_pd(v_K3, v_av20);
        v_cv33 = _mm256_add_pd(v_tv23, v_tv24);

        v_out3 = _mm256_add_pd(v_cv32, v_cv33);
        v_out9 = _mm256_sub_pd(v_cv32, v_cv33);

        v_tv25 = _mm256_mul_pd(v_K2, v_cv29);
        v_cv34 = _mm256_sub_pd(v_av16, v_tv25);
        v_cv35 = _mm256_sub_pd(v_tv23, v_tv24);

        v_out4  = _mm256_add_pd(v_cv34, v_cv35);
        v_out10 = _mm256_sub_pd(v_cv34, v_cv35);

        v_tv26 = _mm256_mul_pd(v_K2, v_av17);
        v_cv36 = _mm256_add_pd(v_tv26, v_t7);
        v_av21 = _mm256_sub_pd(v_t4, v_t5);
        v_tv27 = _mm256_mul_pd(v_K1, v_av21);
        v_tv28 = _mm256_mul_pd(v_K4, v_av18);
        v_av22 = _mm256_add_pd(v_t8, v_t9);
        v_tv29 = _mm256_mul_pd(v_K3, v_av22);

        v_cv37 = _mm256_add_pd(v_cv36, v_tv27);

        v_cv38 = _mm256_add_pd(v_tv21, v_tv28);
        v_cv38 = CONJ_256_D(v_cv38);
        v_cv39 = _mm256_sub_pd(v_cv38, v_tv29);

        v_out2 = _mm256_add_pd(v_cv37, v_cv39);
        v_out7 = _mm256_sub_pd(v_cv37, v_cv39);

        v_cv40 = _mm256_sub_pd(v_cv36, v_tv27);
        v_cv41 = _mm256_add_pd(v_cv38, v_tv29);

        v_out6  = _mm256_add_pd(v_cv40, v_cv41);
        v_out11 = _mm256_sub_pd(v_cv40, v_cv41);

        curr_out = out_r;
        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_256_D(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_256_D(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER2_256_D(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER2_256_D(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER2_256_D(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER2_256_D(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER2_256_D(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER2_256_D(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER2_256_D(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER2_256_D(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER2_256_D(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER2_256_D(curr_out, v_out_stride, v_out12);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    if (n & 1)
    {
        // Registers to hold input data points
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9,
            v_t10, v_t11, v_t12;

        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,v_cv41;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m128d v_K1  = _mm512_castpd512_pd128(v_C1);
        __m128d v_K2  = _mm512_castpd512_pd128(v_C2);
        __m128d v_K3  = _mm512_castpd512_pd128(v_C3);
        __m128d v_K4  = _mm512_castpd512_pd128(v_C4);
        __m128d v_K5  = _mm512_castpd512_pd128(v_C5);
        __m128d v_K6  = _mm512_castpd512_pd128(v_C6);
        __m128d v_K7  = _mm512_castpd512_pd128(v_C7);
        __m128d v_K8  = _mm512_castpd512_pd128(v_C8);
        __m128d v_K9  = _mm512_castpd512_pd128(v_C9);
        __m128d v_K10 = _mm512_castpd512_pd128(v_C10);

        __m128d v_D1  = _mm512_castpd512_pd128(v_DG1);
        __m128d v_D2  = _mm512_castpd512_pd128(v_DG2);
        __m128d v_D3  = _mm512_castpd512_pd128(v_DG3);
        __m128d v_D4  = _mm512_castpd512_pd128(v_DG4);
        __m128d v_D5  = _mm512_castpd512_pd128(v_DG5);
        __m128d v_D6  = _mm512_castpd512_pd128(v_DG6);
        __m128d v_D7  = _mm512_castpd512_pd128(v_DG7);
        __m128d v_D8  = _mm512_castpd512_pd128(v_DG8);
        __m128d v_D9  = _mm512_castpd512_pd128(v_DG9);
        __m128d v_D10 = _mm512_castpd512_pd128(v_DG10);
        __m128d v_D11 = _mm512_castpd512_pd128(v_DG11);
        __m128d v_D12 = _mm512_castpd512_pd128(v_DG12);

        __m128d v_ZERO_128 = _mm512_castpd512_pd128(v_ZERO);

        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_128_D(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_128_D(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_128_D(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_128_D(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_128_D(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_128_D(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_128_D(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_128_D(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_128_D(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_128_D(curr_in, v_in12);

        v_av1 = _mm_add_pd(v_in1, v_in12);
        v_av2 = _mm_add_pd(v_in4, v_in3);
        v_av3 = _mm_add_pd(v_in9, v_in10);
        v_av4 = _mm_add_pd(v_in2, v_in6);
        v_av5 = _mm_add_pd(v_in11, v_in7);
        v_av6 = _mm_add_pd(v_in8, v_in5);

        v_av7  = _mm_sub_pd(v_in1, v_in12);
        v_av8  = _mm_sub_pd(v_in4, v_in3);
        v_av9  = _mm_sub_pd(v_in9, v_in10);
        v_av10 = _mm_sub_pd(v_in2, v_in6);
        v_av11 = _mm_sub_pd(v_in11, v_in7);
        v_av12 = _mm_sub_pd(v_in8, v_in5);

        v_cv1 = _mm_add_pd(v_av2, v_av3);
        v_cv2 = _mm_add_pd(v_av4, v_av5);

        v_cv3 = _mm_add_pd(v_av1, v_cv1);
        v_cv4 = _mm_add_pd(v_av6, v_cv2);

        v_tv1 = _mm_mul_pd(v_cv3, v_D12);
        v_tv2 = _mm_mul_pd(v_cv4, v_D1);
        v_cv5 = _mm_add_pd(v_tv1, v_tv2);
        v_t1  = _mm_sub_pd(v_in0, v_cv5);

        v_tv1 = _mm_mul_pd(v_cv3, v_D1);
        v_tv2 = _mm_mul_pd(v_cv4, v_D12);
        v_cv5 = _mm_add_pd(v_tv1, v_tv2);
        v_t7  = _mm_sub_pd(v_in0, v_cv5);

        v_cv5 = _mm_sub_pd(v_cv1, v_cv2);
        v_tv3 = _mm_mul_pd(v_K2, v_cv5);
        v_cv6 = _mm_add_pd(v_av10, v_av11);
        v_cv7 = _mm_add_pd(v_av8, v_av9);
        v_cv8 = _mm_add_pd(v_cv6, v_cv7);
        v_tv4 = _mm_mul_pd(v_K6, v_cv8);

        v_cv9  = _mm_sub_pd(v_av1, v_av6);
        v_cv10 = _mm_sub_pd(v_cv9, v_tv3);
        v_tv5  = _mm_mul_pd(v_D4, v_cv10);
        v_t2   = _mm_add_pd(v_tv4, v_tv5);

        v_tv4 = _mm_mul_pd(v_K5, v_cv8);
        v_tv5 = _mm_mul_pd(v_D5, v_cv10);
        v_t8  = _mm_sub_pd(v_tv4, v_tv5);

        v_cv11 = _mm_add_pd(v_cv1, v_cv2);
        v_tv6  = _mm_mul_pd(v_K2, v_cv11);
        v_cv12 = _mm_add_pd(v_av1, v_av6);
        v_cv13 = _mm_sub_pd(v_cv12, v_tv6);
        v_cv14 = _mm_sub_pd(v_cv6, v_cv7);

        v_tv7 = _mm_mul_pd(v_K8, v_cv14);
        v_tv8 = _mm_mul_pd(v_D8, v_cv13);
        v_t3  = _mm_add_pd(v_tv7, v_tv8);

        v_tv7 = _mm_mul_pd(v_K7, v_cv14);
        v_tv8 = _mm_mul_pd(v_D9, v_cv13);
        v_t9  = _mm_sub_pd(v_tv7, v_tv8);

        v_cv15 = _mm_sub_pd(v_av2, v_av3);
        v_cv16 = _mm_sub_pd(v_av4, v_av5);
        v_cv17 = _mm_sub_pd(v_av8, v_av9);
        v_cv18 = _mm_sub_pd(v_av10, v_av11);
        v_tv9  = _mm_mul_pd(v_K4, v_cv16);
        v_tv10 = _mm_mul_pd(v_K3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = _mm_xor_pd(v_ZERO_128, v_av12);
        v_cv19 = _mm_add_pd(v_tv9, v_tv11);
        v_cv20 = _mm_add_pd(v_cv19, v_tv10);

        v_tv12 = _mm_mul_pd(v_K2, v_cv17);
        v_tv13 = _mm_mul_pd(v_K1, v_cv18);
        v_cv21 = _mm_add_pd(v_av7, v_tv12);
        v_cv22 = _mm_add_pd(v_cv21, v_tv13);

        v_tv14 = _mm_mul_pd(v_D2, v_cv20);
        v_tv15 = _mm_mul_pd(v_D3, v_cv22);
        v_t4   = _mm_add_pd(v_tv14, v_tv15);
        v_t4   = CONJ_128_D(SWAP_RI_128_D(v_t4));

        v_tv14 = _mm_mul_pd(v_D3, v_cv20);
        v_tv15 = _mm_mul_pd(v_D2, v_cv22);
        v_t10  = _mm_sub_pd(v_tv15, v_tv14);
        v_t10  = SWAP_RI_128_D(v_t10);

        v_cv23 = _mm_sub_pd(v_cv19, v_tv10);
        v_cv24 = _mm_sub_pd(v_cv21, v_tv13);

        v_tv16 = _mm_mul_pd(v_D10, v_cv23);
        v_tv17 = _mm_mul_pd(v_D11, v_cv24);
        v_t5   = _mm_add_pd(v_tv16, v_tv17);
        v_t5   = SWAP_RI_128_D(CONJ_128_D(v_t5));

        v_tv16 = _mm_mul_pd(v_D11, v_cv23);
        v_tv17 = _mm_mul_pd(v_D10, v_cv24);
        v_t11  = _mm_sub_pd(v_tv16, v_tv17);
        v_t11  = SWAP_RI_128_D(v_t11);

        v_cv25 = _mm_sub_pd(v_cv16, v_av12);
        v_cv26 = _mm_sub_pd(v_av7, v_cv17);

        v_tv18 = _mm_mul_pd(v_K9, v_cv25);
        v_tv19 = _mm_mul_pd(v_D7, v_cv26);
        v_t6   = _mm_add_pd(v_tv18, v_tv19);
        v_t6   = CONJ_128_D(SWAP_RI_128_D(v_t6));

        v_tv18 = _mm_mul_pd(v_K10, v_cv25);
        v_tv19 = _mm_mul_pd(v_D6, v_cv26);
        v_t12  = _mm_sub_pd(v_tv19, v_tv18);
        v_t12  = SWAP_RI_128_D(v_t12);

        v_cv27 = _mm_add_pd(v_cv11, v_cv12);
        v_out0 = _mm_add_pd(v_in0, v_cv27);

        v_av13 = _mm_add_pd(v_t2, v_t3);
        v_av14 = _mm_add_pd(v_t4, v_t5);
        v_av15 = _mm_add_pd(v_t1, v_t6);

        v_cv28 = _mm_add_pd(v_av13, v_av14);
        v_out1 = _mm_add_pd(v_cv28, v_av15);

        v_av16  = _mm_sub_pd(v_t1, v_t6);
        v_cv29  = _mm_sub_pd(v_av13, v_av14);
        v_out12 = _mm_add_pd(v_cv29, v_av16);

        v_av17 = _mm_sub_pd(v_t2, v_t3);
        v_cv30 = _mm_sub_pd(v_t7, v_av17);
        v_av18 = _mm_add_pd(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = _mm_xor_pd(v_ZERO_128, v_av18);
        v_tv21 = _mm_xor_pd(v_ZERO_128, v_t12);
        v_cv31 = _mm_sub_pd(v_tv20, v_tv21);
        v_cv31 = CONJ_128_D(v_cv31);

        v_out8 = _mm_add_pd(v_cv30, v_cv31);
        v_out5 = _mm_sub_pd(v_cv30, v_cv31);

        v_tv22 = _mm_mul_pd(v_K2, v_cv28);
        v_cv32 = _mm_sub_pd(v_av15, v_tv22);

        v_av19 = _mm_sub_pd(v_t10, v_t11);
        v_av20 = _mm_sub_pd(v_t8, v_t9);
        v_tv23 = _mm_mul_pd(v_K3, v_av19);
        v_tv23 = CONJ_128_D(v_tv23);
        v_tv24 = _mm_mul_pd(v_K3, v_av20);
        v_cv33 = _mm_add_pd(v_tv23, v_tv24);

        v_out3 = _mm_add_pd(v_cv32, v_cv33);
        v_out9 = _mm_sub_pd(v_cv32, v_cv33);

        v_tv25 = _mm_mul_pd(v_K2, v_cv29);
        v_cv34 = _mm_sub_pd(v_av16, v_tv25);
        v_cv35 = _mm_sub_pd(v_tv23, v_tv24);

        v_out4  = _mm_add_pd(v_cv34, v_cv35);
        v_out10 = _mm_sub_pd(v_cv34, v_cv35);

        v_tv26 = _mm_mul_pd(v_K2, v_av17);
        v_cv36 = _mm_add_pd(v_tv26, v_t7);
        v_av21 = _mm_sub_pd(v_t4, v_t5);
        v_tv27 = _mm_mul_pd(v_K1, v_av21);
        v_tv28 = _mm_mul_pd(v_K4, v_av18);
        v_av22 = _mm_add_pd(v_t8, v_t9);
        v_tv29 = _mm_mul_pd(v_K3, v_av22);

        v_cv37 = _mm_add_pd(v_cv36, v_tv27);

        v_cv38 = _mm_add_pd(v_tv21, v_tv28);
        v_cv38 = CONJ_128_D(v_cv38);
        v_cv39 = _mm_sub_pd(v_cv38, v_tv29);

        v_out2 = _mm_add_pd(v_cv37, v_cv39);
        v_out7 = _mm_sub_pd(v_cv37, v_cv39);

        v_cv40 = _mm_sub_pd(v_cv36, v_tv27);
        v_cv41 = _mm_add_pd(v_cv38, v_tv29);

        v_out6  = _mm_add_pd(v_cv40, v_cv41);
        v_out11 = _mm_sub_pd(v_cv40, v_cv41);

        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_128_D(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_128_D(curr_out, v_out5);
        curr_out = out_r + out_strides[6];
        ST_128_D(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_128_D(curr_out, v_out7);
        curr_out = out_r + out_strides[8];
        ST_128_D(curr_out, v_out8);
        curr_out = out_r + out_strides[9];
        ST_128_D(curr_out, v_out9);
        curr_out = out_r + out_strides[10];
        ST_128_D(curr_out, v_out10);
        curr_out = out_r + out_strides[11];
        ST_128_D(curr_out, v_out11);
        curr_out = out_r + out_strides[12];
        ST_128_D(curr_out, v_out12);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft13avx512(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft13avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft13avx512fp64;
    }
    else
    {
        return NULL;
    }
}
