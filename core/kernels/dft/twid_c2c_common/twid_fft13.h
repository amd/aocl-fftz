// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft13.h
 *
 *  @brief The ISA generic kernel template for the radix 13 twiddle kernel
 *
 *  This file contains the DIT twiddle radix-13 FFT implementations for
 *  single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

// This header has no include guards.
// This is intentional.
// The functions defined in this file are not usable by default.
// They are "instantiated" only when "included" in another file.

#include "core/kernels/simd_includes/generic_kernels_common.h"

#define RADIX 13

static FFTZ_VOID TWID_KNAME_FP32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_13[10] = {
        0.866025403784438646763723170752936183471402627f,
        0.500000000000000000000000000000000000000000000f,
        0.866025403784438646763723170752936183471402627f,
        0.500000000000000000000000000000000000000000000f,
        0.447320117602511140667282045633987571324387014f,  // 2*DGC[3]*CRTM[3]
        0.265966249214837287587521063842185948798330267f,  // 2*DGC[4]*CRTM[3]
        0.131467828262610852858973617628781241523254441f,  // 2*DGC[7]*CRTM[3]
        0.503537032863766627246873853868466977093348562f,  // 2*DGC[8]*CRTM[3]
        0.575140729474003121368385547455453388461001608f,  // 2*DGC[5]*CRTM[2]
        0.174138601152135905005660794929264742616964676f}; // 2*DGC[6]*CRTM[2]

    const FFTZ_FLOAT DGC[12] = {
        0.383795939621999107759935105622541328854274714f,   // DGC[0] + DGC[11]
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

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_FLOAT *in_h2_r = (FFTZ_FLOAT *)in_real;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_FLOAT *out_h2_r = (FFTZ_FLOAT *)out_real;
#endif

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)tws->TW;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_S;

    FFTZ_FLOAT *tw_ptr = tw;

    VREGTYPE_S v_C1 = BCAST_S(CRTM_13[0]);
    VREGTYPE_S v_C2 = BCAST_S(CRTM_13[1]);
    VREGTYPE_S v_C3 = BCAST_S(CRTM_13[2]);
    VREGTYPE_S v_C4 = BCAST_S(CRTM_13[3]);
    VREGTYPE_S v_C5 = BCAST_S(CRTM_13[4]);
    VREGTYPE_S v_C6 = BCAST_S(CRTM_13[5]);
    VREGTYPE_S v_C7 = BCAST_S(CRTM_13[6]);
    VREGTYPE_S v_C8 = BCAST_S(CRTM_13[7]);
    VREGTYPE_S v_C9 = BCAST_S(CRTM_13[8]);
    VREGTYPE_S v_C10 = BCAST_S(CRTM_13[9]);

    VREGTYPE_S v_DG1 = BCAST_S(DGC[0]);
    VREGTYPE_S v_DG2 = BCAST_S(DGC[1]);
    VREGTYPE_S v_DG3 = BCAST_S(DGC[2]);
    VREGTYPE_S v_DG4 = BCAST_S(DGC[3]);
    VREGTYPE_S v_DG5 = BCAST_S(DGC[4]);
    VREGTYPE_S v_DG6 = BCAST_S(DGC[5]);
    VREGTYPE_S v_DG7 = BCAST_S(DGC[6]);
    VREGTYPE_S v_DG8 = BCAST_S(DGC[7]);
    VREGTYPE_S v_DG9 = BCAST_S(DGC[8]);
    VREGTYPE_S v_DG10 = BCAST_S(DGC[9]);
    VREGTYPE_S v_DG11 = BCAST_S(DGC[10]);
    VREGTYPE_S v_DG12 = BCAST_S(DGC[11]);

    VREGTYPE_S v_ZERO = NEG_ZERO_S(flag);

    FFTZ_INTP count;

#if defined(KERNEL_DIRECTION_BWD)
    v_C3 = NEG_S(v_C3, 1);
    v_C4 = NEG_S(v_C4, 1);
    v_C5 = NEG_S(v_C5, 1);
    v_C7 = NEG_S(v_C7, 1);
    v_C9 = NEG_S(v_C9, 1);
    v_DG3 = NEG_S(v_DG3, 1);
    v_DG5 = NEG_S(v_DG5, 1);
    v_DG7 = NEG_S(v_DG7, 1);
    v_DG9 = NEG_S(v_DG9, 1);
    v_DG11 = NEG_S(v_DG11, 1);
#endif
    for (count = 0; count < N; count++)
    {
        // Registers to hold input data points
        VREGTYPE_S v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        VREGTYPE_S v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal
        // constants
        VREGTYPE_S v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        VREGTYPE_S v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9, v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16,
            v_av17, v_av18, v_av19, v_av20, v_av21, v_av22;
        VREGTYPE_S v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9, v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16,
            v_cv17, v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24,
            v_cv25, v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32,
            v_cv33, v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,
            v_cv41;
        VREGTYPE_S v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
            v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16,
            v_tv17, v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24,
            v_tv25, v_tv26, v_tv27, v_tv28, v_tv29;

        LOAD_IN_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw_ptr,
                  load_multi_cols, 0);
        v_in7 = IN_H2_S(v_in7);
        LOAD_IN_H2_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                  load_multi_cols, 0);
        v_in8 = IN_H2_S(v_in8);
        LOAD_IN_H2_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                  load_multi_cols, 0);
        v_in9 = IN_H2_S(v_in9);
        LOAD_IN_H2_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw_ptr,
                  load_multi_cols, 0);
        v_in10 = IN_H2_S(v_in10);
        LOAD_IN_H2_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw_ptr,
                  load_multi_cols, 0);
        v_in11 = IN_H2_S(v_in11);
        LOAD_IN_H2_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12, tw_ptr,
                  load_multi_cols, 0);
        v_in12 = IN_H2_S(v_in12);
#else
        LOAD_IN_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

        GATHER_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = ADD_S(v_in1, v_in12);
        v_av2 = ADD_S(v_in4, v_in3);
        v_av3 = ADD_S(v_in9, v_in10);
        v_av4 = ADD_S(v_in2, v_in6);
        v_av5 = ADD_S(v_in11, v_in7);
        v_av6 = ADD_S(v_in8, v_in5);

        v_av7 = SUB_S(v_in1, v_in12);
        v_av8 = SUB_S(v_in4, v_in3);
        v_av9 = SUB_S(v_in9, v_in10);
        v_av10 = SUB_S(v_in2, v_in6);
        v_av11 = SUB_S(v_in11, v_in7);
        v_av12 = SUB_S(v_in8, v_in5);

        v_cv1 = ADD_S(v_av2, v_av3);
        v_cv2 = ADD_S(v_av4, v_av5);
        v_cv3 = ADD_S(v_av1, v_cv1);
        v_cv4 = ADD_S(v_av6, v_cv2);

        v_tv1 = MUL_S(v_cv3, v_DG12);
        v_tv2 = MUL_S(v_cv4, v_DG1);
        v_cv5 = ADD_S(v_tv1, v_tv2);
        v_t1 = SUB_S(v_in0, v_cv5);

        v_tv1 = MUL_S(v_cv3, v_DG1);
        v_tv2 = MUL_S(v_cv4, v_DG12);
        v_cv5 = ADD_S(v_tv1, v_tv2);
        v_t7 = SUB_S(v_in0, v_cv5);

        v_cv5 = SUB_S(v_cv1, v_cv2);
        v_tv3 = MUL_S(v_C2, v_cv5);
        v_cv6 = ADD_S(v_av10, v_av11);
        v_cv7 = ADD_S(v_av8, v_av9);
        v_cv8 = ADD_S(v_cv6, v_cv7);
        v_tv4 = MUL_S(v_C6, v_cv8);

        v_cv9 = SUB_S(v_av1, v_av6);
        v_cv10 = SUB_S(v_cv9, v_tv3);
        v_tv5 = MUL_S(v_DG4, v_cv10);
        v_t2 = ADD_S(v_tv4, v_tv5);

        v_tv4 = MUL_S(v_C5, v_cv8);
        v_tv5 = MUL_S(v_DG5, v_cv10);
        v_t8 = SUB_S(v_tv4, v_tv5);

        v_cv11 = ADD_S(v_cv1, v_cv2);
        v_tv6 = MUL_S(v_C2, v_cv11);
        v_cv12 = ADD_S(v_av1, v_av6);
        v_cv13 = SUB_S(v_cv12, v_tv6);
        v_cv14 = SUB_S(v_cv6, v_cv7);

        v_tv7 = MUL_S(v_C8, v_cv14);
        v_tv8 = MUL_S(v_DG8, v_cv13);
        v_t3 = ADD_S(v_tv7, v_tv8);

        v_tv7 = MUL_S(v_C7, v_cv14);
        v_tv8 = MUL_S(v_DG9, v_cv13);
        v_t9 = SUB_S(v_tv7, v_tv8);

        v_cv15 = SUB_S(v_av2, v_av3);
        v_cv16 = SUB_S(v_av4, v_av5);
        v_cv17 = SUB_S(v_av8, v_av9);
        v_cv18 = SUB_S(v_av10, v_av11);
        v_tv9 = MUL_S(v_C4, v_cv16);
        v_tv10 = MUL_S(v_C3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = XOR_S(v_ZERO, v_av12);
        v_cv19 = ADD_S(v_tv9, v_tv11);
        v_cv20 = ADD_S(v_cv19, v_tv10);

        v_tv12 = MUL_S(v_C2, v_cv17);
        v_tv13 = MUL_S(v_C1, v_cv18);
        v_cv21 = ADD_S(v_av7, v_tv12);
        v_cv22 = ADD_S(v_cv21, v_tv13);

        v_tv14 = MUL_S(v_DG2, v_cv20);
        v_tv15 = MUL_S(v_DG3, v_cv22);
        v_t4 = ADD_S(v_tv14, v_tv15);
        v_t4 = CONJ_S(SWAP_RI_S(v_t4));

        v_tv14 = MUL_S(v_DG3, v_cv20);
        v_tv15 = MUL_S(v_DG2, v_cv22);
        v_t10 = SUB_S(v_tv15, v_tv14);
        v_t10 = SWAP_RI_S(v_t10);

        v_cv23 = SUB_S(v_cv19, v_tv10);
        v_cv24 = SUB_S(v_cv21, v_tv13);

        v_tv16 = MUL_S(v_DG10, v_cv23);
        v_tv17 = MUL_S(v_DG11, v_cv24);
        v_t5 = ADD_S(v_tv16, v_tv17);
        v_t5 = SWAP_RI_S(CONJ_S(v_t5));

        v_tv16 = MUL_S(v_DG11, v_cv23);
        v_tv17 = MUL_S(v_DG10, v_cv24);
        v_t11 = SUB_S(v_tv16, v_tv17);
        v_t11 = SWAP_RI_S(v_t11);

        v_cv25 = SUB_S(v_cv16, v_av12);
        v_cv26 = SUB_S(v_av7, v_cv17);

        v_tv18 = MUL_S(v_C9, v_cv25);
        v_tv19 = MUL_S(v_DG7, v_cv26);
        v_t6 = ADD_S(v_tv18, v_tv19);
        v_t6 = CONJ_S(SWAP_RI_S(v_t6));

        v_tv18 = MUL_S(v_C10, v_cv25);
        v_tv19 = MUL_S(v_DG6, v_cv26);
        v_t12 = SUB_S(v_tv19, v_tv18);
        v_t12 = SWAP_RI_S(v_t12);

        v_cv27 = ADD_S(v_cv11, v_cv12);
        v_out0 = ADD_S(v_in0, v_cv27);

        v_av13 = ADD_S(v_t2, v_t3);
        v_av14 = ADD_S(v_t4, v_t5);
        v_av15 = ADD_S(v_t1, v_t6);

        v_cv28 = ADD_S(v_av13, v_av14);
        v_out1 = ADD_S(v_cv28, v_av15);

        v_av16 = SUB_S(v_t1, v_t6);
        v_cv29 = SUB_S(v_av13, v_av14);
        v_out12 = ADD_S(v_cv29, v_av16);

        v_av17 = SUB_S(v_t2, v_t3);
        v_cv30 = SUB_S(v_t7, v_av17);
        v_av18 = ADD_S(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = XOR_S(v_ZERO, v_av18);
        v_tv21 = XOR_S(v_ZERO, v_t12);
        v_cv31 = SUB_S(v_tv20, v_tv21);
        v_cv31 = CONJ_S(v_cv31);

        v_out8 = ADD_S(v_cv30, v_cv31);
        v_out5 = SUB_S(v_cv30, v_cv31);

        v_tv22 = MUL_S(v_C2, v_cv28);
        v_cv32 = SUB_S(v_av15, v_tv22);

        v_av19 = SUB_S(v_t10, v_t11);
        v_av20 = SUB_S(v_t8, v_t9);
        v_tv23 = MUL_S(v_C3, v_av19);
        v_tv23 = CONJ_S(v_tv23);
        v_tv24 = MUL_S(v_C3, v_av20);
        v_cv33 = ADD_S(v_tv23, v_tv24);

        v_out3 = ADD_S(v_cv32, v_cv33);
        v_out9 = SUB_S(v_cv32, v_cv33);

        v_tv25 = MUL_S(v_C2, v_cv29);
        v_cv34 = SUB_S(v_av16, v_tv25);
        v_cv35 = SUB_S(v_tv23, v_tv24);

        v_out4 = ADD_S(v_cv34, v_cv35);
        v_out10 = SUB_S(v_cv34, v_cv35);

        v_tv26 = MUL_S(v_C2, v_av17);
        v_cv36 = ADD_S(v_tv26, v_t7);
        v_av21 = SUB_S(v_t4, v_t5);
        v_tv27 = MUL_S(v_C1, v_av21);
        v_tv28 = MUL_S(v_C4, v_av18);
        v_av22 = ADD_S(v_t8, v_t9);
        v_tv29 = MUL_S(v_C3, v_av22);

        v_cv37 = ADD_S(v_cv36, v_tv27);
        v_cv38 = ADD_S(v_tv21, v_tv28);
        v_cv38 = CONJ_S(v_cv38);
        v_cv39 = SUB_S(v_cv38, v_tv29);

        v_out2 = ADD_S(v_cv37, v_cv39);
        v_out7 = SUB_S(v_cv37, v_cv39);

        v_cv40 = SUB_S(v_cv36, v_tv27);
        v_cv41 = ADD_S(v_cv38, v_tv29);

        v_out6 = ADD_S(v_cv40, v_cv41);
        v_out11 = SUB_S(v_cv40, v_cv41);

        SCATTER_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_S(v_out7);
        STORE_OUT_H2_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7,
                       tw_ptr, load_multi_cols, 0);
        v_out8 = OUT_H2_S(v_out8);
        STORE_OUT_H2_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                       tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_S(v_out9);
        STORE_OUT_H2_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                       tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_S(v_out10);
        STORE_OUT_H2_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                       tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_S(v_out11);
        STORE_OUT_H2_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                       tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_S(v_out12);
        STORE_OUT_H2_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                       tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#endif

        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_S * DATA_STRIDE;
        in_r += NUM_SETS_S * v_in_stride;
        out_r += NUM_SETS_S * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_S * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_S * v_out_h2_stride;
#endif
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_S)
    {
        // Registers to hold input data points
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m256 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        __m256 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m256 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40, v_cv41;
        __m256 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m256 v_K1 = CAST_512_TO_256_S(v_C1);
        __m256 v_K2 = CAST_512_TO_256_S(v_C2);
        __m256 v_K3 = CAST_512_TO_256_S(v_C3);
        __m256 v_K4 = CAST_512_TO_256_S(v_C4);
        __m256 v_K5 = CAST_512_TO_256_S(v_C5);
        __m256 v_K6 = CAST_512_TO_256_S(v_C6);
        __m256 v_K7 = CAST_512_TO_256_S(v_C7);
        __m256 v_K8 = CAST_512_TO_256_S(v_C8);
        __m256 v_K9 = CAST_512_TO_256_S(v_C9);
        __m256 v_K10 = CAST_512_TO_256_S(v_C10);

        __m256 v_D1 = CAST_512_TO_256_S(v_DG1);
        __m256 v_D2 = CAST_512_TO_256_S(v_DG2);
        __m256 v_D3 = CAST_512_TO_256_S(v_DG3);
        __m256 v_D4 = CAST_512_TO_256_S(v_DG4);
        __m256 v_D5 = CAST_512_TO_256_S(v_DG5);
        __m256 v_D6 = CAST_512_TO_256_S(v_DG6);
        __m256 v_D7 = CAST_512_TO_256_S(v_DG7);
        __m256 v_D8 = CAST_512_TO_256_S(v_DG8);
        __m256 v_D9 = CAST_512_TO_256_S(v_DG9);
        __m256 v_D10 = CAST_512_TO_256_S(v_DG10);
        __m256 v_D11 = CAST_512_TO_256_S(v_DG11);
        __m256 v_D12 = CAST_512_TO_256_S(v_DG12);

        __m256 v_ZERO_256 = CAST_512_TO_256_S(v_ZERO);

        LOAD_IN_256_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw_ptr,
                      load_multi_cols, 0);
        v_in7 = IN_H2_256_S(v_in7);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_256_S(v_in8);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_256_S(v_in9);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_256_S(v_in10);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_256_S(v_in11);
        LOAD_IN_H2_256_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_256_S(v_in12);
#else
        LOAD_IN_256_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER4_256_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm256_add_ps(v_in1, v_in12);
        v_av2 = _mm256_add_ps(v_in4, v_in3);
        v_av3 = _mm256_add_ps(v_in9, v_in10);
        v_av4 = _mm256_add_ps(v_in2, v_in6);
        v_av5 = _mm256_add_ps(v_in11, v_in7);
        v_av6 = _mm256_add_ps(v_in8, v_in5);

        v_av7 = _mm256_sub_ps(v_in1, v_in12);
        v_av8 = _mm256_sub_ps(v_in4, v_in3);
        v_av9 = _mm256_sub_ps(v_in9, v_in10);
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
        v_t1 = _mm256_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm256_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm256_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm256_add_ps(v_tv1, v_tv2);
        v_t7 = _mm256_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm256_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm256_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm256_add_ps(v_av10, v_av11);
        v_cv7 = _mm256_add_ps(v_av8, v_av9);
        v_cv8 = _mm256_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm256_mul_ps(v_K6, v_cv8);

        v_cv9 = _mm256_sub_ps(v_av1, v_av6);
        v_cv10 = _mm256_sub_ps(v_cv9, v_tv3);
        v_tv5 = _mm256_mul_ps(v_D4, v_cv10);
        v_t2 = _mm256_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm256_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm256_mul_ps(v_D5, v_cv10);
        v_t8 = _mm256_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm256_add_ps(v_cv1, v_cv2);
        v_tv6 = _mm256_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm256_add_ps(v_av1, v_av6);
        v_cv13 = _mm256_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm256_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm256_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm256_mul_ps(v_D8, v_cv13);
        v_t3 = _mm256_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm256_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm256_mul_ps(v_D9, v_cv13);
        v_t9 = _mm256_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm256_sub_ps(v_av2, v_av3);
        v_cv16 = _mm256_sub_ps(v_av4, v_av5);
        v_cv17 = _mm256_sub_ps(v_av8, v_av9);
        v_cv18 = _mm256_sub_ps(v_av10, v_av11);
        v_tv9 = _mm256_mul_ps(v_K4, v_cv16);
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
        v_t4 = _mm256_add_ps(v_tv14, v_tv15);
        v_t4 = CONJ_256_S(SWAP_RI_256_S(v_t4));

        v_tv14 = _mm256_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm256_mul_ps(v_D2, v_cv22);
        v_t10 = _mm256_sub_ps(v_tv15, v_tv14);
        v_t10 = SWAP_RI_256_S(v_t10);

        v_cv23 = _mm256_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm256_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm256_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm256_mul_ps(v_D11, v_cv24);
        v_t5 = _mm256_add_ps(v_tv16, v_tv17);
        v_t5 = SWAP_RI_256_S(CONJ_256_S(v_t5));

        v_tv16 = _mm256_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm256_mul_ps(v_D10, v_cv24);
        v_t11 = _mm256_sub_ps(v_tv16, v_tv17);
        v_t11 = SWAP_RI_256_S(v_t11);

        v_cv25 = _mm256_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm256_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm256_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm256_mul_ps(v_D7, v_cv26);
        v_t6 = _mm256_add_ps(v_tv18, v_tv19);
        v_t6 = CONJ_256_S(SWAP_RI_256_S(v_t6));

        v_tv18 = _mm256_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm256_mul_ps(v_D6, v_cv26);
        v_t12 = _mm256_sub_ps(v_tv19, v_tv18);
        v_t12 = SWAP_RI_256_S(v_t12);

        v_cv27 = _mm256_add_ps(v_cv11, v_cv12);
        v_out0 = _mm256_add_ps(v_in0, v_cv27);

        v_av13 = _mm256_add_ps(v_t2, v_t3);
        v_av14 = _mm256_add_ps(v_t4, v_t5);
        v_av15 = _mm256_add_ps(v_t1, v_t6);

        v_cv28 = _mm256_add_ps(v_av13, v_av14);
        v_out1 = _mm256_add_ps(v_cv28, v_av15);

        v_av16 = _mm256_sub_ps(v_t1, v_t6);
        v_cv29 = _mm256_sub_ps(v_av13, v_av14);
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

        v_out4 = _mm256_add_ps(v_cv34, v_cv35);
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

        v_out6 = _mm256_add_ps(v_cv40, v_cv41);
        v_out11 = _mm256_sub_ps(v_cv40, v_cv41);

        SCATTER4_256_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_256_S(v_out7);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7,
                        tw_ptr, load_multi_cols, 0);
        v_out8 = OUT_H2_256_S(v_out8);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_256_S(v_out9);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_256_S(v_out10);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_256_S(v_out11);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_256_S(v_out12);
        STORE_OUT_H2_256_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_256_S * DATA_STRIDE;

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_256_S * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_256_S * v_out_h2_stride;
#endif
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets >= NUM_SETS_128_S)
    {
        // Registers to hold input data points
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40, v_cv41;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
        __m128 v_K8 = CAST_512_TO_128_S(v_C8);
        __m128 v_K9 = CAST_512_TO_128_S(v_C9);
        __m128 v_K10 = CAST_512_TO_128_S(v_C10);
        __m128 v_D1 = CAST_512_TO_128_S(v_DG1);
        __m128 v_D2 = CAST_512_TO_128_S(v_DG2);
        __m128 v_D3 = CAST_512_TO_128_S(v_DG3);
        __m128 v_D4 = CAST_512_TO_128_S(v_DG4);
        __m128 v_D5 = CAST_512_TO_128_S(v_DG5);
        __m128 v_D6 = CAST_512_TO_128_S(v_DG6);
        __m128 v_D7 = CAST_512_TO_128_S(v_DG7);
        __m128 v_D8 = CAST_512_TO_128_S(v_DG8);
        __m128 v_D9 = CAST_512_TO_128_S(v_DG9);
        __m128 v_D10 = CAST_512_TO_128_S(v_DG10);
        __m128 v_D11 = CAST_512_TO_128_S(v_DG11);
        __m128 v_D12 = CAST_512_TO_128_S(v_DG12);
        __m128 v_ZERO_128 = CAST_512_TO_128_S(v_ZERO);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
        __m128 v_K8 = CAST_256_TO_128_S(v_C8);
        __m128 v_K9 = CAST_256_TO_128_S(v_C9);
        __m128 v_K10 = CAST_256_TO_128_S(v_C10);
        __m128 v_D1 = CAST_256_TO_128_S(v_DG1);
        __m128 v_D2 = CAST_256_TO_128_S(v_DG2);
        __m128 v_D3 = CAST_256_TO_128_S(v_DG3);
        __m128 v_D4 = CAST_256_TO_128_S(v_DG4);
        __m128 v_D5 = CAST_256_TO_128_S(v_DG5);
        __m128 v_D6 = CAST_256_TO_128_S(v_DG6);
        __m128 v_D7 = CAST_256_TO_128_S(v_DG7);
        __m128 v_D8 = CAST_256_TO_128_S(v_DG8);
        __m128 v_D9 = CAST_256_TO_128_S(v_DG9);
        __m128 v_D10 = CAST_256_TO_128_S(v_DG10);
        __m128 v_D11 = CAST_256_TO_128_S(v_DG11);
        __m128 v_D12 = CAST_256_TO_128_S(v_DG12);
        __m128 v_ZERO_128 = CAST_256_TO_128_S(v_ZERO);
#endif

        LOAD_IN_128_S(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw_ptr,
                      load_multi_cols, 0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
        LOAD_IN_H2_128_S(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_128_S(v_in12);
#else
        LOAD_IN_128_S(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_128_S(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        GATHER2_128_S(in_r, v_in_stride, v_in0, is_contiguous_in);

        v_av1 = _mm_add_ps(v_in1, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in3);
        v_av3 = _mm_add_ps(v_in9, v_in10);
        v_av4 = _mm_add_ps(v_in2, v_in6);
        v_av5 = _mm_add_ps(v_in11, v_in7);
        v_av6 = _mm_add_ps(v_in8, v_in5);

        v_av7 = _mm_sub_ps(v_in1, v_in12);
        v_av8 = _mm_sub_ps(v_in4, v_in3);
        v_av9 = _mm_sub_ps(v_in9, v_in10);
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
        v_t1 = _mm_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t7 = _mm_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm_add_ps(v_av10, v_av11);
        v_cv7 = _mm_add_ps(v_av8, v_av9);
        v_cv8 = _mm_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm_mul_ps(v_K6, v_cv8);

        v_cv9 = _mm_sub_ps(v_av1, v_av6);
        v_cv10 = _mm_sub_ps(v_cv9, v_tv3);
        v_tv5 = _mm_mul_ps(v_D4, v_cv10);
        v_t2 = _mm_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm_mul_ps(v_D5, v_cv10);
        v_t8 = _mm_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm_add_ps(v_cv1, v_cv2);
        v_tv6 = _mm_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm_add_ps(v_av1, v_av6);
        v_cv13 = _mm_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm_mul_ps(v_D8, v_cv13);
        v_t3 = _mm_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm_mul_ps(v_D9, v_cv13);
        v_t9 = _mm_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm_sub_ps(v_av2, v_av3);
        v_cv16 = _mm_sub_ps(v_av4, v_av5);
        v_cv17 = _mm_sub_ps(v_av8, v_av9);
        v_cv18 = _mm_sub_ps(v_av10, v_av11);
        v_tv9 = _mm_mul_ps(v_K4, v_cv16);
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
        v_t4 = _mm_add_ps(v_tv14, v_tv15);
        v_t4 = CONJ_128_S(SWAP_RI_128_S(v_t4));

        v_tv14 = _mm_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm_mul_ps(v_D2, v_cv22);
        v_t10 = _mm_sub_ps(v_tv15, v_tv14);
        v_t10 = SWAP_RI_128_S(v_t10);

        v_cv23 = _mm_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm_mul_ps(v_D11, v_cv24);
        v_t5 = _mm_add_ps(v_tv16, v_tv17);
        v_t5 = SWAP_RI_128_S(CONJ_128_S(v_t5));

        v_tv16 = _mm_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm_mul_ps(v_D10, v_cv24);
        v_t11 = _mm_sub_ps(v_tv16, v_tv17);
        v_t11 = SWAP_RI_128_S(v_t11);

        v_cv25 = _mm_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm_mul_ps(v_D7, v_cv26);
        v_t6 = _mm_add_ps(v_tv18, v_tv19);
        v_t6 = CONJ_128_S(SWAP_RI_128_S(v_t6));

        v_tv18 = _mm_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm_mul_ps(v_D6, v_cv26);
        v_t12 = _mm_sub_ps(v_tv19, v_tv18);
        v_t12 = SWAP_RI_128_S(v_t12);

        v_cv27 = _mm_add_ps(v_cv11, v_cv12);
        v_out0 = _mm_add_ps(v_in0, v_cv27);

        v_av13 = _mm_add_ps(v_t2, v_t3);
        v_av14 = _mm_add_ps(v_t4, v_t5);
        v_av15 = _mm_add_ps(v_t1, v_t6);

        v_cv28 = _mm_add_ps(v_av13, v_av14);
        v_out1 = _mm_add_ps(v_cv28, v_av15);

        v_av16 = _mm_sub_ps(v_t1, v_t6);
        v_cv29 = _mm_sub_ps(v_av13, v_av14);
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

        v_out4 = _mm_add_ps(v_cv34, v_cv35);
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

        v_out6 = _mm_add_ps(v_cv40, v_cv41);
        v_out11 = _mm_sub_ps(v_cv40, v_cv41);

        SCATTER2_128_S(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7,
                        tw_ptr, load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_128_S(v_out12);
        STORE_OUT_H2_128_S(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_128_S(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_S(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_128_S * DATA_STRIDE;

        in_r += (v_in_stride << 1);
        out_r += (v_out_stride << 1);
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += (v_in_h2_stride << 1);
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += (v_out_h2_stride << 1);
#endif
        remaining_sets = remaining_sets - NUM_SETS_128_S;
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256) ||                \
    defined(KERNEL_USE_AVX128)
    if (remaining_sets & 1)
    {
        // Registers to hold input data points
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40, v_cv41;
        __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

#if defined(KERNEL_USE_AVX512)
        __m128 v_K1 = CAST_512_TO_128_S(v_C1);
        __m128 v_K2 = CAST_512_TO_128_S(v_C2);
        __m128 v_K3 = CAST_512_TO_128_S(v_C3);
        __m128 v_K4 = CAST_512_TO_128_S(v_C4);
        __m128 v_K5 = CAST_512_TO_128_S(v_C5);
        __m128 v_K6 = CAST_512_TO_128_S(v_C6);
        __m128 v_K7 = CAST_512_TO_128_S(v_C7);
        __m128 v_K8 = CAST_512_TO_128_S(v_C8);
        __m128 v_K9 = CAST_512_TO_128_S(v_C9);
        __m128 v_K10 = CAST_512_TO_128_S(v_C10);
        __m128 v_D1 = CAST_512_TO_128_S(v_DG1);
        __m128 v_D2 = CAST_512_TO_128_S(v_DG2);
        __m128 v_D3 = CAST_512_TO_128_S(v_DG3);
        __m128 v_D4 = CAST_512_TO_128_S(v_DG4);
        __m128 v_D5 = CAST_512_TO_128_S(v_DG5);
        __m128 v_D6 = CAST_512_TO_128_S(v_DG6);
        __m128 v_D7 = CAST_512_TO_128_S(v_DG7);
        __m128 v_D8 = CAST_512_TO_128_S(v_DG8);
        __m128 v_D9 = CAST_512_TO_128_S(v_DG9);
        __m128 v_D10 = CAST_512_TO_128_S(v_DG10);
        __m128 v_D11 = CAST_512_TO_128_S(v_DG11);
        __m128 v_D12 = CAST_512_TO_128_S(v_DG12);
        __m128 v_ZERO_128 = CAST_512_TO_128_S(v_ZERO);
#elif defined(KERNEL_USE_AVX256)
        __m128 v_K1 = CAST_256_TO_128_S(v_C1);
        __m128 v_K2 = CAST_256_TO_128_S(v_C2);
        __m128 v_K3 = CAST_256_TO_128_S(v_C3);
        __m128 v_K4 = CAST_256_TO_128_S(v_C4);
        __m128 v_K5 = CAST_256_TO_128_S(v_C5);
        __m128 v_K6 = CAST_256_TO_128_S(v_C6);
        __m128 v_K7 = CAST_256_TO_128_S(v_C7);
        __m128 v_K8 = CAST_256_TO_128_S(v_C8);
        __m128 v_K9 = CAST_256_TO_128_S(v_C9);
        __m128 v_K10 = CAST_256_TO_128_S(v_C10);
        __m128 v_D1 = CAST_256_TO_128_S(v_DG1);
        __m128 v_D2 = CAST_256_TO_128_S(v_DG2);
        __m128 v_D3 = CAST_256_TO_128_S(v_DG3);
        __m128 v_D4 = CAST_256_TO_128_S(v_DG4);
        __m128 v_D5 = CAST_256_TO_128_S(v_DG5);
        __m128 v_D6 = CAST_256_TO_128_S(v_DG6);
        __m128 v_D7 = CAST_256_TO_128_S(v_DG7);
        __m128 v_D8 = CAST_256_TO_128_S(v_DG8);
        __m128 v_D9 = CAST_256_TO_128_S(v_DG9);
        __m128 v_D10 = CAST_256_TO_128_S(v_DG10);
        __m128 v_D11 = CAST_256_TO_128_S(v_DG11);
        __m128 v_D12 = CAST_256_TO_128_S(v_DG12);
        __m128 v_ZERO_128 = CAST_256_TO_128_S(v_ZERO);
#elif defined(KERNEL_USE_AVX128)
        __m128 v_K1 = v_C1;
        __m128 v_K2 = v_C2;
        __m128 v_K3 = v_C3;
        __m128 v_K4 = v_C4;
        __m128 v_K5 = v_C5;
        __m128 v_K6 = v_C6;
        __m128 v_K7 = v_C7;
        __m128 v_K8 = v_C8;
        __m128 v_K9 = v_C9;
        __m128 v_K10 = v_C10;
        __m128 v_D1 = v_DG1;
        __m128 v_D2 = v_DG2;
        __m128 v_D3 = v_DG3;
        __m128 v_D4 = v_DG4;
        __m128 v_D5 = v_DG5;
        __m128 v_D6 = v_DG6;
        __m128 v_D7 = v_DG7;
        __m128 v_D8 = v_DG8;
        __m128 v_D9 = v_DG9;
        __m128 v_D10 = v_DG10;
        __m128 v_D11 = v_DG11;
        __m128 v_D12 = v_DG12;
        __m128 v_ZERO_128 = v_ZERO;
#endif

        LOAD_IN_64_S(in_r, in_strides, 1, v_in1, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 2, v_in2, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 3, v_in3, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 4, v_in4, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 5, v_in5, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 6, v_in6, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 7, v_in7, tw_ptr, load_multi_cols,
                        0);
        v_in7 = IN_H2_128_S(v_in7);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 8, v_in8, tw_ptr, load_multi_cols,
                        0);
        v_in8 = IN_H2_128_S(v_in8);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 9, v_in9, tw_ptr, load_multi_cols,
                        0);
        v_in9 = IN_H2_128_S(v_in9);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 10, v_in10, tw_ptr,
                        load_multi_cols, 0);
        v_in10 = IN_H2_128_S(v_in10);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 11, v_in11, tw_ptr,
                        load_multi_cols, 0);
        v_in11 = IN_H2_128_S(v_in11);
        LOAD_IN_H2_64_S(in_h2_r, in_strides, 12, v_in12, tw_ptr,
                        load_multi_cols, 0);
        v_in12 = IN_H2_128_S(v_in12);
#else
        LOAD_IN_64_S(in_r, in_strides, 7, v_in7, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 8, v_in8, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 9, v_in9, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 10, v_in10, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 11, v_in11, tw_ptr, load_multi_cols,
                     is_contiguous_in);
        LOAD_IN_64_S(in_r, in_strides, 12, v_in12, tw_ptr, load_multi_cols,
                     is_contiguous_in);
#endif

        LD_LOW_128_S(in_r, v_in0);

        v_av1 = _mm_add_ps(v_in1, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in3);
        v_av3 = _mm_add_ps(v_in9, v_in10);
        v_av4 = _mm_add_ps(v_in2, v_in6);
        v_av5 = _mm_add_ps(v_in11, v_in7);
        v_av6 = _mm_add_ps(v_in8, v_in5);

        v_av7 = _mm_sub_ps(v_in1, v_in12);
        v_av8 = _mm_sub_ps(v_in4, v_in3);
        v_av9 = _mm_sub_ps(v_in9, v_in10);
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
        v_t1 = _mm_sub_ps(v_in0, v_cv5);

        v_tv1 = _mm_mul_ps(v_cv3, v_D1);
        v_tv2 = _mm_mul_ps(v_cv4, v_D12);
        v_cv5 = _mm_add_ps(v_tv1, v_tv2);
        v_t7 = _mm_sub_ps(v_in0, v_cv5);

        v_cv5 = _mm_sub_ps(v_cv1, v_cv2);
        v_tv3 = _mm_mul_ps(v_K2, v_cv5);
        v_cv6 = _mm_add_ps(v_av10, v_av11);
        v_cv7 = _mm_add_ps(v_av8, v_av9);
        v_cv8 = _mm_add_ps(v_cv6, v_cv7);
        v_tv4 = _mm_mul_ps(v_K6, v_cv8);

        v_cv9 = _mm_sub_ps(v_av1, v_av6);
        v_cv10 = _mm_sub_ps(v_cv9, v_tv3);
        v_tv5 = _mm_mul_ps(v_D4, v_cv10);
        v_t2 = _mm_add_ps(v_tv4, v_tv5);

        v_tv4 = _mm_mul_ps(v_K5, v_cv8);
        v_tv5 = _mm_mul_ps(v_D5, v_cv10);
        v_t8 = _mm_sub_ps(v_tv4, v_tv5);

        v_cv11 = _mm_add_ps(v_cv1, v_cv2);
        v_tv6 = _mm_mul_ps(v_K2, v_cv11);
        v_cv12 = _mm_add_ps(v_av1, v_av6);
        v_cv13 = _mm_sub_ps(v_cv12, v_tv6);
        v_cv14 = _mm_sub_ps(v_cv6, v_cv7);

        v_tv7 = _mm_mul_ps(v_K8, v_cv14);
        v_tv8 = _mm_mul_ps(v_D8, v_cv13);
        v_t3 = _mm_add_ps(v_tv7, v_tv8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv14);
        v_tv8 = _mm_mul_ps(v_D9, v_cv13);
        v_t9 = _mm_sub_ps(v_tv7, v_tv8);

        v_cv15 = _mm_sub_ps(v_av2, v_av3);
        v_cv16 = _mm_sub_ps(v_av4, v_av5);
        v_cv17 = _mm_sub_ps(v_av8, v_av9);
        v_cv18 = _mm_sub_ps(v_av10, v_av11);
        v_tv9 = _mm_mul_ps(v_K4, v_cv16);
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
        v_t4 = _mm_add_ps(v_tv14, v_tv15);
        v_t4 = CONJ_128_S(SWAP_RI_128_S(v_t4));

        v_tv14 = _mm_mul_ps(v_D3, v_cv20);
        v_tv15 = _mm_mul_ps(v_D2, v_cv22);
        v_t10 = _mm_sub_ps(v_tv15, v_tv14);
        v_t10 = SWAP_RI_128_S(v_t10);

        v_cv23 = _mm_sub_ps(v_cv19, v_tv10);
        v_cv24 = _mm_sub_ps(v_cv21, v_tv13);

        v_tv16 = _mm_mul_ps(v_D10, v_cv23);
        v_tv17 = _mm_mul_ps(v_D11, v_cv24);
        v_t5 = _mm_add_ps(v_tv16, v_tv17);
        v_t5 = SWAP_RI_128_S(CONJ_128_S(v_t5));

        v_tv16 = _mm_mul_ps(v_D11, v_cv23);
        v_tv17 = _mm_mul_ps(v_D10, v_cv24);
        v_t11 = _mm_sub_ps(v_tv16, v_tv17);
        v_t11 = SWAP_RI_128_S(v_t11);

        v_cv25 = _mm_sub_ps(v_cv16, v_av12);
        v_cv26 = _mm_sub_ps(v_av7, v_cv17);

        v_tv18 = _mm_mul_ps(v_K9, v_cv25);
        v_tv19 = _mm_mul_ps(v_D7, v_cv26);
        v_t6 = _mm_add_ps(v_tv18, v_tv19);
        v_t6 = CONJ_128_S(SWAP_RI_128_S(v_t6));

        v_tv18 = _mm_mul_ps(v_K10, v_cv25);
        v_tv19 = _mm_mul_ps(v_D6, v_cv26);
        v_t12 = _mm_sub_ps(v_tv19, v_tv18);
        v_t12 = SWAP_RI_128_S(v_t12);

        v_cv27 = _mm_add_ps(v_cv11, v_cv12);
        v_out0 = _mm_add_ps(v_in0, v_cv27);

        v_av13 = _mm_add_ps(v_t2, v_t3);
        v_av14 = _mm_add_ps(v_t4, v_t5);
        v_av15 = _mm_add_ps(v_t1, v_t6);

        v_cv28 = _mm_add_ps(v_av13, v_av14);
        v_out1 = _mm_add_ps(v_cv28, v_av15);

        v_av16 = _mm_sub_ps(v_t1, v_t6);
        v_cv29 = _mm_sub_ps(v_av13, v_av14);
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

        v_out4 = _mm_add_ps(v_cv34, v_cv35);
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

        v_out6 = _mm_add_ps(v_cv40, v_cv41);
        v_out11 = _mm_sub_ps(v_cv40, v_cv41);

        ST_LOW_128_S(out_r, v_out0);
        STORE_OUT_64_S(out_r, out_strides, 1, v_out1, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 2, v_out2, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 3, v_out3, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 4, v_out4, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 5, v_out5, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 6, v_out6, tw_ptr, load_multi_cols,
                       is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_128_S(v_out7);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 7, v_out7, tw_ptr,
                       load_multi_cols, 0);
        v_out8 = OUT_H2_128_S(v_out8);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 8, v_out8, tw_ptr,
                       load_multi_cols, 0);
        v_out9 = OUT_H2_128_S(v_out9);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 9, v_out9, tw_ptr,
                       load_multi_cols, 0);
        v_out10 = OUT_H2_128_S(v_out10);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 10, v_out10, tw_ptr,
                       load_multi_cols, 0);
        v_out11 = OUT_H2_128_S(v_out11);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 11, v_out11, tw_ptr,
                       load_multi_cols, 0);
        v_out12 = OUT_H2_128_S(v_out12);
        STORE_OUT_H2_64_S(out_h2_r, out_strides, 12, v_out12, tw_ptr,
                       load_multi_cols, 0);
#else
        STORE_OUT_64_S(out_r, out_strides, 7, v_out7, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 8, v_out8, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 9, v_out9, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 10, v_out10, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 11, v_out11, tw_ptr, load_multi_cols,
                       is_contiguous_out);
        STORE_OUT_64_S(out_r, out_strides, 12, v_out12, tw_ptr, load_multi_cols,
                       is_contiguous_out);
#endif
    }
#endif

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID TWID_KNAME_FP64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                 FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                 FFTZ_INTP n, aoclfftz_strides_t *strides,
                                 FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_13[10] = {
        0.866025403784438646763723170752936183471402627,
        0.500000000000000000000000000000000000000000000,
        0.866025403784438646763723170752936183471402627,
        0.500000000000000000000000000000000000000000000,
        0.447320117602511140667282045633987571324387014,  // 2*DGC[3]*CRTM[3]
        0.265966249214837287587521063842185948798330267,  // 2*DGC[4]*CRTM[3]
        0.131467828262610852858973617628781241523254441,  // 2*DGC[7]*CRTM[3]
        0.503537032863766627246873853868466977093348562,  // 2*DGC[8]*CRTM[3]
        0.575140729474003121368385547455453388461001608,  // 2*DGC[5]*CRTM[2]
        0.174138601152135905005660794929264742616964676}; // 2*DGC[6]*CRTM[2]

    const FFTZ_DOUBLE DGC[12] = {
        0.383795939621999107759935105622541328854274714,   // DGC[0] + DGC[11]
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

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_DOUBLE *in_h2_r = (FFTZ_DOUBLE *)in_real;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_DOUBLE *out_h2_r = (FFTZ_DOUBLE *)out_real;
#endif

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
#if defined(KERNEL_VARIANT_C2R)
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
#endif

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)tws->TW;
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    FFTZ_INTP N = n / NUM_SETS_D;
    FFTZ_INTP count;

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    FFTZ_INTP remaining_sets = n % NUM_SETS_D;
#endif

    VREGTYPE_D v_C1 = BCAST_D(CRTM_13[0]);
    VREGTYPE_D v_C2 = BCAST_D(CRTM_13[1]);
    VREGTYPE_D v_C3 = BCAST_D(CRTM_13[2]);
    VREGTYPE_D v_C4 = BCAST_D(CRTM_13[3]);
    VREGTYPE_D v_C5 = BCAST_D(CRTM_13[4]);
    VREGTYPE_D v_C6 = BCAST_D(CRTM_13[5]);
    VREGTYPE_D v_C7 = BCAST_D(CRTM_13[6]);
    VREGTYPE_D v_C8 = BCAST_D(CRTM_13[7]);
    VREGTYPE_D v_C9 = BCAST_D(CRTM_13[8]);
    VREGTYPE_D v_C10 = BCAST_D(CRTM_13[9]);

    VREGTYPE_D v_DG1 = BCAST_D(DGC[0]);
    VREGTYPE_D v_DG2 = BCAST_D(DGC[1]);
    VREGTYPE_D v_DG3 = BCAST_D(DGC[2]);
    VREGTYPE_D v_DG4 = BCAST_D(DGC[3]);
    VREGTYPE_D v_DG5 = BCAST_D(DGC[4]);
    VREGTYPE_D v_DG6 = BCAST_D(DGC[5]);
    VREGTYPE_D v_DG7 = BCAST_D(DGC[6]);
    VREGTYPE_D v_DG8 = BCAST_D(DGC[7]);
    VREGTYPE_D v_DG9 = BCAST_D(DGC[8]);
    VREGTYPE_D v_DG10 = BCAST_D(DGC[9]);
    VREGTYPE_D v_DG11 = BCAST_D(DGC[10]);
    VREGTYPE_D v_DG12 = BCAST_D(DGC[11]);

    VREGTYPE_D v_ZERO = NEG_ZERO_D(flag);

#if defined(KERNEL_DIRECTION_BWD)
    v_C3 = NEG_D(v_C3, 1);
    v_C4 = NEG_D(v_C4, 1);
    v_C5 = NEG_D(v_C5, 1);
    v_C7 = NEG_D(v_C7, 1);
    v_C9 = NEG_D(v_C9, 1);
    v_DG3 = NEG_D(v_DG3, 1);
    v_DG5 = NEG_D(v_DG5, 1);
    v_DG7 = NEG_D(v_DG7, 1);
    v_DG9 = NEG_D(v_DG9, 1);
    v_DG11 = NEG_D(v_DG11, 1);
#endif
    FFTZ_DOUBLE *tw_ptr = tw;

    for (count = 0; count < N; count++)
    {
        // Registers to hold input data points
        VREGTYPE_D v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7,
            v_in8, v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        VREGTYPE_D v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6,
            v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal
        // constants
        VREGTYPE_D v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        VREGTYPE_D v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_av9, v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16,
            v_av17, v_av18, v_av19, v_av20, v_av21, v_av22;
        VREGTYPE_D v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8,
            v_cv9, v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16,
            v_cv17, v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24,
            v_cv25, v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32,
            v_cv33, v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40,
            v_cv41;
        VREGTYPE_D v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
            v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16,
            v_tv17, v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24,
            v_tv25, v_tv26, v_tv27, v_tv28, v_tv29;

        GATHER_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        LOAD_IN_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw_ptr,
                  load_multi_cols, 0);
        v_in7 = IN_H2_D(v_in7);
        LOAD_IN_H2_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                  load_multi_cols, 0);
        v_in8 = IN_H2_D(v_in8);
        LOAD_IN_H2_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                  load_multi_cols, 0);
        v_in9 = IN_H2_D(v_in9);
        LOAD_IN_H2_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10, tw_ptr,
                  load_multi_cols, 0);
        v_in10 = IN_H2_D(v_in10);
        LOAD_IN_H2_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11, tw_ptr,
                  load_multi_cols, 0);
        v_in11 = IN_H2_D(v_in11);
        LOAD_IN_H2_D(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12, tw_ptr,
                  load_multi_cols, 0);
        v_in12 = IN_H2_D(v_in12);
#else
        LOAD_IN_D(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                  load_multi_cols, is_contiguous_in);
        LOAD_IN_D(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                  load_multi_cols, is_contiguous_in);
#endif

        v_av1 = ADD_D(v_in1, v_in12);
        v_av2 = ADD_D(v_in4, v_in3);
        v_av3 = ADD_D(v_in9, v_in10);
        v_av4 = ADD_D(v_in2, v_in6);
        v_av5 = ADD_D(v_in11, v_in7);
        v_av6 = ADD_D(v_in8, v_in5);

        v_av7 = SUB_D(v_in1, v_in12);
        v_av8 = SUB_D(v_in4, v_in3);
        v_av9 = SUB_D(v_in9, v_in10);
        v_av10 = SUB_D(v_in2, v_in6);
        v_av11 = SUB_D(v_in11, v_in7);
        v_av12 = SUB_D(v_in8, v_in5);

        v_cv1 = ADD_D(v_av2, v_av3);
        v_cv2 = ADD_D(v_av4, v_av5);

        v_cv3 = ADD_D(v_av1, v_cv1);
        v_cv4 = ADD_D(v_av6, v_cv2);

        v_tv1 = MUL_D(v_cv3, v_DG12);
        v_tv2 = MUL_D(v_cv4, v_DG1);
        v_cv5 = ADD_D(v_tv1, v_tv2);
        v_t1 = SUB_D(v_in0, v_cv5);

        v_tv1 = MUL_D(v_cv3, v_DG1);
        v_tv2 = MUL_D(v_cv4, v_DG12);
        v_cv5 = ADD_D(v_tv1, v_tv2);
        v_t7 = SUB_D(v_in0, v_cv5);

        v_cv5 = SUB_D(v_cv1, v_cv2);
        v_tv3 = MUL_D(v_C2, v_cv5);
        v_cv6 = ADD_D(v_av10, v_av11);
        v_cv7 = ADD_D(v_av8, v_av9);
        v_cv8 = ADD_D(v_cv6, v_cv7);
        v_tv4 = MUL_D(v_C6, v_cv8);

        v_cv9 = SUB_D(v_av1, v_av6);
        v_cv10 = SUB_D(v_cv9, v_tv3);
        v_tv5 = MUL_D(v_DG4, v_cv10);
        v_t2 = ADD_D(v_tv4, v_tv5);

        v_tv4 = MUL_D(v_C5, v_cv8);
        v_tv5 = MUL_D(v_DG5, v_cv10);
        v_t8 = SUB_D(v_tv4, v_tv5);

        v_cv11 = ADD_D(v_cv1, v_cv2);
        v_tv6 = MUL_D(v_C2, v_cv11);
        v_cv12 = ADD_D(v_av1, v_av6);
        v_cv13 = SUB_D(v_cv12, v_tv6);
        v_cv14 = SUB_D(v_cv6, v_cv7);

        v_tv7 = MUL_D(v_C8, v_cv14);
        v_tv8 = MUL_D(v_DG8, v_cv13);
        v_t3 = ADD_D(v_tv7, v_tv8);

        v_tv7 = MUL_D(v_C7, v_cv14);
        v_tv8 = MUL_D(v_DG9, v_cv13);
        v_t9 = SUB_D(v_tv7, v_tv8);

        v_cv15 = SUB_D(v_av2, v_av3);
        v_cv16 = SUB_D(v_av4, v_av5);
        v_cv17 = SUB_D(v_av8, v_av9);
        v_cv18 = SUB_D(v_av10, v_av11);
        v_tv9 = MUL_D(v_C4, v_cv16);
        v_tv10 = MUL_D(v_C3, v_cv15);

        // use xor insead of multipling 1.0/-1.0
        v_tv11 = XOR_D(v_ZERO, v_av12);
        v_cv19 = ADD_D(v_tv9, v_tv11);
        v_cv20 = ADD_D(v_cv19, v_tv10);

        v_tv12 = MUL_D(v_C2, v_cv17);
        v_tv13 = MUL_D(v_C1, v_cv18);
        v_cv21 = ADD_D(v_av7, v_tv12);
        v_cv22 = ADD_D(v_cv21, v_tv13);

        v_tv14 = MUL_D(v_DG2, v_cv20);
        v_tv15 = MUL_D(v_DG3, v_cv22);
        v_t4 = ADD_D(v_tv14, v_tv15);
        v_t4 = CONJ_D(SWAP_RI_D(v_t4));

        v_tv14 = MUL_D(v_DG3, v_cv20);
        v_tv15 = MUL_D(v_DG2, v_cv22);
        v_t10 = SUB_D(v_tv15, v_tv14);
        v_t10 = SWAP_RI_D(v_t10);

        v_cv23 = SUB_D(v_cv19, v_tv10);
        v_cv24 = SUB_D(v_cv21, v_tv13);

        v_tv16 = MUL_D(v_DG10, v_cv23);
        v_tv17 = MUL_D(v_DG11, v_cv24);
        v_t5 = ADD_D(v_tv16, v_tv17);
        v_t5 = SWAP_RI_D(CONJ_D(v_t5));

        v_tv16 = MUL_D(v_DG11, v_cv23);
        v_tv17 = MUL_D(v_DG10, v_cv24);
        v_t11 = SUB_D(v_tv16, v_tv17);
        v_t11 = SWAP_RI_D(v_t11);

        v_cv25 = SUB_D(v_cv16, v_av12);
        v_cv26 = SUB_D(v_av7, v_cv17);

        v_tv18 = MUL_D(v_C9, v_cv25);
        v_tv19 = MUL_D(v_DG7, v_cv26);
        v_t6 = ADD_D(v_tv18, v_tv19);
        v_t6 = CONJ_D(SWAP_RI_D(v_t6));

        v_tv18 = MUL_D(v_C10, v_cv25);
        v_tv19 = MUL_D(v_DG6, v_cv26);
        v_t12 = SUB_D(v_tv19, v_tv18);
        v_t12 = SWAP_RI_D(v_t12);

        v_cv27 = ADD_D(v_cv11, v_cv12);
        v_out0 = ADD_D(v_in0, v_cv27);

        v_av13 = ADD_D(v_t2, v_t3);
        v_av14 = ADD_D(v_t4, v_t5);
        v_av15 = ADD_D(v_t1, v_t6);

        v_cv28 = ADD_D(v_av13, v_av14);
        v_out1 = ADD_D(v_cv28, v_av15);

        v_av16 = SUB_D(v_t1, v_t6);
        v_cv29 = SUB_D(v_av13, v_av14);
        v_out12 = ADD_D(v_cv29, v_av16);

        v_av17 = SUB_D(v_t2, v_t3);
        v_cv30 = SUB_D(v_t7, v_av17);
        v_av18 = ADD_D(v_t10, v_t11);

        // use xor insead of multipling 1.0/-1.0
        v_tv20 = XOR_D(v_ZERO, v_av18);
        v_tv21 = XOR_D(v_ZERO, v_t12);
        v_cv31 = SUB_D(v_tv20, v_tv21);
        v_cv31 = CONJ_D(v_cv31);

        v_out8 = ADD_D(v_cv30, v_cv31);
        v_out5 = SUB_D(v_cv30, v_cv31);

        v_tv22 = MUL_D(v_C2, v_cv28);
        v_cv32 = SUB_D(v_av15, v_tv22);

        v_av19 = SUB_D(v_t10, v_t11);
        v_av20 = SUB_D(v_t8, v_t9);
        v_tv23 = MUL_D(v_C3, v_av19);
        v_tv23 = CONJ_D(v_tv23);
        v_tv24 = MUL_D(v_C3, v_av20);
        v_cv33 = ADD_D(v_tv23, v_tv24);

        v_out3 = ADD_D(v_cv32, v_cv33);
        v_out9 = SUB_D(v_cv32, v_cv33);

        v_tv25 = MUL_D(v_C2, v_cv29);
        v_cv34 = SUB_D(v_av16, v_tv25);
        v_cv35 = SUB_D(v_tv23, v_tv24);

        v_out4 = ADD_D(v_cv34, v_cv35);
        v_out10 = SUB_D(v_cv34, v_cv35);

        v_tv26 = MUL_D(v_C2, v_av17);
        v_cv36 = ADD_D(v_tv26, v_t7);
        v_av21 = SUB_D(v_t4, v_t5);
        v_tv27 = MUL_D(v_C1, v_av21);
        v_tv28 = MUL_D(v_C4, v_av18);
        v_av22 = ADD_D(v_t8, v_t9);
        v_tv29 = MUL_D(v_C3, v_av22);

        v_cv37 = ADD_D(v_cv36, v_tv27);

        v_cv38 = ADD_D(v_tv21, v_tv28);
        v_cv38 = CONJ_D(v_cv38);
        v_cv39 = SUB_D(v_cv38, v_tv29);

        v_out2 = ADD_D(v_cv37, v_cv39);
        v_out7 = SUB_D(v_cv37, v_cv39);

        v_cv40 = SUB_D(v_cv36, v_tv27);
        v_cv41 = ADD_D(v_cv38, v_tv29);

        v_out6 = ADD_D(v_cv40, v_cv41);
        v_out11 = SUB_D(v_cv40, v_cv41);

        SCATTER_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_D(v_out7);
        STORE_OUT_H2_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7,
                       tw_ptr, load_multi_cols, 0);
        v_out8 = OUT_H2_D(v_out8);
        STORE_OUT_H2_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                       tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_D(v_out9);
        STORE_OUT_H2_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                       tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_D(v_out10);
        STORE_OUT_H2_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                       tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_D(v_out11);
        STORE_OUT_H2_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                       tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_D(v_out12);
        STORE_OUT_H2_D(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                       tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_D(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                    load_multi_cols, is_contiguous_out);
        STORE_OUT_D(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                    load_multi_cols, is_contiguous_out);
#endif

        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_D * DATA_STRIDE;
        in_r += NUM_SETS_D * v_in_stride;
        out_r += NUM_SETS_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_D * v_out_h2_stride;
#endif
    }

    // The following contains code that performs the FFT on the tail cases.
    // These tails are conditionally "instantiated" based on the ISA requested
    // by the "includee" file.

#if defined(KERNEL_USE_AVX512)
    if (remaining_sets >= NUM_SETS_256_D)
    {
        // Registers to hold input data points
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m256d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        __m256d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m256d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40, v_cv41;
        __m256d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

        __m256d v_K1 = CAST_512_TO_256_D(v_C1);
        __m256d v_K2 = CAST_512_TO_256_D(v_C2);
        __m256d v_K3 = CAST_512_TO_256_D(v_C3);
        __m256d v_K4 = CAST_512_TO_256_D(v_C4);
        __m256d v_K5 = CAST_512_TO_256_D(v_C5);
        __m256d v_K6 = CAST_512_TO_256_D(v_C6);
        __m256d v_K7 = CAST_512_TO_256_D(v_C7);
        __m256d v_K8 = CAST_512_TO_256_D(v_C8);
        __m256d v_K9 = CAST_512_TO_256_D(v_C9);
        __m256d v_K10 = CAST_512_TO_256_D(v_C10);
        __m256d v_D1 = CAST_512_TO_256_D(v_DG1);
        __m256d v_D2 = CAST_512_TO_256_D(v_DG2);
        __m256d v_D3 = CAST_512_TO_256_D(v_DG3);
        __m256d v_D4 = CAST_512_TO_256_D(v_DG4);
        __m256d v_D5 = CAST_512_TO_256_D(v_DG5);
        __m256d v_D6 = CAST_512_TO_256_D(v_DG6);
        __m256d v_D7 = CAST_512_TO_256_D(v_DG7);
        __m256d v_D8 = CAST_512_TO_256_D(v_DG8);
        __m256d v_D9 = CAST_512_TO_256_D(v_DG9);
        __m256d v_D10 = CAST_512_TO_256_D(v_DG10);
        __m256d v_D11 = CAST_512_TO_256_D(v_DG11);
        __m256d v_D12 = CAST_512_TO_256_D(v_DG12);
        __m256d v_ZERO_256 = CAST_512_TO_256_D(v_ZERO);

        GATHER2_256_D(in_r, v_in_stride, v_in0, is_contiguous_in);

        LOAD_IN_256_D(in_r, in_strides, 1, v_in_stride, v_in1, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 2, v_in_stride, v_in2, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 3, v_in_stride, v_in3, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 4, v_in_stride, v_in4, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 5, v_in_stride, v_in5, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 6, v_in_stride, v_in6, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 7, v_in_h2_stride, v_in7, tw_ptr,
                      load_multi_cols, 0);
        v_in7 = IN_H2_256_D(v_in7);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 8, v_in_h2_stride, v_in8, tw_ptr,
                      load_multi_cols, 0);
        v_in8 = IN_H2_256_D(v_in8);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 9, v_in_h2_stride, v_in9, tw_ptr,
                      load_multi_cols, 0);
        v_in9 = IN_H2_256_D(v_in9);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 10, v_in_h2_stride, v_in10,
                         tw_ptr, load_multi_cols, 0);
        v_in10 = IN_H2_256_D(v_in10);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 11, v_in_h2_stride, v_in11,
                         tw_ptr, load_multi_cols, 0);
        v_in11 = IN_H2_256_D(v_in11);
        LOAD_IN_H2_256_D(in_h2_r, in_strides, 12, v_in_h2_stride, v_in12,
                         tw_ptr, load_multi_cols, 0);
        v_in12 = IN_H2_256_D(v_in12);
#else
        LOAD_IN_256_D(in_r, in_strides, 7, v_in_stride, v_in7, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 8, v_in_stride, v_in8, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 9, v_in_stride, v_in9, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 10, v_in_stride, v_in10, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 11, v_in_stride, v_in11, tw_ptr,
                      load_multi_cols, is_contiguous_in);
        LOAD_IN_256_D(in_r, in_strides, 12, v_in_stride, v_in12, tw_ptr,
                      load_multi_cols, is_contiguous_in);
#endif

        v_av1 = _mm256_add_pd(v_in1, v_in12);
        v_av2 = _mm256_add_pd(v_in4, v_in3);
        v_av3 = _mm256_add_pd(v_in9, v_in10);
        v_av4 = _mm256_add_pd(v_in2, v_in6);
        v_av5 = _mm256_add_pd(v_in11, v_in7);
        v_av6 = _mm256_add_pd(v_in8, v_in5);

        v_av7 = _mm256_sub_pd(v_in1, v_in12);
        v_av8 = _mm256_sub_pd(v_in4, v_in3);
        v_av9 = _mm256_sub_pd(v_in9, v_in10);
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
        v_t1 = _mm256_sub_pd(v_in0, v_cv5);

        v_tv1 = _mm256_mul_pd(v_cv3, v_D1);
        v_tv2 = _mm256_mul_pd(v_cv4, v_D12);
        v_cv5 = _mm256_add_pd(v_tv1, v_tv2);
        v_t7 = _mm256_sub_pd(v_in0, v_cv5);

        v_cv5 = _mm256_sub_pd(v_cv1, v_cv2);
        v_tv3 = _mm256_mul_pd(v_K2, v_cv5);
        v_cv6 = _mm256_add_pd(v_av10, v_av11);
        v_cv7 = _mm256_add_pd(v_av8, v_av9);
        v_cv8 = _mm256_add_pd(v_cv6, v_cv7);
        v_tv4 = _mm256_mul_pd(v_K6, v_cv8);

        v_cv9 = _mm256_sub_pd(v_av1, v_av6);
        v_cv10 = _mm256_sub_pd(v_cv9, v_tv3);
        v_tv5 = _mm256_mul_pd(v_D4, v_cv10);
        v_t2 = _mm256_add_pd(v_tv4, v_tv5);

        v_tv4 = _mm256_mul_pd(v_K5, v_cv8);
        v_tv5 = _mm256_mul_pd(v_D5, v_cv10);
        v_t8 = _mm256_sub_pd(v_tv4, v_tv5);

        v_cv11 = _mm256_add_pd(v_cv1, v_cv2);
        v_tv6 = _mm256_mul_pd(v_K2, v_cv11);
        v_cv12 = _mm256_add_pd(v_av1, v_av6);
        v_cv13 = _mm256_sub_pd(v_cv12, v_tv6);
        v_cv14 = _mm256_sub_pd(v_cv6, v_cv7);

        v_tv7 = _mm256_mul_pd(v_K8, v_cv14);
        v_tv8 = _mm256_mul_pd(v_D8, v_cv13);
        v_t3 = _mm256_add_pd(v_tv7, v_tv8);

        v_tv7 = _mm256_mul_pd(v_K7, v_cv14);
        v_tv8 = _mm256_mul_pd(v_D9, v_cv13);
        v_t9 = _mm256_sub_pd(v_tv7, v_tv8);

        v_cv15 = _mm256_sub_pd(v_av2, v_av3);
        v_cv16 = _mm256_sub_pd(v_av4, v_av5);
        v_cv17 = _mm256_sub_pd(v_av8, v_av9);
        v_cv18 = _mm256_sub_pd(v_av10, v_av11);
        v_tv9 = _mm256_mul_pd(v_K4, v_cv16);
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
        v_t4 = _mm256_add_pd(v_tv14, v_tv15);
        v_t4 = CONJ_256_D(SWAP_RI_256_D(v_t4));

        v_tv14 = _mm256_mul_pd(v_D3, v_cv20);
        v_tv15 = _mm256_mul_pd(v_D2, v_cv22);
        v_t10 = _mm256_sub_pd(v_tv15, v_tv14);
        v_t10 = SWAP_RI_256_D(v_t10);

        v_cv23 = _mm256_sub_pd(v_cv19, v_tv10);
        v_cv24 = _mm256_sub_pd(v_cv21, v_tv13);

        v_tv16 = _mm256_mul_pd(v_D10, v_cv23);
        v_tv17 = _mm256_mul_pd(v_D11, v_cv24);
        v_t5 = _mm256_add_pd(v_tv16, v_tv17);
        v_t5 = SWAP_RI_256_D(CONJ_256_D(v_t5));

        v_tv16 = _mm256_mul_pd(v_D11, v_cv23);
        v_tv17 = _mm256_mul_pd(v_D10, v_cv24);
        v_t11 = _mm256_sub_pd(v_tv16, v_tv17);
        v_t11 = SWAP_RI_256_D(v_t11);

        v_cv25 = _mm256_sub_pd(v_cv16, v_av12);
        v_cv26 = _mm256_sub_pd(v_av7, v_cv17);

        v_tv18 = _mm256_mul_pd(v_K9, v_cv25);
        v_tv19 = _mm256_mul_pd(v_D7, v_cv26);
        v_t6 = _mm256_add_pd(v_tv18, v_tv19);
        v_t6 = CONJ_256_D(SWAP_RI_256_D(v_t6));

        v_tv18 = _mm256_mul_pd(v_K10, v_cv25);
        v_tv19 = _mm256_mul_pd(v_D6, v_cv26);
        v_t12 = _mm256_sub_pd(v_tv19, v_tv18);
        v_t12 = SWAP_RI_256_D(v_t12);

        v_cv27 = _mm256_add_pd(v_cv11, v_cv12);
        v_out0 = _mm256_add_pd(v_in0, v_cv27);

        v_av13 = _mm256_add_pd(v_t2, v_t3);
        v_av14 = _mm256_add_pd(v_t4, v_t5);
        v_av15 = _mm256_add_pd(v_t1, v_t6);

        v_cv28 = _mm256_add_pd(v_av13, v_av14);
        v_out1 = _mm256_add_pd(v_cv28, v_av15);

        v_av16 = _mm256_sub_pd(v_t1, v_t6);
        v_cv29 = _mm256_sub_pd(v_av13, v_av14);
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

        v_out4 = _mm256_add_pd(v_cv34, v_cv35);
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

        v_out6 = _mm256_add_pd(v_cv40, v_cv41);
        v_out11 = _mm256_sub_pd(v_cv40, v_cv41);

        SCATTER2_256_D(out_r, v_out_stride, v_out0, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 1, v_out_stride, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 2, v_out_stride, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 3, v_out_stride, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 4, v_out_stride, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 5, v_out_stride, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 6, v_out_stride, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_256_D(v_out7);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 7, v_out_h2_stride, v_out7,
                        tw_ptr, load_multi_cols, 0);
        v_out8 = OUT_H2_256_D(v_out8);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 8, v_out_h2_stride, v_out8,
                        tw_ptr, load_multi_cols, 0);
        v_out9 = OUT_H2_256_D(v_out9);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 9, v_out_h2_stride, v_out9,
                        tw_ptr, load_multi_cols, 0);
        v_out10 = OUT_H2_256_D(v_out10);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 10, v_out_h2_stride, v_out10,
                        tw_ptr, load_multi_cols, 0);
        v_out11 = OUT_H2_256_D(v_out11);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 11, v_out_h2_stride, v_out11,
                        tw_ptr, load_multi_cols, 0);
        v_out12 = OUT_H2_256_D(v_out12);
        STORE_OUT_H2_256_D(out_h2_r, out_strides, 12, v_out_h2_stride, v_out12,
                        tw_ptr, load_multi_cols, 0);
#else
        STORE_OUT_256_D(out_r, out_strides, 7, v_out_stride, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 8, v_out_stride, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 9, v_out_stride, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 10, v_out_stride, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 11, v_out_stride, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_256_D(out_r, out_strides, 12, v_out_stride, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
        tw_ptr += load_multi_cols * (RADIX - 1) * NUM_SETS_256_D * DATA_STRIDE;

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
#if defined(KERNEL_VARIANT_C2R)
        in_h2_r += NUM_SETS_256_D * v_in_h2_stride;
#elif defined(KERNEL_VARIANT_R2C)
        out_h2_r += NUM_SETS_256_D * v_out_h2_stride;
#endif
    }
#endif

#if defined(KERNEL_USE_AVX512) || defined(KERNEL_USE_AVX256)
    if (remaining_sets & 1)
    {
        // Registers to hold input data points
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
            v_in9, v_in10, v_in11, v_in12;
        // Registers to hold output data points
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
            v_out8, v_out9, v_out10, v_out11, v_out12;
        // Registers to hold intrim outputs after multiplying diagonal constants
        __m128d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
            v_t11, v_t12;

        __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8, v_av9,
            v_av10, v_av11, v_av12, v_av13, v_av14, v_av15, v_av16, v_av17,
            v_av18, v_av19, v_av20, v_av21, v_av22;
        __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7, v_cv8, v_cv9,
            v_cv10, v_cv11, v_cv12, v_cv13, v_cv14, v_cv15, v_cv16, v_cv17,
            v_cv18, v_cv19, v_cv20, v_cv21, v_cv22, v_cv23, v_cv24, v_cv25,
            v_cv26, v_cv27, v_cv28, v_cv29, v_cv30, v_cv31, v_cv32, v_cv33,
            v_cv34, v_cv35, v_cv36, v_cv37, v_cv38, v_cv39, v_cv40, v_cv41;
        __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8, v_tv9,
            v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16, v_tv17,
            v_tv18, v_tv19, v_tv20, v_tv21, v_tv22, v_tv23, v_tv24, v_tv25,
            v_tv26, v_tv27, v_tv28, v_tv29;

#if defined(KERNEL_USE_AVX512)
        __m128d v_K1 = CAST_512_TO_128_D(v_C1);
        __m128d v_K2 = CAST_512_TO_128_D(v_C2);
        __m128d v_K3 = CAST_512_TO_128_D(v_C3);
        __m128d v_K4 = CAST_512_TO_128_D(v_C4);
        __m128d v_K5 = CAST_512_TO_128_D(v_C5);
        __m128d v_K6 = CAST_512_TO_128_D(v_C6);
        __m128d v_K7 = CAST_512_TO_128_D(v_C7);
        __m128d v_K8 = CAST_512_TO_128_D(v_C8);
        __m128d v_K9 = CAST_512_TO_128_D(v_C9);
        __m128d v_K10 = CAST_512_TO_128_D(v_C10);
        __m128d v_D1 = CAST_512_TO_128_D(v_DG1);
        __m128d v_D2 = CAST_512_TO_128_D(v_DG2);
        __m128d v_D3 = CAST_512_TO_128_D(v_DG3);
        __m128d v_D4 = CAST_512_TO_128_D(v_DG4);
        __m128d v_D5 = CAST_512_TO_128_D(v_DG5);
        __m128d v_D6 = CAST_512_TO_128_D(v_DG6);
        __m128d v_D7 = CAST_512_TO_128_D(v_DG7);
        __m128d v_D8 = CAST_512_TO_128_D(v_DG8);
        __m128d v_D9 = CAST_512_TO_128_D(v_DG9);
        __m128d v_D10 = CAST_512_TO_128_D(v_DG10);
        __m128d v_D11 = CAST_512_TO_128_D(v_DG11);
        __m128d v_D12 = CAST_512_TO_128_D(v_DG12);
        __m128d v_ZERO_128 = CAST_512_TO_128_D(v_ZERO);
#elif defined(KERNEL_USE_AVX256)
        __m128d v_K1 = CAST_256_TO_128_D(v_C1);
        __m128d v_K2 = CAST_256_TO_128_D(v_C2);
        __m128d v_K3 = CAST_256_TO_128_D(v_C3);
        __m128d v_K4 = CAST_256_TO_128_D(v_C4);
        __m128d v_K5 = CAST_256_TO_128_D(v_C5);
        __m128d v_K6 = CAST_256_TO_128_D(v_C6);
        __m128d v_K7 = CAST_256_TO_128_D(v_C7);
        __m128d v_K8 = CAST_256_TO_128_D(v_C8);
        __m128d v_K9 = CAST_256_TO_128_D(v_C9);
        __m128d v_K10 = CAST_256_TO_128_D(v_C10);
        __m128d v_D1 = CAST_256_TO_128_D(v_DG1);
        __m128d v_D2 = CAST_256_TO_128_D(v_DG2);
        __m128d v_D3 = CAST_256_TO_128_D(v_DG3);
        __m128d v_D4 = CAST_256_TO_128_D(v_DG4);
        __m128d v_D5 = CAST_256_TO_128_D(v_DG5);
        __m128d v_D6 = CAST_256_TO_128_D(v_DG6);
        __m128d v_D7 = CAST_256_TO_128_D(v_DG7);
        __m128d v_D8 = CAST_256_TO_128_D(v_DG8);
        __m128d v_D9 = CAST_256_TO_128_D(v_DG9);
        __m128d v_D10 = CAST_256_TO_128_D(v_DG10);
        __m128d v_D11 = CAST_256_TO_128_D(v_DG11);
        __m128d v_D12 = CAST_256_TO_128_D(v_DG12);
        __m128d v_ZERO_128 = CAST_256_TO_128_D(v_ZERO);
#endif

        LD_128_D(in_r, v_in0);

        LOAD_IN_128_D(in_r, in_strides, 1, 0, v_in1, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 2, 0, v_in2, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 3, 0, v_in3, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 4, 0, v_in4, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 5, 0, v_in5, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 6, 0, v_in6, tw_ptr, load_multi_cols,
                      is_contiguous_in);
#if defined(KERNEL_VARIANT_C2R)
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 7, 0, v_in7, tw_ptr,
                         load_multi_cols, 0);
        v_in7 = IN_H2_128_D(v_in7);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 8, 0, v_in8, tw_ptr,
                         load_multi_cols, 0);
        v_in8 = IN_H2_128_D(v_in8);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 9, 0, v_in9, tw_ptr,
                         load_multi_cols, 0);
        v_in9 = IN_H2_128_D(v_in9);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 10, 0, v_in10, tw_ptr,
                      load_multi_cols, 0);
        v_in10 = IN_H2_128_D(v_in10);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 11, 0, v_in11, tw_ptr,
                      load_multi_cols, 0);
        v_in11 = IN_H2_128_D(v_in11);
        LOAD_IN_H2_128_D(in_h2_r, in_strides, 12, 0, v_in12, tw_ptr,
                      load_multi_cols, 0);
        v_in12 = IN_H2_128_D(v_in12);
#else
        LOAD_IN_128_D(in_r, in_strides, 7, 0, v_in7, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 8, 0, v_in8, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 9, 0, v_in9, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 10, 0, v_in10, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 11, 0, v_in11, tw_ptr, load_multi_cols,
                      is_contiguous_in);
        LOAD_IN_128_D(in_r, in_strides, 12, 0, v_in12, tw_ptr, load_multi_cols,
                      is_contiguous_in);
#endif

        v_av1 = _mm_add_pd(v_in1, v_in12);
        v_av2 = _mm_add_pd(v_in4, v_in3);
        v_av3 = _mm_add_pd(v_in9, v_in10);
        v_av4 = _mm_add_pd(v_in2, v_in6);
        v_av5 = _mm_add_pd(v_in11, v_in7);
        v_av6 = _mm_add_pd(v_in8, v_in5);

        v_av7 = _mm_sub_pd(v_in1, v_in12);
        v_av8 = _mm_sub_pd(v_in4, v_in3);
        v_av9 = _mm_sub_pd(v_in9, v_in10);
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
        v_t1 = _mm_sub_pd(v_in0, v_cv5);

        v_tv1 = _mm_mul_pd(v_cv3, v_D1);
        v_tv2 = _mm_mul_pd(v_cv4, v_D12);
        v_cv5 = _mm_add_pd(v_tv1, v_tv2);
        v_t7 = _mm_sub_pd(v_in0, v_cv5);

        v_cv5 = _mm_sub_pd(v_cv1, v_cv2);
        v_tv3 = _mm_mul_pd(v_K2, v_cv5);
        v_cv6 = _mm_add_pd(v_av10, v_av11);
        v_cv7 = _mm_add_pd(v_av8, v_av9);
        v_cv8 = _mm_add_pd(v_cv6, v_cv7);
        v_tv4 = _mm_mul_pd(v_K6, v_cv8);

        v_cv9 = _mm_sub_pd(v_av1, v_av6);
        v_cv10 = _mm_sub_pd(v_cv9, v_tv3);
        v_tv5 = _mm_mul_pd(v_D4, v_cv10);
        v_t2 = _mm_add_pd(v_tv4, v_tv5);

        v_tv4 = _mm_mul_pd(v_K5, v_cv8);
        v_tv5 = _mm_mul_pd(v_D5, v_cv10);
        v_t8 = _mm_sub_pd(v_tv4, v_tv5);

        v_cv11 = _mm_add_pd(v_cv1, v_cv2);
        v_tv6 = _mm_mul_pd(v_K2, v_cv11);
        v_cv12 = _mm_add_pd(v_av1, v_av6);
        v_cv13 = _mm_sub_pd(v_cv12, v_tv6);
        v_cv14 = _mm_sub_pd(v_cv6, v_cv7);

        v_tv7 = _mm_mul_pd(v_K8, v_cv14);
        v_tv8 = _mm_mul_pd(v_D8, v_cv13);
        v_t3 = _mm_add_pd(v_tv7, v_tv8);

        v_tv7 = _mm_mul_pd(v_K7, v_cv14);
        v_tv8 = _mm_mul_pd(v_D9, v_cv13);
        v_t9 = _mm_sub_pd(v_tv7, v_tv8);

        v_cv15 = _mm_sub_pd(v_av2, v_av3);
        v_cv16 = _mm_sub_pd(v_av4, v_av5);
        v_cv17 = _mm_sub_pd(v_av8, v_av9);
        v_cv18 = _mm_sub_pd(v_av10, v_av11);
        v_tv9 = _mm_mul_pd(v_K4, v_cv16);
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
        v_t4 = _mm_add_pd(v_tv14, v_tv15);
        v_t4 = CONJ_128_D(SWAP_RI_128_D(v_t4));

        v_tv14 = _mm_mul_pd(v_D3, v_cv20);
        v_tv15 = _mm_mul_pd(v_D2, v_cv22);
        v_t10 = _mm_sub_pd(v_tv15, v_tv14);
        v_t10 = SWAP_RI_128_D(v_t10);

        v_cv23 = _mm_sub_pd(v_cv19, v_tv10);
        v_cv24 = _mm_sub_pd(v_cv21, v_tv13);

        v_tv16 = _mm_mul_pd(v_D10, v_cv23);
        v_tv17 = _mm_mul_pd(v_D11, v_cv24);
        v_t5 = _mm_add_pd(v_tv16, v_tv17);
        v_t5 = SWAP_RI_128_D(CONJ_128_D(v_t5));

        v_tv16 = _mm_mul_pd(v_D11, v_cv23);
        v_tv17 = _mm_mul_pd(v_D10, v_cv24);
        v_t11 = _mm_sub_pd(v_tv16, v_tv17);
        v_t11 = SWAP_RI_128_D(v_t11);

        v_cv25 = _mm_sub_pd(v_cv16, v_av12);
        v_cv26 = _mm_sub_pd(v_av7, v_cv17);

        v_tv18 = _mm_mul_pd(v_K9, v_cv25);
        v_tv19 = _mm_mul_pd(v_D7, v_cv26);
        v_t6 = _mm_add_pd(v_tv18, v_tv19);
        v_t6 = CONJ_128_D(SWAP_RI_128_D(v_t6));

        v_tv18 = _mm_mul_pd(v_K10, v_cv25);
        v_tv19 = _mm_mul_pd(v_D6, v_cv26);
        v_t12 = _mm_sub_pd(v_tv19, v_tv18);
        v_t12 = SWAP_RI_128_D(v_t12);

        v_cv27 = _mm_add_pd(v_cv11, v_cv12);
        v_out0 = _mm_add_pd(v_in0, v_cv27);

        v_av13 = _mm_add_pd(v_t2, v_t3);
        v_av14 = _mm_add_pd(v_t4, v_t5);
        v_av15 = _mm_add_pd(v_t1, v_t6);

        v_cv28 = _mm_add_pd(v_av13, v_av14);
        v_out1 = _mm_add_pd(v_cv28, v_av15);

        v_av16 = _mm_sub_pd(v_t1, v_t6);
        v_cv29 = _mm_sub_pd(v_av13, v_av14);
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

        v_out4 = _mm_add_pd(v_cv34, v_cv35);
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

        v_out6 = _mm_add_pd(v_cv40, v_cv41);
        v_out11 = _mm_sub_pd(v_cv40, v_cv41);

        ST_128_D(out_r, v_out0);
        STORE_OUT_128_D(out_r, out_strides, 1, 0, v_out1, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 2, 0, v_out2, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 3, 0, v_out3, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 4, 0, v_out4, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 5, 0, v_out5, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 6, 0, v_out6, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#if defined(KERNEL_VARIANT_R2C)
        v_out7 = OUT_H2_128_D(v_out7);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 7, 0, v_out7, tw_ptr,
                        load_multi_cols, 0);
        v_out8 = OUT_H2_128_D(v_out8);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 8, 0, v_out8, tw_ptr,
                        load_multi_cols, 0);
        v_out9 = OUT_H2_128_D(v_out9);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 9, 0, v_out9, tw_ptr,
                        load_multi_cols, 0);
        v_out10 = OUT_H2_128_D(v_out10);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 10, 0, v_out10, tw_ptr,
                        load_multi_cols, 0);
        v_out11 = OUT_H2_128_D(v_out11);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 11, 0, v_out11, tw_ptr,
                        load_multi_cols, 0);
        v_out12 = OUT_H2_128_D(v_out12);
        STORE_OUT_H2_128_D(out_h2_r, out_strides, 12, 0, v_out12, tw_ptr,
                        load_multi_cols, 0);
#else
        STORE_OUT_128_D(out_r, out_strides, 7, 0, v_out7, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 8, 0, v_out8, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 9, 0, v_out9, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 10, 0, v_out10, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 11, 0, v_out11, tw_ptr,
                        load_multi_cols, is_contiguous_out);
        STORE_OUT_128_D(out_r, out_strides, 12, 0, v_out12, tw_ptr,
                        load_multi_cols, is_contiguous_out);
#endif
    }
#endif
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ REGISTER_KERNEL(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return TWID_KNAME_FP32;
    }
    else if (precision == DT_DOUBLE)
    {
        return TWID_KNAME_FP64;
    }
    else
    {
        return NULL;
    }
}

ops_cycles_t GET_OPS_COUNT(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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
#undef RADIX
