/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
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

/** @file fft48c.c
 *
 *  @brief Radix-48 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-48 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Avinash Thakur
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 196, 616, 192, 0, 0},
                                                     {0, 196, 616, 192, 0, 0}};

ops_cycles_t get_ops_cnt_fft48c(UINT8 precision, UINT8 direction)
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

static VOID fft48c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    FLOAT *in_r, *in_i, *out_r, *out_i;
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

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FLOAT *)in_imag;
        in_i = (FLOAT *)in_real;
        out_r = (FLOAT *)out_imag;
        out_i = (FLOAT *)out_real;
    }
    else
    {
        in_r = (FLOAT *)in_real;
        in_i = (FLOAT *)in_imag;
        out_r = (FLOAT *)out_real;
        out_i = (FLOAT *)out_imag;
    }

    // Constant coefficients
    FLOAT x69 = 0.13052619222005159154840622789548901019374070481173f;
    FLOAT x80 = 0.9914448613738104111445575269285628712777382744481f;
    FLOAT x90 = 0.79335334029123516457977696150129927662867592105191f;
    FLOAT x101 = 0.60876142900872063941609754289816400451639371196248f;
    FLOAT x112 = 0.86602540378443864676372317075293618347140262690519f;
    FLOAT x125 = 0.70710678118654752440084436210484903928483593768847f;
    FLOAT x138 = 0.9659258262890682867497431997288973676339048390084f;
    FLOAT x150 = 0.25881904510252076234889883762404832834906890131993f;
    FLOAT x162 = 0.5f;
    FLOAT x181 = 0.92387953251128675612818318939678828682241662586364f;
    FLOAT x193 = 0.38268343236508977172845998403039886676134456248563f;
    FLOAT x465 = 0.60876142900872063941609754289816400451639371196247f;
    FLOAT x472 = 0.96592582628906828674974319972889736763390483900841f;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT rl0, im0, rl1, im1, rl2, im2, rl3, im3, rl4, im4, rl5, im5, rl6,
            im6, rl7, im7, rl8, im8, rl9, im9, rl10, im10, rl11, im11, rl12,
            im12, rl13, im13, rl14, im14, rl15, im15, rl16, im16, rl17, im17,
            rl18, im18, rl19, im19, rl20, im20, rl21, im21, rl22, im22, rl23,
            im23, rl24, im24, rl25, im25, rl26, im26, rl27, im27, rl28, im28,
            rl29, im29, rl30, im30, rl31, im31, rl32, im32, rl33, im33, rl34,
            im34, rl35, im35, rl36, im36, rl37, im37, rl38, im38, rl39, im39,
            rl40, im40, rl41, im41, rl42, im42, rl43, im43, rl44, im44, rl45,
            im45, rl46, im46, rl47, im47;

        rl0 = in_r[in_strides[0]];
        im0 = in_i[in_strides[0]];
        rl1 = in_r[in_strides[1]];
        im1 = in_i[in_strides[1]];
        rl2 = in_r[in_strides[2]];
        im2 = in_i[in_strides[2]];
        rl3 = in_r[in_strides[3]];
        im3 = in_i[in_strides[3]];
        rl4 = in_r[in_strides[4]];
        im4 = in_i[in_strides[4]];
        rl5 = in_r[in_strides[5]];
        im5 = in_i[in_strides[5]];
        rl6 = in_r[in_strides[6]];
        im6 = in_i[in_strides[6]];
        rl7 = in_r[in_strides[7]];
        im7 = in_i[in_strides[7]];
        rl8 = in_r[in_strides[8]];
        im8 = in_i[in_strides[8]];
        rl9 = in_r[in_strides[9]];
        im9 = in_i[in_strides[9]];
        rl10 = in_r[in_strides[10]];
        im10 = in_i[in_strides[10]];
        rl11 = in_r[in_strides[11]];
        im11 = in_i[in_strides[11]];
        rl12 = in_r[in_strides[12]];
        im12 = in_i[in_strides[12]];
        rl13 = in_r[in_strides[13]];
        im13 = in_i[in_strides[13]];
        rl14 = in_r[in_strides[14]];
        im14 = in_i[in_strides[14]];
        rl15 = in_r[in_strides[15]];
        im15 = in_i[in_strides[15]];
        rl16 = in_r[in_strides[16]];
        im16 = in_i[in_strides[16]];
        rl17 = in_r[in_strides[17]];
        im17 = in_i[in_strides[17]];
        rl18 = in_r[in_strides[18]];
        im18 = in_i[in_strides[18]];
        rl19 = in_r[in_strides[19]];
        im19 = in_i[in_strides[19]];
        rl20 = in_r[in_strides[20]];
        im20 = in_i[in_strides[20]];
        rl21 = in_r[in_strides[21]];
        im21 = in_i[in_strides[21]];
        rl22 = in_r[in_strides[22]];
        im22 = in_i[in_strides[22]];
        rl23 = in_r[in_strides[23]];
        im23 = in_i[in_strides[23]];
        rl24 = in_r[in_strides[24]];
        im24 = in_i[in_strides[24]];
        rl25 = in_r[in_strides[25]];
        im25 = in_i[in_strides[25]];
        rl26 = in_r[in_strides[26]];
        im26 = in_i[in_strides[26]];
        rl27 = in_r[in_strides[27]];
        im27 = in_i[in_strides[27]];
        rl28 = in_r[in_strides[28]];
        im28 = in_i[in_strides[28]];
        rl29 = in_r[in_strides[29]];
        im29 = in_i[in_strides[29]];
        rl30 = in_r[in_strides[30]];
        im30 = in_i[in_strides[30]];
        rl31 = in_r[in_strides[31]];
        im31 = in_i[in_strides[31]];
        rl32 = in_r[in_strides[32]];
        im32 = in_i[in_strides[32]];
        rl33 = in_r[in_strides[33]];
        im33 = in_i[in_strides[33]];
        rl34 = in_r[in_strides[34]];
        im34 = in_i[in_strides[34]];
        rl35 = in_r[in_strides[35]];
        im35 = in_i[in_strides[35]];
        rl36 = in_r[in_strides[36]];
        im36 = in_i[in_strides[36]];
        rl37 = in_r[in_strides[37]];
        im37 = in_i[in_strides[37]];
        rl38 = in_r[in_strides[38]];
        im38 = in_i[in_strides[38]];
        rl39 = in_r[in_strides[39]];
        im39 = in_i[in_strides[39]];
        rl40 = in_r[in_strides[40]];
        im40 = in_i[in_strides[40]];
        rl41 = in_r[in_strides[41]];
        im41 = in_i[in_strides[41]];
        rl42 = in_r[in_strides[42]];
        im42 = in_i[in_strides[42]];
        rl43 = in_r[in_strides[43]];
        im43 = in_i[in_strides[43]];
        rl44 = in_r[in_strides[44]];
        im44 = in_i[in_strides[44]];
        rl45 = in_r[in_strides[45]];
        im45 = in_i[in_strides[45]];
        rl46 = in_r[in_strides[46]];
        im46 = in_i[in_strides[46]];
        rl47 = in_r[in_strides[47]];
        im47 = in_i[in_strides[47]];

        FLOAT x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
            x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27,
            x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
            x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53,
            x54, x55, x56, x57, x58, x61, x62, x64, x66, x67, x68, x72, x73,
            x75, x77, x78, x79, x83, x85, x87, x88, x89, x92, x94, x95, x98,
            x99, x100, x102, x104, x106, x107, x109, x111, x113, x115, x117,
            x118, x120, x122, x123, x124, x126, x127, x129, x131, x133, x135,
            x136, x137, x140, x142, x143, x145, x147, x148, x149, x151, x152,
            x155, x156, x159, x160, x161, x163, x165, x167, x168, x169, x171,
            x173, x174, x176, x178, x179, x180, x183, x185, x186, x188, x190,
            x191, x192, x194, x195, x197, x199, x200, x201, x203, x204, x206,
            x207, x208, x210, x211, x213, x214, x215, x217, x219, x220, x221,
            x222, x223, x225, x227, x228, x229, x231, x232, x234, x235, x236,
            x238, x240, x241, x243, x245, x246, x247, x249, x251, x252, x253,
            x254, x256, x258, x259, x260, x262, x264, x265, x266, x268, x270,
            x271, x272, x273, x274, x275, x276, x277, x278, x280, x281, x282,
            x284, x285, x286, x287, x288, x289, x290, x292, x293, x294, x295,
            x296, x297, x298, x299, x300, x301, x302, x303, x304, x306, x307,
            x308, x309, x311, x312, x313, x314, x315, x316, x318, x319, x320,
            x321, x322, x324, x325, x326, x328, x329, x330, x331, x332, x334,
            x335, x336, x337, x338, x339, x340, x341, x342, x343, x344, x345,
            x347, x348, x349, x350, x352, x353, x354, x355, x356, x357, x359,
            x360, x362, x363, x364, x365, x366, x368, x369, x370, x371, x372,
            x373, x374, x375, x376, x377, x378, x379, x381, x382, x383, x384,
            x385, x386, x388, x389, x390, x391, x392, x393, x394, x396, x397,
            x398, x399, x400, x401, x402, x403, x404, x405, x406, x407, x409,
            x410, x411, x412, x414, x415, x416, x417, x418, x419, x420, x421,
            x422, x423, x424, x425, x426, x428, x429, x430, x431, x432, x433,
            x434, x435, x436, x437, x439, x440, x441, x442, x443, x444, x446,
            x447, x448, x449, x450, x451, x452, x453, x454, x455, x456, x457,
            x458, x463, x466, x467, x468, x469, x471, x473, x474, x475, x476,
            x477, x478, x479, x480, x481, x482, x483, x484, x485, x486, x487,
            x488, x489, x490, x492, x493, x495, x496, x497, x498, x499, x500,
            x501, x502, x503, x504, x505, x506, x507, x508, x509, x510, x511,
            x512, x513, x515, x516, x517, x518, x519, x520, x522, x523, x525,
            x526, x527, x528, x529, x530, x531, x532, x533, x534, x535, x536,
            x537, x538, x539, x540, x541, x542, x543, x544, x545, x547, x548,
            x549, x550, x551, x552, x553, x554, x555, x557, x558, x559, x560,
            x561, x562, x563, x564, x565, x566, x567, x568, x569, x570, x571,
            x572, x573, x574, x575, x576, x577, x578, x579, x580, x581, x582,
            x583, x584, x585, x586, x587, x588, x589, x590, x591, x592, x593,
            x594, x595, x596, x597, x598, x600, x601, x602, x603, x604, x605,
            x606, x607, x608, x609, x610, x611, x612, x613, x614, x615, x616,
            x617, x618, x619, x620, x621, x622, x623, x624, x625, x626, x627,
            x628, x629, x630, x631, x632, x633, x634, x635, x636, x637, x638,
            x639, x640, x641, x642, x643, x644, x645, x646, x647, x648, x649,
            x650, x651, x652, x653, x654, x655, x656, x657, x658, x659, x660,
            x662, x663, x664, x665, x666, x667, x668, x669, x670, x671, x672,
            x673, x674, x675, x676, x677, x678, x679, x680, x681, x682, x683,
            x684, x685, x686, x687, x688, x689, x690, x691, x692, x693, x694,
            x695, x696, x697;

        x0 = rl39 + rl9;
        x1 = rl3 + rl45;
        x2 = rl12 + rl36;
        x3 = rl0 + rl24;
        x4 = x2 + x3;
        x5 = rl18 + rl30 + rl42 + rl6;
        x6 = x4 + x5;
        x7 = rl15 + rl21 + rl27 + rl33 + x0 + x1 + x6;
        x8 = rl43 + rl5;
        x9 = rl41 + rl7;
        x10 = rl19 + rl29 + x8 + x9;
        x11 = rl23 + rl47;
        x12 = rl31 + x11;
        x13 = rl11 + rl35;
        x14 = rl13 + rl37 + x13;
        x15 = rl1 + rl25;
        x16 = x14 + x15;
        x17 = rl17 + x10 + x12 + x16;
        x18 = rl22 + rl46;
        x19 = rl26 + rl34;
        x20 = rl10 + rl14 + rl2 + rl38 + x18 + x19;
        x21 = rl20 + rl44;
        x22 = rl32 + x21;
        x23 = rl40 + rl8;
        x24 = rl16 + x23;
        x25 = rl28 + rl4;
        x26 = x24 + x25;
        x27 = x20 + x22 + x26;
        x28 = x17 + x27;
        x29 = im21 + im27;
        x30 = im39 + im45;
        x31 = im15 + im33;
        x32 = im9 + x31;
        x33 = im18 + im30;
        x34 = im42 + im6;
        x35 = im12 + im36;
        x36 = im0 + im24;
        x37 = x35 + x36;
        x38 = x33 + x34 + x37;
        x39 = im3 + x29 + x30 + x32 + x38;
        x40 = im19 + im29;
        x41 = im17 + im31;
        x42 = im41 + im7 + x41;
        x43 = im43 + im5;
        x44 = im11 + im35;
        x45 = im1 + im23 + im25 + im47;
        x46 = im13 + im37;
        x47 = x44 + x45 + x46;
        x48 = x40 + x42 + x43 + x47;
        x49 = im32 + im8;
        x50 = im2 + im46;
        x51 = im10 + im38;
        x52 = im14 + x51;
        x53 = im22 + im26 + im34 + x50 + x52;
        x54 = im20 + im4;
        x55 = im16 + im28 + im40;
        x56 = im44 + x54 + x55;
        x57 = x49 + x53 + x56;
        x58 = x48 + x57;
        x61 = im1 - im25;
        x62 = rl37 - rl13 + x61;
        x64 = im23 - im47;
        x66 = rl11 - rl35;
        x67 = x64 + x66;
        x68 = x62 + x67;
        x72 = im11 - im35;
        x73 = rl47 - rl23 + x72;
        x75 = im13 - im37;
        x77 = rl1 - rl25;
        x78 = x75 + x77;
        x79 = x73 + x78;
        x83 = -rl19 - rl29;
        x85 = im17 - im31;
        x87 = im7 - im41;
        x88 = x85 + x87;
        x89 = x8 + x83 + x88;
        x92 = im19 - im29;
        x94 = im5 - im43;
        x95 = x92 + x94;
        x98 = -rl17 - rl31;
        x99 = x9 + x98;
        x100 = x95 + x99;
        x102 = x100 * x101 + x68 * x69 + x79 * x80 + x89 * x90;
        x104 = im16 - im40;
        x106 = rl4 - rl28;
        x107 = x104 + x106;
        x109 = im8 - im32;
        x111 = rl44 - rl20;
        x113 = x112 * (x107 + x109 + x111);
        x115 = im6 - im42;
        x117 = im18 - im30;
        x118 = x115 + x117;
        x120 = rl6 - rl18;
        x122 = rl42 - rl30;
        x123 = x120 + x122;
        x124 = x118 + x123;
        x126 = x124 * x125;
        x127 = x113 + x126;
        x129 = rl2 - rl22;
        x131 = rl46 - rl26;
        x133 = im10 - im38;
        x135 = im14 - im34;
        x136 = x133 + x135;
        x137 = x129 + x131 + x136;
        x140 = im2 - im46;
        x142 = im22 - im26;
        x143 = x140 + x142;
        x145 = rl10 - rl14;
        x147 = rl38 - rl34;
        x148 = x145 + x147;
        x149 = x143 + x148;
        x151 = x137 * x138 + x149 * x150;
        x152 = x127 + x151;
        x155 = -im28 - im44;
        x156 = x155 + x54;
        x159 = -rl16 - rl32;
        x160 = x159 + x23;
        x161 = x156 + x160;
        x163 = x161 * x162;
        x165 = im12 - im36;
        x167 = rl0 - rl24;
        x168 = x165 + x167;
        x169 = x163 + x168;
        x171 = im15 - im33;
        x173 = im9 - im39;
        x174 = x171 + x173;
        x176 = rl45 - rl27;
        x178 = rl3 - rl21;
        x179 = x176 + x178;
        x180 = x174 + x179;
        x183 = im3 - im45;
        x185 = im21 - im27;
        x186 = x183 + x185;
        x188 = rl9 - rl15;
        x190 = rl39 - rl33;
        x191 = x188 + x190;
        x192 = x186 + x191;
        x194 = x180 * x181 + x192 * x193;
        x195 = x169 + x194;
        x197 = rl26 - rl2;
        x199 = -im14 - im34;
        x200 = x199 + x51;
        x201 = rl46 - rl22 + x197 + x200;
        x203 = -im22 - im26;
        x204 = x203 + x50;
        x206 = rl34 - rl10;
        x207 = rl38 - rl14 + x206;
        x208 = x204 + x207;
        x210 = -im42 - im6;
        x211 = x210 + x33;
        x213 = rl18 - rl42;
        x214 = rl6 - rl30 + x213;
        x215 = x125 * (x211 + x214);
        x217 = rl16 - rl40;
        x219 = im28 - im4;
        x220 = x217 + x219;
        x221 = im20 - im44;
        x222 = x112 * (rl8 - rl32 + x220 + x221);
        x223 = x215 + x222;
        x225 = rl19 - rl43;
        x227 = -im41 - im7;
        x228 = x227 + x41;
        x229 = rl5 - rl29 + x225 + x228;
        x231 = -im43 - im5;
        x232 = x231 + x40;
        x234 = rl17 - rl41;
        x235 = rl7 - rl31 + x234;
        x236 = x232 + x235;
        x238 = rl35 - rl11;
        x240 = im47 - im23;
        x241 = x238 + x240 + x62;
        x243 = rl25 - rl1;
        x245 = im37 - im13;
        x246 = x243 + x245 + x73;
        x247 = x101 * x229 + x236 * x90 - x241 * x80 - x246 * x69;
        x249 = rl20 - rl44;
        x251 = im32 - im8;
        x252 = x249 + x251;
        x253 = x107 + x252;
        x254 = x162 * x253;
        x256 = im24 - im0;
        x258 = rl12 - rl36;
        x259 = x256 + x258;
        x260 = x254 + x259;
        x262 = rl21 - rl45;
        x264 = -im39 - im9;
        x265 = x264 + x31;
        x266 = rl3 - rl27 + x262 + x265;
        x268 = rl15 - rl39;
        x270 = -im3 - im45;
        x271 = x270 + x29;
        x272 = rl9 - rl33 + x268 + x271;
        x273 = x181 * x272 + x193 * x266;
        x274 = x260 + x273;
        x275 = im34 - im22;
        x276 = im26 - im14;
        x277 = x133 + x140 + x276;
        x278 = x275 + x277;
        x280 = x25 - rl8;
        x281 = x159 - rl40;
        x282 = x21 + x278 + x280 + x281;
        x284 = rl33 - rl3;
        x285 = rl15 - rl45;
        x286 = -rl21 - rl27;
        x287 = x171 - im9;
        x288 = x185 - im3;
        x289 = x287 + x288 + x30;
        x290 = x125 * (x0 + x284 + x285 + x286 + x289);
        x292 = rl14 - rl2;
        x293 = rl38 - rl26 + x292;
        x294 = im16 - im4 + im40;
        x295 = x294 - im28;
        x296 = im20 + im44;
        x297 = -im32 - im8;
        x298 = x295 + x296 + x297;
        x299 = x112 * (rl10 - rl22 + rl34 - rl46 + x293 + x298);
        x300 = x290 + x299;
        x301 = im41 - im7 + x85;
        x302 = im43 - im5;
        x303 = x302 + x92;
        x304 = x301 + x303;
        x306 = -rl1 - rl23 - rl25 - rl47;
        x307 = x14 + x306;
        x308 = x304 + x307;
        x309 = rl19 - rl41;
        x311 = rl29 - rl7;
        x312 = im1 - im23 + im25 - im47;
        x313 = -im13 - im37;
        x314 = x312 + x313 + x44;
        x315 = x309 + x311 + x314 + x8 + x98;
        x316 = x138 * x308 - x150 * x315;
        x318 = -rl0 - rl24;
        x319 = x2 + x318;
        x320 = im42 - im6;
        x321 = x117 + x320;
        x322 = x319 + x321;
        x324 = -rl13 - rl37;
        x325 = x13 + x324;
        x326 = -rl23 - rl47 + x15;
        x328 = -im19 - im29;
        x329 = x231 + x328;
        x330 = x329 + x42;
        x331 = x325 + x326 + x330;
        x332 = x150 * x331;
        x334 = rl17 - rl5;
        x335 = rl43 - rl29;
        x336 = rl41 - rl31;
        x337 = -im11 - im35;
        x338 = x313 + x337;
        x339 = x338 + x45;
        x340 = rl19 - rl7 + x334 + x335 + x336 + x339;
        x341 = x138 * x340;
        x342 = rl32 - rl20 - rl44;
        x343 = x342 - im46;
        x344 = -rl16 - rl40;
        x345 = x25 + x344;
        x347 = im34 - im2;
        x348 = x203 + x347 + x52;
        x349 = x112 * (rl8 + x343 + x345 + x348);
        x350 = rl27 - rl39;
        x352 = im39 - im21;
        x353 = x270 - im27;
        x354 = x32 + x352 + x353;
        x355 = rl33 - rl45 + x178 + x188 + x350 + x354;
        x356 = x125 * x355;
        x357 = x349 + x356;
        x359 = im32 - im20 + im8;
        x360 = x155 + x294 + x359;
        x362 = -rl38 - rl46;
        x363 = x129 + x145 + x19 + x362;
        x364 = x360 + x363;
        x365 = x162 * x364;
        x366 = rl30 - rl42 + x120;
        x368 = -im0 - im24 + x35;
        x369 = x365 + x366 + x368;
        x370 = rl23 - rl47 + x72;
        x371 = x243 + x370 + x75;
        x372 = im29 - im19;
        x373 = im27 - im21 - im3 + im45 + x302 + x372;
        x374 = x191 + x371 + x373 + x99;
        x375 = x181 * x374;
        x376 = rl13 - rl37 + x61;
        x377 = x238 + x376 + x64;
        x378 = rl29 - rl5;
        x379 = x225 + x378 + x88;
        x381 = im33 - im15;
        x382 = im39 - im9 + x381;
        x383 = x179 + x377 + x379 + x382;
        x384 = x193 * x383;
        x385 = rl22 - rl46;
        x386 = x136 + x197 + x385;
        x388 = im30 - im18;
        x389 = im26 - im2;
        x390 = x125 * (im46 - im22 + x123 + x148 + x320 + x386 + x388 + x389);
        x391 = rl24 - rl0;
        x392 = x165 + x391;
        x393 = im44 - im20;
        x394 = x160 + x219 + x390 + x392 + x393;
        x396 = rl33 - rl9;
        x397 = rl39 - rl15 + x271 + x396;
        x398 = x245 + x370 + x77;
        x399 = x328 + x43;
        x400 = x235 + x397 + x398 + x399;
        x401 = x193 * x400;
        x402 = x240 + x376 + x66;
        x403 = rl27 - rl3;
        x404 = rl45 - rl21 + x265 + x403;
        x405 = rl43 - rl19 + x228 + x378;
        x406 = x402 + x404 + x405;
        x407 = x181 * x406;
        x409 = rl2 - rl26 + x200 + x385;
        x410 = -im18 - im30;
        x411 = x34 + x410;
        x412 = x125 * (im22 - im46 + x207 + x214 + x389 + x409 + x411);
        x414 = rl36 - rl12;
        x415 = x256 + x414;
        x416 = x109 + x249;
        x417 = im40 - im16 + x106 + x412 + x415 + x416;
        x418 = -rl11 - rl35;
        x419 = rl31 + x418;
        x420 = x306 + x324;
        x421 = rl17 + x49;
        x422 = im46 - im14 + x133 + x142 + x347;
        x423 = -im16 - im4 - im40;
        x424 = x296 - im28 + x423;
        x425 = x422 + x424;
        x426 = x112 * (x10 + x419 + x420 + x421 + x425);
        x428 = -rl28 - rl4;
        x429 = x428 - rl8;
        x430 = -rl20 - rl44;
        x431 = x281 + x429 + x430;
        x432 = x372 + x94;
        x433 = x312 + x337 + x46;
        x434 = x301 + x432 + x433;
        x435 = x20 + x431 + x434;
        x436 = im27 + x183;
        x437 = x287 + x352 + x436;
        x439 = -rl18 - rl30 - rl42 - rl6 + x4;
        x440 = x162 * x435 + x437 + x439;
        x441 = x22 - rl2;
        x442 = rl8 + x344 + x428;
        x443 = rl22 + x131;
        x444 = rl34 - rl38 + x145 + x443;
        x446 = -im17 - im31 + x227 + x329;
        x447 = x446 + x47;
        x448 = x112 * (x441 + x442 + x444 + x447);
        x449 = -rl1 - rl25;
        x450 = rl7 - rl17;
        x451 = x309 - rl5 + x335 + x450;
        x452 = x12 + x325 + x449 + x451;
        x453 = x297 - im20;
        x454 = x155 + x423 + x453;
        x455 = x454 + x53;
        x456 = x210 + x37 + x410;
        x457 = rl21 - rl39 + x176 + x188 + x284 + x456;
        x458 = x162 * (x452 + x455) + x457;
        x463 = x101 * x62 + x78 * x90;
        x466 = x465 * x67 + x73 * x90;
        x467 = -x100 * x69 + x463 + x466 - x80 * x89;
        x468 = x180 * x193;
        x469 = x181 * x192 - x468;
        x471 = x137 * x150;
        x473 = x149 * x472;
        x474 = x471 - x126 + x473;
        x475 = x465 * x73;
        x476 = x229 * x69;
        x477 = x236 * x80;
        x478 = x67 * x90;
        x479 = x193 * x272;
        x480 = x181 * x266;
        x481 = x479 - x480;
        x482 = x62 * x90;
        x483 = x101 * x78;
        x484 = x482 - x483;
        x485 = x150 * x208 + x201 * x472;
        x486 = im0 - im24;
        x487 = x414 + x486;
        x488 = x487 - x254;
        x489 = x223 + x485 + x488;
        x490 = x24 + x275 + x428;
        x492 = rl31 - rl43;
        x493 = x314 + x334 + x492 + x83 + x9;
        x495 = x125 * x493;
        x496 = rl21 - rl9;
        x497 = -rl15 - rl33;
        x498 = x1 + x350 + x496 + x497;
        x499 = x289 + x498;
        x500 = im31 - im17 + x87;
        x501 = x125 * (x307 + x432 + x499 + x500);
        x502 = -rl12 - rl36;
        x503 = x3 + x502;
        x504 = x321 + x503;
        x505 = -rl10 - rl34;
        x506 = x18 + x505;
        x507 = x11 + x418;
        x508 = rl13 + rl37;
        x509 = x330 + x449 + x507 + x508;
        x510 = rl5 - rl19;
        x511 = rl29 - rl41 + x339 + x450 + x492 + x510;
        x512 = rl15 - rl3 + x176 + x190 + x354 + x496;
        x513 = x125 * x509 + x125 * (x511 + x512);
        x515 = -im12 - im36 + x36;
        x516 = x366 + x515;
        x517 = rl28 - rl4 + x104;
        x518 = x112 * (x416 + x517);
        x519 = rl14 - rl38;
        x520 = x143 + x206 + x519;
        x522 = rl30 - rl6;
        x523 = x125 * (x118 + x213 + x522);
        x525 = x138 * x520 + x150 * x386 - x523;
        x526 = rl32 - rl8;
        x527 = x156 + x217 + x526;
        x528 = x162 * x527;
        x529 = im36 - im12;
        x530 = x167 + x529;
        x531 = x530 - x528;
        x532 = x262 + x403;
        x533 = x174 + x532;
        x534 = x268 + x396;
        x535 = x186 + x534;
        x536 = x181 * x533 + x193 * x535;
        x537 = rl31 - rl7;
        x538 = x234 + x537;
        x539 = x538 + x95;
        x540 = -x101 * x371 + x377 * x90 + x379 * x69 + x536 - x539 * x80;
        x541 = rl42 - rl18 + x522;
        x542 = x125 * (x211 + x541);
        x543 = rl10 - rl34 + x204 + x519;
        x544 = x138 * x409 + x150 * x543 + x542;
        x545 = x112 * (rl40 - rl16 + x219 + x221 + x526);
        x547 = x111 + x251 + x517;
        x548 = x162 * x547;
        x549 = x415 + x548;
        x550 = x549 - x545;
        x551 = x544 + x550;
        x552 = rl41 - rl17 + x537;
        x553 = x232 + x552;
        x554 = x181 * x397 + x193 * x404;
        x555 = x101 * x402 - x398 * x90 - x405 * x80 + x553 * x69 + x554;
        x557 = x49 - im10;
        x558 = x303 + x433 + x500;
        x559 = x112 * (im26 - im22 + im38 + x135 + x140 + x424 + x557 + x558);
        x560 = -rl14 - rl22 - rl26 + x362 + x505;
        x561 = -rl3 - rl39 - rl45 - rl9 + x286 + x497 + x6;
        x562 = x162 * (x17 - rl2 + x431 + x560) + x561;
        x563 = x326 + x508;
        x564 = x419 + x451 + x563;
        x565 = x147 - rl10;
        x566 = rl14 + rl26 - rl46 + x129 + x565;
        x567 = x112 * (x22 + x442 + x564 + x566);
        x568 = -im2 - im38 - im46 + x199 + x203;
        x569 = x568 - im10;
        x570 = -im15 - im21;
        x571 = x264 - im33 + x353 + x38 + x570;
        x572 = x162 * (x454 + x48 + x569) + x571;
        x573 = x382 + x532 + x68 + x89;
        x574 = x373 + x534 + x538 + x79;
        x575 = x181 * x573 + x193 * x574;
        x576 = x124 + x149;
        x577 = x125 * x137;
        x578 = x125 * x576 - x577;
        x579 = x125 * (x201 + x411 + x541);
        x580 = x125 * x208;
        x581 = x229 + x241 + x266;
        x582 = x246 + x272 + x399 + x552;
        x583 = x181 * x582 + x193 * x581;
        x584 = x315 * x472;
        x585 = x150 * x308;
        x586 = x115 + x388;
        x587 = x503 + x586;
        x588 = x162 * x282 + x587;
        x589 = x150 * x340 - x331 * x472;
        x590 = rl18 - rl6 + x122;
        x591 = x515 + x590;
        x592 = x377 * x80;
        x593 = x539 * x90;
        x594 = x371 * x69;
        x595 = x101 * x379;
        x596 = x193 * x533;
        x597 = x181 * x535;
        x598 = x596 - x597;
        x600 = x531 - x518;
        x601 = x138 * x386;
        x602 = x150 * x520;
        x603 = x523 + x601 + x602;
        x604 = x600 + x603;
        x605 = x398 * x80;
        x606 = x181 * x404;
        x607 = x101 * x553;
        x608 = x402 * x69;
        x609 = x405 * x90;
        x610 = x545 + x549;
        x611 = x138 * x543 + x150 * x409 - x542;
        x612 = x610 + x611;
        x613 = im45 + x173 + x288 + x381;
        x614 = x439 + x613;
        x615 = x26 + x434 + x441 + x560;
        x616 = rl27 - rl9 + x178 + x190 + x285 + x456;
        x617 = x69 * x79;
        x618 = x101 * x89;
        x619 = x68 * x80;
        x620 = x100 * x90;
        x621 = x126 - x113;
        x622 = x151 + x621;
        x623 = x391 + x529;
        x624 = x623 - x163;
        x625 = x469 + x624;
        x626 = x241 * x69;
        x627 = x229 * x90;
        x628 = x246 * x80;
        x629 = x101 * x236;
        x630 = x138 * x208 + x150 * x201 - x215;
        x631 = x222 + x260 + x630;
        x632 = rl2 - rl14 + rl26 - rl38 + x298 + x506;
        x633 = x112 * x632;
        x634 = x277 + x342 + x490;
        x635 = x162 * x634;
        x636 = x319 + x586 + x635;
        x637 = x633 + x636;
        x638 = x125 * x499;
        x639 = x15 + x304 + x324 + x507;
        x640 = x138 * x493 - x150 * x639 + x638;
        x641 = rl16 - rl32 + rl40;
        x642 = x21 - im46 + x348 + x429 + x641;
        x643 = x125 * x512;
        x644 = -x138 * x509 + x150 * x511 + x643;
        x645 = x292 + x360 + x443 + x565;
        x646 = x162 * x645;
        x647 = x368 + x590 + x646;
        x648 = x181 * x383 + x193 * x374;
        x649 = x390 + x527 + x530;
        x650 = x181 * x400 + x193 * x406;
        x651 = x258 + x486;
        x652 = x412 + x547 + x651;
        x653 = x112 * (im4 - im44 + x422 + x453 + x55 + x558);
        x654 = -x162 * x28 + x7;
        x655 = x280 + x430 + x641;
        x656 = x112 * (x444 - rl2 + x564 + x655);
        x657 = -x162 * x58 + x39;
        x658 = x478 - x475;
        x659 = -x100 * x80 + x484 + x658 + x69 * x89;
        x660 = -x471 - x473;
        x662 = x215 - x222 + x485;
        x663 = -x229 * x80 + x236 * x69 + x463;
        x664 = x125 * x308 + x125 * (im33 + x173 + x315 + x436 + x498 + x570);
        x665 = x125 * x340 + x125 * (x331 + x355);
        x666 = x379 * x80;
        x667 = x539 * x69;
        x668 = x101 * x377;
        x669 = x371 * x90;
        x670 = x392 + x528;
        x671 = x670 - x518;
        x672 = x525 + x671;
        x673 = x193 * x397 - x606;
        x674 = x101 * x398 + x402 * x90 + x405 * x69 + x553 * x80 + x673;
        x675 = x181 * x574;
        x676 = x193 * x573;
        x677 = x161 + x623;
        x678 = x578 + x677;
        x679 = x193 * x582;
        x680 = x181 * x581;
        x681 = x580 - x579;
        x682 = x253 + x487 + x681;
        x683 = -x150 * x493 + x472 * x639 + x638;
        x684 = x112 * x642;
        x685 = x647 + x684;
        x686 = x150 * x509 - x472 * x511 + x643;
        x687 = x101 * x539 + x371 * x80 + x377 * x69 + x379 * x90 + x536;
        x688 = x101 * x405 - x398 * x69 - x402 * x80 + x553 * x90 + x554;
        x689 = x651 - x548;
        x690 = -rl41 - rl43 - rl5 - rl7 + x83 + x98;
        x691 = x194 + x624;
        x692 = x162 * x615;
        x693 = x112 * (x11 + x16 + x425 + x49 + x690);
        x694 = x318 + x5 + x502;
        x695 = x112 * (x447 + x566 + x655);
        x696 = x311 - rl43 + x336 + x418 + x510 + x563;
        x697 = x162 * (rl17 + x455 + x696) + x616;

        out_r[out_strides[0]] = x28 + x7;
        out_i[out_strides[0]] = x39 + x58;
        out_r[out_strides[1]] = x102 + x152 + x195;
        out_i[out_strides[1]] = x138 * x208 + x150 * x201 - x223 - x247 - x274;
        out_r[out_strides[2]] = x162 * x282 - x300 - x316 - x322;
        out_i[out_strides[2]] = x341 - x332 - x357 - x369;
        out_r[out_strides[3]] = x384 - x375 - x394;
        out_i[out_strides[3]] = x407 - x401 - x417;
        out_r[out_strides[4]] = x440 - x426;
        out_i[out_strides[4]] = x448 + x458;
        out_r[out_strides[5]] = x169 - x113 + x467 + x469 + x474;
        out_i[out_strides[5]] = x475 + x476 + x477 - x478 + x481 + x484 + x489;
        out_r[out_strides[6]] = im10 + im2 - im38 + x276 + x343 + x490 + x495 - x501 + x504;
        out_i[out_strides[6]] = x293 + x360 + x506 + x513 + x516;
        out_r[out_strides[7]] = x518 + x525 + x531 + x540;
        out_i[out_strides[7]] = x555 - x551;
        out_r[out_strides[8]] = x559 + x562;
        out_i[out_strides[8]] = x572 - x567;
        out_r[out_strides[9]] = x168 + x220 + x393 + x526 + x575 + x578;
        out_i[out_strides[9]] = x107 + x252 + x487 + x579 - x580 + x583;
        out_r[out_strides[10]] = x300 + x584 - x585 + x588;
        out_i[out_strides[10]] = x357 - x365 + x589 + x591;
        out_r[out_strides[11]] = x592 + x593 - x594 - x595 + x598 + x604;
        out_i[out_strides[11]] = x193 * x397 - x605 - x606 - x607 + x608 + x609 - x612;
        out_r[out_strides[12]] = x614 + x615;
        out_i[out_strides[12]] = x452 + x557 + x56 + x568 + x616;
        out_r[out_strides[13]] = -x617 - x618 + x619 + x620 - x622 - x625;
        out_i[out_strides[13]] = -x481 - x626 - x627 + x628 + x629 - x631;
        out_r[out_strides[14]] = x640 - x637;
        out_i[out_strides[14]] = x112 * x642 - x644 - x647;
        out_r[out_strides[15]] = x648 + x649;
        out_i[out_strides[15]] = x652 - x650;
        out_r[out_strides[16]] = x653 + x654;
        out_i[out_strides[16]] = x657 - x656;
        out_r[out_strides[17]] = x195 + x621 + x659 + x660;
        out_i[out_strides[17]] = -x274 + x465 * x67 - x662 - x663 + x73 * x90;
        out_r[out_strides[18]] = rl8 - x278 + x342 - x345 + x587 + x664;
        out_i[out_strides[18]] = x295 - im44 + x359 + x363 + x591 - x665;
        out_r[out_strides[19]] = -x598 - x666 - x667 + x668 + x669 - x672;
        out_i[out_strides[19]] = x544 - x550 - x674;
        out_r[out_strides[20]] = x426 + x440;
        out_i[out_strides[20]] = x458 - x448;
        out_r[out_strides[21]] = x676 - x675 - x678;
        out_i[out_strides[21]] = x679 - x680 + x682;
        out_r[out_strides[22]] = x112 * x632 - x636 - x683;
        out_i[out_strides[22]] = x686 - x685;
        out_r[out_strides[23]] = x600 - x523 - x601 - x602 + x687;
        out_i[out_strides[23]] = x611 - x545 + x688 + x689;
        out_r[out_strides[24]] = x27 + x418 + x420 + x561 + x690;
        out_i[out_strides[24]] = -im1 - im23 - im25 - im47 + x338 + x446 + x57 + x571;
        out_r[out_strides[25]] = x152 - x102 - x691;
        out_i[out_strides[25]] = x247 - x222 + x273 + x488 + x630;
        out_r[out_strides[26]] = x290 - x299 + x316 + x588;
        out_i[out_strides[26]] = x332 - x341 - x349 + x356 - x369;
        out_r[out_strides[27]] = x375 - x384 - x394;
        out_i[out_strides[27]] = x401 - x407 - x417;
        out_r[out_strides[28]] = -x437 - x692 - x693 - x694;
        out_i[out_strides[28]] = x697 - x695;
        out_r[out_strides[29]] = -x127 - x467 - x625 - x660;
        out_i[out_strides[29]] = -x476 - x477 - x481 - x482 + x483 + x489 + x658;
        out_r[out_strides[30]] = x501 - x495 + x504 + x634;
        out_i[out_strides[30]] = x516 - x513 + x645;
        out_r[out_strides[31]] = x138 * x520 + x150 * x386 - x523 - x540 - x671;
        out_i[out_strides[31]] = -x551 - x555;
        out_r[out_strides[32]] = x654 - x653;
        out_i[out_strides[32]] = x656 + x657;
        out_r[out_strides[33]] = x125 * x576 - x575 - x577 - x677;
        out_i[out_strides[33]] = x253 - x259 - x583 - x681;
        out_r[out_strides[34]] = x299 - x290 - x584 + x585 + x588;
        out_i[out_strides[34]] = x349 - x356 - x369 - x589;
        out_r[out_strides[35]] = -x592 - x593 + x594 + x595 - x598 + x604;
        out_i[out_strides[35]] = x605 + x607 - x608 - x609 - x612 - x673;
        out_r[out_strides[36]] = -x435 - x613 - x694;
        out_i[out_strides[36]] = x421 + x457 + x56 + x569 + x696;
        out_r[out_strides[37]] = x181 * x192 - x468 + x617 + x618 - x619 - x620 - x622 - x624;
        out_i[out_strides[37]] = x481 + x626 + x627 - x628 - x629 - x631;
        out_r[out_strides[38]] = -x637 - x640;
        out_i[out_strides[38]] = x516 + x644 - x646 + x684;
        out_r[out_strides[39]] = x649 - x648;
        out_i[out_strides[39]] = x650 + x652;
        out_r[out_strides[40]] = x562 - x559;
        out_i[out_strides[40]] = x567 + x572;
        out_r[out_strides[41]] = -x113 - x474 - x659 - x691;
        out_i[out_strides[41]] = x273 - x260 - x466 - x662 + x663;
        out_r[out_strides[42]] = -x282 - x322 - x664;
        out_i[out_strides[42]] = x364 + x591 + x665;
        out_r[out_strides[43]] = x598 + x666 + x667 - x668 - x669 - x672;
        out_i[out_strides[43]] = x544 + x545 + x674 + x689;
        out_r[out_strides[44]] = x614 - x692 + x693;
        out_i[out_strides[44]] = x695 + x697;
        out_r[out_strides[45]] = x675 - x676 - x678;
        out_i[out_strides[45]] = x680 - x679 + x682;
        out_r[out_strides[46]] = x504 + x633 - x635 + x683;
        out_i[out_strides[46]] = -x685 - x686;
        out_r[out_strides[47]] = -x518 - x603 - x670 - x687;
        out_i[out_strides[47]] = x138 * x543 + x150 * x409 - x542 - x610 - x688;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft48c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    DOUBLE *in_r, *in_i, *out_r, *out_i;
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

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (DOUBLE *)in_imag;
        in_i = (DOUBLE *)in_real;
        out_r = (DOUBLE *)out_imag;
        out_i = (DOUBLE *)out_real;
    }
    else
    {
        in_r = (DOUBLE *)in_real;
        in_i = (DOUBLE *)in_imag;
        out_r = (DOUBLE *)out_real;
        out_i = (DOUBLE *)out_imag;
    }

    // Constant coefficients
    DOUBLE x69 = 0.13052619222005159154840622789548901019374070481173;
    DOUBLE x80 = 0.9914448613738104111445575269285628712777382744481;
    DOUBLE x90 = 0.79335334029123516457977696150129927662867592105191;
    DOUBLE x101 = 0.60876142900872063941609754289816400451639371196248;
    DOUBLE x112 = 0.86602540378443864676372317075293618347140262690519;
    DOUBLE x125 = 0.70710678118654752440084436210484903928483593768847;
    DOUBLE x138 = 0.9659258262890682867497431997288973676339048390084;
    DOUBLE x150 = 0.25881904510252076234889883762404832834906890131993;
    DOUBLE x162 = 0.5;
    DOUBLE x181 = 0.92387953251128675612818318939678828682241662586364;
    DOUBLE x193 = 0.38268343236508977172845998403039886676134456248563;
    DOUBLE x465 = 0.60876142900872063941609754289816400451639371196247;
    DOUBLE x472 = 0.96592582628906828674974319972889736763390483900841;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE rl0, im0, rl1, im1, rl2, im2, rl3, im3, rl4, im4, rl5, im5, rl6,
            im6, rl7, im7, rl8, im8, rl9, im9, rl10, im10, rl11, im11, rl12,
            im12, rl13, im13, rl14, im14, rl15, im15, rl16, im16, rl17, im17,
            rl18, im18, rl19, im19, rl20, im20, rl21, im21, rl22, im22, rl23,
            im23, rl24, im24, rl25, im25, rl26, im26, rl27, im27, rl28, im28,
            rl29, im29, rl30, im30, rl31, im31, rl32, im32, rl33, im33, rl34,
            im34, rl35, im35, rl36, im36, rl37, im37, rl38, im38, rl39, im39,
            rl40, im40, rl41, im41, rl42, im42, rl43, im43, rl44, im44, rl45,
            im45, rl46, im46, rl47, im47;

        rl0 = in_r[in_strides[0]];
        im0 = in_i[in_strides[0]];
        rl1 = in_r[in_strides[1]];
        im1 = in_i[in_strides[1]];
        rl2 = in_r[in_strides[2]];
        im2 = in_i[in_strides[2]];
        rl3 = in_r[in_strides[3]];
        im3 = in_i[in_strides[3]];
        rl4 = in_r[in_strides[4]];
        im4 = in_i[in_strides[4]];
        rl5 = in_r[in_strides[5]];
        im5 = in_i[in_strides[5]];
        rl6 = in_r[in_strides[6]];
        im6 = in_i[in_strides[6]];
        rl7 = in_r[in_strides[7]];
        im7 = in_i[in_strides[7]];
        rl8 = in_r[in_strides[8]];
        im8 = in_i[in_strides[8]];
        rl9 = in_r[in_strides[9]];
        im9 = in_i[in_strides[9]];
        rl10 = in_r[in_strides[10]];
        im10 = in_i[in_strides[10]];
        rl11 = in_r[in_strides[11]];
        im11 = in_i[in_strides[11]];
        rl12 = in_r[in_strides[12]];
        im12 = in_i[in_strides[12]];
        rl13 = in_r[in_strides[13]];
        im13 = in_i[in_strides[13]];
        rl14 = in_r[in_strides[14]];
        im14 = in_i[in_strides[14]];
        rl15 = in_r[in_strides[15]];
        im15 = in_i[in_strides[15]];
        rl16 = in_r[in_strides[16]];
        im16 = in_i[in_strides[16]];
        rl17 = in_r[in_strides[17]];
        im17 = in_i[in_strides[17]];
        rl18 = in_r[in_strides[18]];
        im18 = in_i[in_strides[18]];
        rl19 = in_r[in_strides[19]];
        im19 = in_i[in_strides[19]];
        rl20 = in_r[in_strides[20]];
        im20 = in_i[in_strides[20]];
        rl21 = in_r[in_strides[21]];
        im21 = in_i[in_strides[21]];
        rl22 = in_r[in_strides[22]];
        im22 = in_i[in_strides[22]];
        rl23 = in_r[in_strides[23]];
        im23 = in_i[in_strides[23]];
        rl24 = in_r[in_strides[24]];
        im24 = in_i[in_strides[24]];
        rl25 = in_r[in_strides[25]];
        im25 = in_i[in_strides[25]];
        rl26 = in_r[in_strides[26]];
        im26 = in_i[in_strides[26]];
        rl27 = in_r[in_strides[27]];
        im27 = in_i[in_strides[27]];
        rl28 = in_r[in_strides[28]];
        im28 = in_i[in_strides[28]];
        rl29 = in_r[in_strides[29]];
        im29 = in_i[in_strides[29]];
        rl30 = in_r[in_strides[30]];
        im30 = in_i[in_strides[30]];
        rl31 = in_r[in_strides[31]];
        im31 = in_i[in_strides[31]];
        rl32 = in_r[in_strides[32]];
        im32 = in_i[in_strides[32]];
        rl33 = in_r[in_strides[33]];
        im33 = in_i[in_strides[33]];
        rl34 = in_r[in_strides[34]];
        im34 = in_i[in_strides[34]];
        rl35 = in_r[in_strides[35]];
        im35 = in_i[in_strides[35]];
        rl36 = in_r[in_strides[36]];
        im36 = in_i[in_strides[36]];
        rl37 = in_r[in_strides[37]];
        im37 = in_i[in_strides[37]];
        rl38 = in_r[in_strides[38]];
        im38 = in_i[in_strides[38]];
        rl39 = in_r[in_strides[39]];
        im39 = in_i[in_strides[39]];
        rl40 = in_r[in_strides[40]];
        im40 = in_i[in_strides[40]];
        rl41 = in_r[in_strides[41]];
        im41 = in_i[in_strides[41]];
        rl42 = in_r[in_strides[42]];
        im42 = in_i[in_strides[42]];
        rl43 = in_r[in_strides[43]];
        im43 = in_i[in_strides[43]];
        rl44 = in_r[in_strides[44]];
        im44 = in_i[in_strides[44]];
        rl45 = in_r[in_strides[45]];
        im45 = in_i[in_strides[45]];
        rl46 = in_r[in_strides[46]];
        im46 = in_i[in_strides[46]];
        rl47 = in_r[in_strides[47]];
        im47 = in_i[in_strides[47]];

        DOUBLE x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
            x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27,
            x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
            x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53,
            x54, x55, x56, x57, x58, x61, x62, x64, x66, x67, x68, x72, x73,
            x75, x77, x78, x79, x83, x85, x87, x88, x89, x92, x94, x95, x98,
            x99, x100, x102, x104, x106, x107, x109, x111, x113, x115, x117,
            x118, x120, x122, x123, x124, x126, x127, x129, x131, x133, x135,
            x136, x137, x140, x142, x143, x145, x147, x148, x149, x151, x152,
            x155, x156, x159, x160, x161, x163, x165, x167, x168, x169, x171,
            x173, x174, x176, x178, x179, x180, x183, x185, x186, x188, x190,
            x191, x192, x194, x195, x197, x199, x200, x201, x203, x204, x206,
            x207, x208, x210, x211, x213, x214, x215, x217, x219, x220, x221,
            x222, x223, x225, x227, x228, x229, x231, x232, x234, x235, x236,
            x238, x240, x241, x243, x245, x246, x247, x249, x251, x252, x253,
            x254, x256, x258, x259, x260, x262, x264, x265, x266, x268, x270,
            x271, x272, x273, x274, x275, x276, x277, x278, x280, x281, x282,
            x284, x285, x286, x287, x288, x289, x290, x292, x293, x294, x295,
            x296, x297, x298, x299, x300, x301, x302, x303, x304, x306, x307,
            x308, x309, x311, x312, x313, x314, x315, x316, x318, x319, x320,
            x321, x322, x324, x325, x326, x328, x329, x330, x331, x332, x334,
            x335, x336, x337, x338, x339, x340, x341, x342, x343, x344, x345,
            x347, x348, x349, x350, x352, x353, x354, x355, x356, x357, x359,
            x360, x362, x363, x364, x365, x366, x368, x369, x370, x371, x372,
            x373, x374, x375, x376, x377, x378, x379, x381, x382, x383, x384,
            x385, x386, x388, x389, x390, x391, x392, x393, x394, x396, x397,
            x398, x399, x400, x401, x402, x403, x404, x405, x406, x407, x409,
            x410, x411, x412, x414, x415, x416, x417, x418, x419, x420, x421,
            x422, x423, x424, x425, x426, x428, x429, x430, x431, x432, x433,
            x434, x435, x436, x437, x439, x440, x441, x442, x443, x444, x446,
            x447, x448, x449, x450, x451, x452, x453, x454, x455, x456, x457,
            x458, x463, x466, x467, x468, x469, x471, x473, x474, x475, x476,
            x477, x478, x479, x480, x481, x482, x483, x484, x485, x486, x487,
            x488, x489, x490, x492, x493, x495, x496, x497, x498, x499, x500,
            x501, x502, x503, x504, x505, x506, x507, x508, x509, x510, x511,
            x512, x513, x515, x516, x517, x518, x519, x520, x522, x523, x525,
            x526, x527, x528, x529, x530, x531, x532, x533, x534, x535, x536,
            x537, x538, x539, x540, x541, x542, x543, x544, x545, x547, x548,
            x549, x550, x551, x552, x553, x554, x555, x557, x558, x559, x560,
            x561, x562, x563, x564, x565, x566, x567, x568, x569, x570, x571,
            x572, x573, x574, x575, x576, x577, x578, x579, x580, x581, x582,
            x583, x584, x585, x586, x587, x588, x589, x590, x591, x592, x593,
            x594, x595, x596, x597, x598, x600, x601, x602, x603, x604, x605,
            x606, x607, x608, x609, x610, x611, x612, x613, x614, x615, x616,
            x617, x618, x619, x620, x621, x622, x623, x624, x625, x626, x627,
            x628, x629, x630, x631, x632, x633, x634, x635, x636, x637, x638,
            x639, x640, x641, x642, x643, x644, x645, x646, x647, x648, x649,
            x650, x651, x652, x653, x654, x655, x656, x657, x658, x659, x660,
            x662, x663, x664, x665, x666, x667, x668, x669, x670, x671, x672,
            x673, x674, x675, x676, x677, x678, x679, x680, x681, x682, x683,
            x684, x685, x686, x687, x688, x689, x690, x691, x692, x693, x694,
            x695, x696, x697;

        x0 = rl39 + rl9;
        x1 = rl3 + rl45;
        x2 = rl12 + rl36;
        x3 = rl0 + rl24;
        x4 = x2 + x3;
        x5 = rl18 + rl30 + rl42 + rl6;
        x6 = x4 + x5;
        x7 = rl15 + rl21 + rl27 + rl33 + x0 + x1 + x6;
        x8 = rl43 + rl5;
        x9 = rl41 + rl7;
        x10 = rl19 + rl29 + x8 + x9;
        x11 = rl23 + rl47;
        x12 = rl31 + x11;
        x13 = rl11 + rl35;
        x14 = rl13 + rl37 + x13;
        x15 = rl1 + rl25;
        x16 = x14 + x15;
        x17 = rl17 + x10 + x12 + x16;
        x18 = rl22 + rl46;
        x19 = rl26 + rl34;
        x20 = rl10 + rl14 + rl2 + rl38 + x18 + x19;
        x21 = rl20 + rl44;
        x22 = rl32 + x21;
        x23 = rl40 + rl8;
        x24 = rl16 + x23;
        x25 = rl28 + rl4;
        x26 = x24 + x25;
        x27 = x20 + x22 + x26;
        x28 = x17 + x27;
        x29 = im21 + im27;
        x30 = im39 + im45;
        x31 = im15 + im33;
        x32 = im9 + x31;
        x33 = im18 + im30;
        x34 = im42 + im6;
        x35 = im12 + im36;
        x36 = im0 + im24;
        x37 = x35 + x36;
        x38 = x33 + x34 + x37;
        x39 = im3 + x29 + x30 + x32 + x38;
        x40 = im19 + im29;
        x41 = im17 + im31;
        x42 = im41 + im7 + x41;
        x43 = im43 + im5;
        x44 = im11 + im35;
        x45 = im1 + im23 + im25 + im47;
        x46 = im13 + im37;
        x47 = x44 + x45 + x46;
        x48 = x40 + x42 + x43 + x47;
        x49 = im32 + im8;
        x50 = im2 + im46;
        x51 = im10 + im38;
        x52 = im14 + x51;
        x53 = im22 + im26 + im34 + x50 + x52;
        x54 = im20 + im4;
        x55 = im16 + im28 + im40;
        x56 = im44 + x54 + x55;
        x57 = x49 + x53 + x56;
        x58 = x48 + x57;
        x61 = im1 - im25;
        x62 = rl37 - rl13 + x61;
        x64 = im23 - im47;
        x66 = rl11 - rl35;
        x67 = x64 + x66;
        x68 = x62 + x67;
        x72 = im11 - im35;
        x73 = rl47 - rl23 + x72;
        x75 = im13 - im37;
        x77 = rl1 - rl25;
        x78 = x75 + x77;
        x79 = x73 + x78;
        x83 = -rl19 - rl29;
        x85 = im17 - im31;
        x87 = im7 - im41;
        x88 = x85 + x87;
        x89 = x8 + x83 + x88;
        x92 = im19 - im29;
        x94 = im5 - im43;
        x95 = x92 + x94;
        x98 = -rl17 - rl31;
        x99 = x9 + x98;
        x100 = x95 + x99;
        x102 = x100 * x101 + x68 * x69 + x79 * x80 + x89 * x90;
        x104 = im16 - im40;
        x106 = rl4 - rl28;
        x107 = x104 + x106;
        x109 = im8 - im32;
        x111 = rl44 - rl20;
        x113 = x112 * (x107 + x109 + x111);
        x115 = im6 - im42;
        x117 = im18 - im30;
        x118 = x115 + x117;
        x120 = rl6 - rl18;
        x122 = rl42 - rl30;
        x123 = x120 + x122;
        x124 = x118 + x123;
        x126 = x124 * x125;
        x127 = x113 + x126;
        x129 = rl2 - rl22;
        x131 = rl46 - rl26;
        x133 = im10 - im38;
        x135 = im14 - im34;
        x136 = x133 + x135;
        x137 = x129 + x131 + x136;
        x140 = im2 - im46;
        x142 = im22 - im26;
        x143 = x140 + x142;
        x145 = rl10 - rl14;
        x147 = rl38 - rl34;
        x148 = x145 + x147;
        x149 = x143 + x148;
        x151 = x137 * x138 + x149 * x150;
        x152 = x127 + x151;
        x155 = -im28 - im44;
        x156 = x155 + x54;
        x159 = -rl16 - rl32;
        x160 = x159 + x23;
        x161 = x156 + x160;
        x163 = x161 * x162;
        x165 = im12 - im36;
        x167 = rl0 - rl24;
        x168 = x165 + x167;
        x169 = x163 + x168;
        x171 = im15 - im33;
        x173 = im9 - im39;
        x174 = x171 + x173;
        x176 = rl45 - rl27;
        x178 = rl3 - rl21;
        x179 = x176 + x178;
        x180 = x174 + x179;
        x183 = im3 - im45;
        x185 = im21 - im27;
        x186 = x183 + x185;
        x188 = rl9 - rl15;
        x190 = rl39 - rl33;
        x191 = x188 + x190;
        x192 = x186 + x191;
        x194 = x180 * x181 + x192 * x193;
        x195 = x169 + x194;
        x197 = rl26 - rl2;
        x199 = -im14 - im34;
        x200 = x199 + x51;
        x201 = rl46 - rl22 + x197 + x200;
        x203 = -im22 - im26;
        x204 = x203 + x50;
        x206 = rl34 - rl10;
        x207 = rl38 - rl14 + x206;
        x208 = x204 + x207;
        x210 = -im42 - im6;
        x211 = x210 + x33;
        x213 = rl18 - rl42;
        x214 = rl6 - rl30 + x213;
        x215 = x125 * (x211 + x214);
        x217 = rl16 - rl40;
        x219 = im28 - im4;
        x220 = x217 + x219;
        x221 = im20 - im44;
        x222 = x112 * (rl8 - rl32 + x220 + x221);
        x223 = x215 + x222;
        x225 = rl19 - rl43;
        x227 = -im41 - im7;
        x228 = x227 + x41;
        x229 = rl5 - rl29 + x225 + x228;
        x231 = -im43 - im5;
        x232 = x231 + x40;
        x234 = rl17 - rl41;
        x235 = rl7 - rl31 + x234;
        x236 = x232 + x235;
        x238 = rl35 - rl11;
        x240 = im47 - im23;
        x241 = x238 + x240 + x62;
        x243 = rl25 - rl1;
        x245 = im37 - im13;
        x246 = x243 + x245 + x73;
        x247 = x101 * x229 + x236 * x90 - x241 * x80 - x246 * x69;
        x249 = rl20 - rl44;
        x251 = im32 - im8;
        x252 = x249 + x251;
        x253 = x107 + x252;
        x254 = x162 * x253;
        x256 = im24 - im0;
        x258 = rl12 - rl36;
        x259 = x256 + x258;
        x260 = x254 + x259;
        x262 = rl21 - rl45;
        x264 = -im39 - im9;
        x265 = x264 + x31;
        x266 = rl3 - rl27 + x262 + x265;
        x268 = rl15 - rl39;
        x270 = -im3 - im45;
        x271 = x270 + x29;
        x272 = rl9 - rl33 + x268 + x271;
        x273 = x181 * x272 + x193 * x266;
        x274 = x260 + x273;
        x275 = im34 - im22;
        x276 = im26 - im14;
        x277 = x133 + x140 + x276;
        x278 = x275 + x277;
        x280 = x25 - rl8;
        x281 = x159 - rl40;
        x282 = x21 + x278 + x280 + x281;
        x284 = rl33 - rl3;
        x285 = rl15 - rl45;
        x286 = -rl21 - rl27;
        x287 = x171 - im9;
        x288 = x185 - im3;
        x289 = x287 + x288 + x30;
        x290 = x125 * (x0 + x284 + x285 + x286 + x289);
        x292 = rl14 - rl2;
        x293 = rl38 - rl26 + x292;
        x294 = im16 - im4 + im40;
        x295 = x294 - im28;
        x296 = im20 + im44;
        x297 = -im32 - im8;
        x298 = x295 + x296 + x297;
        x299 = x112 * (rl10 - rl22 + rl34 - rl46 + x293 + x298);
        x300 = x290 + x299;
        x301 = im41 - im7 + x85;
        x302 = im43 - im5;
        x303 = x302 + x92;
        x304 = x301 + x303;
        x306 = -rl1 - rl23 - rl25 - rl47;
        x307 = x14 + x306;
        x308 = x304 + x307;
        x309 = rl19 - rl41;
        x311 = rl29 - rl7;
        x312 = im1 - im23 + im25 - im47;
        x313 = -im13 - im37;
        x314 = x312 + x313 + x44;
        x315 = x309 + x311 + x314 + x8 + x98;
        x316 = x138 * x308 - x150 * x315;
        x318 = -rl0 - rl24;
        x319 = x2 + x318;
        x320 = im42 - im6;
        x321 = x117 + x320;
        x322 = x319 + x321;
        x324 = -rl13 - rl37;
        x325 = x13 + x324;
        x326 = -rl23 - rl47 + x15;
        x328 = -im19 - im29;
        x329 = x231 + x328;
        x330 = x329 + x42;
        x331 = x325 + x326 + x330;
        x332 = x150 * x331;
        x334 = rl17 - rl5;
        x335 = rl43 - rl29;
        x336 = rl41 - rl31;
        x337 = -im11 - im35;
        x338 = x313 + x337;
        x339 = x338 + x45;
        x340 = rl19 - rl7 + x334 + x335 + x336 + x339;
        x341 = x138 * x340;
        x342 = rl32 - rl20 - rl44;
        x343 = x342 - im46;
        x344 = -rl16 - rl40;
        x345 = x25 + x344;
        x347 = im34 - im2;
        x348 = x203 + x347 + x52;
        x349 = x112 * (rl8 + x343 + x345 + x348);
        x350 = rl27 - rl39;
        x352 = im39 - im21;
        x353 = x270 - im27;
        x354 = x32 + x352 + x353;
        x355 = rl33 - rl45 + x178 + x188 + x350 + x354;
        x356 = x125 * x355;
        x357 = x349 + x356;
        x359 = im32 - im20 + im8;
        x360 = x155 + x294 + x359;
        x362 = -rl38 - rl46;
        x363 = x129 + x145 + x19 + x362;
        x364 = x360 + x363;
        x365 = x162 * x364;
        x366 = rl30 - rl42 + x120;
        x368 = -im0 - im24 + x35;
        x369 = x365 + x366 + x368;
        x370 = rl23 - rl47 + x72;
        x371 = x243 + x370 + x75;
        x372 = im29 - im19;
        x373 = im27 - im21 - im3 + im45 + x302 + x372;
        x374 = x191 + x371 + x373 + x99;
        x375 = x181 * x374;
        x376 = rl13 - rl37 + x61;
        x377 = x238 + x376 + x64;
        x378 = rl29 - rl5;
        x379 = x225 + x378 + x88;
        x381 = im33 - im15;
        x382 = im39 - im9 + x381;
        x383 = x179 + x377 + x379 + x382;
        x384 = x193 * x383;
        x385 = rl22 - rl46;
        x386 = x136 + x197 + x385;
        x388 = im30 - im18;
        x389 = im26 - im2;
        x390 = x125 * (im46 - im22 + x123 + x148 + x320 + x386 + x388 + x389);
        x391 = rl24 - rl0;
        x392 = x165 + x391;
        x393 = im44 - im20;
        x394 = x160 + x219 + x390 + x392 + x393;
        x396 = rl33 - rl9;
        x397 = rl39 - rl15 + x271 + x396;
        x398 = x245 + x370 + x77;
        x399 = x328 + x43;
        x400 = x235 + x397 + x398 + x399;
        x401 = x193 * x400;
        x402 = x240 + x376 + x66;
        x403 = rl27 - rl3;
        x404 = rl45 - rl21 + x265 + x403;
        x405 = rl43 - rl19 + x228 + x378;
        x406 = x402 + x404 + x405;
        x407 = x181 * x406;
        x409 = rl2 - rl26 + x200 + x385;
        x410 = -im18 - im30;
        x411 = x34 + x410;
        x412 = x125 * (im22 - im46 + x207 + x214 + x389 + x409 + x411);
        x414 = rl36 - rl12;
        x415 = x256 + x414;
        x416 = x109 + x249;
        x417 = im40 - im16 + x106 + x412 + x415 + x416;
        x418 = -rl11 - rl35;
        x419 = rl31 + x418;
        x420 = x306 + x324;
        x421 = rl17 + x49;
        x422 = im46 - im14 + x133 + x142 + x347;
        x423 = -im16 - im4 - im40;
        x424 = x296 - im28 + x423;
        x425 = x422 + x424;
        x426 = x112 * (x10 + x419 + x420 + x421 + x425);
        x428 = -rl28 - rl4;
        x429 = x428 - rl8;
        x430 = -rl20 - rl44;
        x431 = x281 + x429 + x430;
        x432 = x372 + x94;
        x433 = x312 + x337 + x46;
        x434 = x301 + x432 + x433;
        x435 = x20 + x431 + x434;
        x436 = im27 + x183;
        x437 = x287 + x352 + x436;
        x439 = -rl18 - rl30 - rl42 - rl6 + x4;
        x440 = x162 * x435 + x437 + x439;
        x441 = x22 - rl2;
        x442 = rl8 + x344 + x428;
        x443 = rl22 + x131;
        x444 = rl34 - rl38 + x145 + x443;
        x446 = -im17 - im31 + x227 + x329;
        x447 = x446 + x47;
        x448 = x112 * (x441 + x442 + x444 + x447);
        x449 = -rl1 - rl25;
        x450 = rl7 - rl17;
        x451 = x309 - rl5 + x335 + x450;
        x452 = x12 + x325 + x449 + x451;
        x453 = x297 - im20;
        x454 = x155 + x423 + x453;
        x455 = x454 + x53;
        x456 = x210 + x37 + x410;
        x457 = rl21 - rl39 + x176 + x188 + x284 + x456;
        x458 = x162 * (x452 + x455) + x457;
        x463 = x101 * x62 + x78 * x90;
        x466 = x465 * x67 + x73 * x90;
        x467 = -x100 * x69 + x463 + x466 - x80 * x89;
        x468 = x180 * x193;
        x469 = x181 * x192 - x468;
        x471 = x137 * x150;
        x473 = x149 * x472;
        x474 = x471 - x126 + x473;
        x475 = x465 * x73;
        x476 = x229 * x69;
        x477 = x236 * x80;
        x478 = x67 * x90;
        x479 = x193 * x272;
        x480 = x181 * x266;
        x481 = x479 - x480;
        x482 = x62 * x90;
        x483 = x101 * x78;
        x484 = x482 - x483;
        x485 = x150 * x208 + x201 * x472;
        x486 = im0 - im24;
        x487 = x414 + x486;
        x488 = x487 - x254;
        x489 = x223 + x485 + x488;
        x490 = x24 + x275 + x428;
        x492 = rl31 - rl43;
        x493 = x314 + x334 + x492 + x83 + x9;
        x495 = x125 * x493;
        x496 = rl21 - rl9;
        x497 = -rl15 - rl33;
        x498 = x1 + x350 + x496 + x497;
        x499 = x289 + x498;
        x500 = im31 - im17 + x87;
        x501 = x125 * (x307 + x432 + x499 + x500);
        x502 = -rl12 - rl36;
        x503 = x3 + x502;
        x504 = x321 + x503;
        x505 = -rl10 - rl34;
        x506 = x18 + x505;
        x507 = x11 + x418;
        x508 = rl13 + rl37;
        x509 = x330 + x449 + x507 + x508;
        x510 = rl5 - rl19;
        x511 = rl29 - rl41 + x339 + x450 + x492 + x510;
        x512 = rl15 - rl3 + x176 + x190 + x354 + x496;
        x513 = x125 * x509 + x125 * (x511 + x512);
        x515 = -im12 - im36 + x36;
        x516 = x366 + x515;
        x517 = rl28 - rl4 + x104;
        x518 = x112 * (x416 + x517);
        x519 = rl14 - rl38;
        x520 = x143 + x206 + x519;
        x522 = rl30 - rl6;
        x523 = x125 * (x118 + x213 + x522);
        x525 = x138 * x520 + x150 * x386 - x523;
        x526 = rl32 - rl8;
        x527 = x156 + x217 + x526;
        x528 = x162 * x527;
        x529 = im36 - im12;
        x530 = x167 + x529;
        x531 = x530 - x528;
        x532 = x262 + x403;
        x533 = x174 + x532;
        x534 = x268 + x396;
        x535 = x186 + x534;
        x536 = x181 * x533 + x193 * x535;
        x537 = rl31 - rl7;
        x538 = x234 + x537;
        x539 = x538 + x95;
        x540 = -x101 * x371 + x377 * x90 + x379 * x69 + x536 - x539 * x80;
        x541 = rl42 - rl18 + x522;
        x542 = x125 * (x211 + x541);
        x543 = rl10 - rl34 + x204 + x519;
        x544 = x138 * x409 + x150 * x543 + x542;
        x545 = x112 * (rl40 - rl16 + x219 + x221 + x526);
        x547 = x111 + x251 + x517;
        x548 = x162 * x547;
        x549 = x415 + x548;
        x550 = x549 - x545;
        x551 = x544 + x550;
        x552 = rl41 - rl17 + x537;
        x553 = x232 + x552;
        x554 = x181 * x397 + x193 * x404;
        x555 = x101 * x402 - x398 * x90 - x405 * x80 + x553 * x69 + x554;
        x557 = x49 - im10;
        x558 = x303 + x433 + x500;
        x559 = x112 * (im26 - im22 + im38 + x135 + x140 + x424 + x557 + x558);
        x560 = -rl14 - rl22 - rl26 + x362 + x505;
        x561 = -rl3 - rl39 - rl45 - rl9 + x286 + x497 + x6;
        x562 = x162 * (x17 - rl2 + x431 + x560) + x561;
        x563 = x326 + x508;
        x564 = x419 + x451 + x563;
        x565 = x147 - rl10;
        x566 = rl14 + rl26 - rl46 + x129 + x565;
        x567 = x112 * (x22 + x442 + x564 + x566);
        x568 = -im2 - im38 - im46 + x199 + x203;
        x569 = x568 - im10;
        x570 = -im15 - im21;
        x571 = x264 - im33 + x353 + x38 + x570;
        x572 = x162 * (x454 + x48 + x569) + x571;
        x573 = x382 + x532 + x68 + x89;
        x574 = x373 + x534 + x538 + x79;
        x575 = x181 * x573 + x193 * x574;
        x576 = x124 + x149;
        x577 = x125 * x137;
        x578 = x125 * x576 - x577;
        x579 = x125 * (x201 + x411 + x541);
        x580 = x125 * x208;
        x581 = x229 + x241 + x266;
        x582 = x246 + x272 + x399 + x552;
        x583 = x181 * x582 + x193 * x581;
        x584 = x315 * x472;
        x585 = x150 * x308;
        x586 = x115 + x388;
        x587 = x503 + x586;
        x588 = x162 * x282 + x587;
        x589 = x150 * x340 - x331 * x472;
        x590 = rl18 - rl6 + x122;
        x591 = x515 + x590;
        x592 = x377 * x80;
        x593 = x539 * x90;
        x594 = x371 * x69;
        x595 = x101 * x379;
        x596 = x193 * x533;
        x597 = x181 * x535;
        x598 = x596 - x597;
        x600 = x531 - x518;
        x601 = x138 * x386;
        x602 = x150 * x520;
        x603 = x523 + x601 + x602;
        x604 = x600 + x603;
        x605 = x398 * x80;
        x606 = x181 * x404;
        x607 = x101 * x553;
        x608 = x402 * x69;
        x609 = x405 * x90;
        x610 = x545 + x549;
        x611 = x138 * x543 + x150 * x409 - x542;
        x612 = x610 + x611;
        x613 = im45 + x173 + x288 + x381;
        x614 = x439 + x613;
        x615 = x26 + x434 + x441 + x560;
        x616 = rl27 - rl9 + x178 + x190 + x285 + x456;
        x617 = x69 * x79;
        x618 = x101 * x89;
        x619 = x68 * x80;
        x620 = x100 * x90;
        x621 = x126 - x113;
        x622 = x151 + x621;
        x623 = x391 + x529;
        x624 = x623 - x163;
        x625 = x469 + x624;
        x626 = x241 * x69;
        x627 = x229 * x90;
        x628 = x246 * x80;
        x629 = x101 * x236;
        x630 = x138 * x208 + x150 * x201 - x215;
        x631 = x222 + x260 + x630;
        x632 = rl2 - rl14 + rl26 - rl38 + x298 + x506;
        x633 = x112 * x632;
        x634 = x277 + x342 + x490;
        x635 = x162 * x634;
        x636 = x319 + x586 + x635;
        x637 = x633 + x636;
        x638 = x125 * x499;
        x639 = x15 + x304 + x324 + x507;
        x640 = x138 * x493 - x150 * x639 + x638;
        x641 = rl16 - rl32 + rl40;
        x642 = x21 - im46 + x348 + x429 + x641;
        x643 = x125 * x512;
        x644 = -x138 * x509 + x150 * x511 + x643;
        x645 = x292 + x360 + x443 + x565;
        x646 = x162 * x645;
        x647 = x368 + x590 + x646;
        x648 = x181 * x383 + x193 * x374;
        x649 = x390 + x527 + x530;
        x650 = x181 * x400 + x193 * x406;
        x651 = x258 + x486;
        x652 = x412 + x547 + x651;
        x653 = x112 * (im4 - im44 + x422 + x453 + x55 + x558);
        x654 = -x162 * x28 + x7;
        x655 = x280 + x430 + x641;
        x656 = x112 * (x444 - rl2 + x564 + x655);
        x657 = -x162 * x58 + x39;
        x658 = x478 - x475;
        x659 = -x100 * x80 + x484 + x658 + x69 * x89;
        x660 = -x471 - x473;
        x662 = x215 - x222 + x485;
        x663 = -x229 * x80 + x236 * x69 + x463;
        x664 = x125 * x308 + x125 * (im33 + x173 + x315 + x436 + x498 + x570);
        x665 = x125 * x340 + x125 * (x331 + x355);
        x666 = x379 * x80;
        x667 = x539 * x69;
        x668 = x101 * x377;
        x669 = x371 * x90;
        x670 = x392 + x528;
        x671 = x670 - x518;
        x672 = x525 + x671;
        x673 = x193 * x397 - x606;
        x674 = x101 * x398 + x402 * x90 + x405 * x69 + x553 * x80 + x673;
        x675 = x181 * x574;
        x676 = x193 * x573;
        x677 = x161 + x623;
        x678 = x578 + x677;
        x679 = x193 * x582;
        x680 = x181 * x581;
        x681 = x580 - x579;
        x682 = x253 + x487 + x681;
        x683 = -x150 * x493 + x472 * x639 + x638;
        x684 = x112 * x642;
        x685 = x647 + x684;
        x686 = x150 * x509 - x472 * x511 + x643;
        x687 = x101 * x539 + x371 * x80 + x377 * x69 + x379 * x90 + x536;
        x688 = x101 * x405 - x398 * x69 - x402 * x80 + x553 * x90 + x554;
        x689 = x651 - x548;
        x690 = -rl41 - rl43 - rl5 - rl7 + x83 + x98;
        x691 = x194 + x624;
        x692 = x162 * x615;
        x693 = x112 * (x11 + x16 + x425 + x49 + x690);
        x694 = x318 + x5 + x502;
        x695 = x112 * (x447 + x566 + x655);
        x696 = x311 - rl43 + x336 + x418 + x510 + x563;
        x697 = x162 * (rl17 + x455 + x696) + x616;

        out_r[out_strides[0]] = x28 + x7;
        out_i[out_strides[0]] = x39 + x58;
        out_r[out_strides[1]] = x102 + x152 + x195;
        out_i[out_strides[1]] = x138 * x208 + x150 * x201 - x223 - x247 - x274;
        out_r[out_strides[2]] = x162 * x282 - x300 - x316 - x322;
        out_i[out_strides[2]] = x341 - x332 - x357 - x369;
        out_r[out_strides[3]] = x384 - x375 - x394;
        out_i[out_strides[3]] = x407 - x401 - x417;
        out_r[out_strides[4]] = x440 - x426;
        out_i[out_strides[4]] = x448 + x458;
        out_r[out_strides[5]] = x169 - x113 + x467 + x469 + x474;
        out_i[out_strides[5]] = x475 + x476 + x477 - x478 + x481 + x484 + x489;
        out_r[out_strides[6]] = im10 + im2 - im38 + x276 + x343 + x490 + x495 - x501 + x504;
        out_i[out_strides[6]] = x293 + x360 + x506 + x513 + x516;
        out_r[out_strides[7]] = x518 + x525 + x531 + x540;
        out_i[out_strides[7]] = x555 - x551;
        out_r[out_strides[8]] = x559 + x562;
        out_i[out_strides[8]] = x572 - x567;
        out_r[out_strides[9]] = x168 + x220 + x393 + x526 + x575 + x578;
        out_i[out_strides[9]] = x107 + x252 + x487 + x579 - x580 + x583;
        out_r[out_strides[10]] = x300 + x584 - x585 + x588;
        out_i[out_strides[10]] = x357 - x365 + x589 + x591;
        out_r[out_strides[11]] = x592 + x593 - x594 - x595 + x598 + x604;
        out_i[out_strides[11]] = x193 * x397 - x605 - x606 - x607 + x608 + x609 - x612;
        out_r[out_strides[12]] = x614 + x615;
        out_i[out_strides[12]] = x452 + x557 + x56 + x568 + x616;
        out_r[out_strides[13]] = -x617 - x618 + x619 + x620 - x622 - x625;
        out_i[out_strides[13]] = -x481 - x626 - x627 + x628 + x629 - x631;
        out_r[out_strides[14]] = x640 - x637;
        out_i[out_strides[14]] = x112 * x642 - x644 - x647;
        out_r[out_strides[15]] = x648 + x649;
        out_i[out_strides[15]] = x652 - x650;
        out_r[out_strides[16]] = x653 + x654;
        out_i[out_strides[16]] = x657 - x656;
        out_r[out_strides[17]] = x195 + x621 + x659 + x660;
        out_i[out_strides[17]] = -x274 + x465 * x67 - x662 - x663 + x73 * x90;
        out_r[out_strides[18]] = rl8 - x278 + x342 - x345 + x587 + x664;
        out_i[out_strides[18]] = x295 - im44 + x359 + x363 + x591 - x665;
        out_r[out_strides[19]] = -x598 - x666 - x667 + x668 + x669 - x672;
        out_i[out_strides[19]] = x544 - x550 - x674;
        out_r[out_strides[20]] = x426 + x440;
        out_i[out_strides[20]] = x458 - x448;
        out_r[out_strides[21]] = x676 - x675 - x678;
        out_i[out_strides[21]] = x679 - x680 + x682;
        out_r[out_strides[22]] = x112 * x632 - x636 - x683;
        out_i[out_strides[22]] = x686 - x685;
        out_r[out_strides[23]] = x600 - x523 - x601 - x602 + x687;
        out_i[out_strides[23]] = x611 - x545 + x688 + x689;
        out_r[out_strides[24]] = x27 + x418 + x420 + x561 + x690;
        out_i[out_strides[24]] = -im1 - im23 - im25 - im47 + x338 + x446 + x57 + x571;
        out_r[out_strides[25]] = x152 - x102 - x691;
        out_i[out_strides[25]] = x247 - x222 + x273 + x488 + x630;
        out_r[out_strides[26]] = x290 - x299 + x316 + x588;
        out_i[out_strides[26]] = x332 - x341 - x349 + x356 - x369;
        out_r[out_strides[27]] = x375 - x384 - x394;
        out_i[out_strides[27]] = x401 - x407 - x417;
        out_r[out_strides[28]] = -x437 - x692 - x693 - x694;
        out_i[out_strides[28]] = x697 - x695;
        out_r[out_strides[29]] = -x127 - x467 - x625 - x660;
        out_i[out_strides[29]] = -x476 - x477 - x481 - x482 + x483 + x489 + x658;
        out_r[out_strides[30]] = x501 - x495 + x504 + x634;
        out_i[out_strides[30]] = x516 - x513 + x645;
        out_r[out_strides[31]] = x138 * x520 + x150 * x386 - x523 - x540 - x671;
        out_i[out_strides[31]] = -x551 - x555;
        out_r[out_strides[32]] = x654 - x653;
        out_i[out_strides[32]] = x656 + x657;
        out_r[out_strides[33]] = x125 * x576 - x575 - x577 - x677;
        out_i[out_strides[33]] = x253 - x259 - x583 - x681;
        out_r[out_strides[34]] = x299 - x290 - x584 + x585 + x588;
        out_i[out_strides[34]] = x349 - x356 - x369 - x589;
        out_r[out_strides[35]] = -x592 - x593 + x594 + x595 - x598 + x604;
        out_i[out_strides[35]] = x605 + x607 - x608 - x609 - x612 - x673;
        out_r[out_strides[36]] = -x435 - x613 - x694;
        out_i[out_strides[36]] = x421 + x457 + x56 + x569 + x696;
        out_r[out_strides[37]] = x181 * x192 - x468 + x617 + x618 - x619 - x620 - x622 - x624;
        out_i[out_strides[37]] = x481 + x626 + x627 - x628 - x629 - x631;
        out_r[out_strides[38]] = -x637 - x640;
        out_i[out_strides[38]] = x516 + x644 - x646 + x684;
        out_r[out_strides[39]] = x649 - x648;
        out_i[out_strides[39]] = x650 + x652;
        out_r[out_strides[40]] = x562 - x559;
        out_i[out_strides[40]] = x567 + x572;
        out_r[out_strides[41]] = -x113 - x474 - x659 - x691;
        out_i[out_strides[41]] = x273 - x260 - x466 - x662 + x663;
        out_r[out_strides[42]] = -x282 - x322 - x664;
        out_i[out_strides[42]] = x364 + x591 + x665;
        out_r[out_strides[43]] = x598 + x666 + x667 - x668 - x669 - x672;
        out_i[out_strides[43]] = x544 + x545 + x674 + x689;
        out_r[out_strides[44]] = x614 - x692 + x693;
        out_i[out_strides[44]] = x695 + x697;
        out_r[out_strides[45]] = x675 - x676 - x678;
        out_i[out_strides[45]] = x680 - x679 + x682;
        out_r[out_strides[46]] = x504 + x633 - x635 + x683;
        out_i[out_strides[46]] = -x685 - x686;
        out_r[out_strides[47]] = -x518 - x603 - x670 - x687;
        out_i[out_strides[47]] = x138 * x543 + x150 * x409 - x542 - x610 - x688;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft48c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft48c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft48c_fp64;
    }
    else
    {
        return NULL;
    }
}
