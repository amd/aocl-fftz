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

/** @file rfft4c.c
 *
 *  @brief Radix-4 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-4 real-to-halfcomplex fused of two
 *  different implementations (differs in DFT weight matrix) using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 0, 0, 0, 0},
                                                     {0, 0, 0, 0, 0, 0}};

ops_cycles_t get_ops_cnt_r2hcf_rfft4c(INT32 precision)
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

static VOID r2hcf_rfft4c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
    /* TO BE IMPLEMENTED */
}

static VOID r2hcf_rfft4c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
    /* TO BE IMPLEMENTED */
}

static VOID r2hcf_rfft4c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
    /* TO BE IMPLEMENTED */
}

static VOID r2hcf_rfft4c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
    /* TO BE IMPLEMENTED */
}

kfft_ register_kernel_r2hcf_rfft4c(INT32 precision, INT32 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft4c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft4c_fp64_fwd;
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft4c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft4c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
