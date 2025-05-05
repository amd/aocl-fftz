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

/** @file data_conversion.h
 *
 *  @brief Convert half complex to full complex values and vice-versa.
 *
 *  This file contains the utility functions that expand half complex outputs
 *  from R2C fft to full complex and vice-versa.
 *
 *  @author D. Vijay Krishna
 *  @author Jeevanantham N
 */

// Increment N-dimensional index with carry propagation
#define INCREMENT_NDIM_INDEX(indices_arr, dim_rank, dim_sizes, dim0_limit,     \
                             done_flag)                                        \
    do                                                                         \
    {                                                                          \
        UINT8 carry = 1;                                                       \
        for (INT32 dim_idx = dim_rank - 1; dim_idx >= 0 && carry; dim_idx--)   \
        {                                                                      \
            indices_arr[dim_idx]++;                                            \
            if (dim_idx == 0)                                                  \
            {                                                                  \
                if (indices_arr[dim_idx] >= dim0_limit)                        \
                {                                                              \
                    done_flag = 1;                                             \
                    carry = 0;                                                 \
                }                                                              \
                else                                                           \
                {                                                              \
                    carry = 0;                                                 \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                if (indices_arr[dim_idx] >= dim_sizes[dim_idx])                \
                {                                                              \
                    indices_arr[dim_idx] = 0;                                  \
                    carry = 1;                                                 \
                }                                                              \
                else                                                           \
                {                                                              \
                    carry = 0;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// Universal N-dimensional half-complex to full-complex conversion
// Works for any number of dimensions: 1D, 2D, 3D, 4D, 5D, etc.
//
#define convert_half_complex_to_complex_impl(out, in, dims, dim_rank,          \
                                             batches, map, dt_t)               \
    do                                                                         \
    {                                                                          \
        dt_t *in_t = (dt_t *)in;                                               \
        dt_t *out_t = (dt_t *)out;                                             \
        /* Extract dimension information */                                    \
        INTP dim0 = dims[0].n;       /* Last dimension (memory) */             \
        INTP dim0_hc = dim0 / 2 + 1; /* Half-complex size */                   \
                                                                               \
        /* Allocate arrays for dimension sizes and strides */                  \
        INTP *dim_sizes = NULL;                                                \
        INTP *strides_full = NULL;                                             \
        INTP *strides_hc = NULL;                                               \
        INTP *copy_indices = NULL;                                             \
        ALLOC_UNALIGN_UNINIT(dim_sizes, INTP, dim_rank * sizeof(INTP))         \
        ALLOC_UNALIGN_UNINIT(strides_full, INTP, dim_rank * sizeof(INTP))      \
        ALLOC_UNALIGN_UNINIT(strides_hc, INTP, dim_rank * sizeof(INTP))        \
        ALLOC_UNALIGN_UNINIT(copy_indices, INTP, dim_rank * sizeof(INTP))      \
        if (!dim_sizes || !strides_full || !strides_hc || !copy_indices)       \
        {                                                                      \
            printf("ERROR: Failed to allocate memory for dimension arrays\n"); \
            FREE_UNALIGN_ALLOCATED_MEM(dim_sizes)                              \
            FREE_UNALIGN_ALLOCATED_MEM(strides_full)                           \
            FREE_UNALIGN_ALLOCATED_MEM(strides_hc)                             \
            FREE_UNALIGN_ALLOCATED_MEM(copy_indices)                           \
            break;                                                             \
        }                                                                      \
                                                                               \
        /* Calculate strides and total sizes for full and half-complex */      \
        INTP stride_full = 1;                                                  \
        INTP stride_hc = 1;                                                    \
        INTP n_full = 1;                                                       \
        INTP n_hc = 1;                                                         \
                                                                               \
        /* Build dimension sizes and strides (innermost to outermost) */       \
        for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)                 \
        {                                                                      \
            INTP dim_size = (dim_idx == 0) ? dim0 : dims[dim_idx].n;           \
            INTP dim_size_hc = (dim_idx == 0) ? dim0_hc : dims[dim_idx].n;     \
                                                                               \
            dim_sizes[dim_idx] = dim_size;                                     \
            strides_full[dim_idx] = stride_full;                               \
            strides_hc[dim_idx] = stride_hc;                                   \
                                                                               \
            stride_full *= dim_size;                                           \
            stride_hc *= dim_size_hc;                                          \
            n_full *= dim_size;                                                \
            n_hc *= dim_size_hc;                                               \
        }                                                                      \
                                                                               \
        /* Process each batch */                                               \
        for (INTP batch_idx = 0; batch_idx < batches; batch_idx++)             \
        {                                                                      \
            INTP in_base = batch_idx * n_hc;                                   \
            INTP out_base = batch_idx * n_full;                                \
                                                                               \
            /* Step 1: Copy stored half-complex data to output */              \
            for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)             \
            {                                                                  \
                copy_indices[dim_idx] = 0;                                     \
            }                                                                  \
                                                                               \
            UINT8 done_copying = 0;                                            \
            while (!done_copying)                                              \
            {                                                                  \
                /* Compute HC linear index */                                  \
                INTP hc_idx = 0;                                               \
                for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)         \
                {                                                              \
                    hc_idx += copy_indices[dim_idx] * strides_hc[dim_idx];     \
                }                                                              \
                                                                               \
                /* Compute Full linear index (same multi-indices) */           \
                INTP full_idx = 0;                                             \
                for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)         \
                {                                                              \
                    full_idx +=                                                \
                        copy_indices[dim_idx] * strides_full[dim_idx];         \
                }                                                              \
                                                                               \
                /* Copy real and imaginary parts */                            \
                INTP src_idx = map ? map[in_base + hc_idx] * DATA_STRIDE       \
                                   : (in_base + hc_idx) * DATA_STRIDE;         \
                INTP dst_idx = (out_base + full_idx) * DATA_STRIDE;            \
                out_t[dst_idx] = in_t[src_idx];                                \
                out_t[dst_idx + 1] = in_t[src_idx + 1];                        \
                                                                               \
                /* Increment multi-dimensional index */                        \
                INCREMENT_NDIM_INDEX(copy_indices, dim_rank, dim_sizes,        \
                                     dim0_hc, done_copying);                   \
            }                                                                  \
                                                                               \
            /* Step 2: Fill missing region via Hermitian symmetry */           \
            if (dim0_hc < dim0)                                                \
            {                                                                  \
                INTP *indices = NULL;                                          \
                INTP *mirror_indices = NULL;                                   \
                ALLOC_UNALIGN_UNINIT(indices, INTP, dim_rank * sizeof(INTP))   \
                ALLOC_UNALIGN_UNINIT(mirror_indices, INTP,                     \
                                     dim_rank * sizeof(INTP))                  \
                if (!indices || !mirror_indices)                               \
                {                                                              \
                    printf("ERROR: Failed to allocate memory for indices and " \
                            "mirror_indices\n");                               \
                    FREE_UNALIGN_ALLOCATED_MEM(indices)                        \
                    FREE_UNALIGN_ALLOCATED_MEM(mirror_indices)                 \
                    FREE_UNALIGN_ALLOCATED_MEM(dim_sizes)                      \
                    FREE_UNALIGN_ALLOCATED_MEM(strides_full)                   \
                    FREE_UNALIGN_ALLOCATED_MEM(strides_hc)                     \
                    FREE_UNALIGN_ALLOCATED_MEM(copy_indices)                   \
                    break;                                                     \
                }                                                              \
                                                                               \
                /* Start at first missing index (dim0_hc onwards in dim0) */   \
                for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)         \
                {                                                              \
                    indices[dim_idx] = (dim_idx == 0) ? dim0_hc : 0;           \
                }                                                              \
                                                                               \
                UINT8 done = 0;                                                \
                while (!done)                                                  \
                {                                                              \
                    /* Compute missing position linear index */                \
                    INTP miss_idx = 0;                                         \
                    for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)     \
                    {                                                          \
                        miss_idx +=                                            \
                            indices[dim_idx] * strides_full[dim_idx];          \
                    }                                                          \
                                                                               \
                    /* Hermitian mirror: F[i] = conj(F[-i mod N]) */           \
                    for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)     \
                    {                                                          \
                        mirror_indices[dim_idx] =                              \
                            (dim_sizes[dim_idx] - indices[dim_idx]) %          \
                            dim_sizes[dim_idx];                                \
                    }                                                          \
                                                                               \
                    /* Compute mirror linear index */                          \
                    INTP mirr_idx = 0;                                         \
                    for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)     \
                    {                                                          \
                        mirr_idx +=                                            \
                            mirror_indices[dim_idx] * strides_full[dim_idx];   \
                    }                                                          \
                                                                               \
                    /* Set conjugate: real stays same, imaginary negated */    \
                    INTP dst_idx = (out_base + miss_idx) * DATA_STRIDE;        \
                    INTP src_idx = (out_base + mirr_idx) * DATA_STRIDE;        \
                    out_t[dst_idx] = out_t[src_idx];                           \
                    out_t[dst_idx + 1] = -out_t[src_idx + 1];                  \
                                                                               \
                    /* Increment multi-dimensional index */                    \
                    INCREMENT_NDIM_INDEX(indices, dim_rank, dim_sizes, dim0,   \
                                         done);                                \
                }                                                              \
                FREE_UNALIGN_ALLOCATED_MEM(indices)                            \
                FREE_UNALIGN_ALLOCATED_MEM(mirror_indices)                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* Cleanup */                                                          \
        FREE_UNALIGN_ALLOCATED_MEM(dim_sizes)                                  \
        FREE_UNALIGN_ALLOCATED_MEM(strides_full)                               \
        FREE_UNALIGN_ALLOCATED_MEM(strides_hc)                                 \
        FREE_UNALIGN_ALLOCATED_MEM(copy_indices)                               \
    } while (0)

// Converts complex data to half-complex data in a same buffer
#define convert_complex_to_half_complex_impl(buffer, n, batches, map, dt_t)    \
    do                                                                         \
    {                                                                          \
        dt_t *buffer_t = (dt_t *)buffer;                                       \
        for (INTP b = 0; b < batches; b++)                                     \
        {                                                                      \
            buffer_t[map[b * n] * DATA_STRIDE + 1] = 0.0;                      \
            for (INTP i = 1, j = n - 1; i <= (n - 1) / 2; i++, j--)            \
            {                                                                  \
                INTP src_idx = map[b * n + i] * DATA_STRIDE;                   \
                INTP dst_idx = map[b * n + j] * DATA_STRIDE;                   \
                buffer_t[dst_idx] = buffer_t[src_idx];                         \
                buffer_t[dst_idx + 1] = -buffer_t[src_idx + 1];                \
            }                                                                  \
            if (n % 2 == 0)                                                    \
            {                                                                  \
                buffer_t[map[b * n + (n / 2)] * DATA_STRIDE + 1] = 0.0;        \
            }                                                                  \
        }                                                                      \
    } while (0)

#define convert_complex_to_half_complex(buffer, n, batches, map, dt_t)         \
    do                                                                         \
    {                                                                          \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            convert_complex_to_half_complex_impl(buffer, n, batches, map,      \
                                                 FLOAT);                       \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            convert_complex_to_half_complex_impl(buffer, n, batches, map,      \
                                                 DOUBLE);                      \
        }                                                                      \
    } while (0)

// Expands real to complex interleaved format
#define convert_real_to_complex_impl(out, in, n, batches, map, dt_t)           \
    do                                                                         \
    {                                                                          \
        dt_t *in_t = (dt_t *)in;                                               \
        dt_t *out_t = (dt_t *)out;                                             \
        for (INTP b = 0; b < batches; b++)                                     \
        {                                                                      \
            for (INTP i = 0; i < n; i++)                                       \
            {                                                                  \
                INTP idx = b * n + i;                                          \
                out_t[idx * DATA_STRIDE] = in_t[map[idx]];                     \
                out_t[idx * DATA_STRIDE + 1] = 0.0;                            \
            }                                                                  \
        }                                                                      \
    } while (0)

#define convert_real_to_complex(out, in, n, batches, map, dt_t)                \
    do                                                                         \
    {                                                                          \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            convert_real_to_complex_impl(out, in, n, batches, map, FLOAT);     \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            convert_real_to_complex_impl(out, in, n, batches, map, DOUBLE);    \
        }                                                                      \
    } while (0)

#define set_zero_for_dc_and_nyquist_nd(out, n, dim0, batches, map, dt_t)       \
    do                                                                         \
    {                                                                          \
        INTP num_batches = batches;                                            \
        INTP outer_dim = n / dim0;                                             \
        INTP dim0_hc = dim0 / 2 + 1;                                           \
        INTP n_hc = n / 2 + 1;                                                 \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            FLOAT *out_f = (FLOAT *)out;                                       \
            for (INTP b = 0; b < num_batches; b++)                             \
            {                                                                  \
                for (INTP i = 0; i < outer_dim; i++)                           \
                {                                                              \
                    out_f[map[b * n_hc + i * dim0_hc] * DATA_STRIDE + 1] =     \
                        0.0f;                                                  \
                    out_f[map[b * n_hc + (i + 1) * dim0_hc - 1] *              \
                              DATA_STRIDE +                                    \
                          1] = 0.0f;                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            DOUBLE *out_d = (DOUBLE *)out;                                     \
            for (INTP b = 0; b < num_batches; b++)                             \
            {                                                                  \
                for (INTP i = 0; i < outer_dim; i++)                           \
                {                                                              \
                    out_d[map[b * n_hc + i * dim0_hc] * DATA_STRIDE + 1] =     \
                        0.0;                                                   \
                    out_d[map[b * n_hc + (i + 1) * dim0_hc - 1] *              \
                              DATA_STRIDE +                                    \
                          1] = 0.0;                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// Macro wrapper that dispatches to the N-dimensional conversion implementation
// based on the data type

#define convert_half_complex_to_complex(out, in, dims, dim_rank, batches,      \
                                        map, dt_t)                             \
    do                                                                         \
    {                                                                          \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            convert_half_complex_to_complex_impl(out, in, dims, dim_rank,      \
                                                 batches, map, FLOAT);         \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            convert_half_complex_to_complex_impl(out, in, dims, dim_rank,      \
                                                 batches, map, DOUBLE);        \
        }                                                                      \
    } while (0)
