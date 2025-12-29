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

/**
 * Enforces Hermitian symmetry on N-dimensional half-complex C2R input.
 * - Sets imaginary part to zero at DC and Nyquist positions
 * - Applies conjugate symmetry: X[k] = conj(X[N-k])
 */
#define make_hc_as_hermitian_symmetric_impl(out, dims, dim_rank, batches,      \
                                            map, dt_t)                         \
    do                                                                         \
    {                                                                          \
        dt_t *out_buf = (dt_t *)out;                                           \
                                                                               \
        /* Allocate arrays */                                                  \
        INTP *dim_sizes = NULL;                                                \
        INTP *hc_strides = NULL;                                               \
        INTP *cur_pos = NULL;                                                  \
        INTP *conj_pos = NULL;                                                 \
        UINT8 *has_nyquist = NULL;                                             \
        ALLOC_UNALIGN_UNINIT(dim_sizes, INTP, dim_rank * sizeof(INTP))         \
        ALLOC_UNALIGN_UNINIT(hc_strides, INTP, dim_rank * sizeof(INTP))        \
        ALLOC_UNALIGN_UNINIT(cur_pos, INTP, dim_rank * sizeof(INTP))           \
        ALLOC_UNALIGN_UNINIT(conj_pos, INTP, dim_rank * sizeof(INTP))          \
        ALLOC_UNALIGN_UNINIT(has_nyquist, UINT8, dim_rank)                     \
                                                                               \
        if (!dim_sizes || !hc_strides || !cur_pos || !conj_pos || !has_nyquist)\
        {                                                                      \
            printf("ERR: alloc fail in make_hc_as_hermitian_symmetric_impl\n");\
            FREE_UNALIGN_ALLOCATED_MEM(dim_sizes)                              \
            FREE_UNALIGN_ALLOCATED_MEM(hc_strides)                             \
            FREE_UNALIGN_ALLOCATED_MEM(cur_pos)                                \
            FREE_UNALIGN_ALLOCATED_MEM(conj_pos)                               \
            FREE_UNALIGN_ALLOCATED_MEM(has_nyquist)                            \
            break;                                                             \
        }                                                                      \
                                                                               \
        /* Setup: compute sizes, strides, and Nyquist flags */                 \
        INTP total_hc_elements = 1;                                            \
        INTP cumulative_stride = 1;                                            \
        INTP hc_size = 0;                                                      \
        for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)                 \
        {                                                                      \
            dim_sizes[dim_idx] = dims[dim_idx].n;                              \
            hc_size = (dim_idx == 0) ? (dims[dim_idx].n / 2 + 1)               \
                                     : dims[dim_idx].n;                        \
            hc_strides[dim_idx] = cumulative_stride;                           \
            cumulative_stride *= hc_size;                                      \
            total_hc_elements *= hc_size;                                      \
            has_nyquist[dim_idx] = (dims[dim_idx].n % 2 == 0);                 \
        }                                                                      \
                                                                               \
        /* Process each batch */                                               \
        for (INTP batch_idx = 0; batch_idx < batches; batch_idx++)             \
        {                                                                      \
            INTP batch_offset = batch_idx * total_hc_elements;                 \
                                                                               \
            /*                                                                 \
             * PART 1: Set imag=0 at real positions (DC/Nyquist combos)        \
             * Bitmask enumerates all 2^dim_rank combinations:                 \
             *   bit=0 -> DC (index 0), bit=1 -> Nyquist (index N/2)           \
             */                                                                \
            INTP n_masks = 1 << dim_rank;                                      \
            for (INTP mask = 0; mask < n_masks; mask++)                        \
            {                                                                  \
                UINT8 is_valid = 1;                                            \
                INTP linear_idx = 0;                                           \
                for (INT32 dim_idx = 0; dim_idx < dim_rank; dim_idx++)         \
                {                                                              \
                    UINT8 use_nyq = (mask >> dim_idx) & 1;                     \
                    if (use_nyq && !has_nyquist[dim_idx])                      \
                    {                                                          \
                        is_valid = 0;                                          \
                        break;                                                 \
                    }                                                          \
                    INTP pos = use_nyq ? (dim_sizes[dim_idx] / 2) : 0;         \
                    linear_idx += pos * hc_strides[dim_idx];                   \
                }                                                              \
                if (is_valid)                                                  \
                {                                                              \
                    INTP idx = map[batch_offset + linear_idx];                 \
                    out_buf[idx * DATA_STRIDE + 1] = (dt_t)0.0;                \
                }                                                              \
            }                                                                  \
                                                                               \
            /*                                                                 \
             * PART 2: Enforce conjugate symmetry on pairs                     \
             * For each fixed column (DC/Nyquist in dim0), iterate             \
             * through dims 1..N-1 and apply: conj_pos = conj(cur_pos)         \
             */                                                                \
            INT32 n_fixed_cols = 1 + (has_nyquist[0] ? 1 : 0);                 \
            for (INT32 col_iter = 0; col_iter < n_fixed_cols; col_iter++)      \
            {                                                                  \
                INTP col_idx = (col_iter == 0) ? 0 : (dim_sizes[0] / 2);       \
                cur_pos[0] = col_idx;                                          \
                conj_pos[0] = col_idx;                                         \
                                                                               \
                /* 1D: No conjugate pairs exist */                             \
                if (dim_rank == 1)                                             \
                {                                                              \
                    continue;                                                  \
                }                                                              \
                                                                               \
                /* Reset position for dims 1..N-1 */                           \
                for (INT32 dim_idx = 1; dim_idx < dim_rank; dim_idx++)         \
                {                                                              \
                    cur_pos[dim_idx] = 0;                                      \
                }                                                              \
                                                                               \
                UINT8 is_done = 0;                                             \
                while (!is_done)                                               \
                {                                                              \
                    /* Skip if position is real (all dims at DC/Nyquist) */    \
                    UINT8 is_real_pos = 1;                                     \
                    for (INT32 dim_idx = 1; dim_idx < dim_rank; dim_idx++)     \
                    {                                                          \
                        INTP p = cur_pos[dim_idx];                             \
                        UINT8 at_dc = (p == 0);                                \
                        UINT8 at_nyq = has_nyquist[dim_idx] &&                 \
                                       (p == dim_sizes[dim_idx] / 2);          \
                        if (!at_dc && !at_nyq)                                 \
                        {                                                      \
                            is_real_pos = 0;                                   \
                            break;                                             \
                        }                                                      \
                    }                                                          \
                                                                               \
                    if (!is_real_pos)                                          \
                    {                                                          \
                        /* Compute linear index for current position */        \
                        INTP cur_idx = cur_pos[0] * hc_strides[0];             \
                        for (INT32 dim_idx = 1; dim_idx < dim_rank; dim_idx++) \
                        {                                                      \
                            cur_idx += cur_pos[dim_idx] * hc_strides[dim_idx]; \
                        }                                                      \
                                                                               \
                        /* Compute conjugate position: (N - k) % N */          \
                        INTP conj_idx = conj_pos[0] * hc_strides[0];           \
                        for (INT32 dim_idx = 1; dim_idx < dim_rank; dim_idx++) \
                        {                                                      \
                            conj_pos[dim_idx] = (dim_sizes[dim_idx] -          \
                                                 cur_pos[dim_idx]) %           \
                                                dim_sizes[dim_idx];            \
                            conj_idx += conj_pos[dim_idx] *                    \
                                        hc_strides[dim_idx];                   \
                        }                                                      \
                        /*                                                     \
                         * Process only if cur_idx < conj_idx                  \
                         * (each pair once)                                    \
                         */                                                    \
                        if (cur_idx < conj_idx)                                \
                        {                                                      \
                            INTP src_idx = map[batch_offset + cur_idx];        \
                            INTP dst_idx = map[batch_offset + conj_idx];       \
                            out_buf[dst_idx * DATA_STRIDE] =                   \
                                out_buf[src_idx * DATA_STRIDE];                \
                            out_buf[dst_idx * DATA_STRIDE + 1] =               \
                                -out_buf[src_idx * DATA_STRIDE + 1];           \
                        }                                                      \
                    }                                                          \
                                                                               \
                    /* Odometer-style increment for dims 1..N-1 */             \
                    UINT8 carry_flag = 1;                                      \
                    for (INT32 dim_idx = 1; dim_idx < dim_rank && carry_flag;  \
                         dim_idx++)                                            \
                    {                                                          \
                        cur_pos[dim_idx]++;                                    \
                        if (cur_pos[dim_idx] >= dim_sizes[dim_idx])            \
                        {                                                      \
                            cur_pos[dim_idx] = 0;                              \
                        }                                                      \
                        else                                                   \
                        {                                                      \
                            carry_flag = 0;                                    \
                        }                                                      \
                    }                                                          \
                    if (carry_flag)                                            \
                    {                                                          \
                        is_done = 1;                                           \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* Cleanup: Free allocated memory */                                   \
        FREE_UNALIGN_ALLOCATED_MEM(dim_sizes)                                  \
        FREE_UNALIGN_ALLOCATED_MEM(hc_strides)                                 \
        FREE_UNALIGN_ALLOCATED_MEM(cur_pos)                                    \
        FREE_UNALIGN_ALLOCATED_MEM(conj_pos)                                   \
        FREE_UNALIGN_ALLOCATED_MEM(has_nyquist)                                \
    } while (0)

#define make_hc_as_hermitian_symmetric(out, dims, dim_rank, batches, map,      \
                                       dt_t)                                   \
    do                                                                         \
    {                                                                          \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            make_hc_as_hermitian_symmetric_impl(out, dims, dim_rank, batches,  \
                                                map, FLOAT);                   \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            make_hc_as_hermitian_symmetric_impl(out, dims, dim_rank, batches,  \
                                                map, DOUBLE);                  \
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
