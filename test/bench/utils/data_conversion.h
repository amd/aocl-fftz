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
 */

// Expands half-complex data to complex data
#define convert_half_complex_to_complex_impl(out, in, n, batches, map, dt_t)   \
    do                                                                         \
    {                                                                          \
        dt_t *in_t = (dt_t *)in;                                               \
        dt_t *out_t = (dt_t *)out;                                             \
        INTP in_idx = 0;                                                       \
        INTP out_idx = 0;                                                      \
        INTP mirror_idx = 0;                                                   \
        for (INTP b = 0; b < batches; b++)                                     \
        {                                                                      \
            INTP idx = map[in_idx] * DATA_STRIDE;                              \
            /* DC point */                                                     \
            out_t[out_idx] = in_t[idx];                                        \
            out_t[out_idx + 1] = 0.0;                                          \
            in_idx++;                                                          \
            out_idx += DATA_STRIDE;                                            \
            /* half complex points */                                          \
            for (INTP i = 1; i < (n / 2 + (n & 0x1)); i++)                     \
            {                                                                  \
                INTP idx = map[in_idx] * DATA_STRIDE;                          \
                out_t[out_idx] = in_t[idx];                                    \
                out_t[out_idx + 1] = in_t[idx + 1];                            \
                in_idx++;                                                      \
                out_idx += DATA_STRIDE;                                        \
            }                                                                  \
            mirror_idx = in_idx - 1;                                           \
            /* Nyquist point */                                                \
            if (n % 2 == 0)                                                    \
            {                                                                  \
                INTP idx = map[in_idx] * 2;                                    \
                out_t[out_idx] = in_t[idx];                                    \
                out_t[out_idx + 1] = 0.0;                                      \
                in_idx++;                                                      \
                out_idx += DATA_STRIDE;                                        \
            }                                                                  \
            /* mirror complex points */                                        \
            for (INTP i = 1; i < (n / 2 + (n & 0x1)); i++)                     \
            {                                                                  \
                INTP idx = map[mirror_idx] * 2;                                \
                out_t[out_idx] = in_t[idx];                                    \
                out_t[out_idx + 1] = -in_t[idx + 1];                           \
                mirror_idx--;                                                  \
                out_idx += DATA_STRIDE;                                        \
            }                                                                  \
        }                                                                      \
    } while (0)

#define convert_half_complex_to_complex(out, in, n, batches, map, dt_t)        \
    do                                                                         \
    {                                                                          \
        if (dt_t == FLOAT_P)                                                   \
        {                                                                      \
            convert_half_complex_to_complex_impl(out, in, n, batches, map,     \
                                                 FLOAT);                       \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            convert_half_complex_to_complex_impl(out, in, n, batches, map,     \
                                                 DOUBLE);                      \
        }                                                                      \
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
