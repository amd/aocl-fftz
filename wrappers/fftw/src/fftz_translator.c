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

/** @file fftz_translator.c
 *
 *  @brief Contains utility functions required for FFTW wrapper.
 *
 *  This file provides utility functions for FFTW wrapper.
 */

#include "fftz_translator.h"

aoclfftz_flags_t init_flag(INT32 sign, VOID *in, VOID *out, fft_type_t ffttype)
{
    direction_t direction = (sign == FFTW_FORWARD) ? FORWARD : BACKWARD;
    INT8 place = (in == out) ? 0 : 1; // 0-INPLACE, 1-OUTOFPLACE;
    INT8 order = 0;                   // INORDER

    aoclfftz_flags_t flags = {0};
    flags.fft_type = ffttype;
    flags.fft_direction = direction;
    flags.storage_order = order;
    flags.fft_placement = place;
    flags.transpose_mode = 0; // Not supported, use default value 0

    return flags;
}

// Initialize problem descriptor for double datatype
fftw_plan get_handle_d(dv_desc *dv_desc, INT32 sign, VOID *in, VOID *out,
                       fft_type_t ffttype)
{
    aoclfftz_prob_desc_d *problem = NULL;
    ALLOC_ALIGN_UNINIT(problem, aoclfftz_prob_desc_d, sizeof(aoclfftz_prob_desc_d));

    INIT_PD(problem, dv_desc, sign, in, out, ffttype);

    problem->in = (DOUBLE *)in;
    problem->out = (DOUBLE *)out;

    fftw_plan handle = (fftw_plan)aoclfftz_setup_d(problem);
    DESTROY_DESC(problem, dv_desc);

    return handle;
}

// Initialize problem descriptor for float datatype
fftwf_plan get_handle_f(dv_desc *dv_desc, INT32 sign, VOID *in, VOID *out,
                        fft_type_t ffttype)
{
    aoclfftz_prob_desc_f *problem = NULL;
    ALLOC_ALIGN_UNINIT(problem, aoclfftz_prob_desc_f, sizeof(aoclfftz_prob_desc_f));

    INIT_PD(problem, dv_desc, sign, in, out, ffttype);

    problem->in = (FLOAT *)in;
    problem->out = (FLOAT *)out;

    fftwf_plan handle = (fftwf_plan)aoclfftz_setup_f(problem);
    DESTROY_DESC(problem, dv_desc);

    return handle;
}

fftw_plan get_handle_d_64_(dv_desc_64_ *dv_desc, INT32 sign, VOID *in,
                           VOID *out, fft_type_t ffttype)
{
    aoclfftz_prob_desc_d_64_ *problem = NULL;
    ALLOC_ALIGN_UNINIT(problem, aoclfftz_prob_desc_d_64_,
                  sizeof(aoclfftz_prob_desc_d_64_));

    INIT_PD(problem, dv_desc, sign, in, out, ffttype);

    problem->in = (DOUBLE *)in;
    problem->out = (DOUBLE *)out;

    fftw_plan handle = (fftw_plan)aoclfftz_setup_d_64_(problem);
    DESTROY_DESC(problem, dv_desc);

    return handle;
}

fftwf_plan get_handle_f_64_(dv_desc_64_ *dv_desc, INT32 sign, VOID *in,
                            VOID *out, fft_type_t ffttype)
{
    aoclfftz_prob_desc_f_64_ *problem = NULL;
    ALLOC_ALIGN_UNINIT(problem, aoclfftz_prob_desc_f_64_,
                  sizeof(aoclfftz_prob_desc_f_64_));

    INIT_PD(problem, dv_desc, sign, in, out, ffttype);

    problem->in = (FLOAT *)in;
    problem->out = (FLOAT *)out;

    fftwf_plan handle = (fftwf_plan)aoclfftz_setup_f_64_(problem);
    DESTROY_DESC(problem, dv_desc);

    return handle;
}

dv_desc *get_dv_desc(INT32 rank, const INT32 *n)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    p_dv_desc->dims[0].in_stride = 1;
    p_dv_desc->dims[0].out_stride = 1;
    for (INT32 i = 1; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride  = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_r2c_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    p_dv_desc->dims[0].in_stride = 1;
    p_dv_desc->dims[0].out_stride = 1;
    if (rank > 1)
    {
        if (is_inplace)
        {
            p_dv_desc->dims[1].in_stride = p_dv_desc->dims[1].out_stride =
                (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                p_dv_desc->dims[0].out_stride;
        }
        else
        {
            p_dv_desc->dims[1].in_stride =
                p_dv_desc->dims[0].n * p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride = (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                                            p_dv_desc->dims[0].out_stride;
        }
    }
    for (INT32 i = 2; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride  = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_c2r_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    p_dv_desc->dims[0].in_stride = 1;
    p_dv_desc->dims[0].out_stride = 1;
    if (rank > 1)
    {
        if (is_inplace)
        {
            p_dv_desc->dims[1].in_stride = p_dv_desc->dims[1].out_stride =
                (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                p_dv_desc->dims[0].out_stride;
        }
        else
        {
            p_dv_desc->dims[1].in_stride = (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                                           p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride =
                p_dv_desc->dims[0].n * p_dv_desc->dims[0].out_stride;
        }
    }
    for (INT32 i = 2; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride  = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_many_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                               const INT32 *inembed, INT32 istride, INT32 idist,
                               const INT32 *onembed, INT32 ostride, INT32 odist)
{
    if (inembed == NULL)
    {
        inembed = n;
    }
    if (onembed == NULL)
    {
        onembed = n;
    }

    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = howmany;
    // FIXME: FFTZ requires in_stride to be non-zero. Until FFTZ is fixed, the
    // wrapper handles the zero and makes it 1
    p_dv_desc->vecs[0].in_stride = idist == 0 ? 1 : idist;
    p_dv_desc->vecs[0].out_stride = odist == 0 ? 1 : odist;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    p_dv_desc->dims[0].in_stride = istride;
    p_dv_desc->dims[0].out_stride = ostride;
    for (INT32 i = 1; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride = p_dv_desc->dims[i - 1].in_stride *
                                       inembed[rank - i];
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                        onembed[rank - i];
    }

    return p_dv_desc;
}

dv_desc *get_many_r2c_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                               const INT32 *inembed, INT32 istride, INT32 idist,
                               const INT32 *onembed, INT32 ostride, INT32 odist, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = howmany;
    p_dv_desc->vecs[0].in_stride = idist;
    p_dv_desc->vecs[0].out_stride = odist;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    /* ---------------- set in_stride for dims ---------------- */
    if (inembed != NULL)
    {
        for (INT32 i = 0; i < rank; i++)
        {
            if (i == 0)
            {
                p_dv_desc->dims[i].in_stride = istride;
            }
            else
            {
                p_dv_desc->dims[i].in_stride =
                    p_dv_desc->dims[i - 1].in_stride * inembed[rank - i];
            }
        }
    }
    else
    {
        // set in_stride and out_stride for rank 0
        p_dv_desc->dims[0].in_stride = istride;
        // set in_stride for rank 1
        if (rank > 1)
        {
            if (is_inplace)
            {
                p_dv_desc->dims[1].in_stride =
                    p_dv_desc->dims[0].in_stride *
                    ((p_dv_desc->dims[0].n / 2) + 1) * 2;
            }
            else // FIXME: move it to final else case?
            {
                p_dv_desc->dims[1].in_stride =
                    p_dv_desc->dims[0].in_stride * p_dv_desc->dims[0].n;
            }
        }
        // set in_stride for rank > 1
        for (INT32 i = 2; i < rank; i++)
        {
            p_dv_desc->dims[i].in_stride = p_dv_desc->dims[i - 1].in_stride *
                                           p_dv_desc->dims[i - 1].n;
        }
    }

    /* ---------------- set out_stride for dims ---------------- */
    if (onembed != NULL)
    {
        for (INT32 i = 0; i < rank; i++)
        {
            if (i == 0)
            {
                p_dv_desc->dims[i].out_stride = ostride;
            }
            else
            {
                p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                                onembed[rank - i];
            }
        }
    }
    else
    {
        // set in_stride and out_stride for rank 0
        p_dv_desc->dims[0].out_stride = ostride;
        // set out_stride for rank 1
        if (rank > 1)
        {
            p_dv_desc->dims[1].out_stride = p_dv_desc->dims[0].out_stride *
                                            ((p_dv_desc->dims[0].n * 2) + 1);
        }
        // set out_stride for rank > 1
        for (INT32 i = 2; i < rank; i++)
        {
            p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                            p_dv_desc->dims[i - 1].n;
        }
    }
    return p_dv_desc;
}

dv_desc *get_many_c2r_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                               const INT32 *inembed, INT32 istride, INT32 idist,
                               const INT32 *onembed, INT32 ostride, INT32 odist, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    p_dv_desc->vecs[0].n = howmany;
    p_dv_desc->vecs[0].in_stride = idist;
    p_dv_desc->vecs[0].out_stride = odist;

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[rank - i - 1];
    }

    /* ---------------- set in_stride for dims ---------------- */
    if (inembed != NULL)
    {
        for (INT32 i = 0; i < rank; i++)
        {
            if (i == 0)
            {
                p_dv_desc->dims[i].in_stride = istride;
            }
            else
            {
                p_dv_desc->dims[i].in_stride =
                    p_dv_desc->dims[i - 1].in_stride * inembed[rank - i];
            }
        }
    }
    else
    {
        // set in_stride and out_stride for rank 0
        p_dv_desc->dims[0].in_stride = istride;
        // set in_stride for rank 1
        if (rank > 1)
        {
            p_dv_desc->dims[1].in_stride =
                ((p_dv_desc->dims[0].in_stride * p_dv_desc->dims[0].n) * 2) + 1;
        }
        // set in_stride for rank > 1
        for (INT32 i = 2; i < rank; i++)
        {
            p_dv_desc->dims[i].in_stride = p_dv_desc->dims[i - 1].in_stride *
                                           p_dv_desc->dims[i - 1].n;
        }
    }

    /* ---------------- set out_stride for dims ---------------- */
    if (onembed != NULL)
    {
        for (INT32 i = 0; i < rank; i++)
        {
            if (i == 0)
            {
                p_dv_desc->dims[i].out_stride = ostride;
            }
            else
            {
                p_dv_desc->dims[i].out_stride =
                    p_dv_desc->dims[i - 1].out_stride * onembed[rank - i];
            }
        }
    }
    else
    {
        // set in_stride and out_stride for rank 0
        p_dv_desc->dims[0].out_stride = ostride;
        // set out_stride for rank 1
        if (rank > 1)
        {
            if (is_inplace)
            {
                p_dv_desc->dims[1].out_stride =
                   (((p_dv_desc->dims[0].out_stride * p_dv_desc->dims[0].n) / 2)
                      + 1) * 2;
            }
            else // FIXME: move it to final else case?
            {
                p_dv_desc->dims[1].out_stride =
                    p_dv_desc->dims[0].out_stride * p_dv_desc->dims[0].n;
            }
        }
        // set out_stride for rank > 1
        for (INT32 i = 2; i < rank; i++)
        {
            p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                            p_dv_desc->dims[i - 1].n;
        }
    }
    return p_dv_desc;
}

dv_desc *get_guru_dv_desc(INT32 rank, const fftw_iodim *dims,
                         INT32 howmany_rank, const fftw_iodim *howmany_dims)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));

    p_dv_desc->vec_rank = howmany_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * howmany_rank);
    for (INT32 i = 0; i < howmany_rank; i++)
    {
        p_dv_desc->vecs[i].n = howmany_dims[howmany_rank - i - 1].n;
        p_dv_desc->vecs[i].in_stride = howmany_dims[howmany_rank - i - 1].in_stride;
        p_dv_desc->vecs[i].out_stride = howmany_dims[howmany_rank - i - 1].out_stride;
    }

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                  sizeof(aoclfftz_dim_t) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = dims[rank - i - 1].n;
        p_dv_desc->dims[i].in_stride = dims[rank - i - 1].in_stride;
        p_dv_desc->dims[i].out_stride = dims[rank - i - 1].out_stride;
    }

    return p_dv_desc;
}

dv_desc_64_ *get_guru_64_dv_desc(INT32 rank, const fftw_iodim64 *dims,
                                INT32 howmany_rank,
                                const fftw_iodim64 *howmany_dims)
{
    dv_desc_64_ *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc_64_, sizeof(dv_desc_64_));

    p_dv_desc->vec_rank = howmany_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t_64_,
                  sizeof(aoclfftz_dim_t_64_) * howmany_rank);
    for (INT32 i = 0; i < howmany_rank; i++)
    {
        p_dv_desc->vecs[i].n = howmany_dims[howmany_rank - i - 1].n;
        p_dv_desc->vecs[i].in_stride = howmany_dims[howmany_rank - i - 1].in_stride;
        p_dv_desc->vecs[i].out_stride = howmany_dims[howmany_rank - i - 1].out_stride;
    }

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t_64_,
                  sizeof(aoclfftz_dim_t_64_) * rank);
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = dims[rank - i - 1].n;
        p_dv_desc->dims[i].in_stride = dims[rank - i - 1].in_stride;
        p_dv_desc->dims[i].out_stride = dims[rank - i - 1].out_stride;
    }

    return p_dv_desc;
}
