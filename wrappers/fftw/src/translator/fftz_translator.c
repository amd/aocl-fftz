// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fftz_translator.c
 *
 *  @brief Contains utility functions required for FFTW wrapper.
 *
 *  This file provides utility functions for FFTW wrapper.
 */

#include "src/translator/fftz_translator.h"

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
    if (problem == NULL)
    {
        return NULL;
    }

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
    if (problem == NULL)
    {
        return NULL;
    }
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
    if (problem == NULL)
    {
        return NULL;
    }

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
    if (problem == NULL)
    {
        return NULL;
    }
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
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    p_dv_desc->dim_rank = (rank > 0) ? rank : 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * p_dv_desc->dim_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_r2c_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    INT32 effective_rank = (rank > 0) ? rank : 1;
    p_dv_desc->dim_rank = effective_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * effective_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
            p_dv_desc->dims[1].in_stride =
                (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride =
                (p_dv_desc->dims[0].n / 2 + 1) *
                p_dv_desc->dims[0].out_stride;
        }
        else
        {
            p_dv_desc->dims[1].in_stride =
                p_dv_desc->dims[0].n * p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride =
                (p_dv_desc->dims[0].n / 2 + 1) *
                p_dv_desc->dims[0].out_stride;
        }
    }
    for (INT32 i = 2; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride  = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_c2r_dv_desc(INT32 rank, const INT32 *n, INT32 is_inplace)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    // get_*_dv_desc is used for single batch multi dimensional problems,
    // so setting vec_ranks and vecs values as 1.
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = 1;
    p_dv_desc->vecs[0].in_stride = 1;
    p_dv_desc->vecs[0].out_stride = 1;

    INT32 effective_rank = (rank > 0) ? rank : 1;
    p_dv_desc->dim_rank = effective_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * effective_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
            p_dv_desc->dims[1].in_stride =
                (p_dv_desc->dims[0].n / 2 + 1) *
                p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride =
                (p_dv_desc->dims[0].n / 2 + 1) * 2 *
                p_dv_desc->dims[0].out_stride;
        }
        else
        {
            p_dv_desc->dims[1].in_stride =
                (p_dv_desc->dims[0].n / 2 + 1) *
                p_dv_desc->dims[0].in_stride;
            p_dv_desc->dims[1].out_stride =
                p_dv_desc->dims[0].n * p_dv_desc->dims[0].out_stride;
        }
    }
    for (INT32 i = 2; i < rank; i++)
    {
        p_dv_desc->dims[i].in_stride  = p_dv_desc->dims[i - 1].in_stride *
                                        p_dv_desc->dims[i - 1].n;
        p_dv_desc->dims[i].out_stride = p_dv_desc->dims[i - 1].out_stride *
                                        p_dv_desc->dims[i - 1].n;
    }

    return p_dv_desc;
}

dv_desc *get_many_dv_desc(INT32 rank, const INT32 *n, INT32 howmany,
                               const INT32 *inembed, INT32 istride, INT32 idist,
                               const INT32 *onembed, INT32 ostride, INT32 odist)
{
    inembed = (inembed == NULL) ? n : inembed;
    onembed = (onembed == NULL) ? n : onembed;

    istride = (istride == 0) ? 1 : istride;
    ostride = (ostride == 0) ? 1 : ostride;

    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = howmany;
    p_dv_desc->vecs[0].in_stride = idist == 0 ? 1 : idist;
    p_dv_desc->vecs[0].out_stride = odist == 0 ? 1 : odist;

    INT32 effective_rank = (rank > 0) ? rank : 1;
    p_dv_desc->dim_rank = effective_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * effective_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
    istride = (istride == 0) ? 1 : istride;
    ostride = (ostride == 0) ? 1 : ostride;

    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }
    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = howmany;
    p_dv_desc->vecs[0].in_stride = (idist == 0) ? 1 : idist;
    p_dv_desc->vecs[0].out_stride = (odist == 0) ? 1 : odist;

    INT32 effective_rank = (rank > 0) ? rank : 1;
    p_dv_desc->dim_rank = effective_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * effective_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
        p_dv_desc->dims[0].in_stride = istride;
        if (rank > 1)
        {
            if (is_inplace)
            {
                p_dv_desc->dims[1].in_stride =
                    p_dv_desc->dims[0].in_stride *
                    ((p_dv_desc->dims[0].n / 2) + 1) * 2;
            }
            else
            {
                p_dv_desc->dims[1].in_stride =
                    p_dv_desc->dims[0].in_stride * p_dv_desc->dims[0].n;
            }
        }
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
        p_dv_desc->dims[0].out_stride = ostride;
        if (rank > 1)
        {
            p_dv_desc->dims[1].out_stride = p_dv_desc->dims[0].out_stride *
                                            ((p_dv_desc->dims[0].n / 2) + 1);
        }
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
    istride = (istride == 0) ? 1 : istride;
    ostride = (ostride == 0) ? 1 : ostride;

    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    p_dv_desc->vec_rank = 1;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t, sizeof(aoclfftz_dim_t));
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    p_dv_desc->vecs[0].n = howmany;
    p_dv_desc->vecs[0].in_stride = idist == 0 ? 1 : idist;
    p_dv_desc->vecs[0].out_stride = odist == 0 ? 1 : odist;

    INT32 effective_rank = (rank > 0) ? rank : 1;
    p_dv_desc->dim_rank = effective_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * effective_rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }

    if (rank == 0)
    {
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
        return p_dv_desc;
    }

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
        p_dv_desc->dims[0].in_stride = istride;
        if (rank > 1)
        {
            p_dv_desc->dims[1].in_stride =
                p_dv_desc->dims[0].in_stride *
                ((p_dv_desc->dims[0].n / 2) + 1);
        }
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
        p_dv_desc->dims[0].out_stride = ostride;
        if (rank > 1)
        {
            if (is_inplace)
            {
                p_dv_desc->dims[1].out_stride =
                    p_dv_desc->dims[0].out_stride *
                    ((p_dv_desc->dims[0].n / 2) + 1) * 2;
            }
            else
            {
                p_dv_desc->dims[1].out_stride =
                    p_dv_desc->dims[0].out_stride * p_dv_desc->dims[0].n;
            }
        }
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
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    if (howmany_rank > 0)
    {
        p_dv_desc->vec_rank = howmany_rank;
        ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t,
                           sizeof(aoclfftz_dim_t) * howmany_rank);
        if (p_dv_desc->vecs == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        for (INT32 i = 0; i < howmany_rank; i++)
        {
            p_dv_desc->vecs[i].n = howmany_dims[howmany_rank - i - 1].n;
            p_dv_desc->vecs[i].in_stride = howmany_dims[howmany_rank - i - 1].is;
            p_dv_desc->vecs[i].out_stride = howmany_dims[howmany_rank - i - 1].os;
        }
    }
    else
    {
        p_dv_desc->vec_rank = 1;
        ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t,
                           sizeof(aoclfftz_dim_t));
        if (p_dv_desc->vecs == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        p_dv_desc->vecs[0].n = 1;
        p_dv_desc->vecs[0].in_stride = 1;
        p_dv_desc->vecs[0].out_stride = 1;
    }

    if (rank > 0)
    {
        p_dv_desc->dim_rank = rank;
        ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                           sizeof(aoclfftz_dim_t) * rank);
        if (p_dv_desc->dims == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        for (INT32 i = 0; i < rank; i++)
        {
            p_dv_desc->dims[i].n = dims[rank - i - 1].n;
            p_dv_desc->dims[i].in_stride = dims[rank - i - 1].is;
            p_dv_desc->dims[i].out_stride = dims[rank - i - 1].os;
        }
    }
    else
    {
        p_dv_desc->dim_rank = 1;
        ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                           sizeof(aoclfftz_dim_t));
        if (p_dv_desc->dims == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
    }

    return p_dv_desc;
}

dv_desc_64_ *get_guru_64_dv_desc(INT32 rank, const fftw_iodim64 *dims,
                                INT32 howmany_rank,
                                const fftw_iodim64 *howmany_dims)
{
    dv_desc_64_ *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc_64_, sizeof(dv_desc_64_));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }

    if (howmany_rank > 0)
    {
        p_dv_desc->vec_rank = howmany_rank;
        ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t_64_,
                           sizeof(aoclfftz_dim_t_64_) * howmany_rank);
        if (p_dv_desc->vecs == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        for (INT32 i = 0; i < howmany_rank; i++)
        {
            p_dv_desc->vecs[i].n = howmany_dims[howmany_rank - i - 1].n;
            p_dv_desc->vecs[i].in_stride = howmany_dims[howmany_rank - i - 1].is;
            p_dv_desc->vecs[i].out_stride = howmany_dims[howmany_rank - i - 1].os;
        }
    }
    else
    {
        p_dv_desc->vec_rank = 1;
        ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t_64_,
                           sizeof(aoclfftz_dim_t_64_));
        if (p_dv_desc->vecs == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        p_dv_desc->vecs[0].n = 1;
        p_dv_desc->vecs[0].in_stride = 1;
        p_dv_desc->vecs[0].out_stride = 1;
    }

    if (rank > 0)
    {
        p_dv_desc->dim_rank = rank;
        ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t_64_,
                           sizeof(aoclfftz_dim_t_64_) * rank);
        if (p_dv_desc->dims == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        for (INT32 i = 0; i < rank; i++)
        {
            p_dv_desc->dims[i].n = dims[rank - i - 1].n;
            p_dv_desc->dims[i].in_stride = dims[rank - i - 1].is;
            p_dv_desc->dims[i].out_stride = dims[rank - i - 1].os;
        }
    }
    else
    {
        p_dv_desc->dim_rank = 1;
        ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t_64_,
                           sizeof(aoclfftz_dim_t_64_));
        if (p_dv_desc->dims == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
            FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
            return NULL;
        }
        p_dv_desc->dims[0].n = 1;
        p_dv_desc->dims[0].in_stride = 1;
        p_dv_desc->dims[0].out_stride = 1;
    }

    return p_dv_desc;
}

/**
 * @brief Reverses the order of elements in an integer array.
 *
 * This function copies the elements of the input array p_array into the output
 * array p_reversed_array, in reverse order. The size of the arrays is specified
 * by size.
 *
 * @param[in]  p_array           Pointer to the input integer array.
 * @param[out] p_reversed_array  Pointer to the output array for reversed
 *                               elements.
 * @param[in]  size              Number of elements in the array.
 */
VOID reverse_array(const INT32 *p_array, INT32 *p_reversed_array, INT32 size)
{
    for (INT32 dst_idx = 0, src_idx = size - 1; dst_idx < size;
         dst_idx++, src_idx--)
    {
        p_reversed_array[dst_idx] = p_array[src_idx];
    }
}

/**
 * @brief Constructs a dv_desc structure for a Fortran-ordered GURU FFT
 * interface.
 *
 * This function allocates and initializes a dv_desc structure for representing
 * multi-dimensional FFT problems as described by the Fortran GURU interface,
 * filling in dimension and vector descriptors using the provided arrays for
 * sizes and strides.
 *
 * Note: The C API uses an array of structures (each containing n, is, os
 *       fields), while the Fortran API provides the same information as
 *       separate arrays for sizes, input strides, and output strides.
 *
 * @param[in]  rank         The number of FFT dimensions.
 * @param[in]  n            Array of dimension sizes, of length rank.
 * @param[in]  is           Array of input strides, of length rank.
 * @param[in]  os           Array of output strides, of length rank.
 * @param[in]  howmany_rank The number of vector dimensions ("howmany"
 *                          dimensions).
 * @param[in]  h_n          Array of howmany dimension sizes, of length
 *                          howmany_rank.
 * @param[in]  h_is         Array of howmany input strides, of length
 *                          howmany_rank.
 * @param[in]  h_os         Array of howmany output strides, of length
 *                          howmany_rank.
 *
 * @return Pointer to the allocated and initialized dv_desc structure, or NULL
 *         if allocation fails.
 */
dv_desc *get_fortran_guru_dv_desc(INT32 rank, const INT32 *n, const INT32 *is,
                                  const INT32 *os, INT32 howmany_rank,
                                  const INT32 *h_n, const INT32 *h_is,
                                  const INT32 *h_os)
{
    dv_desc *p_dv_desc = NULL;
    ALLOC_ALIGN_UNINIT(p_dv_desc, dv_desc, sizeof(dv_desc));
    if (p_dv_desc == NULL)
    {
        return NULL;
    }
    p_dv_desc->vec_rank = howmany_rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->vecs, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * howmany_rank);
    if (p_dv_desc->vecs == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    for (INT32 i = 0; i < howmany_rank; i++)
    {
        p_dv_desc->vecs[i].n = h_n[i];
        p_dv_desc->vecs[i].in_stride = (h_is[i] == 0) ? 1 : h_is[i];
        p_dv_desc->vecs[i].out_stride = (h_os[i] == 0) ? 1 : h_os[i];
    }

    p_dv_desc->dim_rank = rank;
    ALLOC_ALIGN_UNINIT(p_dv_desc->dims, aoclfftz_dim_t,
                       sizeof(aoclfftz_dim_t) * rank);
    if (p_dv_desc->dims == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc->vecs);
        FREE_ALIGN_ALLOCATED_MEM(p_dv_desc);
        return NULL;
    }
    for (INT32 i = 0; i < rank; i++)
    {
        p_dv_desc->dims[i].n = n[i];
        p_dv_desc->dims[i].in_stride = (is[i] == 0) ? 1 : is[i];
        p_dv_desc->dims[i].out_stride = (os[i] == 0) ? 1 : os[i];
    }

    return p_dv_desc;
}
