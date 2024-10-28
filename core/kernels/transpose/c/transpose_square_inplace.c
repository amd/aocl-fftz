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

/** @file transpose_inplace_square.c
 *
 *  @brief Implementations of transpose kernels
 *
 *  Implementations of transpose kernels that use different algorithms to
 *  perform the transpose operation on square matrices, in place.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/transpose/transpose_kernels.h"

#define TYPE_GENERIC_IMPLEMENTATION

#define TRANSPOSE_DT FLOAT
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT DOUBLE
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT aoclfftz_complex_f_t
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#undef TRANSPOSE_DT

#define TRANSPOSE_DT aoclfftz_complex_d_t
#include "core/kernels/transpose/c/transpose_square_inplace_generic.h"
#undef TRANSPOSE_DT

#undef TYPE_GENERIC_IMPLEMENTATION
