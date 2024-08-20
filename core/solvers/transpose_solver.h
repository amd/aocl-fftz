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

/** @file transpose_solver.c
 *
 *  @brief Header file for common transpose solver macros
 *
 *  This file contains the macros required by the transpose solver.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef TRANSPOSE_SOLVER_H
#define TRANSPOSE_SOLVER_H

// Number of rows/cols above which the recursive square transpose is used
#define SEL_REC_MINDIM_FLOAT 3800
#define SEL_REC_MINDIM_DOUBLE 3800
#define SEL_REC_MINDIM_aoclfftz_complex_f_t 3800
#define SEL_REC_MINDIM_aoclfftz_complex_d_t 4900

// Set the value of a variable based on the data type
#define SET_VAR(type_enum_var, var_prefix, destination)                        \
do                                                                             \
{                                                                              \
    switch ((type_enum_var))                                                   \
    {                                                                          \
    case TYPE_FLOAT:                                                           \
        (destination) = CONCAT(var_prefix, FLOAT);                             \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = CONCAT(var_prefix, DOUBLE);                            \
        break;                                                                 \
    case TYPE_FLOATCOMPLEX:                                                    \
        (destination) = CONCAT(var_prefix, aoclfftz_complex_f_t);              \
        break;                                                                 \
    case TYPE_DOUBLECOMPLEX:                                                   \
        (destination) = CONCAT(var_prefix, aoclfftz_complex_d_t);              \
        break;                                                                 \
    }                                                                          \
} while (0);

// Set the value of a function pointer based on the data type
#define SET_FNPTR(type_enum_var, destination, fn_prefix, fn_suffix)            \
do                                                                             \
{                                                                              \
    switch ((type_enum_var))                                                   \
    {                                                                          \
    case TYPE_FLOAT:                                                           \
        (destination) = FUNC(fn_prefix, FLOAT, fn_suffix);                     \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = FUNC(fn_prefix, DOUBLE, fn_suffix);                    \
        break;                                                                 \
    case TYPE_FLOATCOMPLEX:                                                    \
        (destination) = FUNC(fn_prefix, aoclfftz_complex_f_t, fn_suffix);      \
        break;                                                                 \
    case TYPE_DOUBLECOMPLEX:                                                   \
        (destination) = FUNC(fn_prefix, aoclfftz_complex_d_t, fn_suffix);      \
        break;                                                                 \
    }                                                                          \
} while (0);

#endif // TRANSPOSE_SOLVER_H
