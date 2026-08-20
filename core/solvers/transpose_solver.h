// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
#define SEL_REC_MINDIM_FFTZ_FLOAT 3800
#define SEL_REC_MINDIM_FFTZ_DOUBLE 3800
#define SEL_REC_MINDIM_aoclfftz_complex_f_t 3800
#define SEL_REC_MINDIM_aoclfftz_complex_d_t 4900

// Set the value of a variable based on the data type
#define SET_VAR(type_enum_var, var_prefix, destination)                        \
do                                                                             \
{                                                                              \
    switch ((type_enum_var))                                                   \
    {                                                                          \
    case TYPE_FLOAT:                                                           \
        (destination) = CONCAT(var_prefix, FFTZ_FLOAT); \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = CONCAT(var_prefix, FFTZ_DOUBLE); \
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
        (destination) = FUNC(fn_prefix, FFTZ_FLOAT, fn_suffix); \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = FUNC(fn_prefix, FFTZ_DOUBLE, fn_suffix); \
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
