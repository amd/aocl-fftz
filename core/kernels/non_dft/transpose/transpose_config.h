// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_config.h
 *
 *  @brief Parameters for chosing the optimal transpose kernels
 *
 *  This file contains the various parameters that are used to obtain the
 *  optimal transpose kernel.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef TRANSPOSE_CONFIG_H
#define TRANSPOSE_CONFIG_H

// Minimum number or rows (or columns) for recursive square transpose to be used
#define REC_MIN_FFTZ_FLOAT 128
#define REC_MIN_FFTZ_DOUBLE 64
#define REC_MIN_aoclfftz_complex_f_t 64
#define REC_MIN_aoclfftz_complex_d_t 32

// Defines the block dimension (block-rows x block-columns) for the block based
// transpose algorithm
#define BLOCK_DIM_FFTZ_FLOAT 16
#define BLOCK_DIM_FFTZ_DOUBLE 8
#define BLOCK_DIM_aoclfftz_complex_f_t 8
#define BLOCK_DIM_aoclfftz_complex_d_t 4

#endif // TRANSPOSE_CONFIG_H
