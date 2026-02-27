/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file aoclfftz_core_wrapper.c
 *
 *  @brief Contains wrapper function definitions for core funtions
 *  with dllexport attribute.
 *
 *  This file contains the wrapper function definitions for core functions
 *  with `__declspec(dllexport)` attribute for Windows compatibility.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Ashwin K. Godbole
 */

#include "aoclfftz_core_wrapper.h"

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

ops_cycles_t get_ops_cnt_fft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft2c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft3c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft4c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft5c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft6c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft7c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft8c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft9c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft9c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft10c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft11c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft11c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft12c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft13c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft13c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft14c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft15c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft16c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft20c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft20c(precision, direction);
}
ops_cycles_t get_ops_cnt_fft48c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft48c(precision, direction);
}

#ifdef ENABLE_AVX128
ops_cycles_t get_ops_cnt_fft2avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft2avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft3avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft3avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft4avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft4avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft5avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft5avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft6avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft6avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft7avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft7avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft8avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft8avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft9avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft9avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft10avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft10avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft11avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft11avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft12avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft12avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft13avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft13avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft14avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft14avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft15avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft15avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft16avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft16avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft20avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft20avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_fft48avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft48avx128(precision, direction);
}
#endif

#ifdef ENABLE_AVX256
ops_cycles_t get_ops_cnt_fft2avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft2avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft3avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft3avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft4avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft4avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft5avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft5avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft6avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft6avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft7avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft7avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft8avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft8avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft9avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft9avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft10avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft10avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft11avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft11avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft12avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft12avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft13avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft13avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft14avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft14avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft15avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft15avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft16avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft16avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft20avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft20avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_fft48avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft48avx256(precision, direction);
}
#endif

#ifdef ENABLE_AVX512
ops_cycles_t get_ops_cnt_fft2avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft2avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft3avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft3avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft4avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft4avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft5avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft5avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft6avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft6avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft7avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft7avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft8avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft8avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft9avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft9avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft10avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft10avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft11avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft11avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft12avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft12avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft13avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft13avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft14avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft14avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft15avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft15avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft16avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft16avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft20avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft20avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_fft48avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_fft48avx512(precision, direction);
}
#endif

ops_cycles_t get_ops_cnt_twid_fft2c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft2c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft3c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft3c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft4c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft4c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft5c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft5c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft6c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft6c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft7c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft7c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft8c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft8c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft9c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft9c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft10c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft10c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft11c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft11c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft12c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft12c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft13c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft13c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft14c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft14c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft15c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft15c(prec, dir);
}
ops_cycles_t get_ops_cnt_twid_fft16c_wrapper(UINT8 prec, UINT8 dir)
{
    return get_ops_cnt_twid_fft16c(prec, dir);
}

#ifdef ENABLE_AVX128
// ops_cycles_t get_ops_cnt_twid_fft2avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft2avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft3avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft3avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft4avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft4avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft5avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft5avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft6avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft6avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft7avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft7avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft8avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft8avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft9avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft9avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft10avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft10avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft11avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft11avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft12avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft12avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft13avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft13avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft14avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft14avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft15avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft15avx128(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft16avx128_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft16avx128(prec, dir);
// }
#endif

#ifdef ENABLE_AVX256
// ops_cycles_t get_ops_cnt_twid_fft2avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft2avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft3avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft3avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft4avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft4avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft5avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft5avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft6avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft6avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft7avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft7avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft8avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft8avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft9avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft9avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft10avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft10avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft11avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft11avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft12avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft12avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft13avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft13avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft14avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft14avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft15avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft15avx256(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft16avx256_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft16avx256(prec, dir);
// }
#endif

#ifdef ENABLE_AVX512
// ops_cycles_t get_ops_cnt_twid_fft2avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft2avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft3avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft3avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft4avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft4avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft5avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft5avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft6avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft6avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft7avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft7avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft8avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft8avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft9avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft9avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft10avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft10avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft11avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft11avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft12avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft12avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft13avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft13avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft14avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft14avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft15avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft15avx512(prec, dir);
// }
// ops_cycles_t get_ops_cnt_twid_fft16avx512_wrapper(UINT8 prec, UINT8 dir)
// {
//     return get_ops_cnt_twid_fft16avx512(prec, dir);
// }
#endif

ops_cycles_t get_ops_cnt_r2hc_rfft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft2c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft3c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft4c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft5c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft6c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft7c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft8c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft10c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft12c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft14c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft15c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft16c(precision, direction);
}

ops_cycles_t get_ops_cnt_r2hcf_rfft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft2c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft3c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft4c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft5c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft6c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft7c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft8c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft10c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft12c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft14c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft15c(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft16c(precision, direction);
}


#ifdef ENABLE_AVX128
ops_cycles_t get_ops_cnt_r2hc_rfft2avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft2avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft3avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft3avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft4avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft4avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft5avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft5avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft6avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft6avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft7avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft7avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft8avx128_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft8avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft10avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft10avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft12avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft12avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft14avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft14avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft15avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft15avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft16avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft16avx128(precision, direction);
}


ops_cycles_t get_ops_cnt_r2hcf_rfft2avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft2avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft3avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft4avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft5avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft6avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft7avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx128_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft8avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx128_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft10avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx128_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft12avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx128_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft14avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx128_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft15avx128(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx128_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft16avx128(precision, direction);
}
#endif

#ifdef ENABLE_AVX256
ops_cycles_t get_ops_cnt_r2hc_rfft2avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft2avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft3avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft3avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft4avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft4avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft5avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft5avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft6avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft6avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft7avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft7avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft8avx256_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft8avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft10avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft10avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft12avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft12avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft14avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft14avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft15avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft15avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft16avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft16avx256(precision, direction);
}

ops_cycles_t get_ops_cnt_r2hcf_rfft2avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft2avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft3avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft4avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft5avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft6avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft7avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx256_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft8avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx256_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft10avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx256_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft12avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx256_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft14avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx256_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft15avx256(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx256_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft16avx256(precision, direction);
}
#endif

#ifdef ENABLE_AVX512
ops_cycles_t get_ops_cnt_r2hc_rfft2avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft2avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft3avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft3avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft4avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft4avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft5avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft5avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft6avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft6avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft7avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft7avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft8avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft8avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft10avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft10avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft12avx512_wrapper(UINT8 precision,
                                                  UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft12avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft14avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft14avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft15avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft15avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hc_rfft16avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hc_rfft16avx512(precision, direction);
}

ops_cycles_t get_ops_cnt_r2hcf_rfft2avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft2avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft3avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft4avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft5avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft6avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft7avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx512_wrapper(UINT8 precision,
                                                   UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft8avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx512_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft10avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft12avx512_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft12avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx512_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft14avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx512_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft15avx512(precision, direction);
}
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx512_wrapper(UINT8 precision,
                                                    UINT8 direction)
{
    return get_ops_cnt_r2hcf_rfft16avx512(precision, direction);
}
#endif

/* ---------------- kernels : register_kernel_fft* ---------------- */

kfft_ register_kernel_fft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft2c(precision, direction);
}
kfft_ register_kernel_fft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft3c(precision, direction);
}
kfft_ register_kernel_fft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft4c(precision, direction);
}
kfft_ register_kernel_fft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft5c(precision, direction);
}
kfft_ register_kernel_fft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft6c(precision, direction);
}
kfft_ register_kernel_fft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft7c(precision, direction);
}
kfft_ register_kernel_fft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft8c(precision, direction);
}
kfft_ register_kernel_fft9c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft9c(precision, direction);
}
kfft_ register_kernel_fft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft10c(precision, direction);
}
kfft_ register_kernel_fft11c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft11c(precision, direction);
}
kfft_ register_kernel_fft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft12c(precision, direction);
}
kfft_ register_kernel_fft13c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft13c(precision, direction);
}
kfft_ register_kernel_fft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft14c(precision, direction);
}
kfft_ register_kernel_fft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft15c(precision, direction);
}
kfft_ register_kernel_fft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft16c(precision, direction);
}
kfft_ register_kernel_fft20c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft20c(precision, direction);
}
kfft_ register_kernel_fft48c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft48c(precision, direction);
}

#ifdef ENABLE_AVX128
kfft_ register_kernel_fft2avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft2avx128(precision, direction);
}
kfft_ register_kernel_fft3avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft3avx128(precision, direction);
}
kfft_ register_kernel_fft4avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft4avx128(precision, direction);
}
kfft_ register_kernel_fft5avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft5avx128(precision, direction);
}
kfft_ register_kernel_fft6avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft6avx128(precision, direction);
}
kfft_ register_kernel_fft7avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft7avx128(precision, direction);
}
kfft_ register_kernel_fft8avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft8avx128(precision, direction);
}
kfft_ register_kernel_fft9avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft9avx128(precision, direction);
}
kfft_ register_kernel_fft10avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft10avx128(precision, direction);
}
kfft_ register_kernel_fft11avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft11avx128(precision, direction);
}
kfft_ register_kernel_fft12avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft12avx128(precision, direction);
}
kfft_ register_kernel_fft13avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft13avx128(precision, direction);
}
kfft_ register_kernel_fft14avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft14avx128(precision, direction);
}
kfft_ register_kernel_fft15avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft15avx128(precision, direction);
}
kfft_ register_kernel_fft16avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft16avx128(precision, direction);
}
kfft_ register_kernel_fft20avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft20avx128(precision, direction);
}
kfft_ register_kernel_fft48avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft48avx128(precision, direction);
}
#endif

#ifdef ENABLE_AVX256
kfft_ register_kernel_fft2avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft2avx256(precision, direction);
}
kfft_ register_kernel_fft3avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft3avx256(precision, direction);
}
kfft_ register_kernel_fft4avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft4avx256(precision, direction);
}
kfft_ register_kernel_fft5avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft5avx256(precision, direction);
}
kfft_ register_kernel_fft6avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft6avx256(precision, direction);
}
kfft_ register_kernel_fft7avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft7avx256(precision, direction);
}
kfft_ register_kernel_fft8avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft8avx256(precision, direction);
}
kfft_ register_kernel_fft9avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft9avx256(precision, direction);
}
kfft_ register_kernel_fft10avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft10avx256(precision, direction);
}
kfft_ register_kernel_fft11avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft11avx256(precision, direction);
}
kfft_ register_kernel_fft12avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft12avx256(precision, direction);
}
kfft_ register_kernel_fft13avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft13avx256(precision, direction);
}
kfft_ register_kernel_fft14avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft14avx256(precision, direction);
}
kfft_ register_kernel_fft15avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft15avx256(precision, direction);
}
kfft_ register_kernel_fft16avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft16avx256(precision, direction);
}
kfft_ register_kernel_fft20avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft20avx256(precision, direction);
}
kfft_ register_kernel_fft48avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft48avx256(precision, direction);
}
#endif

#ifdef ENABLE_AVX512
kfft_ register_kernel_fft2avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft2avx512(precision, direction);
}
kfft_ register_kernel_fft3avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft3avx512(precision, direction);
}
kfft_ register_kernel_fft4avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft4avx512(precision, direction);
}
kfft_ register_kernel_fft5avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft5avx512(precision, direction);
}
kfft_ register_kernel_fft6avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft6avx512(precision, direction);
}
kfft_ register_kernel_fft7avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft7avx512(precision, direction);
}
kfft_ register_kernel_fft8avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft8avx512(precision, direction);
}
kfft_ register_kernel_fft9avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft9avx512(precision, direction);
}
kfft_ register_kernel_fft10avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft10avx512(precision, direction);
}
kfft_ register_kernel_fft11avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft11avx512(precision, direction);
}
kfft_ register_kernel_fft12avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft12avx512(precision, direction);
}
kfft_ register_kernel_fft13avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft13avx512(precision, direction);
}
kfft_ register_kernel_fft14avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft14avx512(precision, direction);
}
kfft_ register_kernel_fft15avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft15avx512(precision, direction);
}
kfft_ register_kernel_fft16avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft16avx512(precision, direction);
}
kfft_ register_kernel_fft20avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft20avx512(precision, direction);
}
kfft_ register_kernel_fft48avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_fft48avx512(precision, direction);
}
#endif

kfft_ register_kernel_twid_fft2c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft2c(prec, dir);
}
kfft_ register_kernel_twid_fft3c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft3c(prec, dir);
}
kfft_ register_kernel_twid_fft4c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft4c(prec, dir);
}
kfft_ register_kernel_twid_fft5c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft5c(prec, dir);
}
kfft_ register_kernel_twid_fft6c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft6c(prec, dir);
}
kfft_ register_kernel_twid_fft7c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft7c(prec, dir);
}
kfft_ register_kernel_twid_fft8c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft8c(prec, dir);
}
kfft_ register_kernel_twid_fft9c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft9c(prec, dir);
}
kfft_ register_kernel_twid_fft10c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft10c(prec, dir);
}
kfft_ register_kernel_twid_fft11c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft11c(prec, dir);
}
kfft_ register_kernel_twid_fft12c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft12c(prec, dir);
}
kfft_ register_kernel_twid_fft13c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft13c(prec, dir);
}
kfft_ register_kernel_twid_fft14c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft14c(prec, dir);
}
kfft_ register_kernel_twid_fft15c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft15c(prec, dir);
}
kfft_ register_kernel_twid_fft16c_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft16c(prec, dir);
}

#ifdef ENABLE_AVX128
kfft_ register_kernel_twid_fft2avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft2avx128(prec, dir);
}
kfft_ register_kernel_twid_fft3avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft3avx128(prec, dir);
}
kfft_ register_kernel_twid_fft4avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft4avx128(prec, dir);
}
kfft_ register_kernel_twid_fft5avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft5avx128(prec, dir);
}
kfft_ register_kernel_twid_fft6avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft6avx128(prec, dir);
}
kfft_ register_kernel_twid_fft7avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft7avx128(prec, dir);
}
kfft_ register_kernel_twid_fft8avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft8avx128(prec, dir);
}
kfft_ register_kernel_twid_fft9avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft9avx128(prec, dir);
}
kfft_ register_kernel_twid_fft10avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft10avx128(prec, dir);
}
kfft_ register_kernel_twid_fft11avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft11avx128(prec, dir);
}
kfft_ register_kernel_twid_fft12avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft12avx128(prec, dir);
}
kfft_ register_kernel_twid_fft13avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft13avx128(prec, dir);
}
kfft_ register_kernel_twid_fft14avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft14avx128(prec, dir);
}
kfft_ register_kernel_twid_fft15avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft15avx128(prec, dir);
}
kfft_ register_kernel_twid_fft16avx128_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft16avx128(prec, dir);
}
#endif

#ifdef ENABLE_AVX256
kfft_ register_kernel_twid_fft2avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft2avx256(prec, dir);
}
kfft_ register_kernel_twid_fft3avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft3avx256(prec, dir);
}
kfft_ register_kernel_twid_fft4avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft4avx256(prec, dir);
}
kfft_ register_kernel_twid_fft5avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft5avx256(prec, dir);
}
kfft_ register_kernel_twid_fft6avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft6avx256(prec, dir);
}
kfft_ register_kernel_twid_fft7avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft7avx256(prec, dir);
}
kfft_ register_kernel_twid_fft8avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft8avx256(prec, dir);
}
kfft_ register_kernel_twid_fft9avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft9avx256(prec, dir);
}
kfft_ register_kernel_twid_fft10avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft10avx256(prec, dir);
}
kfft_ register_kernel_twid_fft11avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft11avx256(prec, dir);
}
kfft_ register_kernel_twid_fft12avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft12avx256(prec, dir);
}
kfft_ register_kernel_twid_fft13avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft13avx256(prec, dir);
}
kfft_ register_kernel_twid_fft14avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft14avx256(prec, dir);
}
kfft_ register_kernel_twid_fft15avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft15avx256(prec, dir);
}
kfft_ register_kernel_twid_fft16avx256_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft16avx256(prec, dir);
}
#endif

#ifdef ENABLE_AVX512
kfft_ register_kernel_twid_fft2avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft2avx512(prec, dir);
}
kfft_ register_kernel_twid_fft3avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft3avx512(prec, dir);
}
kfft_ register_kernel_twid_fft4avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft4avx512(prec, dir);
}
kfft_ register_kernel_twid_fft5avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft5avx512(prec, dir);
}
kfft_ register_kernel_twid_fft6avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft6avx512(prec, dir);
}
kfft_ register_kernel_twid_fft7avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft7avx512(prec, dir);
}
kfft_ register_kernel_twid_fft8avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft8avx512(prec, dir);
}
kfft_ register_kernel_twid_fft9avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft9avx512(prec, dir);
}
kfft_ register_kernel_twid_fft10avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft10avx512(prec, dir);
}
kfft_ register_kernel_twid_fft11avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft11avx512(prec, dir);
}
kfft_ register_kernel_twid_fft12avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft12avx512(prec, dir);
}
kfft_ register_kernel_twid_fft13avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft13avx512(prec, dir);
}
kfft_ register_kernel_twid_fft14avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft14avx512(prec, dir);
}
kfft_ register_kernel_twid_fft15avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft15avx512(prec, dir);
}
kfft_ register_kernel_twid_fft16avx512_wrapper(UINT8 prec, UINT8 dir)
{
    return register_kernel_twid_fft16avx512(prec, dir);
}
#endif

kfft_ register_kernel_r2hc_rfft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft2c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft3c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft4c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft5c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft6c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft7c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft8c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft10c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft12c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft14c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft15c(precision, direction);
}
kfft_ register_kernel_r2hc_rfft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft16c(precision, direction);
}

kfft_ register_kernel_r2hcf_rfft2c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft2c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft3c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft3c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft4c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft4c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft5c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft5c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft6c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft6c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft7c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft7c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft8c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft8c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft10c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft10c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft12c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft12c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft14c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft14c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft15c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft15c(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft16c_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hcf_rfft16c(precision, direction);
}

#ifdef ENABLE_AVX128
// R2HC - AVX128 register kernel wrapper
kfft_ register_kernel_r2hc_rfft2avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft2avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft3avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft3avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft4avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft4avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft5avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft5avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft6avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft6avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft7avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft7avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft8avx128_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft8avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft10avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft10avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft12avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft12avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft14avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft14avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft15avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft15avx128(precision, direction);
}
kfft_ register_kernel_r2hc_rfft16avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft16avx128(precision, direction);
}

// R2HC-Fused - AVX128 register kernel wrapper
kfft_ register_kernel_r2hcf_rfft2avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft2avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft3avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft3avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft4avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft4avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft5avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft5avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft6avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft6avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft7avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft7avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft8avx128_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft8avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft10avx128_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft10avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft12avx128_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft12avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft14avx128_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft14avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft15avx128_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft15avx128(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft16avx128_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft16avx128(precision, direction);
}
#endif

#ifdef ENABLE_AVX256
// R2HC - AVX256 register kernel wrapper
kfft_ register_kernel_r2hc_rfft2avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft2avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft3avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft3avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft4avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft4avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft5avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft5avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft6avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft6avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft7avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft7avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft8avx256_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft8avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft10avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft10avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft12avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft12avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft14avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft14avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft15avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft15avx256(precision, direction);
}
kfft_ register_kernel_r2hc_rfft16avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft16avx256(precision, direction);
}

// R2HC-Fused - AVX256 register kernel wrapper
kfft_ register_kernel_r2hcf_rfft2avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft2avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft3avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft3avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft4avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft4avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft5avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft5avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft6avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft6avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft7avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft7avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft8avx256_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft8avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft10avx256_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft10avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft12avx256_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft12avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft14avx256_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft14avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft15avx256_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft15avx256(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft16avx256_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft16avx256(precision, direction);
}
#endif

#ifdef ENABLE_AVX512
// R2HC - AVX512 register kernel wrapper
kfft_ register_kernel_r2hc_rfft2avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft2avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft3avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft3avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft4avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft4avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft5avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft5avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft6avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft6avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft7avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft7avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft8avx512_wrapper(UINT8 precision, UINT8 direction)
{
    return register_kernel_r2hc_rfft8avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft10avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft10avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft12avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft12avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft14avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft14avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft15avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft15avx512(precision, direction);
}
kfft_ register_kernel_r2hc_rfft16avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hc_rfft16avx512(precision, direction);
}

// R2HC-Fused - AVX512 register kernel wrapper
kfft_ register_kernel_r2hcf_rfft2avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft2avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft3avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft3avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft4avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft4avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft5avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft5avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft6avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft6avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft7avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft7avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft8avx512_wrapper(UINT8 precision,
                                                UINT8 direction)
{
    return register_kernel_r2hcf_rfft8avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft10avx512_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft10avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft12avx512_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft12avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft14avx512_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft14avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft15avx512_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft15avx512(precision, direction);
}
kfft_ register_kernel_r2hcf_rfft16avx512_wrapper(UINT8 precision,
                                                 UINT8 direction)
{
    return register_kernel_r2hcf_rfft16avx512(precision, direction);
}
#endif

/* ---------------- kernels : permuted_copy_* ---------------- */

VOID permuted_copy_c_fp32_wrapper(VOID *in, VOID *out, INTP n, INTP radix,
                                  aoclfftz_strides_t *strides,
                                  UINT8 data_stride)
{
    permuted_copy_c_fp32(in, out, n, radix, strides, data_stride);
}
VOID permuted_copy_c_fp64_wrapper(VOID *in, VOID *out, INTP n, INTP radix,
                                  aoclfftz_strides_t *strides,
                                  UINT8 data_stride)
{
    permuted_copy_c_fp64(in, out, n, radix, strides, data_stride);
}

/* ---------------- memory allocators/destroys ---------------- */

aoclfftz_decomp_scheme_t *alloc_decomp_scheme_wrapper(INT32 vec_rank,
                                                      INT32 dim_rank)
{
    return alloc_decomp_scheme(vec_rank, dim_rank);
}
aoclfftz_solution_t *alloc_solution_wrapper(INT32 vec_rank, INT32 dim_rank)
{
    return alloc_solution(vec_rank, dim_rank);
}
aoclfftz_selector_t *alloc_selector_wrapper(INT32 vec_rank, INT32 dim_rank,
                                            VOID *scratch_space,
                                            kernel_tables_t *kernel_tables)
{
    return alloc_selector(vec_rank, dim_rank, scratch_space, kernel_tables,
                          0 /*unused*/);
}
VOID *alloc_twiddle_buffer_wrapper(INTP size, UINT32 dt_prec)
{
    return alloc_twiddle_buffer(size, dt_prec);
}

VOID destroy_selector_wrapper(aoclfftz_selector_t *sel)
{
    destroy_selector(sel);
}
VOID destroy_solution_wrapper(aoclfftz_solution_t *sol)
{
    destroy_solution(sol, 1);
}
VOID destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    destroy_decomp_scheme(decomp_scheme);
}
VOID destroy_handle_wrapper(VOID *handle)
{
    destroy_handle(handle);
}

/* ---------------- fuse vector wrapper ---------------- */
VOID fuse_vecs_wrapper(aoclfftz_solution_t *sol)
{
    fuse_vecs(sol);
}

/* ---------------- strides wrapper ---------------- */
VOID populate_stride_array_wrapper(INTP *strides, INTP stride_val, INTP n,
                                   UINT8 compute_half_complex,
                                   UINT8 adjust_to_full_complex)
{
    populate_stride_array(strides, stride_val, n, compute_half_complex,
                          adjust_to_full_complex);
}

/* ---------------- fused strides wrapper ---------------- */
VOID prepare_fused_kernel_strides_wrapper(INTP *strides, INTP radix,
                                          INTP offset)
{
    prepare_fused_kernel_strides(strides, radix, offset);
}

// Transpose wrappers
#define TRANSPOSE_WRAPPER_DEFN(kernel_name, TYPE, isa)                         \
    VOID CONCAT(FUNC(kernel_name, TYPE, isa), _wrapper)(TRANSPOSE_KERNEL_ARGS) \
    {                                                                          \
        FUNC(kernel_name, TYPE, c)(in_ptr, out_ptr, row_metadata,              \
                                   column_metadata, aux_mem);                  \
    }                                                                          \


#define TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(kernel_name, isa)                     \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, FLOAT, isa)                            \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, DOUBLE, isa)                           \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_f_t, isa)             \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_d_t, isa)

TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tiq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tisq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tiq_recursive_buf, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tir_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tisr_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tos_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tos_blocked, c)

// for the gtests, we want to use the in-memory twiddle factors
// so we define IN_MEMORY_TWIDDLE_FACTORS to 1 if not explicitly set/defined
#if !defined(IN_MEMORY_TWIDDLE_FACTORS)
    #define IN_MEMORY_TWIDDLE_FACTORS 1
#elif IN_MEMORY_TWIDDLE_FACTORS == 0
    #undef IN_MEMORY_TWIDDLE_FACTORS
    #define IN_MEMORY_TWIDDLE_FACTORS 1
#endif

// twiddle buffer setup wrappers
EXPORT_SYM_DYN VOID compute_twiddle_buffer_float_wrapper(VOID *twiddle_buffer,
                                                       INTP r, INTP m)
{
    compute_twiddle_buffer(twiddle_buffer, r, m, DT_FLOAT);
}

EXPORT_SYM_DYN VOID compute_twiddle_buffer_double_wrapper(VOID *twiddle_buffer,
                                                        INTP r, INTP m)
{
    compute_twiddle_buffer(twiddle_buffer, r, m, DT_DOUBLE);
}
