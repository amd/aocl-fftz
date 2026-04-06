// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file types.h
 *  @brief AOCL-FFTZ Library - Typedef declarations.
 *
 *  This file contains the typedef declarations of different data types.
 *
 *  @author S. Biplab Raut
 */

/**
* @defgroup std_types Types AOCL-FFTZ - Types
* @brief Type Definitions of standard datatypes.
*/

#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stddef.h>

typedef int64_t INT64;          /**< fixed signed 64 bits int @ingroup std_types */
typedef int32_t INT32;          /**< fixed signed 32 bits int : Use it for signed int @ingroup std_types */
typedef ptrdiff_t INTP;         /**< portable signed int type : 32 bits (ILP32), 64 bits (LP64) @ingroup std_types */
typedef uint64_t UINT64;        /**< fixed unsigned 64 bits int @ingroup std_types */
typedef uint32_t UINT32;        /**< fixed unsigned 32 bits int : Use it for unsigned int @ingroup std_types */
typedef size_t UINTP;           /**< portable unsigned int type : 32 bits (ILP32), 64 bits (LP64) @ingroup std_types */
typedef char CHAR;              /**< signed character data type : 8 bits @ingroup std_types */
typedef unsigned char UCHAR;    /**< unsigned character data type : 8 bits @ingroup std_types */
typedef short SHORT;            /**< signed short integer : 16 bits @ingroup std_types */
typedef unsigned short USHORT;  /**< unsigned short integer : 16 bits @ingroup std_types */
typedef void VOID;              /**< void type @ingroup std_types */
typedef float FLOAT32;          /**< single precision floating point : 32 bits @ingroup std_types */
typedef float FLOAT;            /**< single precision floating point : 32 bits @ingroup std_types */
typedef double FLOAT64;         /**< double precision floating point : 64 bits @ingroup std_types */
typedef double DOUBLE;          /**< double precision floating point : 64 bits @ingroup std_types */
typedef uint8_t UINT8;          /**< unsigned 8 bits integer @ingroup std_types */
typedef int8_t INT8;            /**< signed 8 bits integer @ingroup std_types */

#endif // TYPES_H
