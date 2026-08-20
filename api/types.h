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

typedef int64_t FFTZ_INT64; /**< fixed signed 64 bits int @ingroup std_types */
/** fixed signed 32 bits int : Use it for signed int @ingroup std_types */
typedef int32_t FFTZ_INT32;
/** portable signed int type : 32 bits (ILP32), 64 bits (LP64) @ingroup
 * std_types */
typedef ptrdiff_t FFTZ_INTP;
/** fixed unsigned 64 bits int @ingroup std_types */
typedef uint64_t FFTZ_UINT64;
/** fixed unsigned 32 bits int : Use it for unsigned int @ingroup std_types */
typedef uint32_t FFTZ_UINT32;
/** portable unsigned int type : 32 bits (ILP32), 64 bits (LP64) @ingroup
 * std_types */
typedef size_t FFTZ_UINTP;
/** signed character data type : 8 bits @ingroup std_types */
typedef char FFTZ_CHAR;
/** unsigned character data type : 8 bits @ingroup std_types */
typedef unsigned char FFTZ_UCHAR;
/** signed short integer : 16 bits @ingroup std_types */
typedef short FFTZ_SHORT;
/** unsigned short integer : 16 bits @ingroup std_types */
typedef unsigned short FFTZ_USHORT;
typedef void FFTZ_VOID; /**< void type @ingroup std_types */
/** single precision floating point : 32 bits @ingroup std_types */
typedef float FFTZ_FLOAT32;
/** single precision floating point : 32 bits @ingroup std_types */
typedef float FFTZ_FLOAT;
/** double precision floating point : 64 bits @ingroup std_types */
typedef double FFTZ_FLOAT64;
/** double precision floating point : 64 bits @ingroup std_types */
typedef double FFTZ_DOUBLE;
typedef uint8_t FFTZ_UINT8; /**< unsigned 8 bits integer @ingroup std_types */
typedef int8_t FFTZ_INT8; /**< signed 8 bits integer @ingroup std_types */

#endif // TYPES_H
