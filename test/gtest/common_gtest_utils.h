// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file common_gtest_utils.h
 *
 * @brief Common utility functions for GTest tests (api & kernel).
 *
 * This file contains common utility functions used across different
 * test files including special value generators for double & float testing.
 *
 * @author Niranjan Reddy
 *
 */

#ifndef AOCLFFTZ_COMMON_GTEST_UTILS_H
#define AOCLFFTZ_COMMON_GTEST_UTILS_H

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <typeinfo>

extern "C"
{
#include "api/types.h"
}

// Define true minimum values for float and double if not already defined
#ifndef DBL_TRUE_MIN
#define DBL_TRUE_MIN 4.9406564584124654e-324
#endif
#ifndef FLT_TRUE_MIN
#define FLT_TRUE_MIN 1.40129846e-45F
#endif

#define NUM_FP_SPECIAL_VALUE_CASES 11

#define TINY_VALUES_ONLY_COUNT 8
#define LARGE_VALUES_ONLY_COUNT 2

// Enum to specify input value generation strategy
enum class InputValueStrategy
{
    FULL_ZERO,            // Full zero values
    MID_RANGE,            // Values in range [-10.0, 10.0)
    NEAR_EDGE,            // Near-edge values
    SPECIAL_VALUES,       // (NaN, Inf, MAX, MIN) - for robustness testing
    SPECIAL_EXCEPT_NAN,   // (Inf, MAX, MIN) - for robustness testing, but not NaN
    TINY_VALUES_ONLY,     // Subnormal and small normal values
    LARGE_VALUES_ONLY     // Only large values (close to but not exceeding MAX)
};

/**
 * @brief Structure to hold output validation statistics
 */
typedef struct
{
    UINTP nan_count;      /**< Count of NaN values */
    UINTP inf_count;      /**< Count of Infinity values (both +Inf and -Inf) */
    UINTP zero_count;     /**< Count of zero values (both +0 and -0) */
    UINTP nonzero_count;  /**< Count of non-zero finite values */
} output_validation_stats;

/**
 * @brief Generate only tiny values (subnormal and very small normal values)
 *
 * @tparam T data type (float32 or float64)
 * @param category input category to pick the type of value
 * @return T return a tiny value
 */
template <class T> T get_subnormal_and_near_underflow_value(INT32 category)
{
    if (typeid(T) == typeid(DOUBLE))
    {
        // DBL_MIN = 2.23e-308
        const DOUBLE normalised_random = DBL_MIN * (1 + (rand() % 1000));
        // DBL_TRUE_MIN = 4.94e-324
        const DOUBLE subnormal_random = DBL_TRUE_MIN * (1 + (rand() % 1000));

        switch (category % TINY_VALUES_ONLY_COUNT)
        {
        case 0:
            return normalised_random;
        case 1:
            return -normalised_random;
        case 2:
            return subnormal_random;
        case 3:
            return -subnormal_random;
        case 4:
            return DBL_TRUE_MIN;
        case 5:
            return -DBL_TRUE_MIN;
        case 6:
            return DBL_MIN;
        default:
            return -DBL_MIN;
        }
    }
    else if (typeid(T) == typeid(FLOAT))
    {
        // FLT_MIN = 1.18e-38
        const FLOAT normalised_random = FLT_MIN * (1 + (rand() % 1000));
        // FLT_TRUE_MIN = 1.40e-45
        const FLOAT subnormal_random = FLT_TRUE_MIN * (1 + (rand() % 1000));

        switch (category % TINY_VALUES_ONLY_COUNT)
        {
        case 0:
            return normalised_random;
        case 1:
            return -normalised_random;
        case 2:
            return subnormal_random;
        case 3:
            return -subnormal_random;
        case 4:
            return FLT_MIN;
        case 5:
            return -FLT_MIN;
        case 6:
            return FLT_TRUE_MIN;
        default:
            return -FLT_TRUE_MIN;
        }
    }
    return 0.0;
}

/**
 * @brief Generate only large values (values close to but not exceeding MAX)
 * 
 * @tparam T data type (float32 or float64)
 * @param category input category to pick the type of value
 * @param in_size size of input array to scale values appropriately
 * @return T return a large value
 */
template <class T> T get_near_overflow_value(INT32 category, INT32 in_size)
{
    if (typeid(T) == typeid(DOUBLE))
    {
        // DBL_MAX = 1.80e+308
        // Scale by input size to prevent overflow during FFT operations
        const DOUBLE large_value = DBL_MAX / (in_size + (rand() % 1000));

        switch (category % LARGE_VALUES_ONLY_COUNT)
        {
        case 0:
            return large_value;
        default:
            return -large_value;
        }
    }
    else if (typeid(T) == typeid(FLOAT))
    {
        // FLT_MAX = 3.40e+38
        // Scale by input size to prevent overflow during FFT operations
        const FLOAT large_value = FLT_MAX / (in_size + (rand() % 1000));

        switch (category % LARGE_VALUES_ONLY_COUNT)
        {
        case 0:
            return large_value;
        default:
            return -large_value;
        }
    }
    return 0.0;
}

/**
 * @brief Generate values near edge cases
 * 
 * This function generates floating-point values that are close to
 * edge cases but not so extreme that they cause overflow in FFT operations.
 * 
 * @tparam T data type (float32 or float64)
 * @param category input category to pick the type of value
 * @param in_size size of input array to scale values appropriately
 * @return T return a near-edge value
 */
template <class T> T get_near_edge_value(INT32 category, INT32 in_size)
{
    // Mix of tiny and large values - reuse existing functions
    // 50% tiny values, 50% large values
    if (category % (TINY_VALUES_ONLY_COUNT * 2) < TINY_VALUES_ONLY_COUNT)
    {
        return get_subnormal_and_near_underflow_value<T>(category);
    }
    else
    {
        return get_near_overflow_value<T>(category, in_size);
    }
}

/**
 * @brief Get the special values like NaN, infinity, negative infinity
 * floating point min/max/subnormal-min values in positive and negative range.
 * This function may generate the above special values based on the given
 * `category` parameter. category mod (NUM_FP_SPECIAL_VALUE_CASES * 2) is used
 * to limit special values to exactly 50% probability (11 special cases
 * out of NUM_FP_SPECIAL_VALUE_CASES * 2). Random values will be in range
 * [-10.0, 10.0) with 3 decimal precision.
 *
 * @tparam T data type (float32 or float64)
 * @param category input category to pick the type of value
 * @return T return the normal or special value
 */
template <class T> T get_fp_special_value(INT32 category)
{
    if (typeid(T) == typeid(DOUBLE))
    {
        // Multiply by 2 to get 50% special values and 50% random values
        switch (category % (NUM_FP_SPECIAL_VALUE_CASES * 2))
        {
        case 0:
            return 0.0L;
        case 1:
            return DBL_TRUE_MIN;
        case 2:
            return -DBL_TRUE_MIN;
        case 3:
            return DBL_MIN;
        case 4:
            return -DBL_MIN;
        case 5:
            return DBL_MAX;
        case 6:
            return -DBL_MAX;
        case 7:
            return INFINITY;
        case 8:
            return -INFINITY;
        case 9:
            return NAN;
        case 10:
            return -NAN;
        default:
            return ((rand() % 20000) / 1000.0) - 10.0;
        }
    }
    else if (typeid(T) == typeid(FLOAT))
    {
        // Multiply by 2 to get 50% special values and 50% random values
        switch (category % (NUM_FP_SPECIAL_VALUE_CASES * 2))
        {
        case 0:
            return 0.0F;
        case 1:
            return FLT_TRUE_MIN;
        case 2:
            return -FLT_TRUE_MIN;
        case 3:
            return FLT_MIN;
        case 4:
            return -FLT_MIN;
        case 5:
            return FLT_MAX;
        case 6:
            return -FLT_MAX;
        case 7:
            return INFINITY;
        case 8:
            return -INFINITY;
        case 9:
            return NAN;
        case 10:
            return -NAN;
        default:
            return ((rand() % 20000) / 1000.0) - 10.0;
        }
    }
    return 0.0;
}

/**
 * @brief Generate test values based on specified strategy
 * 
 * 
 * @tparam T data type (FLOAT or DOUBLE)
 * @param strategy the input value generation strategy to use
 * @param in_size size of input array, used to scale large values to prevent overflow
 * @return T generated value according to the specified strategy
 */
template <class T> T get_value_based_on_strategy(InputValueStrategy strategy, INT32 in_size)
{
    switch (strategy)
    {
    case InputValueStrategy::FULL_ZERO:
        return static_cast<T>(0.0);

    case InputValueStrategy::MID_RANGE:
        // Normal random numbers in range [-10.0, 10.0) with 3 decimal precision
        return static_cast<T>(((rand() % 20000) / 1000.0) - 10.0);

    case InputValueStrategy::NEAR_EDGE:
        return get_near_edge_value<T>(rand(), in_size);

    case InputValueStrategy::SPECIAL_VALUES:
        return get_fp_special_value<T>(rand());

    case InputValueStrategy::TINY_VALUES_ONLY:
        return get_subnormal_and_near_underflow_value<T>(rand());

    case InputValueStrategy::LARGE_VALUES_ONLY:
        return get_near_overflow_value<T>(rand(), in_size);

    case InputValueStrategy::SPECIAL_EXCEPT_NAN:
    {
        T special_value = get_fp_special_value<T>(rand());
        if (std::isnan(special_value))
        {
            special_value = static_cast<T>(0.0);
        }
        return special_value;
    }

    default:
        return static_cast<T>(0.0);
    }
}

/**
 * @brief Analyze the output array and return comprehensive statistics
 * 
 * 
 * @tparam T data type (float32 or float64)
 * @param data pointer to data array
 * @param size number of elements to check
 * @return output_validation_stats structure containing all counts
 */
template <class T> output_validation_stats validate_output_array(T* data, UINTP size)
{
    output_validation_stats stats = {0, 0, 0, 0};
    
    for (UINTP i = 0; i < size; i++)
    {
        if (std::isnan(data[i]))
        {
            stats.nan_count++;
        }
        else if (std::isinf(data[i]))
        {
            stats.inf_count++;
        }
        else if (data[i] == static_cast<T>(0.0))
        {
            stats.zero_count++;
        }
        else
        {
            stats.nonzero_count++;
        }
    }
    
    return stats;
}

#endif // AOCLFFTZ_COMMON_GTEST_UTILS_H
