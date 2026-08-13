# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# @file CMakeLists.txt
#
# @brief CMake based build system for the library and test binary.
#
# Compiles and installs the library and test bench binary with configured
# options. Supports uninstall custom command to remove the installed files.
# Enables CTest for adding and testing Gtest and test bench related tests.
# Finds Doxygen installation and generates the documentation automatically.
#
# @author S. Biplab Raut


cmake_minimum_required(VERSION 3.26.0)

#set the project name
project(AOCL-FFTZ)
message(STATUS "Building AOCL-FFTZ LIBRARY AND TEST SUITE")
# Version used by pkg-config and CMake config
set(FFTZ_VERSION "5.3.2")

string(TIMESTAMP TODAY "%Y%m%d")
add_compile_definitions(AOCL_BUILD_VERSION="Build ${TODAY}")

message(STATUS "C Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS 7.1)
        message(FATAL_ERROR "Minimum GCC version supported is 7.1!")
    endif()
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS 8.0)
        message(FATAL_ERROR "Minimum Clang version supported is 8.0! In case of AOCC, use version 2.0 and above!")
    endif()
else()
    message(FATAL_ERROR "Compiler not supported! Library has only been tested with Clang and GCC.")
endif()

set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

option(AOCL_ENABLE_LOG "Enables logging support within the library (Disabled by default)" OFF)
set(AOCL_TEST_COVERAGE "OFF" CACHE STRING "Enables GTest and AOCL test bench based CTest suite (OFF / STANDARD / EXHAUSTIVE, default: OFF)")
set_property(CACHE AOCL_TEST_COVERAGE PROPERTY STRINGS "OFF" "STANDARD" "EXHAUSTIVE")
option(ACCURACY_WITH_DFT "Enables accuracy mode to run with DFT (Disabled by default)" OFF)
option(ASAN "Enables address sanitizer checks. Supported only on Linux Debug build (Disabled by default)" OFF)
option(BUILD_DOC "Builds documentation for library (Disabled by default)" OFF)
option(BUILD_STATIC_LIBS "Builds static library (Default build type is shared library)" OFF)
option(BUILD_THIRD_PARTY_WRAPPERS "Builds fftz third party wrapper libraries along aocl_fftz library" OFF)
option(CODE_COVERAGE "Enables source code coverage and generates coverage report. Supported only on Linux with GCC compiler (Disabled by default)" OFF)
option(CODE_COVERAGE_FOR_ATG "Enables source code coverage instrumentation for running coverage and parsing tools for use by AI Test case Generation (ATG) later. Supported only on Linux with GCC compiler (Disabled by default)" OFF)
option(ENABLE_APP_INFO_LOGS "Enables info logging for FFT problems used by the application (Independent of AOCL_ENABLE_LOG, Disabled by default)" OFF)
set(ENABLE_INSTRUCTIONS_UPTO "AVX512" CACHE STRING "Specifies maximum AVX instruction set to compile (None / AVX128 / AVX256 / AVX512, default: AVX512)")
set_property(CACHE ENABLE_INSTRUCTIONS_UPTO PROPERTY STRINGS "None" "AVX128" "AVX256" "AVX512")
option(ENABLE_FMA "Enables -ffp-contract=fast (forces FMA generation). Required for Clang/AOCC, implied by GCC at -O3 (Enabled by default)" ON)
option(ENABLE_MULTI_THREADING "Compiles library with multi-threading support using OpenMP (Disabled by default)" OFF)
option(ENABLE_STRICT_WARNINGS "Enables compiler flags to treat all warnings as errors (Enabled by default)" ON)
option(FUZZTEST "Enables Compilation of fuzz test with fuzzing mode. Supported only on Linux Debug build with Clang compiler (Disabled by default)" OFF)
set(SELECT_REAL_FFT_EXECUTION_ORDER "TRUE_RECURSION" CACHE STRING "Real FFT CT execution order: ITERATIVE (legacy Direct-first with SWAP), PARTIAL_RECURSION (recursive CT-first tree with Direct tail-chaining), TRUE_RECURSION (CT-orchestrated recurse-then-combine mirroring Complex FFT; default)")
set_property(CACHE SELECT_REAL_FFT_EXECUTION_ORDER PROPERTY STRINGS "ITERATIVE" "PARTIAL_RECURSION" "TRUE_RECURSION")
option(VALGRIND "Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)" OFF)
if(WIN32)
    set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE FILEPATH "Path to the custom OpenMP library")
else()
    set(OpenMP_libomp_LIBRARY "" CACHE FILEPATH "Path to the custom OpenMP library")
endif()

#list source files to be compiled for the library
set(CORE_PATH core)
file(GLOB API_SRC_FILES api/*.c api/*.cpp api/*.h)
#CONFIURE_DEPENDS option will be used to pick up the new added files and re-run the cmake build if necessary
file(GLOB CORE_SRC_FILES CONFIGURE_DEPENDS ${CORE_PATH}/*.c ${CORE_PATH}/*.h
                        ${CORE_PATH}/common/*.c ${CORE_PATH}/common/*.h
                        ${CORE_PATH}/kernels/*.c ${CORE_PATH}/kernels/*.h
                        ${CORE_PATH}/kernels/non_dft/transpose/*.c ${CORE_PATH}/kernels/non_dft/transpose/*.h
                        ${CORE_PATH}/kernels/non_dft/transpose/c/*.c ${CORE_PATH}/kernels/non_dft/transpose/c/*.h
                        ${CORE_PATH}/solvers/*.c ${CORE_PATH}/solvers/*.h
                        ${CORE_PATH}/solvers/complex/*.c ${CORE_PATH}/solvers/complex/*.h
                        ${CORE_PATH}/solvers/real/*.c ${CORE_PATH}/solvers/real/*.h)
file(GLOB CORE_C_FILES CONFIGURE_DEPENDS
                        ${CORE_PATH}/kernels/dft/c2c/c/*.c ${CORE_PATH}/kernels/dft/c2c/c/*.h
                        ${CORE_PATH}/kernels/non_dft/ele_mul/c/*.c
                        ${CORE_PATH}/kernels/non_dft/type_convert/c/*.c
                        ${CORE_PATH}/kernels/non_dft/perm_copy/c/*.c
                        ${CORE_PATH}/kernels/dft/r2hc/c/*.c ${CORE_PATH}/kernels/dft/r2hc/c/*.h
                        ${CORE_PATH}/kernels/dft/r2hc_fused/c/*.c ${CORE_PATH}/kernels/dft/r2hc_fused/c/*.h
                        ${CORE_PATH}/kernels/dft/tw_c2c_fwd/c/*.c
                        ${CORE_PATH}/kernels/dft/tw_c2c_bwd/c/*.c
                        ${CORE_PATH}/kernels/dft/tw_c2c_r2c/c/*.c
                        ${CORE_PATH}/kernels/dft/tw_c2c_c2r/c/*.c)
list(APPEND CORE_SRC_FILES ${CORE_C_FILES})

#Add AVX Kernels based on the flags ENABLE_AVX128, ENABLE_AVX256 & ENABLE_AVX512
if(ENABLE_INSTRUCTIONS_UPTO MATCHES "AVX512")
    set(ENABLE_AVX512 ON)
    add_compile_definitions(ENABLE_AVX512)
    file(GLOB CORE_AVX512_FILES CONFIGURE_DEPENDS
                            ${CORE_PATH}/kernels/dft/c2c/avx512/*.c
                            ${CORE_PATH}/kernels/non_dft/ele_mul/avx512/*.c
                            ${CORE_PATH}/kernels/dft/r2hc/avx512/*.c
                            ${CORE_PATH}/kernels/dft/r2hc_fused/avx512/*.c
                            ${CORE_PATH}/kernels/dft/tw_c2c_fwd/avx512/*.c
                            ${CORE_PATH}/kernels/dft/tw_c2c_bwd/avx512/*.c
                            ${CORE_PATH}/kernels/dft/tw_c2c_r2c/avx512/*.c
                            ${CORE_PATH}/kernels/dft/tw_c2c_c2r/avx512/*.c
                            ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx512.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX512_FILES})
endif()
if(ENABLE_INSTRUCTIONS_UPTO MATCHES "AVX256|AVX512")
    set(ENABLE_AVX256 ON)
    add_compile_definitions(ENABLE_AVX256)
    file(GLOB CORE_AVX256_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/dft/c2c/avx256/*.c
                                                  ${CORE_PATH}/kernels/non_dft/ele_mul/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/r2hc/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/r2hc_fused/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_fwd/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_bwd/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_r2c/avx256/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_c2r/avx256/*.c
                                                  ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx256.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX256_FILES})
endif()
if(ENABLE_INSTRUCTIONS_UPTO MATCHES "AVX128|AVX256|AVX512")
    set(ENABLE_AVX128 ON)
    add_compile_definitions(ENABLE_AVX128)
    file(GLOB CORE_AVX128_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/dft/c2c/avx128/*.c
                                                  ${CORE_PATH}/kernels/non_dft/ele_mul/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/r2hc/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/r2hc_fused/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_fwd/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_bwd/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_r2c/avx128/*.c
                                                  ${CORE_PATH}/kernels/dft/tw_c2c_c2r/avx128/*.c
                                                  ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx128.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX128_FILES})
endif()

file(GLOB SELECTOR_SRC_FILES selector/*.c selector/*.h
                             selector/complex/*.c selector/complex/*.h
                             selector/real/*.c selector/real/*.h)
file(GLOB UTILS_SRC_FILES utils/*.c utils/*.cpp utils/*.h)

#set the expected path variables
include_directories(.)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_DIR}/build/)
set(DEST_LIB_PATH lib)
set(DEST_INC_PATH include)
set(DEST_BIN_PATH bin)

set(INC_FILES api/aoclfftz.h api/types.h)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type selected is Release")
    add_compile_definitions(CFLAGS_SET1="${CMAKE_C_FLAGS_RELEASE}")
else()
    message(STATUS "Build type selected is Debug")
    add_compile_definitions(CFLAGS_SET1="${CMAKE_C_FLAGS_DEBUG}")
endif()

if (ENABLE_FMA AND (ENABLE_INSTRUCTIONS_UPTO MATCHES "None|AVX128"))
    set(ENABLE_FMA OFF)
endif()

if(ENABLE_FMA)
    add_compile_definitions(ENABLE_FMA)
endif()

#set(USER_DEBUG_ON "-g")
message(STATUS "C Compiler used: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler used: ${CMAKE_CXX_COMPILER}")
if (ENABLE_AVX128)
    message(STATUS "AVX128 Instructions: Enabled")
else()
    message(STATUS "AVX128 Instructions: Disabled")
endif()
if (ENABLE_AVX256)
    message(STATUS "AVX256 Instructions: Enabled")
else()
    message(STATUS "AVX256 Instructions: Disabled")
endif()
if (ENABLE_AVX512)
    message(STATUS "AVX512 Instructions: Enabled")
else()
    message(STATUS "AVX512 Instructions: Disabled")
endif()
if (ENABLE_FMA)
    message(STATUS "FMA instructions: Enabled")
endif()

if(WIN32)
    if(ENABLE_AVX512)
        set(AVX512_FLAG /arch:AVX512)
    endif()
    if(ENABLE_AVX256)
        set(AVX256_FLAG /arch:AVX)
    endif()
    if(ENABLE_AVX128)
        set(AVX128_FLAG /arch:AVX)
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")

elseif(UNIX)
    if(ENABLE_FMA)
        if(ENABLE_AVX512)
            set(AVX512_FLAG "-mavx512f -mavx512dq -ffp-contract=fast") # avx512dq for xor, insert and extract instructions
        endif()
        if(ENABLE_AVX256)
            set(AVX256_FLAG "-mavx -mfma -ffp-contract=fast")
        endif()
        if(ENABLE_AVX128)
            set(AVX128_FLAG "-mavx -mfma -ffp-contract=fast")
        endif()
    else()
        if(ENABLE_AVX512)
            set(AVX512_FLAG "-mavx512f -mavx512dq") # avx512dq for xor, insert and extract instructions
        endif()
        if(ENABLE_AVX256)
            set(AVX256_FLAG "-mavx")
        endif()
        if(ENABLE_AVX128)
            set(AVX128_FLAG "-mavx")
        endif()
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
endif()

include(CheckCXXCompilerFlag)
add_compile_definitions(CCompiler="${CMAKE_C_COMPILER}")
add_compile_definitions(CXXCompiler="${CMAKE_CXX_COMPILER}")
add_compile_definitions(CFLAGS_SET2="${CMAKE_C_FLAGS}")

if(ENABLE_MULTI_THREADING)
    add_compile_definitions(MULTI_THREADING)
    file(GLOB MT_SRC_FILES CONFIGURE_DEPENDS
                        ${CORE_PATH}/solvers/complex/multithread/*.c
                        ${CORE_PATH}/solvers/real/multithread/*.c)
    if(WIN32)
        message(STATUS "Custom path to look for OpenMP: ${OpenMP_libomp_LIBRARY}")
    endif()
    if(WIN32 OR NOT OpenMP_libomp_LIBRARY)
        find_package(OpenMP)
        if(OpenMP_FOUND)
            message(STATUS "Using system OpenMP")
        else()
            message(FATAL_ERROR "OpenMP Missing! Please install OpenMP to enable multi-threading")
        endif()
    else()
        message(STATUS "Custom path to look for OpenMP: ${OpenMP_libomp_LIBRARY}")
    endif()
    list(APPEND CORE_SRC_FILES ${MT_SRC_FILES})
endif()

if(ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI_FFTZ="1")
endif()

if(AOCL_ENABLE_LOG)
    add_compile_definitions(AOCL_ENABLE_LOG)
endif()

if(ENABLE_APP_INFO_LOGS)
    add_compile_definitions(ENABLE_APP_INFO_LOGS)
endif()

if(SELECT_REAL_FFT_EXECUTION_ORDER STREQUAL "ITERATIVE")
    add_compile_definitions(REAL_FFT_EXECUTION_ORDER=0)
elseif(SELECT_REAL_FFT_EXECUTION_ORDER STREQUAL "PARTIAL_RECURSION")
    add_compile_definitions(REAL_FFT_EXECUTION_ORDER=1)
elseif(SELECT_REAL_FFT_EXECUTION_ORDER STREQUAL "TRUE_RECURSION")
    add_compile_definitions(REAL_FFT_EXECUTION_ORDER=2)
else()
    message(FATAL_ERROR "Invalid SELECT_REAL_FFT_EXECUTION_ORDER='${SELECT_REAL_FFT_EXECUTION_ORDER}'. Allowed values: ITERATIVE, PARTIAL_RECURSION, TRUE_RECURSION.")
endif()
message(STATUS "Real FFT execution order: ${SELECT_REAL_FFT_EXECUTION_ORDER}")

# set library and sources
# Set target name as aocl_fftz on Linux (shared and static) and on Windows use aocl_fftz_static for static build.
if(BUILD_STATIC_LIBS)
    set(FFTZ_EXPORT_NAME aocl-fftz-static-targets)
    set(FFTZ_EXPORTED_TARGET_NAME aocl_fftz_static)
    if(WIN32)
        set(lib_name aocl_fftz_static)
    else()
        set(lib_name aocl_fftz)
    endif()
else()
    set(lib_name aocl_fftz)
    set(FFTZ_EXPORT_NAME aocl-fftz-shared-targets)
    set(FFTZ_EXPORTED_TARGET_NAME aocl_fftz)
endif()
set(SRC_FILES
        ${UTILS_SRC_FILES}
        ${API_SRC_FILES}
        ${CORE_SRC_FILES}
        ${SELECTOR_SRC_FILES}
    )
set(BENCH_DIR test/bench)
set(TEST_UTILS_DIR test/utils)
set(BENCH_UTILS_DIR test/bench/utils)

#set library dependencies (empty on Windows; libm on UNIX)
set(lib_deps "")
if(UNIX)
    list(APPEND lib_deps m) # math library
endif()

#add macro for math constants
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# NOTE: disable versioning of loops for strides.
#       Certain versions of gcc compiler use -fversion-loops-for-strides
#       optimzation flag when "-O3" is used to compile. This creates a copy of
#       a loop that uses a variable stride, which is specialized for the unit
#       stride case. Since the overhead from this code generated by this flag
#       is not desirable, we switch it off using its counterpart,
#       "-fno-version-loops-for-strides".
# NOTE: The flag needs to be applied only to the C Kernel files directory.
# Check if the compiler supports -fversion-loops-for-strides
check_cxx_compiler_flag("-fversion-loops-for-strides" COMPILER_SUPPORTS_VERSION_LOOPS_FOR_STRIDES)
if(COMPILER_SUPPORTS_VERSION_LOOPS_FOR_STRIDES)
    set_source_files_properties(${CORE_C_FILES} PROPERTIES COMPILE_OPTIONS "-fno-version-loops-for-strides")
endif()

# validate AOCL_TEST_COVERAGE values
if(NOT AOCL_TEST_COVERAGE STREQUAL "OFF"
   AND NOT AOCL_TEST_COVERAGE STREQUAL "STANDARD"
   AND NOT AOCL_TEST_COVERAGE STREQUAL "EXHAUSTIVE")
    message(FATAL_ERROR "Invalid value for AOCL_TEST_COVERAGE: '${AOCL_TEST_COVERAGE}'\nAllowed values are: OFF, STANDARD, EXHAUSTIVE.")
endif()

# validate ENABLE_INSTRUCTIONS_UPTO values
if(NOT ENABLE_INSTRUCTIONS_UPTO STREQUAL "None"
   AND NOT ENABLE_INSTRUCTIONS_UPTO STREQUAL "AVX128"
   AND NOT ENABLE_INSTRUCTIONS_UPTO STREQUAL "AVX256"
   AND NOT ENABLE_INSTRUCTIONS_UPTO STREQUAL "AVX512")
    message(FATAL_ERROR "Invalid value for ENABLE_INSTRUCTIONS_UPTO: '${ENABLE_INSTRUCTIONS_UPTO}'\nAllowed values are: None, AVX128, AVX256, AVX512.")
endif()

# setup to get source code coverage(GCOV)
if(CODE_COVERAGE OR CODE_COVERAGE_FOR_ATG)
    if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        # Switch off GCOV: target not supported
        set(CODE_COVERAGE OFF)
        set(CODE_COVERAGE_FOR_ATG OFF)
        message(WARNING "Target not supported. Code coverage is only supported on Linux OS with GCC compiler")
    endif()
endif()

# setup to run memory check using AddressSanitizer
if(ASAN)
    if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
        # Switch off ASAN: target not supported
        set(ASAN OFF)
        message(WARNING "Target not supported. ASAN is only supported on Linux builds")
    endif()
endif()

# setup to run memory check using Valgrind
if(VALGRIND)
    if((NOT (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_BUILD_TYPE MATCHES "Debug")) OR ASAN)
        set(VALGRIND OFF)
        message(WARNING "Target not supported. VALGRIND is only supported with Linux/Debug build and incompatible with ASAN=ON")
    endif()
endif()

# setup to run fuzztest
if(FUZZTEST)
    if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_BUILD_TYPE STREQUAL "Debug") OR WIN32)
        # Switch off FUZZTEST: target not supported
        set(FUZZTEST OFF)
        message(WARNING "Target not supported. Fuzz testing is supported only in Linux with Clang compiler and in Debug build mode")
    endif()
endif()

# Setting AOCL_TEST_COVERAGE to STANDARD to enable CTests and GTests for VALGRIND / CODE_COVERAGE / FUZZTEST
if((VALGRIND OR CODE_COVERAGE OR CODE_COVERAGE_FOR_ATG OR FUZZTEST) AND (AOCL_TEST_COVERAGE STREQUAL "OFF"))
    set(AOCL_TEST_COVERAGE "STANDARD")
endif()

message(STATUS ASAN : ${ASAN})
message(STATUS VALGRIND : ${VALGRIND})
message(STATUS CODE COVERAGE : ${CODE_COVERAGE})
message(STATUS CODE COVERAGE ATG : ${CODE_COVERAGE_FOR_ATG})
message(STATUS AOCL_ENABLE_LOG : ${AOCL_ENABLE_LOG})
message(STATUS FUZZTEST : ${FUZZTEST})
message(STATUS MULTI-THREADING : ${ENABLE_MULTI_THREADING})

if(ACCURACY_WITH_DFT)
    add_compile_definitions(ENABLE_DFT_REFERENCE)
    message(STATUS "DFT Reference property tests enabled")
endif()

#setup CTest for GoogleTest and test bench
if(NOT AOCL_TEST_COVERAGE STREQUAL "OFF")
    # enabling ctest
    enable_testing()
    set(GTEST_DIR test/gtest)
    add_subdirectory(${GTEST_DIR})
    add_subdirectory(${BENCH_DIR})

    # Set test mode for bench directory
    if(AOCL_TEST_COVERAGE STREQUAL "EXHAUSTIVE")
        message(STATUS "AOCL_TEST_COVERAGE: EXHAUSTIVE - Runs both TEST_SUITE and TEST_SUITE_EXHAUSTIVE")
    elseif(AOCL_TEST_COVERAGE STREQUAL "STANDARD")
        message(STATUS "AOCL_TEST_COVERAGE: STANDARD - Using TEST_SUITE only")
    endif()
else()
    message(STATUS "AOCL_TEST_COVERAGE: OFF - Tests disabled")
endif()

# compile kernel files with only the relevant simd flags.
#
# By default, the properties set are visible only to source files in the current folder
# To make it visible to files across folders,'DIRECTORY' option is used
if(ENABLE_AVX512)
    set_source_files_properties(${CORE_AVX512_FILES} DIRECTORY "${ROOT_DIR}" "${GTEST_DIR}"
                                PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
endif()
if(ENABLE_AVX256)
    set_source_files_properties(${CORE_AVX256_FILES} DIRECTORY "${ROOT_DIR}" "${GTEST_DIR}"
                                PROPERTIES COMPILE_FLAGS "${AVX256_FLAG}")
endif()
if(ENABLE_AVX128)
    set_source_files_properties(${CORE_AVX128_FILES} DIRECTORY "${ROOT_DIR}" "${GTEST_DIR}"
                                PROPERTIES COMPILE_FLAGS "${AVX128_FLAG}")
endif()

#Enabling strict warnings
if(ENABLE_STRICT_WARNINGS)
    if(WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
        set_property(SOURCE ${SRC_FILES} APPEND PROPERTY COMPILE_OPTIONS "-DENABLE_STRICT_WARNINGS")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wpedantic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wpedantic")
        set_property(SOURCE ${SRC_FILES} APPEND PROPERTY COMPILE_OPTIONS "-DENABLE_STRICT_WARNINGS")
    endif()
endif()

# ---------------------------------------------------------------------------
# OBJECT library: compiles all FFTZ sources and exposes the public API headers
# via target_include_directories PUBLIC.  A parent project can aggregate the
# object files into any library using:
#   add_library(<<PARENT>> SHARED $<TARGET_OBJECTS:fftz_objects>)
#   target_link_libraries(<<PARENT>> PUBLIC fftz_objects)
# ---------------------------------------------------------------------------
add_library(fftz_objects OBJECT ${SRC_FILES})
target_include_directories(fftz_objects PUBLIC .)

# Build object files as PIC so they can be aggregated into either a SHARED
# or STATIC library by a parent project, regardless of how this project's
# own ${lib_name} is configured. Required for ELF platforms.
# No-op on Windows (CMake silently ignores POSITION_INDEPENDENT_CODE there).
set_target_properties(fftz_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Standalone library built from the OBJECT target above.
# Used when aocl-fftz is consumed directly (not aggregated by a parent project).
if(BUILD_STATIC_LIBS)
    add_library(${lib_name} STATIC $<TARGET_OBJECTS:fftz_objects>)
    message(STATUS "Library type being built is: Static")
else()
    add_library(${lib_name} SHARED $<TARGET_OBJECTS:fftz_objects>)
    message(STATUS "Library type being built is: Shared")
endif()

# ${lib_name}'s own usage requirements (include dir, libm, OpenMP, coverage,
# ASAN) are set on ${lib_name} directly below; fftz_objects carries the same
# as INTERFACE / PUBLIC props for parent-project aggregation via
# $<TARGET_OBJECTS:fftz_objects> + target_link_libraries(parent PUBLIC fftz_objects).

# Expose the FFTZ public headers to downstream consumers of ${lib_name}.
# BUILD_INTERFACE: header dir inside the source tree during build.
# INSTALL_INTERFACE: relative path under CMAKE_INSTALL_PREFIX after install.
target_include_directories(${lib_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${DEST_INC_PATH}>
)

# AOCL Build-It-Yourself unified build: compile the same sources into a sibling
# OBJECT library whose objects are pulled directly into the single libaocl.
if(AOCL_TB_UNIFIED_BUILD)
    aocl_tb_objectify(${lib_name})
endif()

set_target_properties(${lib_name} PROPERTIES EXPORT_NAME ${FFTZ_EXPORTED_TARGET_NAME})

# Link system deps like libm: PUBLIC for static (consumers need -lm), PRIVATE for shared.
# Also expose them as INTERFACE on fftz_objects so a parent project that aggregates
# $<TARGET_OBJECTS:fftz_objects> into its own library inherits them via
# target_link_libraries(parent PUBLIC fftz_objects).
if(lib_deps)
    target_link_libraries(fftz_objects INTERFACE ${lib_deps})
    if(BUILD_STATIC_LIBS)
        target_link_libraries(${lib_name} PUBLIC ${lib_deps})
    else()
        target_link_libraries(${lib_name} PRIVATE ${lib_deps})
    endif()
endif()

# Add required OpenMP flags
if(ENABLE_MULTI_THREADING)
    if(WIN32)
        # PUBLIC on the OBJECT lib propagates OpenMP::OpenMP_C/CXX as INTERFACE
        # link items to consumers (parent aggregation case).
        target_link_libraries(fftz_objects PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        if(BUILD_STATIC_LIBS)
            # Static libraries must link OpenMP PUBLIC so that apps that link us get the runtime too.
            target_link_libraries(${lib_name} PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(${lib_name} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        endif()
    else()
        # On Unix: For custom path, link that file + -fopenmp and for system OpenMP, link OpenMP::OpenMP_C/OpenMP_CXX
        # pkg-config uses -lomp/-lgomp in Libs.private.
        if(OpenMP_libomp_LIBRARY)
            # PUBLIC compile option so fftz_objects' own sources get -fopenmp AND
            # the flag is propagated as INTERFACE_COMPILE_OPTIONS to consumers.
            target_compile_options(fftz_objects PUBLIC -fopenmp)
            # INTERFACE link options/libs on fftz_objects so parent aggregations
            # get -fopenmp on the link line and the libomp runtime library.
            target_link_options(fftz_objects INTERFACE -fopenmp)
            target_link_libraries(fftz_objects INTERFACE ${OpenMP_libomp_LIBRARY})
            if(BUILD_STATIC_LIBS)
                target_link_options(${lib_name} PUBLIC -fopenmp)
                target_link_libraries(${lib_name} PUBLIC ${OpenMP_libomp_LIBRARY})
            else()
                target_link_options(${lib_name} PRIVATE -fopenmp)
                target_link_libraries(${lib_name} PRIVATE ${OpenMP_libomp_LIBRARY})
            endif()
        else()
            # System OpenMP: PUBLIC on fftz_objects so OpenMP::OpenMP_C/CXX is
            # propagated to consumers (compile flags + runtime link).
            target_link_libraries(fftz_objects PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
            if(BUILD_STATIC_LIBS)
                target_link_libraries(${lib_name} PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
            else()
                target_link_libraries(${lib_name} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
            endif()
        endif()
    endif()
endif()

# add the compiler and linker flags for code coverage
# Compile flags go on fftz_objects (PUBLIC, propagated to consumers).
# Link flags go on both ${lib_name} (PUBLIC, for direct consumers via find_package)
# and fftz_objects (INTERFACE, for parent aggregation of $<TARGET_OBJECTS:fftz_objects>).
if(CODE_COVERAGE OR CODE_COVERAGE_FOR_ATG)
    target_compile_options(fftz_objects PUBLIC --coverage)
    check_cxx_compiler_flag("-fprofile-abs-path" HAS_FPROFILE_ABS_PATH)
    if(HAS_FPROFILE_ABS_PATH)
        target_compile_options(fftz_objects PUBLIC -fprofile-abs-path)
    endif()
    target_link_options(fftz_objects INTERFACE --coverage)
    target_link_options(${lib_name} PUBLIC --coverage)
endif()

# add the compiler and linker flags for AddressSanitizer
if(ASAN)
    target_compile_options(fftz_objects PUBLIC -fsanitize=address -O0 -gdwarf-5 -g3)
    target_link_options(fftz_objects INTERFACE -fsanitize=address)
    target_link_options(${lib_name} PUBLIC -fsanitize=address)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#add test bench executable
set(exe_name aocl_fftz_bench)
file(GLOB BENCH_SRC_FILES ${BENCH_DIR}/*.c ${BENCH_DIR}/*.h ${TEST_UTILS_DIR}/dims_vecs_helper.* ${BENCH_UTILS_DIR}/*.c ${BENCH_UTILS_DIR}/*.h)
if(WIN32)
    message(STATUS "C Compiler flags used for this windows build are: ${CMAKE_C_FLAGS}")
    message(STATUS "C++ Compiler flags used for this windows build are: ${CMAKE_CXX_FLAGS}")
    add_executable(${exe_name} ${BENCH_SRC_FILES})
else()
    message(STATUS "C Compiler flags used for this Unix build are: ${CMAKE_C_FLAGS}")
    message(STATUS "C++ Compiler flags used for this Unix build are: ${CMAKE_CXX_FLAGS}")
    add_executable(${exe_name} ${BENCH_SRC_FILES})
endif()
target_link_libraries(${exe_name} ${lib_name})
target_link_libraries(${exe_name} ${CMAKE_DL_LIBS})
if(lib_deps)
    target_link_libraries(${exe_name} ${lib_deps})
endif()

# Set rpath so the executable finds the shared library in all scenarios:
# - Build directory: ./build/aocl_fftz_bench finds ./build/libaocl_fftz.so
# - Install directory: bin/aocl_fftz_bench finds ../lib/libaocl_fftz.so
if(UNIX AND NOT BUILD_STATIC_LIBS)
    set_target_properties(${exe_name} PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN/../${DEST_LIB_PATH}"
    )
endif()

#Install into destination path by using -DCMAKE_INSTALL_PREFIX=<path> while configuring cmake
message(STATUS "AOCL-FFTZ Library, Interface header files and Test binary will be installed at ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${exe_name} DESTINATION ${DEST_BIN_PATH})
install(TARGETS ${lib_name} EXPORT ${FFTZ_EXPORT_NAME}
    DESTINATION ${DEST_LIB_PATH}
    INCLUDES DESTINATION ${DEST_INC_PATH}
)
install(FILES ${INC_FILES} DESTINATION ${DEST_INC_PATH})

# CMake package config files
include(CMakePackageConfigHelpers)

set(FFTZ_INSTALL_CMAKEDIR "${DEST_LIB_PATH}/cmake/aocl-fftz")
# Config decides at package-load time: find_dependency(OpenMP) when aocl-fftz::aocl_fftz_static has OpenMP in its interface.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aocl-fftz-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/aocl-fftz-config.cmake"
    INSTALL_DESTINATION "${FFTZ_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aocl-fftz-config-version.cmake"
    VERSION "${FFTZ_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/aocl-fftz-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aocl-fftz-config-version.cmake"
    DESTINATION "${FFTZ_INSTALL_CMAKEDIR}"
)

install(EXPORT ${FFTZ_EXPORT_NAME}
    NAMESPACE aocl-fftz::
    DESTINATION "${FFTZ_INSTALL_CMAKEDIR}"
)

# Install .pc files so pkg-config works. CMake projects can use find_package(aocl-fftz) as well.
set(FFTZ_DESCRIPTION "AOCL-FFTZ: high-performance FFT library")
if (BUILD_THIRD_PARTY_WRAPPERS)
    string(APPEND FFTZ_DESCRIPTION " (build includes FFTW-compatible wrappers; see fftw3.pc / fftw3f.pc / fftw3_omp.pc / fftw3f_omp.pc)")
endif()
if(WIN32 AND BUILD_STATIC_LIBS)
    set(FFTZ_LIBS "-laocl_fftz_static")
else()
    set(FFTZ_LIBS "-laocl_fftz")
endif()
# Populate Libs.private in the .pc file with the same libs we actually link (e.g. -lm, -fopenmp
# or custom OpenMP path). That way pkg-config gives downstream the correct link line.
set(FFTZ_LIBS_PRIVATE "")

# On Unix, we must explicitly link libm.
if(UNIX)
    string(APPEND FFTZ_LIBS_PRIVATE " -lm")
endif()

# Add OpenMP to Libs.private only (not Cflags/Libs). If user set a custom lib path, use that
# so downstream links the same runtime we did, and add -fopenmp so pkg-config --static
# consumers get a complete link line. Otherwise use what FindOpenMP selected so
# -lomp vs -lgomp matches the actual link interface.
if(ENABLE_MULTI_THREADING)
    if(NOT WIN32)
        if(OpenMP_libomp_LIBRARY)
            get_filename_component(OPENMP_LIB_REAL "${OpenMP_libomp_LIBRARY}" REALPATH)
            string(APPEND FFTZ_LIBS_PRIVATE " ${OPENMP_LIB_REAL}")
            if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
                string(APPEND FFTZ_LIBS_PRIVATE " -fopenmp")
            endif()
        else()
            if(DEFINED OpenMP_C_LIB_NAMES AND OpenMP_C_LIB_NAMES)
                foreach(_omp_lib ${OpenMP_C_LIB_NAMES})
                    string(APPEND FFTZ_LIBS_PRIVATE " -l${_omp_lib}")
                endforeach()
            else()
                if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    string(APPEND FFTZ_LIBS_PRIVATE " -lomp")
                else()
                    string(APPEND FFTZ_LIBS_PRIVATE " -lgomp")
                endif()
            endif()
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        string(APPEND FFTZ_LIBS_PRIVATE " -fopenmp")
    endif()
endif()
string(STRIP "${FFTZ_LIBS_PRIVATE}" FFTZ_LIBS_PRIVATE)
# Generate and install fftz.pc.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftz.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/fftz.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fftz.pc DESTINATION ${DEST_LIB_PATH}/pkgconfig)

# When BUILD_THIRD_PARTY_WRAPPERS is ON, generate and install FFTW-compatible
# pkg-config files (fftw3.pc, fftw3f.pc, fftw3_omp.pc, fftw3f_omp.pc) so downstreams
# can link against AOCL-FFTZ as if it were FFTW.
if(BUILD_THIRD_PARTY_WRAPPERS)
    set(aoclfftz_fftw3_pc_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw3.pc.in")
    set(aoclfftz_fftw3f_pc_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw3f.pc.in")
    set(aoclfftz_fftw3_omp_pc_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw3_omp.pc.in")
    set(aoclfftz_fftw3f_omp_pc_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw3f_omp.pc.in")
    if(EXISTS "${aoclfftz_fftw3_pc_in}" AND EXISTS "${aoclfftz_fftw3f_pc_in}"
            AND EXISTS "${aoclfftz_fftw3_omp_pc_in}" AND EXISTS "${aoclfftz_fftw3f_omp_pc_in}")
        configure_file(
            ${aoclfftz_fftw3_pc_in}
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3.pc
            @ONLY
        )
        configure_file(
            ${aoclfftz_fftw3f_pc_in}
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3f.pc
            @ONLY
        )
        configure_file(
            ${aoclfftz_fftw3_omp_pc_in}
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3_omp.pc
            @ONLY
        )
        configure_file(
            ${aoclfftz_fftw3f_omp_pc_in}
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3f_omp.pc
            @ONLY
        )
        install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3.pc
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3f.pc
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3_omp.pc
            ${CMAKE_CURRENT_BINARY_DIR}/fftw3f_omp.pc
            DESTINATION ${DEST_LIB_PATH}/pkgconfig
        )
    else()
        message(WARNING
            "FFTW wrapper pkg-config templates not found (fftw3.pc.in, fftw3f.pc.in, "
            "fftw3_omp.pc.in, fftw3f_omp.pc.in); FFTW .pc files will not be installed. "
            "Wrappers will still be built."
        )
    endif()
endif()

if((NOT BUILD_STATIC_LIBS) AND WIN32 AND (NOT AOCL_TEST_COVERAGE STREQUAL "OFF"))
    add_custom_command(
        TARGET ${lib_name}
        POST_BUILD
        COMMENT "Copy DLL to GTest executables path"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> ${CMAKE_BINARY_DIR}/${GTEST_DIR}
        )
endif()

# Uninstall the binary, library and header files using custom command.
# Guarded: when built as part of the unified AOCL project, another component
# may already define a global "uninstall" target.
if(NOT TARGET uninstall)
    add_custom_target("uninstall" COMMENT "Uninstalling the installed files from ${CMAKE_INSTALL_PREFIX}")
    add_custom_command(
        TARGET "uninstall"
        POST_BUILD
        COMMENT "Uninstall the installed files as per install_manifest.txt and from ${ROOT_DIR}/lib/"
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for install_manifest.txt..."
        COMMAND [ -f install_manifest.txt ] && xargs rm -vf < install_manifest.txt || echo 'install_manifest.txt is not present, nothing to uninstall, install the library!'
        )
endif()

# copy library from build directory to lib directory
add_custom_command(
    TARGET ${lib_name}
    POST_BUILD
    COMMENT "Copy library to ${ROOT_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${ROOT_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ROOT_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> ${ROOT_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:${lib_name}> ${ROOT_DIR}/lib
    )

if (BUILD_DOC)
    add_subdirectory(docs)
endif()

if(BUILD_THIRD_PARTY_WRAPPERS)
    set(FFTZ_LIB_TARGET ${lib_name})
    add_subdirectory(wrappers/fftw)
endif()
