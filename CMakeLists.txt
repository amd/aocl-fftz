# Copyright (C) 2023-2025 Advanced Micro Devices. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

option(AOCL_ENABLE_LOG "Enable logging support within the library (Disabled by default)" OFF)
set(AOCL_TEST_COVERAGE "OFF" CACHE STRING "Enable GTest and AOCL test bench based CTest suite (OFF / STANDARD / EXHAUSTIVE, default: OFF)")
set_property(CACHE AOCL_TEST_COVERAGE PROPERTY STRINGS "OFF" "STANDARD" "EXHAUSTIVE")
option(ACCURACY_WITH_DFT "Enables accuracy mode to run with DFT (Disabled by default)" OFF)
option(ASAN "Enables address sanitizer checks. Supported only on Linux Debug build (Disabled by default)" OFF)
option(BUILD_DOC "Build documentation for library (Disabled by default)" OFF)
option(BUILD_STATIC_LIBS "Build static library (Default build type is shared library)" OFF)
option(CODE_COVERAGE "Enables source code coverage and generates coverage report. Supported only on Linux with GCC compiler (Disabled by default)" OFF)
option(ENABLE_AVX128 "Compiles library with AVX 128-bit kernels support (Enabled by default)" ON)
option(ENABLE_AVX256 "Compiles library with AVX 256-bit kernels support (Enabled by default)" ON)
option(ENABLE_AVX512 "Compiles library with AVX 512-bit kernels support (Enabled by default)" ON)
option(ENABLE_FMA "Enable -ffp-contract=fast (forces FMA generation). Required for Clang/AOCC, implied by GCC at -O3 (Enabled by default)" ON)
option(ENABLE_MULTI_THREADING "Compiles library with multi-threading support using OpenMP (Disabled by default)" OFF)
#option(ENABLE_MPI "Compiles library with MPI support (Disabled by default)" OFF)
option(ENABLE_STRICT_WARNINGS "Enable compiler flags to treat all warnings as errors (Enabled by default)" ON)
option(FUZZTEST "Enable Compilation of fuzz test with fuzzing mode. Supported only on Linux Debug build with Clang compiler (Disabled by default)" OFF)
option(VALGRIND "Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)" OFF)
option(BUILD_THIRD_PARTY_WRAPPERS "Build fftz third party wrapper libraries along aocl_fftz library" OFF)

#list source files to be compiled for the library
set(CORE_PATH core)
file(GLOB API_SRC_FILES api/*.c api/*.cpp api/*.h)
#CONFIURE_DEPENDS option will be used to pick up the new added files and re-run the cmake build if necessary
file(GLOB CORE_SRC_FILES CONFIGURE_DEPENDS ${CORE_PATH}/*.c ${CORE_PATH}/*.h
                        ${CORE_PATH}/common/*.c ${CORE_PATH}/common/*.h
                        ${CORE_PATH}/kernels/*.c ${CORE_PATH}/kernels/*.h
                        ${CORE_PATH}/kernels/transpose/*.c ${CORE_PATH}/kernels/transpose/*.h
                        ${CORE_PATH}/kernels/transpose/c/*.c ${CORE_PATH}/kernels/transpose/c/*.h
                        ${CORE_PATH}/solvers/*.c ${CORE_PATH}/solvers/*.h
                        ${CORE_PATH}/solvers/complex/*.c ${CORE_PATH}/solvers/complex/*.h
                        ${CORE_PATH}/solvers/real/*.c ${CORE_PATH}/solvers/real/*.h)
file(GLOB CORE_C_FILES CONFIGURE_DEPENDS
                        ${CORE_PATH}/kernels/c2c/c/*.c ${CORE_PATH}/kernels/c2c/c/*.h
                        ${CORE_PATH}/kernels/r2hc/c/*.c ${CORE_PATH}/kernels/r2hc/c/*.h
                        ${CORE_PATH}/kernels/r2hc_fused/c/*.c ${CORE_PATH}/kernels/r2hc_fused/c/*.h
                        ${CORE_PATH}/kernels/tw_c2c/c/*.c)
list(APPEND CORE_SRC_FILES ${CORE_C_FILES})

#Add AVX Kernels based on the flags ENABLE_AVX128, ENABLE_AVX256 & ENABLE_AVX512
if(ENABLE_AVX512)
    set(ENABLE_AVX256 ON)
    add_compile_definitions(ENABLE_AVX512)
    file(GLOB CORE_AVX512_FILES CONFIGURE_DEPENDS
                            ${CORE_PATH}/kernels/c2c/avx512/*.c
                            ${CORE_PATH}/kernels/r2hc/avx512/*.c
                            ${CORE_PATH}/kernels/r2hc_fused/avx512/*.c
                            ${CORE_PATH}/kernels/tw_c2c/avx512/*.c
                            ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx512.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX512_FILES})
endif()
if(ENABLE_AVX256)
    set(ENABLE_AVX128 ON)
    add_compile_definitions(ENABLE_AVX256)
    file(GLOB CORE_AVX256_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/c2c/avx256/*.c
                                                  ${CORE_PATH}/kernels/r2hc/avx256/*.c
                                                  ${CORE_PATH}/kernels/r2hc_fused/avx256/*.c
                                                  ${CORE_PATH}/kernels/tw_c2c/avx256/*.c
                                                  ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx256.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX256_FILES})
endif()
if(ENABLE_AVX128)
    add_compile_definitions(ENABLE_AVX128)
    file(GLOB CORE_AVX128_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/c2c/avx128/*.c
                                                  ${CORE_PATH}/kernels/r2hc/avx128/*.c
                                                  ${CORE_PATH}/kernels/r2hc_fused/avx128/*.c
                                                  ${CORE_PATH}/kernels/tw_c2c/avx128/*.c
                                                  ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx128.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX128_FILES})
endif()

file(GLOB SELECTOR_SRC_FILES selector/*.c selector/*.h
                             selector/complex/*.c selector/complex/*.h
                             selector/real/*.c selector/real/*.h)
file(GLOB THREADS_SRC_FILES threads/*.c threads/*.h)
file(GLOB UTILS_SRC_FILES utils/*.c utils/*.cpp utils/*.h)

#set the expected path variables
include_directories(.)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_DIR}/build/)
set(DEST_LIB_PATH aocl_fftz/lib)
set(DEST_INC_PATH aocl_fftz/include)
set(DEST_BIN_PATH aocl_fftz/bin)

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

#set(USER_DEBUG_ON "-g")
message(STATUS "C Compiler used: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler used: ${CMAKE_CXX_COMPILER}")
message(STATUS ENABLE_AVX128 : ${ENABLE_AVX128})
message(STATUS ENABLE_AVX256 : ${ENABLE_AVX256})
message(STATUS ENABLE_AVX512 : ${ENABLE_AVX512})

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
    if(ENABLE_AVX512)
        if(ENABLE_FMA)
            set(AVX512_FLAG "-mavx512f -mavx512dq -ffp-contract=fast") # avx512dq for xor, insert and extract instructions
        else()
            set(AVX512_FLAG "-mavx512f -mavx512dq") # avx512dq for xor, insert and extract instructions
        endif()
    endif()
    if(ENABLE_AVX256)
        if(ENABLE_FMA)
            set(AVX256_FLAG "-mavx -mfma -ffp-contract=fast")
        else()
            set(AVX256_FLAG "-mavx -mfma")
        endif()
    endif()
    if(ENABLE_AVX128)
        set(AVX128_FLAG "-mavx")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
endif()

include(CheckCXXCompilerFlag)
add_compile_definitions(CCompiler="${CMAKE_C_COMPILER}")
add_compile_definitions(CXXCompiler="${CMAKE_CXX_COMPILER}")
add_compile_definitions(CFLAGS_SET2="${CMAKE_C_FLAGS}")

if(ENABLE_MULTI_THREADING)
    file(GLOB MT_SRC_FILES CONFIGURE_DEPENDS
                        ${CORE_PATH}/solvers/complex/multithread/*.c
                        ${CORE_PATH}/solvers/real/multithread/*.c)
    if (WIN32)
	set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE STRING "OpenMP library path")
	message(STATUS "OpenMP custom path to look for OpenMP: ${OpenMP_libomp_LIBRARY}")
    endif()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message(STATUS "OpenMP found")
        add_compile_definitions(MULTI_THREADING)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    else()
	message(FATAL_ERROR "OpenMP Missing! Please install OpenMP to enable multi-threading")
    endif()
    list(APPEND CORE_SRC_FILES ${MT_SRC_FILES})
endif()

if(ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI_FFTZ="1")
endif()

if(AOCL_ENABLE_LOG)
    add_compile_definitions(AOCL_ENABLE_LOG)
endif()

#set library and sources
set(lib_name aocl_fftz)
set(SRC_FILES
        ${UTILS_SRC_FILES}
        ${API_SRC_FILES}
        ${CORE_SRC_FILES}
        ${SELECTOR_SRC_FILES}
    )
set(BENCH_DIR test/bench)
set(TEST_UTILS_DIR test/utils)
set(BENCH_UTILS_DIR test/bench/utils)

#set library dependencies
if(UNIX)
    set(lib_deps m) #math library
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

# setup to get source code coverage(GCOV)
if(CODE_COVERAGE)
    if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        # Switch off GCOV: target not supported
        set(CODE_COVERAGE OFF)
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
if((VALGRIND OR CODE_COVERAGE OR FUZZTEST) AND (AOCL_TEST_COVERAGE STREQUAL "OFF"))
    set(AOCL_TEST_COVERAGE "STANDARD")
endif()

message(STATUS ASAN : ${ASAN})
message(STATUS VALGRIND : ${VALGRIND})
message(STATUS CODE COVERAGE : ${CODE_COVERAGE})
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

# add library
if(BUILD_STATIC_LIBS)
    add_library(${lib_name} STATIC ${SRC_FILES})
    message(STATUS "Library type being built is: Static")
else()
    add_library(${lib_name} SHARED ${SRC_FILES})
    message(STATUS "Library type being built is: Shared")
endif()

#add required OpenMP flags
if(ENABLE_MULTI_THREADING)
    if(WIN32)
        target_link_libraries(${lib_name} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    else()
        target_link_options(${lib_name} PUBLIC -fopenmp)
    endif()
endif()

# add the compiler and linker flags for code coverage
if(CODE_COVERAGE)
    target_compile_options(${lib_name} PUBLIC --coverage)
    # Check if the compiler supports -fprofile-abs-path
    check_cxx_compiler_flag("-fprofile-abs-path" HAS_FPROFILE_ABS_PATH)
    if(HAS_FPROFILE_ABS_PATH)
        target_compile_options(${lib_name} PUBLIC -fprofile-abs-path)
    endif()
    target_link_options(${lib_name} PUBLIC --coverage)
endif()

# add the compiler and linker flags for AddressSanitizer
if(ASAN)
    target_compile_options(${lib_name} PUBLIC -fsanitize=address -O0 -gdwarf-5 -g3 )
    target_link_options(${lib_name} PUBLIC -fsanitize=address)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${DEST_LIB_PATH}")

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
target_link_libraries(${exe_name} ${lib_deps})

#Install into destination path by using -DCMAKE_INSTALL_PREFIX=<path> while configuring cmake
message(STATUS "AOCL-FFTZ Library, Interface header files and Test binary will be installed at ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${exe_name} DESTINATION ${DEST_BIN_PATH})
install(TARGETS ${lib_name} DESTINATION ${DEST_LIB_PATH})
install(FILES ${INC_FILES} DESTINATION ${DEST_INC_PATH})
if((NOT BUILD_STATIC_LIBS) AND WIN32 AND (NOT AOCL_TEST_COVERAGE STREQUAL "OFF"))
    add_custom_command(
        TARGET ${lib_name}
        POST_BUILD
        COMMENT "Copy DLL to GTest executables path"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> ${CMAKE_BINARY_DIR}/${GTEST_DIR}
        )
endif()

#Uninstall the binary. library and header files using custom command
add_custom_target("uninstall" COMMENT "Uninstalling the installed files from ${install_dir}")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall the installed files as per install_manifest.txt and from ${ROOT_DIR}/lib/"
    COMMAND ${CMAKE_COMMAND} -E echo "Checking for install_manifest.txt..."
    COMMAND [ -f install_manifest.txt ] && xargs rm -vf < install_manifest.txt || echo 'install_manifest.txt is not present, nothing to uninstall, install the library!'
    )

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
    add_subdirectory(wrappers/fftw)
endif()
