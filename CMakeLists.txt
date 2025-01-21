# Copyright (C) 2023-2024 Advanced Micro Devices. All rights reserved.
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


cmake_minimum_required(VERSION 3.10.0)

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

option(BUILD_STATIC_LIBS "Build static library (Default build type is shared library)" OFF)
option(ENABLE_AVX128 "Compile with AVX 128-bit instruction set support" OFF)
option(ENABLE_AVX256 "Compile with AVX 256-bit instruction set support" OFF)
option(ENABLE_AVX512 "Compile with AVX512 instruction set support" OFF)
#option(ENABLE_MPI "Compile with distributed MPI FFTZ support" OFF)
option(AOCL_TEST_COVERAGE "Enable GTest and AOCL test bench based CTest suite (Disabled by default)" OFF)
option(CODE_COVERAGE "Enable source code coverage (only on Linux with GCC compiler)" OFF)
option(ASAN "Enable Address Sanitizer checks (only on Linux/Debug build)" OFF)
option(VALGRIND "Enable Valgrind checks (only on Linux/Debug. Incompatible with ASAN=ON)" OFF)
option(BUILD_DOC "Build documentation for aocl-fftz (Disabled by default)" OFF)
option(ENABLE_STRICT_WARNINGS "Enable compiler flags to treat all warnings as errors (Enabled by default)" ON)
option(AOCL_ENABLE_LOG "Enable logging support within the library (Disabled by default)" OFF)
option(FUZZTEST "Enable Compilation of fuzz test with fuzzing mode (only on Clang/Debug, Disabled by default)" OFF)
option(ENABLE_MULTI_THREADING "Compile with multi-threading support using OpenMP (Disabled by default)" OFF)

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
                        ${CORE_PATH}/kernels/r2hc_fused/c/*.c ${CORE_PATH}/kernels/r2hc_fused/c/*.h)
list(APPEND CORE_SRC_FILES ${CORE_C_FILES})

#Add AVX Kernels based on the flags ENABLE_AVX128, ENABLE_AVX256 & ENABLE_AVX512
if(ENABLE_AVX512)
    set(ENABLE_AVX256 ON)
    add_compile_definitions(ENABLE_AVX512)
    file(GLOB CORE_AVX512_FILES CONFIGURE_DEPENDS
                            ${CORE_PATH}/kernels/c2c/avx512/*.c
                            ${CORE_PATH}/kernels/r2hc/avx512/*.c
                            ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx512.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX512_FILES})
endif()
if(ENABLE_AVX256)
    set(ENABLE_AVX128 ON)
    add_compile_definitions(ENABLE_AVX256)
    file(GLOB CORE_AVX256_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/c2c/avx256/*.c
                                                  ${CORE_PATH}/kernels/r2hc/avx256/*.c
                                                  ${CORE_PATH}/kernels/r2hc_fused/avx256/*.c
                                                  ${CORE_PATH}/kernels/simd_includes/r2hc_simd_avx256.h)
    list(APPEND CORE_SRC_FILES ${CORE_AVX256_FILES})
endif()
if(ENABLE_AVX128)
    add_compile_definitions(ENABLE_AVX128)
    file(GLOB CORE_AVX128_FILES CONFIGURE_DEPENDS ${CORE_PATH}/kernels/c2c/avx128/*.c
                                                  ${CORE_PATH}/kernels/r2hc/avx128/*.c
                                                  ${CORE_PATH}/kernels/r2hc_fused/avx128/*.c
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
    if(ENABLE_AVX128 OR ENABLE_AVX256)
        set(AVX_FLAG /arch:AVX)
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")

elseif(UNIX)
    if(ENABLE_AVX512)
        set(AVX512_FLAG "-mavx512f -mavx512dq") # avx512dq for xor, insert and extract instructions
    endif()
    if(ENABLE_AVX128 OR ENABLE_AVX256)
        set(AVX_FLAG -mavx)
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
                        ${CORE_PATH}/solvers/complex/multithread/*.c)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        message(STATUS "OpenMP found")
        add_compile_definitions(MULTI_THREADING)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    else()
        message(FATAL_ERROR "OpenMP Missing! Please install OpenMP to enable mult-threading")
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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)
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
    if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_BUILD_TYPE MATCHES "Debug"))
        # Switch off ASAN: target not supported
        set(ASAN OFF)
        message(WARNING "Target not supported. ASAN is only supported on Linux/Debug build")
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

# Enable CTest for any of the cases
if(VALGRIND OR ASAN OR CODE_COVERAGE OR FUZZTEST)
    set(AOCL_TEST_COVERAGE ON)
endif()

message(STATUS ASAN : ${ASAN})
message(STATUS VALGRIND : ${VALGRIND})
message(STATUS CODE COVERAGE : ${CODE_COVERAGE})
message(STATUS AOCL_ENABLE_LOG : ${AOCL_ENABLE_LOG})
message(STATUS FUZZTEST : ${FUZZTEST})
message(STATUS MULTI-THREADING : ${ENABLE_MULTI_THREADING})

#setup CTest for GoogleTest and test bench
if(AOCL_TEST_COVERAGE)
    # enabling ctest
    enable_testing()
    set(GTEST_DIR test/gtest)
    add_subdirectory(${GTEST_DIR})
    add_subdirectory(${BENCH_DIR})
endif()

# compile kernel files with only the relevant simd flags.
#
# By default, the properties set are visible only to source files in the current folder
# To make it visible to files across folders,'DIRECTORY' option is used
if(ENABLE_AVX512)
    set_source_files_properties(${CORE_AVX512_FILES} DIRECTORY "${ROOT_DIR}" "${GTEST_DIR}"
                                PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
endif()
if(ENABLE_AVX128 OR ENABLE_AVX256)
    set_source_files_properties(${CORE_AVX128_FILES} ${CORE_AVX256_FILES} DIRECTORY "${ROOT_DIR}" "${GTEST_DIR}"
                                PROPERTIES COMPILE_FLAGS "${AVX_FLAG}")
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
if((NOT BUILD_STATIC_LIBS) AND WIN32 AND AOCL_TEST_COVERAGE)
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

if (BUILD_DOC)
    find_package(Doxygen 1.10.0)
    set(W_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    if(NOT (DOXYGEN_FOUND))
        message(FATAL_ERROR "Doxygen not found please install doxygen and try again (or) disable docs building -DBUILD_DOC=OFF")
    else()
        execute_process(COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                WORKING_DIRECTORY ${W_DIR})
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/index.html)
        message(STATUS "Documentation generated successfully, to view documentation open docs/html/index.html")
    else()
        message(FATAL_ERROR "Document generation failed.")
    endif()
endif()
