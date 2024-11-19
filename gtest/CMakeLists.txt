# Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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
# @brief CMake based build system for gtest module.
#
# Installs GTest library and link all the gtest based test cases to CTest.
# Rebuild the AOCL-FFTZ library with wrapper files to make it work for gtests.
# Compiles and installs the library and test bench binary with configured
# options. Supports uninstall custom command to remove the installed files.
# Enables CTest for adding and testing Gtest and test bench related tests.
# Finds Doxygen installation and generates the documentation automatically.
#
# @author Srirammaswamy Srinivasan
# @author Ashwin Godbole

include(GoogleTest)
include(ExternalProject)

# to avoid DOWNLOAD_EXTRACT_TIMESTAMP warning from CMake 3.24 onwards
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# fetch, configure and build gtest inside build directory
set(deps_googletest "GTEST")
message("Fetching and Configuring GoogleTest")
set(GTEST_URL "https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip")
set(GTEST_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest/download")
set(GTEST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest/build")
set(GTEST_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest/install")
ExternalProject_Add(
    GTEST
    URL ${GTEST_URL}
    PREFIX  ${GTEST_BUILD_DIR}
    DOWNLOAD_DIR ${GTEST_DOWNLOAD_DIR}
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
        -Dgtest_force_shared_crt=ON # For Windows: Prevent overriding the parent project's compiler/linker settings
        -DCMAKE_INSTALL_LIBDIR=lib
)
include_directories(${GTEST_INSTALL_DIR}/include)
link_directories(${GTEST_INSTALL_DIR}/lib)

# fetch, configure and build benchmark inside build directory
set(deps_benchmark "BENCHMARK")
message("Fetching and Configuring Google Benchmark")
set(BENCHMARK_URL "https://github.com/google/benchmark/archive/refs/tags/v1.9.0.zip")
set(BENCHMARK_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/download")
set(BENCHMARK_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/build")
set(BENCHMARK_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/install")
ExternalProject_Add(
    BENCHMARK
    URL ${BENCHMARK_URL}
    PREFIX  ${BENCHMARK_BUILD_DIR}
    DOWNLOAD_DIR ${BENCHMARK_DOWNLOAD_DIR}
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${BENCHMARK_INSTALL_DIR}
        -DBENCHMARK_ENABLE_TESTING=OFF # avoid benchmark tests getting added to FFTZ ctests
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_BUILD_TYPE=Release
)
include_directories(${BENCHMARK_INSTALL_DIR}/include)
link_directories(${BENCHMARK_INSTALL_DIR}/lib)

# gtest/aoclfftz_core_wrapper.h and gtest/aoclfftz_core_wrapper.c files
# will be added to a separately built core library for gtest
# since it uses the core functions directly to create wrapper version of
# functions with __declspec(dllexport) attribute
file(GLOB CORE_WRAPPER_FILES ${CMAKE_SOURCE_DIR}/gtest/aoclfftz_core_wrapper.h
                             ${CMAKE_SOURCE_DIR}/gtest/aoclfftz_core_wrapper.c)

# library will be generated in ${CMAKE_BINARY_DIR}/gtest
set(gtest_lib_name ${lib_name}_gtest)
if(BUILD_STATIC_LIBS)
    add_library(${gtest_lib_name} STATIC ${SRC_FILES} ${CORE_WRAPPER_FILES})
else()
    add_library(${gtest_lib_name} SHARED ${SRC_FILES} ${CORE_WRAPPER_FILES})
endif()

# add the compiler and linker flags for code coverage
if(CODE_COVERAGE)
    target_compile_options(${gtest_lib_name} PUBLIC --coverage -fprofile-abs-path)
    target_link_options(${gtest_lib_name} PUBLIC --coverage)
endif()

# add the compiler and linker flags for AddressSanitizer
if(ASAN)
    target_compile_options(${gtest_lib_name} PUBLIC -fsanitize=address -O0 -gdwarf-5 -g3)
    target_link_options(${gtest_lib_name} PUBLIC -fsanitize=address)
endif()

file(GLOB GTEST_SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/gtest/*/*.cpp)

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    if("${TNAME}" STREQUAL "")
        continue()
    endif()
    add_executable(${TNAME} ${FILE})
    add_dependencies(${TNAME} ${deps_googletest})
    target_link_libraries(${TNAME} ${gtest_lib_name} gtest gtest_main)
endforeach()

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    if("${TNAME}" STREQUAL "")
        continue()
    endif()
    gtest_discover_tests(${TNAME})
endforeach()

add_executable(kernel_perf_test ${CMAKE_SOURCE_DIR}/gtest/kernel/kernel_perf_test.cpp)
add_dependencies(kernel_perf_test ${deps_googletest} ${deps_benchmark})
target_link_libraries(kernel_perf_test ${gtest_lib_name} benchmark)

# Windows specific config dependencies for building & running kernel_perftest with benchmark
if(WIN32)
    # defining BENCHMARK_STATIC_DEFINE to avoid __declspec(dllexport) prefix in benchmark APIs
    target_compile_definitions(kernel_perf_test PRIVATE -DBENCHMARK_STATIC_DEFINE)
    # Benchmark internally uses shlwapi to get CPUInfo in windows, making shlwapi an indirect dependency
    # https://github.com/google/benchmark/blob/main/docs/platform_specific_build_instructions.md
    # TODO : Avoid linking shlwapi if possible
    target_link_libraries(kernel_perf_test shlwapi)
endif()
