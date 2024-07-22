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


include(FetchContent)
include(GoogleTest)

# to avoid DOWNLOAD_EXTRACT_TIMESTAMP warning from CMake 3.24 onwards
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    # Avoid installing gtest libs/headers by default
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message("Fetching and Configuring Google Benchmark")
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY "https://github.com/google/benchmark.git"
        GIT_TAG v1.8.3 # Latest release
    )
    set(BENCHMARK_ENABLE_TESTING OFF) # avoid benchmark tests getting added to FFTZ ctests
    FetchContent_MakeAvailable(benchmark)
endif()

set(BENCHMARK_ENABLE_LIBPFM ON)

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
    target_link_libraries(${TNAME} ${gtest_lib_name} GTest::gtest_main)
endforeach()

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    if("${TNAME}" STREQUAL "")
        continue()
    endif()
    gtest_discover_tests(${TNAME})
endforeach()

add_executable(kernel_perf_test ${CMAKE_SOURCE_DIR}/gtest/kernel/kernel_perf_test.cpp)
target_link_libraries(kernel_perf_test ${gtest_lib_name} GTest::gtest_main benchmark::benchmark)
