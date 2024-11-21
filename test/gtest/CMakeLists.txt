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

# Add the necessary package for pthread
find_package(Threads REQUIRED)

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
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
include_directories(${BENCHMARK_INSTALL_DIR}/include)
link_directories(${BENCHMARK_INSTALL_DIR}/lib)

# test/gtest/aoclfftz_core_wrapper.h and test/gtest/aoclfftz_core_wrapper.c files
# will be added to a separately built core library for gtest
# since it uses the core functions directly to create wrapper version of
# functions with __declspec(dllexport) attribute
file(GLOB CORE_WRAPPER_FILES ${CMAKE_SOURCE_DIR}/${GTEST_DIR}/aoclfftz_core_wrapper.h
                             ${CMAKE_SOURCE_DIR}/${GTEST_DIR}/aoclfftz_core_wrapper.c
                             ${CMAKE_SOURCE_DIR}/${TEST_UTILS_DIR}/dims_vecs_helper.*)

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

file(GLOB GTEST_SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/${GTEST_DIR}/*/*.cpp)

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    if("${TNAME}" STREQUAL "")
        continue()
    endif()
    add_executable(${TNAME} ${FILE})
    add_dependencies(${TNAME} ${deps_googletest})
    target_link_libraries(${TNAME} ${gtest_lib_name} gtest gtest_main Threads::Threads)
endforeach()

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    if("${TNAME}" STREQUAL "")
        continue()
    endif()
    gtest_discover_tests(${TNAME})
endforeach()

add_executable(kernel_perf_test ${CMAKE_SOURCE_DIR}/${GTEST_DIR}/kernel/kernel_perf_test.cpp)
add_dependencies(kernel_perf_test ${deps_googletest} ${deps_benchmark})
target_link_libraries(kernel_perf_test ${gtest_lib_name} benchmark Threads::Threads)

# Windows specific config dependencies for building & running kernel_perftest with benchmark
if(WIN32)
    # defining BENCHMARK_STATIC_DEFINE to avoid __declspec(dllexport) prefix in benchmark APIs
    target_compile_definitions(kernel_perf_test PRIVATE -DBENCHMARK_STATIC_DEFINE)
    # Benchmark internally uses shlwapi to get CPUInfo in windows, making shlwapi an indirect dependency
    # https://github.com/google/benchmark/blob/main/docs/platform_specific_build_instructions.md
    # TODO : Avoid linking shlwapi if possible
    target_link_libraries(kernel_perf_test shlwapi)
endif()

if(FUZZTEST)
    set(deps_gfuzztest "GFUZZTEST")
    message("Fetching and Configuring GoogleFuzzTest")
    set(GFUZZTEST_URL "https://github.com/google/fuzztest/archive/refs/tags/2024-10-28.zip")
    set(GFUZZTEST_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/googlefuzztest/download")
    set(GFUZZTEST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/googlefuzztest/build")
    set(GFUZZTEST_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/googlefuzztest/install")
    ExternalProject_Add(
        GFUZZTEST
        URL ${GFUZZTEST_URL}
        PREFIX  ${GFUZZTEST_BUILD_DIR}
        DOWNLOAD_DIR ${GFUZZTEST_DOWNLOAD_DIR}
        CMAKE_GENERATOR ${gen}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${GFUZZTEST_INSTALL_DIR}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DABSL_PROPAGATE_CXX_STD=ON
            -DFUZZTEST_FUZZING_MODE=ON
    )
    # link the dependent libraries of linked library with ld linker
    # --copy-dt-needed-entries: Ensures that all symbols from dependent shared libraries are available.
    # -fuse-ld=bfd: Uses the BFD linker instead of the default one.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--copy-dt-needed-entries -fuse-ld=bfd")
    include_directories(${GFUZZTEST_BUILD_DIR}/src/${deps_gfuzztest})
    include_directories(${GFUZZTEST_INSTALL_DIR}/include)
    # directories needed to link transitive dependent libs
    link_directories(${GFUZZTEST_INSTALL_DIR}/lib)
    link_directories(${GFUZZTEST_BUILD_DIR}/src/${deps_gfuzztest}-build/fuzztest)
    link_directories(${GFUZZTEST_BUILD_DIR}/src/${deps_gfuzztest}-build/common)
    # set fuzzing flags explicitly
    set(FUZZ_FLAGS "-g -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION -UNDEBUG -fsanitize=address -fsanitize-coverage=inline-8bit-counters -fsanitize-coverage=trace-cmp -DADDRESS_SANITIZER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FUZZ_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FUZZ_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    file(GLOB FUZZ_TEST_SRC_FILES
    ${CMAKE_SOURCE_DIR}/test/gtest/fuzz/fuzz_test.cpp
    ${CMAKE_SOURCE_DIR}/test/bench/utils/*.c
    ${CMAKE_SOURCE_DIR}/test/bench/bench_problem.c
    ${CMAKE_SOURCE_DIR}/test/utils/dims_vecs_helper.c
    ${CMAKE_SOURCE_DIR}/test/bench/accuracy.c
    ${CMAKE_SOURCE_DIR}/test/bench/dft_reference.c
    )
    add_executable(fuzz_test ${FUZZ_TEST_SRC_FILES})
    add_dependencies(fuzz_test ${deps_gfuzztest})
    target_link_libraries(fuzz_test PRIVATE aocl_fftz)
    target_link_libraries(fuzz_test PRIVATE fuzztest_fuzztest_gtest_main)
endif()
