# Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

# to avoid DOWNLOAD_EXTRACT_TIMESTAMP warning from CMake 3.26 onwards
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.26.0")
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
    get_filename_component(TNAME ${FILE} NAME_WLE)
    if(NOT "${TNAME}" MATCHES "[A-Za-z0-9_]*_gtest")
        continue()
    endif()
    # FIXME: skipping selector tests for testing since it is not valid after
    # direct-solver changes in post-selection
    if("${TNAME}" MATCHES "selector_")
        message("   skipping ${TNAME}")
        continue()
    endif()
    add_executable(${TNAME} ${FILE})
    add_dependencies(${TNAME} ${deps_googletest})
    target_link_libraries(${TNAME} ${gtest_lib_name} gtest gtest_main Threads::Threads)
    gtest_discover_tests(${TNAME})
endforeach()

if(ENABLE_MULTI_THREADING)
    if(WIN32)
        target_link_libraries(${gtest_lib_name} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    else()
        target_link_options(${gtest_lib_name} PUBLIC -fopenmp)
    endif()
endif()

if(ENABLE_AVX512 OR ENABLE_AVX256 OR ENABLE_AVX128)
    include(CheckCXXSourceRuns)
    
    # Common header code for all AVX detection
    set(AVX_CHECK_HEADER "
    #include <immintrin.h>
    #ifdef _WIN32
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
    ")
    
    # Check for AVX512 support
    if(ENABLE_AVX512)
        set(CHECK_AVX512_CODE "${AVX_CHECK_HEADER}
        int main() {
            unsigned int eax, ebx, ecx, edx;
        #ifdef _WIN32
            int cpuInfo[4];
            __cpuidex(cpuInfo, 7, 0);
            eax = cpuInfo[0];
            ebx = cpuInfo[1];
            ecx = cpuInfo[2];
            edx = cpuInfo[3];
        #else
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
        #endif
            return (ebx & (1 << 16)) ? 0 : 1;  // AVX512F - bit 16 in EBX
        }
        ")

        file(WRITE "${CMAKE_BINARY_DIR}/check_avx512.cpp" "${CHECK_AVX512_CODE}")
        try_run(RUN_RESULT COMPILE_RESULT
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/check_avx512.cpp"
        )

        if(RUN_RESULT EQUAL 0)
            target_compile_definitions(kernel_gtest PRIVATE -DAVX512_SUPPORTED)
        else()
            message(WARNING "AVX512 not supported by hardware. Skipping AVX512 tests.")
        endif()
        file(REMOVE "${CMAKE_BINARY_DIR}/check_avx512.cpp")
    endif()

    # Check for AVX256 support (AVX2)
    if(ENABLE_AVX256)
        set(CHECK_AVX256_CODE "${AVX_CHECK_HEADER}
        int main() {
            unsigned int eax, ebx, ecx, edx;
        #ifdef _WIN32
            int cpuInfo[4];
            __cpuidex(cpuInfo, 1, 0);
            eax = cpuInfo[0];
            ebx = cpuInfo[1];
            ecx = cpuInfo[2];
            edx = cpuInfo[3];
        #else
            __cpuid(1, eax, ebx, ecx, edx);
        #endif
            // check for AVX and FMA as library runs AVX256 kernel with fma instructions
            return ((ecx & 0x18001000) == 0x18001000) ? 0 : 1;  // AVX - bits 27-28 in ECX & FMA - 12th bit in ECX
        }
        ")

        file(WRITE "${CMAKE_BINARY_DIR}/check_avx256.cpp" "${CHECK_AVX256_CODE}")
        try_run(RUN_RESULT COMPILE_RESULT
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/check_avx256.cpp"
        )

        if(RUN_RESULT EQUAL 0)
            target_compile_definitions(kernel_gtest PRIVATE -DAVX256_SUPPORTED)
        else()
            message(WARNING "AVX256 (with FMA) not supported by hardware. Skipping AVX256 tests.")
        endif()
        file(REMOVE "${CMAKE_BINARY_DIR}/check_avx256.cpp")
    endif()

    # Check for AVX support (AVX)
    if(ENABLE_AVX128)
        set(CHECK_AVX_CODE "${AVX_CHECK_HEADER}
        int main() {
            unsigned int eax, ebx, ecx, edx;
        #ifdef _WIN32
            int cpuInfo[4];
            __cpuidex(cpuInfo, 1, 0);
            eax = cpuInfo[0];
            ebx = cpuInfo[1];
            ecx = cpuInfo[2];
            edx = cpuInfo[3];
        #else
            __cpuid(1, eax, ebx, ecx, edx);
        #endif
            return ((ecx & 0x18000000) == 0x18000000) ? 0 : 1;  // AVX - bits 27-28 in ECX
        }
        ")

        file(WRITE "${CMAKE_BINARY_DIR}/check_avx128.cpp" "${CHECK_AVX_CODE}")
        try_run(RUN_RESULT COMPILE_RESULT
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/check_avx128.cpp"
        )

        if(RUN_RESULT EQUAL 0)
            target_compile_definitions(kernel_gtest PRIVATE -DAVX128_SUPPORTED)
        else()
            message(WARNING "AVX128 (AVX) not supported by hardware. Skipping AVX128 tests.")
        endif()
        file(REMOVE "${CMAKE_BINARY_DIR}/check_avx128.cpp")
    endif()
endif()


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
    gtest_discover_tests(fuzz_test)
endif()
