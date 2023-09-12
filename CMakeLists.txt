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
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if (${CMAKE_C_COMPILER_VERSION} VERSION_LESS 7.1)
        message(FATAL_ERROR "Minimum GCC version supported is 7.1!")
    endif ()
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    if (${CMAKE_C_COMPILER_VERSION} VERSION_LESS 11.0)
        message(FATAL_ERROR "Minimum Clang version supported is 11.0! In case of AOCC, use version 2.0 and above!")
endif ()
else ()
    message(FATAL_ERROR "Compiler not supported! Library has only been tested with Clang and GCC.")
endif ()

set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)
option (BUILD_STATIC_LIBS "Build static library (Default build type is shared library)" OFF)
option (ENABLE_AVX "Compile with AVX instruction set support" OFF)
option (ENABLE_AVX2 "Compile with AVX2 instruction set support" OFF)
option (ENABLE_AVX512 "Compile with AVX512 instruction set support" OFF)
option (ENABLE_MPI "Compile with distributed MPI FFTZ support" OFF)
option (AOCL_TEST_COVERAGE "Enable GTest and AOCL test bench based CTest suite (Disabled by default)" OFF)
option (BUILD_DOC "Build documentation for aocl-fftz (Disabled by default)" OFF)
option (ENABLE_STRICT_WARNINGS "Enable compiler flags to treat all warnings as errors (Enabled by default)" ON)

#list source files to be compiled for the library
set (CORE_PATH core)
file(GLOB API_SRC_FILES api/*.c api/*.cpp api/*.h)
file(GLOB CORE_SRC_FILES ${CORE_PATH}/*.c ${CORE_PATH}/*.h
                        ${CORE_PATH}/common/*.c ${CORE_PATH}/common/*.h
                        ${CORE_PATH}/kernels/*.c ${CORE_PATH}/kernels/*.h
                        ${CORE_PATH}/kernels/c/*.c ${CORE_PATH}/kernels/c/*.h
                        ${CORE_PATH}/kernels/avx128/*.c ${CORE_PATH}/kernels/avx128/*.h
                        ${CORE_PATH}/kernels/avx256/*.c ${CORE_PATH}/kernels/avx256/*.h
                        ${CORE_PATH}/kernels/avx512/*.c ${CORE_PATH}/kernels/avx512/*.h
                        ${CORE_PATH}/solvers/*.c ${CORE_PATH}/solvers/*.h)
file(GLOB SELECTOR_SRC_FILES selector/*.c selector/*.h)
file(GLOB THREADS_SRC_FILES threads/*.c threads/*.h)
file(GLOB UTILS_SRC_FILES utils/*.c utils/*.cpp utils/*.h)

#set the expected path variables
include_directories(.)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_DIR}/build/)
set(DEST_LIB_PATH aocl_fftz/lib)
set(DEST_INC_PATH aocl_fftz/include)
set(DEST_BIN_PATH aocl_fftz/bin)

set(INC_FILES api/aoclfftz.h)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type selected is Release")
    add_compile_definitions(CFLAGS_SET1="${CMAKE_C_FLAGS_RELEASE}")
else ()
    message(STATUS "Build type selected is Debug")
    add_compile_definitions(CFLAGS_SET1="${CMAKE_C_FLAGS_DEBUG}")
endif()

#set(USER_DEBUG_ON "-g")
message(STATUS "C Compiler used: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler used: ${CMAKE_CXX_COMPILER}")
message(ENABLE_AVX : ${ENABLE_AVX})
message(ENABLE_AVX2 : ${ENABLE_AVX2})
message(ENABLE_AVX512 : ${ENABLE_AVX512})

if (WIN32)
if (ENABLE_AVX512)
    set (AVX512_FLAG /arch:AVX512)
    message(STATUS "Enabling AVX512 instructions")
endif()
if (ENABLE_AVX2)
    set (AVX2_FLAG /arch:AVX2)
    message(STATUS "Enabling AVX2 instructions")
endif()
if (ENABLE_AVX)
    set (AVX512_FLAG /arch:AVX)
    message(STATUS "Enabling AVX instructions")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} /O2 /DNDEBUG /Oy")

elseif (UNIX)
if (ENABLE_AVX512)
    set (AVX512_FLAG -mavx512f)
    message(STATUS "Enabling AVX512 instructions")
endif()
if (ENABLE_AVX2)
    set (AVX2_FLAG -mavx2)
    message(STATUS "Enabling AVX2 instructions")
endif()
if (ENABLE_AVX)
    set (AVX512_FLAG -mavx)
    message(STATUS "Enabling AVX instructions")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEBUG_ON} -fomit-frame-pointer -fstrict-aliasing")
endif()

add_compile_options(${AVX_FLAG} ${AVX2_FLAG} ${AVX512_FLAG})

add_compile_definitions(CCompiler="${CMAKE_C_COMPILER}")
add_compile_definitions(CXXCompiler="${CMAKE_CXX_COMPILER}")
add_compile_definitions(CFLAGS_SET2="${CMAKE_C_FLAGS}")

if (ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI_FFTZ="1")
endif()

#set library and sources
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)
set (lib_name aocl_fftz)
set(SRC_FILES
        ${UTILS_SRC_FILES}
        ${API_SRC_FILES}
        ${CORE_SRC_FILES}
        ${SELECTOR_SRC_FILES}
    )
set (BENCH_PATH test)

#set library dependencies
set(lib_deps m) #math library

#setup CTest for GoogleTest and test bench
if (AOCL_TEST_COVERAGE)
    # enabling ctest
    enable_testing()
    add_subdirectory(gtest)
    add_subdirectory(test)
endif()

#Enabling strict warnings
if(ENABLE_STRICT_WARNINGS)
	if (WIN32)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
		set_property(SOURCE ${SRC_FILES} APPEND PROPERTY COMPILE_OPTIONS "-DENABLE_STRICT_WARNINGS")
	else()
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wpedantic")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wpedantic")
		set_property(SOURCE ${SRC_FILES} APPEND PROPERTY COMPILE_OPTIONS "-DENABLE_STRICT_WARNINGS")
	endif()
endif()

#add library
if (BUILD_STATIC_LIBS)
	add_library(${lib_name} STATIC ${SRC_FILES})
	message(STATUS "Library type being built is: Static")
else ()
	add_library(${lib_name} SHARED ${SRC_FILES})
	message(STATUS "Library type being built is: Shared")
endif()

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${DEST_LIB_PATH}")

#add test bench executable
set (exe_name aocl_fftz_bench)
if (WIN32)
message(STATUS "C Compiler flags used for this windows build are: ${CMAKE_C_FLAGS}")
message(STATUS "C++ Compiler flags used for this windows build are: ${CMAKE_CXX_FLAGS}")
add_executable(${exe_name} ${BENCH_PATH}/aoclfftz_corebench.c ${BENCH_PATH}/aoclfftz_corebench.h)
else ()
message(STATUS "C Compiler flags used for this Unix build are: ${CMAKE_C_FLAGS}")
message(STATUS "C++ Compiler flags used for this Unix build are: ${CMAKE_CXX_FLAGS}")
add_executable(${exe_name} ${BENCH_PATH}/aoclfftz_corebench.c ${BENCH_PATH}/aoclfftz_corebench.h)
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
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> ${CMAKE_BINARY_DIR}/gtest
        )
endif()

#Uninstall the binary. library and header files using custom command
add_custom_target("uninstall" COMMENT "Uninstalling the installed files from ${install_dir}")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall the installed files as per install_manifest.txt and from ${ROOT_DIR}/lib/"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
    COMMAND rm -vf ${ROOT_DIR}/lib/*
    )

if (BUILD_DOC)
    find_package(Doxygen 1.9.6)
    set(W_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    if(NOT (DOXYGEN_FOUND))
        message(STATUS "Doxygen not found please install doxygen and try again.")
    else()
        execute_process(COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                WORKING_DIRECTORY ${W_DIR}
                COMMAND_ECHO STDOUT)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/index.html)
        message(STATUS "Documentation generated successfully, to view documentation open docs/html/index.html")
    else()
        message(STATUS "Document generation failed.")
    endif()
endif()
