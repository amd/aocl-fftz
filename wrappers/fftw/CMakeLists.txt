# Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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
# @brief CMake based build system for the FFTW Wrapper library.
#
# Compiles and installs the FFTW wrapper library. Requires FFTZ library,
# header files for building library.
#

cmake_minimum_required(VERSION 3.26.0)

project(FFTW-WRAPPER)
message(STATUS "Building FFTW-WRAPPER")

set(lib_name fftw3xc_wrapper_fftz)
option(BUILD_STATIC_LIBS "Build static library (Default build type is shared library)" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(FFTZ_LIB aocl_fftz)

# If the user does not specify aocl_fftz lib path, search for aocl_fftz library
# in project's lib directory and link the directory.
if(NOT BUILD_THIRD_PARTY_WRAPPERS)
    if(NOT FFTZ_LIB_PATH)
        # Find library within the project's lib directory
        find_library(FFTZ_LIB_DIR NAMES ${FFTZ_LIB} PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../lib PATH_SUFFIXES "Release" "Debug" NO_CACHE)
        if(FFTZ_LIB_DIR)
            get_filename_component(FFTZ_LIB_DIR ${FFTZ_LIB_DIR} DIRECTORY)
            set(FFTZ_LIB_PATH "${FFTZ_LIB_DIR}" CACHE PATH "Path to FFTZ library directory" FORCE)
        else()
            set(FFTZ_LIB_PATH "" CACHE PATH "Path to FFTZ library directory" FORCE)
            message(FATAL_ERROR "${FFTZ_LIB} lib not found. Please build ${FFTZ_LIB} lib or provide the FFTZ lib path.")
        endif()

        # Link FFTZ library and include path
        link_directories(${FFTZ_LIB_DIR})
        message("Using ${FFTZ_LIB} found at ${FFTZ_LIB_DIR}")

    else()
        # if the user specifies the aocl_fftz lib path, search for aocl_fftz library in the given path.
        # if not found throw error.
        find_library(FFTZ_LIB_FOUND NAMES ${FFTZ_LIB} PATHS ${FFTZ_LIB_PATH} NO_DEFAULT_PATH NO_CACHE)
        set(FFTZ_LIB_PATH "${FFTZ_LIB_PATH}" CACHE PATH "Path to FFTZ library directory" FORCE)
        if(NOT FFTZ_LIB_FOUND)
            message(FATAL_ERROR "Unable to find ${FFTZ_LIB} lib at provided path: ${FFTZ_LIB_PATH}. Please provide valid library path.")
        endif()

        # Link FFTZ library
        link_directories(${FFTZ_LIB_PATH})
    endif()
endif()

# Link aocl_fftz api path to include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../api)

#set library dependencies
if(UNIX)
    set(lib_deps m) #math library
endif()

if(WIN32)
    #add macro for math constants
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Build fftw_wrapper library
set(SRC_PATH src)
set(INC_FILES ${SRC_PATH}/fftw_wrapper.f ${SRC_PATH}/fftw_wrapper.h)

file(GLOB SRC_FILES ${SRC_PATH}/*.c ${SRC_PATH}/*.h)
include_directories(.)

# Set RPATH to find aocl_fftz library at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../lib")

if(BUILD_STATIC_LIBS)
    add_library(${lib_name} STATIC ${SRC_FILES} ${UTIL_FILES})
    message(STATUS "FFTW-WRAPPER Library type being built is: Static")
    set(full_lib_name ${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    add_library(${lib_name} SHARED ${SRC_FILES} ${UTIL_FILES})
    message(STATUS "FFTW-WRAPPER Library type being built is: Shared")
    set(full_lib_name ${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

target_link_libraries(${lib_name} PRIVATE ${FFTZ_LIB} ${lib_deps})

set(DEST_LIB_PATH wrappers/fftw/lib)
set(DEST_INCLUDE_PATH wrappers/fftw/include)
#Install into destination path by using -DCMAKE_INSTALL_PREFIX=<path> while configuring cmake
message(STATUS "FFTW-WRAPPER Library will be installed at ${CMAKE_INSTALL_PREFIX}/${DEST_LIB_PATH}")
install(TARGETS ${lib_name} DESTINATION ${DEST_LIB_PATH})
install(FILES ${INC_FILES} DESTINATION ${DEST_INCLUDE_PATH})

# create a symlink fftw3.h, fftw3.f and libfftw3xc_wrapper_fftz.so
set(SYMLINK_DEST_PATH ${CMAKE_INSTALL_PREFIX})
set(SYMLINK_SRC_PATH ../wrappers/fftw)

function(create_sym_link sym_name target_name)
    install(CODE "execute_process(COMMAND echo Creating symbolic link ${sym_name} to ${target_name})")
    install(CODE "execute_process(COMMAND rm -f ${sym_name})")
    install(CODE "execute_process(COMMAND ln -s ${target_name} ${sym_name})")
endfunction()

if(NOT WIN32)
    create_sym_link(${SYMLINK_DEST_PATH}/include/fftw3.h ${SYMLINK_SRC_PATH}/include/fftw_wrapper.h)
    create_sym_link(${SYMLINK_DEST_PATH}/include/fftw3.f ${SYMLINK_SRC_PATH}/include/fftw_wrapper.f)
    create_sym_link(${SYMLINK_DEST_PATH}/lib/${full_lib_name} ${SYMLINK_SRC_PATH}/lib/${full_lib_name})
endif()
