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
# @brief CMakeLists.txt to build example programs for AOCL-FFTZ
#
# @author Partiksha

cmake_minimum_required(VERSION 3.26.0)
project(aoclfftz_examples C)

# Set the path to the AOCL-FFTZ include and library directories
set(FFTZ_LIB aocl_fftz)
set(AOCL_FFTZ_INSTALL_PATH "" CACHE PATH "Path to the AOCL-FFTZ installation directory")

# Default paths if not provided
if(NOT AOCL_FFTZ_INSTALL_PATH)
    message(STATUS "AOCL_FFTZ_INSTALL_PATH not provided, using default paths")
    # Default paths within the workspace
    set(AOCL_FFTZ_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
    set(AOCL_FFTZ_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../api")
else()
    message(STATUS "Using AOCL_FFTZ_INSTALL_PATH: ${AOCL_FFTZ_INSTALL_PATH}")
    # Set paths based on the installed structure
    set(AOCL_FFTZ_LIB_DIR "${AOCL_FFTZ_INSTALL_PATH}/aocl_fftz/lib")
    set(AOCL_FFTZ_INCLUDE_DIR "${AOCL_FFTZ_INSTALL_PATH}/aocl_fftz/include")
endif()

# Check if the header exists
if(NOT EXISTS "${AOCL_FFTZ_INCLUDE_DIR}/aoclfftz.h")
    message(FATAL_ERROR "Header not found at ${AOCL_FFTZ_INCLUDE_DIR}/aoclfftz.h")
endif()

# Link and include directories
message(STATUS "Using library directory: ${AOCL_FFTZ_LIB_DIR}")
message(STATUS "Using include directory: ${AOCL_FFTZ_INCLUDE_DIR}")

# Create example_name executable
FUNCTION(add_example_outofsource example_name example_source)
    add_executable(${example_name} ${example_source})
    target_include_directories(${example_name} PRIVATE ${AOCL_FFTZ_INCLUDE_DIR})
    target_link_directories(${example_name} PRIVATE ${AOCL_FFTZ_LIB_DIR})
    if (WIN32)
        target_link_libraries(${example_name} ${FFTZ_LIB})
    else()
        target_link_libraries(${example_name} ${FFTZ_LIB} m)
    endif()
    target_link_libraries(${example_name} ${CMAKE_DL_LIBS})
    message(STATUS "Example added out-of-src-tree : ${example_name}")
ENDFUNCTION()

# Build all example programs starting with example_*
file(GLOB_RECURSE EXAMPLE_FILES "example_*.c*")
foreach(example_path ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_path} NAME_WE)
    add_example_outofsource(${example_name} ${example_path})
endforeach()
