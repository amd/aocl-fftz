# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# @file CMakeLists.txt
# 
# @brief CMakeLists.txt to build example programs for AOCL-FFTZ
#
# @author Partiksha

cmake_minimum_required(VERSION 3.26.0)
project(aoclfftz_examples C)

set(AOCL_FFTZ_INSTALL_PATH "" CACHE PATH "Path to the AOCL-FFTZ installation directory")

# set the expected path variables
include_directories(.)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEST_BIN_PATH bin)

# Create a list to collect all example targets
set(EXAMPLE_TARGETS "" CACHE INTERNAL "List of all example targets")

# Find OpenMP package, Assuming Library is built with ENABLE_MULTI_THREADING=ON
if (WIN32)
    set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE FILEPATH "OpenMP library path")
    message(STATUS "OpenMP custom path to look for OpenMP: ${OpenMP_libomp_LIBRARY}")
endif()
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found - will enable parallel execution")
else()
    message(FATAL_ERROR "OpenMP Missing! Please install OpenMP and ensure that the AOCL-FFTZ library was built with ENABLE_MULTI_THREADING=ON.")
endif()

if(NOT AOCL_FFTZ_INSTALL_PATH)
    message(FATAL_ERROR "Please provide AOCL_FFTZ_INSTALL_PATH to point to the AOCL-FFTZ installation directory.")
else()
    message(STATUS "Using AOCL_FFTZ_INSTALL_PATH: ${AOCL_FFTZ_INSTALL_PATH}")
    # Set paths based on the installed structure
    set(AOCL_FFTZ_LIB_DIR "${AOCL_FFTZ_INSTALL_PATH}/lib")
    set(AOCL_FFTZ_INCLUDE_DIR "${AOCL_FFTZ_INSTALL_PATH}/include")
    # Choose library name from what is actually installed.
    # On Windows, prefer shared (aocl_fftz.lib) when present; fall back to aocl_fftz_static.lib when shared is absent.
    if(NOT DEFINED FFTZ_LIB)
        if(WIN32)
            if(EXISTS "${AOCL_FFTZ_LIB_DIR}/aocl_fftz.lib")
                set(FFTZ_LIB aocl_fftz)
            elseif(EXISTS "${AOCL_FFTZ_LIB_DIR}/aocl_fftz_static.lib")
                set(FFTZ_LIB aocl_fftz_static)
            else()
                set(FFTZ_LIB aocl_fftz)
            endif()
        else()
            set(FFTZ_LIB aocl_fftz)
        endif()
        message(STATUS "Linking against AOCL-FFTZ library: ${FFTZ_LIB}")
    endif()
endif()

# Check if the header exists
if(NOT EXISTS "${AOCL_FFTZ_INCLUDE_DIR}/aoclfftz.h")
    message(FATAL_ERROR "Header not found at ${AOCL_FFTZ_INCLUDE_DIR}/aoclfftz.h")
endif()

# Link and include directories
message(STATUS "Using library directory: ${AOCL_FFTZ_LIB_DIR}")
message(STATUS "Using include directory: ${AOCL_FFTZ_INCLUDE_DIR}")

# Set default install location if CMAKE_INSTALL_PREFIX is not defined
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${ROOT_DIR}" CACHE PATH "Install path prefix" FORCE)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${DEST_BIN_PATH}")

# Create example_name executable
FUNCTION(add_example_outofsource example_name example_source)
    add_executable(${example_name} ${example_source})
    target_include_directories(${example_name} PRIVATE ${AOCL_FFTZ_INCLUDE_DIR})
    target_link_directories(${example_name} PRIVATE ${AOCL_FFTZ_LIB_DIR})
    if (WIN32)
        message(STATUS "Adding OpenMP support for Windows")
        target_link_libraries(${example_name} ${FFTZ_LIB} OpenMP::OpenMP_C)
    else()
        # Check if we're linking against a static library
        if(EXISTS "${AOCL_FFTZ_LIB_DIR}/libaocl_fftz.a")
            message(STATUS "Linking against static library")
            target_link_libraries(${example_name} ${FFTZ_LIB} m OpenMP::OpenMP_C)
        else()
            message(STATUS "Linking against shared library")
            target_link_libraries(${example_name} ${FFTZ_LIB} m)
        endif()
    endif()
    target_link_libraries(${example_name} ${CMAKE_DL_LIBS})

    # Add this target to our list of example targets
    set(EXAMPLE_TARGETS ${EXAMPLE_TARGETS} ${example_name} CACHE INTERNAL "")
    message(STATUS "Example added : ${example_name}")
ENDFUNCTION()

# Build all example programs starting with example_*
file(GLOB_RECURSE EXAMPLE_FILES "example_*.c*")
foreach(example_path ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_path} NAME_WE)
    add_example_outofsource(${example_name} ${example_path})
endforeach()

message(STATUS "AOCL-FFTZ Library examples will be installed at ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${EXAMPLE_TARGETS} DESTINATION ${DEST_BIN_PATH} COMPONENT examples)

# Uninstall the binary files using custom command
add_custom_target("uninstall" COMMENT "Uninstalling the installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall the installed files as per install_manifest.txt"
    COMMAND ${CMAKE_COMMAND} -E echo "Checking for install_manifest.txt..."
    COMMAND [ -f install_manifest.txt ] && xargs rm -vf < install_manifest.txt || echo 'install_manifest.txt is not present, nothing to uninstall!'
)
