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

set(lib_name fftw3)
option(BUILD_STATIC_LIBS "Build static library (Default build type is shared library)" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

if(NOT DEFINED FFTZ_LIB)
     if(WIN32 AND BUILD_STATIC_LIBS)
         set(FFTZ_LIB aocl_fftz_static)
     else()
         set(FFTZ_LIB aocl_fftz)
     endif()
endif()

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
set(API_SRC_PATH api)
set(C_WRAPPER_SRC_PATH src/c)
set(FORTRAN_SRC_PATH src/fortran)
set(TRANSLATOR_SRC_PATH src/translator)
set(INC_FILES ${API_SRC_PATH}/fftw_wrapper.f ${API_SRC_PATH}/fftw_wrapper.f03 ${API_SRC_PATH}/fftw_wrapper.h)

file(GLOB SRC_FILES ${API_SRC_PATH}/*.c ${API_SRC_PATH}/*.h
                    ${C_WRAPPER_SRC_PATH}/*.c ${C_WRAPPER_SRC_PATH}/*.h
                    ${FORTRAN_SRC_PATH}/*.c ${FORTRAN_SRC_PATH}/*.h
                    ${TRANSLATOR_SRC_PATH}/*.c ${TRANSLATOR_SRC_PATH}/*.h)
include_directories(.)

# Set RPATH to $ORIGIN so the wrapper finds libaocl_fftz.so in the same directory at runtime.
# Use install RPATH at build time to avoid CMake rewriting RPATH on install
if(UNIX AND NOT BUILD_STATIC_LIBS)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

if(BUILD_STATIC_LIBS)
    add_library(${lib_name} STATIC ${SRC_FILES} ${UTIL_FILES})
    message(STATUS "FFTW-WRAPPER Library type being built is: Static")
else()
    add_library(${lib_name} SHARED ${SRC_FILES} ${UTIL_FILES})
    message(STATUS "FFTW-WRAPPER Library type being built is: Shared")
endif()

if(UNIX AND NOT BUILD_STATIC_LIBS)
    set_target_properties(${lib_name} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Link to the main AOCL-FFTZ library:
# - If FFTZ_LIB_TARGET is set and names a CMake target, we are built as a subdirectory; link to that target.
# - If FFTZ_LIB_TARGET is set but not a target, fail with an error (invalid value).
# - Otherwise (unset): wrapper built in its dir; link ${FFTZ_LIB} .
if(FFTZ_LIB_TARGET AND TARGET "${FFTZ_LIB_TARGET}")
    if(BUILD_STATIC_LIBS)
        target_link_libraries(${lib_name} PUBLIC
            $<BUILD_INTERFACE:${FFTZ_LIB_TARGET}>
            $<INSTALL_INTERFACE:aocl-fftz::aocl_fftz_static>
            ${lib_deps})
    else()
        target_link_libraries(${lib_name} PRIVATE ${FFTZ_LIB_TARGET} ${lib_deps})
    endif()
elseif(FFTZ_LIB_TARGET)
    message(FATAL_ERROR "FFTZ_LIB_TARGET is set to '${FFTZ_LIB_TARGET}' but it is not a CMake target.")
else()
    # Built from this directory: use ${FFTZ_LIB} and link_directories from the NOT BUILD_THIRD_PARTY_WRAPPERS block above.
    # For static wrapper, link FFTZ as PUBLIC so exported FFTW3Targets.cmake propagates the dependency to consumers.
    if(BUILD_STATIC_LIBS)
        target_link_libraries(${lib_name} PUBLIC
            $<BUILD_INTERFACE:${FFTZ_LIB}>
            $<INSTALL_INTERFACE:aocl-fftz::aocl_fftz_static>
            ${lib_deps})
    else()
        target_link_libraries(${lib_name} PRIVATE ${FFTZ_LIB} ${lib_deps})
    endif()
endif()

# Install destination
set(DEST_LIB_PATH lib)
set(DEST_INCLUDE_PATH include)
set(FFTW3_CMAKE_DIR "${DEST_LIB_PATH}/cmake/FFTW3")
message(STATUS "FFTW-WRAPPER Library will be installed at ${CMAKE_INSTALL_PREFIX}/${DEST_LIB_PATH}")
install(TARGETS ${lib_name} EXPORT FFTW3Targets
    DESTINATION ${DEST_LIB_PATH}
    INCLUDES DESTINATION ${DEST_INCLUDE_PATH}
)

# Provide libfftw3f, libfftw3_omp and libfftw3f_omp for apps that link -lfftw3f / -lfftw3_omp / -lfftw3f_omp.
foreach(FFTW3_ALIAS fftw3f fftw3_omp fftw3f_omp)
    if(BUILD_STATIC_LIBS)
        set(FFTW3_ALIAS_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${FFTW3_ALIAS}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(FFTW3_ALIAS_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}${FFTW3_ALIAS}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
    add_custom_command(TARGET ${lib_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> $<TARGET_FILE_DIR:${lib_name}>/${FFTW3_ALIAS_NAME})
    install(FILES $<TARGET_FILE:${lib_name}> DESTINATION ${DEST_LIB_PATH} RENAME ${FFTW3_ALIAS_NAME})
    if(WIN32 AND NOT BUILD_STATIC_LIBS)
        set(FFTW3_IMPORT_NAME "${CMAKE_IMPORT_LIBRARY_PREFIX}${FFTW3_ALIAS}${CMAKE_IMPORT_LIBRARY_SUFFIX}")
        add_custom_command(TARGET ${lib_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:${lib_name}> $<TARGET_FILE_DIR:${lib_name}>/${FFTW3_IMPORT_NAME})
        install(FILES $<TARGET_LINKER_FILE:${lib_name}> DESTINATION ${DEST_LIB_PATH} RENAME ${FFTW3_IMPORT_NAME})
    endif()
endforeach()

install(FILES ${API_SRC_PATH}/fftw_wrapper.h DESTINATION ${DEST_INCLUDE_PATH} RENAME fftw3.h)
install(FILES ${API_SRC_PATH}/fftw_wrapper.f DESTINATION ${DEST_INCLUDE_PATH} RENAME fftw3.f)
install(FILES ${API_SRC_PATH}/fftw_wrapper.f03 DESTINATION ${DEST_INCLUDE_PATH} RENAME fftw3.f03)

# FFTW3 CMake config so find_package(FFTW3) finds this wrapper
include(CMakePackageConfigHelpers)
install(EXPORT FFTW3Targets
    NAMESPACE FFTW3::
    DESTINATION ${FFTW3_CMAKE_DIR}
    FILE FFTW3Targets.cmake
)
set(FFTW3_CONFIG_IN "${CMAKE_SOURCE_DIR}/cmake/FFTW3Config.cmake.in")
if(EXISTS "${FFTW3_CONFIG_IN}")
    configure_package_config_file(
        "${FFTW3_CONFIG_IN}"
        "${CMAKE_CURRENT_BINARY_DIR}/FFTW3Config.cmake"
        INSTALL_DESTINATION "${FFTW3_CMAKE_DIR}"
    )
    # FFTW3_VERSION must be 3.x so find_package(FFTW3 3) / find_package(FFTW3 3.x) succeeds
    # SameMajorVersion requires requested major to match; real FFTW library is 3.x.
    set(FFTW3_VERSION "3.3.10")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/FFTW3ConfigVersion.cmake"
        VERSION "${FFTW3_VERSION}"
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/FFTW3Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/FFTW3ConfigVersion.cmake"
        DESTINATION "${FFTW3_CMAKE_DIR}"
    )
endif()
