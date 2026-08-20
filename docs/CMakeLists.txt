# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# @file CMakeLists.txt
# 
# @brief CMake based sphinx documentation generator.
#
# Checks if necessary packages are installed for generating sphinx documentation
# if they are not installed then neccessary steps are shown to install the required packages.
#
# @author Partiksha

cmake_minimum_required(VERSION 3.26.0)

project(aoclfftz-doc)

find_package(Doxygen 1.9.1)
set(W_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Build Doxygen documentation
if(NOT (DOXYGEN_FOUND))
    message(STATUS "Doxygen not found please install doxygen and try again.")
else()
    execute_process(COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
            WORKING_DIRECTORY ${W_DIR}
            COMMAND_ECHO STDOUT)
endif()


# Check if xml generated successfully from Doxygen, if yes continue building sphinx documentation
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xml/index.xml)
    message(STATUS "xml generated from Doxygen successfully, at ${W_DIR}/xml/ .")

# Check for python and the documentation packages:
# - Sphinx
# - rocm-docs-core
# - breathe
# - myst_parser

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sphinx"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET)
    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "Python3 package sphinx is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install sphinx"
      )
    endif()
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "from rocm_docs import ROCmDocs"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET)
    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "Python3 package rocm_docs is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install rocm-docs-core"
      )
    endif()


    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import breathe"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET)
    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "Python3 package breathe is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install breathe"
      )
    endif()
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import myst_parser"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET)
    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "Python3 package myst_parser is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install myst_parser"
      )
    endif()

    include (FindPackageHandleStandardArgs)
 
    ## Look for an executable called sphinx-build
    find_program (SPHINX_EXECUTABLE
                  NAMES sphinx-build sphinx-build.exe
                  DOC "Sphinx Documentation Generator"
    )

    if(NOT SPHINX_EXECUTABLE)
      message(FATAL_ERROR "Failed to find sphinx-build executable.")
    endif()
    
    message(STATUS "Building Sphinx documentation.")

    execute_process(
      COMMAND ${SPHINX_EXECUTABLE} -M html ./rst/ ./sphinx
      RESULT_VARIABLE EXIT_CODE
      WORKING_DIRECTORY ${W_DIR})

    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "Sphinx document generation failed: ${EXIT_CODE}"
      )
    else()
      message(STATUS "Sphinx generated successfully at ${W_DIR}/sphinx/html")
      file(REMOVE_RECURSE ${W_DIR}/sphinx/doctrees)
    endif()
else() # !(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs/xml/index.xml)
    message(STATUS "Doxygen document generation failed.")
endif() # (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs/xml/index.xml)
