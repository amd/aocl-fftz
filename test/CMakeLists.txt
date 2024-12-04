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
# @brief CMake based build system for the test binary and code coverage.
#
# Adds CTest for test bench. Generates code coverage report.
#
# @author Ashwin Godbole
# @author Jeya R
# @author Prasandh Sankarankutty

# only run if code coverage is enabled
if(CODE_COVERAGE)
    find_program(GCOV gcov)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if(NOT (GCOV AND LCOV AND GENHTML))
        message(WARNING "Required tools not found: gcov,lcov,genhtml. Code coverage can not be performed")
    endif()

    # set lcov variable to exclude paths during code coverage
    set(LCOV_FILTERS "'/usr/*';'${ROOT_DIR}/build/*'")
    set(LCOV_TEST_FILTERS "'${ROOT_DIR}/gtest/*';'${ROOT_DIR}/test/*'")

    # add custom target for running ctest and generating code coverage report
    add_custom_target(code-coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory coverage/
        COMMAND ctest -j `nproc`
        COMMAND ${LCOV} --rc branch_coverage=1 --capture --directory . --output-file coverage/coverage.info
        # coverage report : library + tests
        COMMAND ${LCOV} --rc branch_coverage=1 --remove  coverage/coverage.info ${LCOV_FILTERS} --output-file coverage/coverage_filtered.info
        COMMAND ${LCOV} --list coverage/coverage_filtered.info
        COMMAND ${GENHTML} --rc branch_coverage=1 coverage/coverage_filtered.info --output-directory coverage/html_report
        # coverage report excluding tests
        COMMAND ${LCOV} --rc branch_coverage=1 --remove  coverage/coverage.info ${LCOV_FILTERS} ${LCOV_TEST_FILTERS} --output-file coverage/coverage_test_filtered.info
        COMMAND ${LCOV} --list coverage/coverage_test_filtered.info
        COMMAND ${GENHTML} --rc branch_coverage=1 coverage/coverage_test_filtered.info --output-directory coverage/html_report_test_filtered
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Code Coverage Report (LCOV) after tests.."
    )
endif()

if(VALGRIND)
    set(MEMORYCHECK_COMMAND_OPTIONS "-q --trace-children=yes --child-silent-after-fork=yes --leak-check=full --show-leak-kinds=all --track-origins=yes")
    # To run all registered tests with valgrind use "ctest -T memcheck"
    include(CTest) # Default to using valgrind
endif()

# Add testcases to the list as 'testcase_name testcase_command'
set(TEST_SUITE
    "DIRECT_PERFORMANCE ./aocl_fftz_bench 7"                                       # performance
    "BATCHED_CT_WITH_SAME_STRIDE ./aocl_fftz_bench 7v18:2:2 -b a"                  # accuracy
    "SIZE_ONE ./aocl_fftz_bench 1 -b a"                                            # size one accuracy
    "SIZE_ONE_BATCHED ./aocl_fftz_bench 5v1:2:3 -b a"                              # size one batched
    "MULTILEVEL_CT_WITH_STRIDES ./aocl_fftz_bench 128:3:2 -b a -m i"               # ILP64 data model
    "CT_WITH_BLUESTEIN ./aocl_fftz_bench 38:4:4 -b a -r i"                         # Inplace
    "BLUESTEIN_WITH_STRIDES ./aocl_fftz_bench 23:5:7 -b a -p f"                    # Float
    "DIRECT_AVX_WITH_SAME_STRIDE_D ./aocl_fftz_bench 15:5:5 -b a -o 2"             # AVX128
    "CT_AVX_WITH_SAME_STRIDE_F ./aocl_fftz_bench 18:5:5 -b a -p f -m i -r i -o 2"  # AVX128
    "DIRECT_AVX2_WITH_SAME_STRIDE_D ./aocl_fftz_bench 8:2:2 -b a -o 3"             # AVX256
    "CT_AVX2_WITH_SAME_STRIDE_F ./aocl_fftz_bench 22:2:2 -b a -p f -o 3 -l 3"      # logger mode INFO
    "2D_BATCH_3D_DIM ./aocl_fftz_bench 3x2v17x19x8 -b a --dir b"                   # Inverse FFT
    "3D_WITH_STRIDE_GREATER_THAN_DIM ./aocl_fftz_bench 2x14:4:5x3 -b a"
    "4D_WITH_STRIDE ./aocl_fftz_bench 2x72x37x13:2:3 -b a"
    "CT_ALL_KERNELS_1 ./aocl_fftz_bench 1287 -b a -o 3"                            # covers 9,11,13 kernels
    "CT_ALL_KERNELS_2 ./aocl_fftz_bench 3360 -b a -o 3"                            # covers every other radix kernels
    "BLUESTEIN_UNIT_STRIDE ./aocl_fftz_bench 29:1:1 -b a"                          # unit strided bluestein to pick memcpy path
    "2D_BATCHED_3D_PERFORMANCE ./aocl_fftz_bench 2x4v200x17x2")

list(LENGTH TEST_SUITE N)
math(EXPR N "${N} - 1")  # Adjust length to get zero-based index

# Iterate over the test cases and add them dynamically
foreach(index RANGE ${N})
    list(GET TEST_SUITE ${index} TEST_CASE)
    # Use separate_arguments to split the string into a list
    set(TEST_CASE_ARGS)
    separate_arguments(TEST_CASE_ARGS NATIVE_COMMAND "${TEST_CASE}")
    list(GET TEST_CASE_ARGS 0 TEST_CASE_NAME)
    list(REMOVE_AT TEST_CASE_ARGS 0)
    # Add the test
    add_test(
        NAME AOCL_FFTZ_TESTBENCH_${TEST_CASE_NAME}
        COMMAND ${TEST_CASE_ARGS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()
