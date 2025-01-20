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
    set(LCOV_TEST_FILTERS "${ROOT_DIR}/test/*")

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
# Test case comments are in the given format                                                bench-type | datatype | data-model | dim | batched | solver | placement | direction | variant | intention(optional)
set(TEST_SUITE
    "C_DIRECT_PERFORMANCE ./aocl_fftz_bench 14 -p f"                                        # perf     |  double  | LP64       | 1D  |  no     | direct    | outplace  | forward   | C       | performance
    "C_BATCHED_CT_WITH_SAME_STRIDE ./aocl_fftz_bench 7v48:2:2 -b a"                         # accuracy |  double  | LP64       | 1D  |  yes    | CT        | outplace  | forward   | C       | C accuracy
    "SIZE_ONE_IN_PLACE ./aocl_fftz_bench 1 -b a -r i"                                       # accuracy |  double  | LP64       | 1D  |  no     | sizeone   | inplace   | forward   | C       | size one inplace
    "SIZE_ONE_FUSABLE_ND_BATCHED ./aocl_fftz_bench 2x5v1 -b a -p f"                         # accuracy |  float   | LP64       | 1D  |  yes    | sizeone   | outplace  | forward   | C       | size one batched
    "SIZE_ONE_NON_FUSABLE_ND_BATCHED ./aocl_fftz_bench 5:24:34x2:12:17x5:2:3v1:2:3 -b a"    # accuracy |  double  | LP64       | 1D  |  yes    | sizeone   | outplace  | forward   | C       | size one batched strided
    "C_MULTILEVEL_CT_WITH_STRIDES ./aocl_fftz_bench 15015:3:2 -b a -m i -p f"               # accuracy |  float   | ILP64      | 1D  |  no     | CT        | outplace  | forward   | C       | C ILP64 data model, covers direct prime kernels
    "C_CT_WITH_BLUESTEIN ./aocl_fftz_bench 170:4:4 -b a -r i"                               # accuracy |  double  | LP64       | 1D  |  no     | CT+BS     | inplace   | forward   | C       | C Inplace
    "C_BLUESTEIN_WITH_STRIDES ./aocl_fftz_bench 23:5:7 -b a -p f --dir b"                   # accuracy |  float   | LP64       | 1D  |  no     | BS        | outplace  | backward  | C       | C Float Inverse
    "AVX512_BLUESTEIN_WITH_STRIDES ./aocl_fftz_bench 823:7:7 -b a -r i -o 4"                # accuracy |  double  | LP64       | 1D  |  no     | BS        | inplace   | forward   | AVX512  | AVX512 Bluestein
    "AVX128_DIRECT_WITH_SAME_STRIDE_D ./aocl_fftz_bench 15:5:5 -b a -o 2"                   # accuracy |  double  | LP64       | 1D  |  no     | direct    | outplace  | forward   | AVX128  | AVX128 direct
    "AVX128_CT_WITH_SAME_STRIDE_F ./aocl_fftz_bench 5005:8:5 -b a -p f -m i -o 2 --dir b"   # accuracy |  float   | ILP64      | 1D  |  no     | CT        | outplace  | backward  | AVX128  | AVX128 Float Inverse ILP64, covers direct prime kernels
    "AVX128_3D_BATCHED_3D_ACCURACY ./aocl_fftz_bench 2x4x8v200x17x2 -b a -o 2"              # accuracy |  double  | LP64       | 3D  |  yes    | NDim      | outplace  | forward   | AVX128  | AVX128 multiple solvers
    "AVX256_DIRECT_WITH_SAME_STRIDE_D ./aocl_fftz_bench 2v16:2:2 -b a -o 3"                 # accuracy |  double  | LP64       | 1D  |  yes    | direct    | outplace  | forward   | AVX256  | AVX256 direct
    "AVX256_CT_WITH_SAME_STRIDE_F ./aocl_fftz_bench 2v40:2:2 -b a -p f -o 3 -l 3"           # accuracy |  float   | LP64       | 1D  |  yes    | CT        | outplace  | forward   | AVX256  | AVX256 Float logger mode INFO
    "AVX256_2D_BATCH_3D_DIM ./aocl_fftz_bench 3x2v56x19x12 -b a --dir b -o 3 -m i"          # accuracy |  double  | ILP64      | 3D  |  yes    | NDim      | outplace  | backward  | AVX256  | AVX256 Inverse FFT ILP64
    "C_3D_WITH_STRIDE_GREATER_THAN_DIM ./aocl_fftz_bench 12x28:4:5x2 -b a -p f"             # accuracy |  float   | LP64       | 3D  |  no     | NDim      | outplace  | forward   | C       | C 3D FFT
    "C_4D_WITH_STRIDE ./aocl_fftz_bench 4x64x9x10:3:3 -b a -p f -r i"                       # accuracy |  float   | LP64       | 4D  |  no     | NDim      | inplace   | forward   | C       | C 4D FFT
    "C_BLUESTEIN_UNIT_STRIDE ./aocl_fftz_bench 29:1:1 -b a"                                 # accuracy |  double  | LP64       | 1D  |  no     | bluestein | outplace  | forward   | C       | C unit strided bluestein to pick memcpy path
    )

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
