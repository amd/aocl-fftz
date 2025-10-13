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
    set(LCOV_TEST_FILTERS "'${ROOT_DIR}/test/*'")

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
# Test case comments are in the given format
#                                                                                                             bench-type | fft-type | datatype | data-model | dim | batched | solver | placement | direction | variant | intention(optional)
set(STANDARD_TEST_SUITE
    "C_DIRECT_PERFORMANCE ./aocl_fftz_bench 14"                                                               # perf     | complex  | double   | LP64       | 1D  |  no     | direct    | outplace  | forward   | C       | performance
    "C_BATCHED_CT_WITH_SAME_STRIDE ./aocl_fftz_bench 7v48:2:2 -b a"                                           # accuracy | complex  | double   | LP64       | 1D  |  yes    | CT        | outplace  | forward   | C       | C accuracy
    "SIZE_ONE_IN_PLACE ./aocl_fftz_bench 1 -b a -r i"                                                         # accuracy | complex  | double   | LP64       | 1D  |  no     | sizeone   | inplace   | forward   | C       | size one inplace
    "SIZE_ONE_FUSABLE_ND_BATCHED ./aocl_fftz_bench 2x5v1 -b a -p f"                                           # accuracy | complex  | float    | LP64       | 1D  |  yes    | sizeone   | outplace  | forward   | C       | size one batched
    "SIZE_ONE_NON_FUSABLE_ND_BATCHED ./aocl_fftz_bench 5:24:34x2:12:17x5:2:3v1:2:3 -b a"                      # accuracy | complex  | double   | LP64       | 1D  |  yes    | sizeone   | outplace  | forward   | C       | size one batched strided
    "C_MULTILEVEL_CT_WITH_STRIDES ./aocl_fftz_bench 15015:3:2 -b a -m i -p f"                                 # accuracy | complex  | float    | ILP64      | 1D  |  no     | CT        | outplace  | forward   | C       | C ILP64 data model, covers direct prime kernels
    "C_CT_WITH_BLUESTEIN ./aocl_fftz_bench 170:4:4 -b a -r i"                                                 # accuracy | complex  | double   | LP64       | 1D  |  no     | CT+BS     | inplace   | forward   | C       | C Inplace
    "C_BLUESTEIN_WITH_STRIDES ./aocl_fftz_bench 23:5:7 -b a -p f --dir b"                                     # accuracy | complex  | float    | LP64       | 1D  |  no     | BS        | outplace  | backward  | C       | C Float Inverse
    "C_DIRECT_BATCHED ./aocl_fftz_bench 7v11:2:3 -b a --dir b"                                                # accuracy | complex  | double   | LP64       | 1D  | yes     | direct    | outplace  | backward  | C       | C batched 1D direct
    "AVX128_2D_CT_WITH_BLUESTEIN_WITH_STRIDES_D ./aocl_fftz_bench -b a 19x190:1:4 -o 1 --dir b"               # accuracy | complex  | double   | LP64       | 2D  | no      | CT+BS     | outplace  | backward  | AVX128  | AVX128 accuracy, covers direct prime kernels
    "AVX128_SIZE_ONE_BATCHED_F ./aocl_fftz_bench 4v1:2:1 -b a -p f -m i -o 1 --dir b"                         # accuracy | complex  | float    | ILP64      | 1D  | yes     | sizeone   | outplace  | backward  | AVX128  | AVX128 size one inplace
    "AVX128_3D_BATCHED_ACCURACY_D ./aocl_fftz_bench 2v2x1x1048 -b a -o 1"                                     # accuracy | complex  | double   | LP64       | 3D  | yes     | NDim      | outplace  | forward   | AVX128  | AVX128 multiple solvers
    "AVX256_DIRECT_WITH_SAME_STRIDE_D ./aocl_fftz_bench 15:5:5 -b a -o 2"                                     # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | outplace  | forward   | AVX256  | AVX256 direct
    "AVX256_CT_WITH_SAME_STRIDE_F ./aocl_fftz_bench 5005:8:5 -b a -p f -m i -o 2 --dir b"                     # accuracy | complex  | float    | ILP64      | 1D  | no      | CT        | outplace  | backward  | AVX256  | AVX256 Float Inverse ILP64, covers direct prime kernels
    "AVX256_3D_BATCHED_3D_ACCURACY ./aocl_fftz_bench 2x4x8v200x17x2 -b a -o 2"                                # accuracy | complex  | double   | LP64       | 3D  | yes     | NDim      | outplace  | forward   | AVX256  | AVX256 multiple solvers
    "AVX512_DIRECT_WITH_SAME_STRIDE_D ./aocl_fftz_bench 2v16:2:2 -b a -o 3"                                   # accuracy | complex  | double   | LP64       | 1D  | yes     | direct    | outplace  | forward   | AVX512  | AVX512 direct
    "AVX512_CT_WITH_SAME_STRIDE_F ./aocl_fftz_bench 2v40:2:2 -b a -p f -o 3"                                  # accuracy | complex  | float    | LP64       | 1D  | yes     | CT        | outplace  | forward   | AVX512  | AVX512 Float logger mode INFO
    "AVX512_2D_BATCH_3D_DIM ./aocl_fftz_bench 3x2v56x19x12 -b a --dir b -o 3 -m i"                            # accuracy | complex  | double   | ILP64      | 3D  | yes     | NDim      | outplace  | backward  | AVX512  | AVX512 Inverse FFT ILP64
    "AVX512_BATCHED_DIRECT ./aocl_fftz_bench 3v14 -b a -p f -m i -o 3"                                        # accuracy | complex  | float    | ILP64      | 1D  | yes     | direct    | outplace  | forward   | AVX512  | AVX512 batched direct
    "C_2D_BATCHED ./aocl_fftz_bench 3v13x49 -b a"                                                             # accuracy | complex  | double   | LP64       | 2D  | yes     | NDim      | outplace  | forward   | C       | C 2D Batched
    "C_1D_BLUESTEIN_IN_PLACE ./aocl_fftz_bench 17 -b a -r i"                                                  # accuracy | complex  | double   | LP64       | 1D  | no      | BS        | inplace   | forward   | C       | C Inplace Bluestein
    "AVX128_4D ./aocl_fftz_bench 2x21x3x60:1:1 -b a -p f -o 1"                                                # accuracy | complex  | float    | LP64       | 4D  | no      | NDim      | outplace  | forward   | AVX128  | AVX128 4D CT
    "AVX256_BATCHED_CT ./aocl_fftz_bench 17v190 -b a -m i -o 2 --dir b"                                       # accuracy | complex  | double   | ILP64      | 1D  | yes     | CT+BS     | outplace  | backward  | AVX256  | AVX256 CT accuracy
    "C_3D_CT ./aocl_fftz_bench 7x2x24 -b a -p f --dir b"                                                      # accuracy | complex  | float    | LP64       | 3D  | no      | NDim      | outplace  | backward  | C       | C 3D accuracy
    "C_1D_BLUESTEIN ./aocl_fftz_bench 31:02:02 -b a -r i --dir b"                                             # accuracy | complex  | double   | LP64       | 1D  | no      | BS        | inplace   | backward  | C       | C Inplace Bluestein
    "AVX128_BATCHED_DIRECT ./aocl_fftz_bench 12v12 -b a -o 1 --dir b"                                         # accuracy | complex  | double   | LP64       | 1D  | yes     | direct    | outplace  | backward  | AVX128  | AVX128 Batched direct accuracy
    "AVX256_SIZEONE_BATCHED ./aocl_fftz_bench 4v1:6:6 -b a -p f -o 2 --dir b"                                 # accuracy | complex  | float    | LP64       | 1D  | yes     | sizeone   | outplace  | backward  | AVX256  | AVX256 Sizeone batched accuracy
    "AVX512_4D_BATCHED ./aocl_fftz_bench 4v33x2x6x3:2:2 -b a -o 3 --dir b"                                    # accuracy | complex  | double   | LP64       | 4D  | yes     | NDim      | outplace  | backward  | AVX512  | AVX512 batched 4D accuracy
    "C_3D_WITH_STRIDE_GREATER_THAN_DIM ./aocl_fftz_bench 12x28:4:5x2 -b a -p f"                               # accuracy | complex  | float    | LP64       | 3D  |  no     | NDim      | outplace  | forward   | C       | C 3D FFT
    "C_4D_WITH_STRIDE ./aocl_fftz_bench 4x64x9x10:3:3 -b a -p f -r i"                                         # accuracy | complex  | float    | LP64       | 4D  |  no     | NDim      | inplace   | forward   | C       | C 4D FFT
    "C_BLUESTEIN_UNIT_STRIDE ./aocl_fftz_bench 29:1:1 -b a"                                                   # accuracy | complex  | double   | LP64       | 1D  |  no     | bluestein | outplace  | forward   | C       | C unit strided bluestein to pick memcpy path
    "C_COL_MAJOR ./aocl_fftz_bench 5:1:1v48:5:5 -b a"                                                         # accuracy | complex  | double   | LP64       | 1D  |  yes    | CT        | outplace  | forward   | C       | C accuracy to check column major input
    "C_BATCHED_UNEVEN_STRIDE ./aocl_fftz_bench 5:25:27v2x3x4 -b a -p f"                                       # accuracy | complex  | float    | LP64       | 1D  |  yes    | NDim      | outplace  | forward   | C       | C with uneven strides in batches
    "C_NO_BATCH_FUSION ./aocl_fftz_bench 2:36:36x2:17:17v4x4 -b a -r i"                                       # accuracy | complex  | double   | LP64       | 1D  |  yes    | NDim      | inplace   | forward   | C       | Avoid fusion of no batch cases
    "C_MT_BATCHED_DIRECT ./aocl_fftz_bench 200v16 -b a -n 2 -r i "                                            # accuracy | complex  | double   | LP64       | 1D  |  yes    | MT+direct | inplace   | forward   | C       | C MT 1D Batched-direct
    "C_MT_BATCHED_CT ./aocl_fftz_bench 20v256 -b a -r i -n 2"                                                 # accuracy | complex  | double   | LP64       | 1D  |  yes    | MT+CT     | inplace   | forward   | C       | C MT 1D Batched CT
    "C_MT_2D ./aocl_fftz_bench 256x128 -b a -n 4"                                                             # accuracy | complex  | double   | LP64       | 2D  |  no     | NDim+MT   | outplace  | forward   | C       | C MT 2D
    "C2C_5D_IP ./aocl_fftz_bench 2x3x4x5x6:2:2 -b a --dir f -p f -r i"                                        # accuracy | complex  | float    | LP64       | 5D  |  no     | NDim      | inplace   | forward   | C       | C MT 5D
    "C2C_5D_CTBS ./aocl_fftz_bench 3x3x3x74x18:4:4 -b a --dir f -p f -o 0"                                    # accuracy | complex  | float    | LP64       | 5D  | no      | CT+BS     | outplace  | forward   | C       | 5D CT+BS, float, outplace, forward
    "C_MT_BATCHED_3D ./aocl_fftz_bench 2v20x10x48 -b a -n 8"                                                  # accuracy | complex  | double   | LP64       | 3D  |  yes    | MT+NDim   | outplace  | forward   | C       | C MT Batched 3D
    "C_MT_MULTILEVEL_CT_WITH_STRIDES ./aocl_fftz_bench 15000:3:2 -b a -p f -n 8"                              # accuracy | complex  | float    | LP64       | 1D  |  no     | CT+MT     | outplace  | forward   | C       | C MT multilevel CT
    "C2C_MT_BATCHED_DIRECT_IN_PLACE ./aocl_fftz_bench 17v9:4:4 -b a -r i -n 2"                                # accuracy | complex  | double   | LP64       | 1D  | yes     | direct    | inplace   | forward   | C       | MT batched direct forward accuracy
    "C2C_MT_4D_DIRECT_IN_PLACE ./aocl_fftz_bench 8x4x32x12:8:8 -b a -p f -r i -o 3 -n 4"                      # accuracy | complex  | float    | LP64       | 4D  | no      | NDim      | inplace   | forward   | AVX512  | MT 4D direct forward accuracy
    "C2C_MT_2D_CT_IN_PLACE ./aocl_fftz_bench 8x100 -b a -r i -o 1 -n 6"                                       # accuracy | complex  | double   | LP64       | 2D  | no      | CT        | inplace   | forward   | AVX128  | MT 2D CT forward accuracy
    "C2C_MT_DIRECT_1D_IN_PLACE ./aocl_fftz_bench 3:02:02 -b a -r i -o 2 -n 8"                                 # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | inplace   | forward   | AVX256  | MT direct forward accuracy
    "C2C_MT_BLUESTEIN_IN_PLACE_FWD ./aocl_fftz_bench 29:01:01 -b a -m i -r i -o 2 -n 10"                      # accuracy | complex  | double   | ILP64      | 1D  | no      | bluestein | inplace   | forward   | AVX256  | MT Bluestein forward accuracy
    "C2C_MT_SIZEONE_IN_PLACE ./aocl_fftz_bench 1v1 -b a -r i -n 12"                                           # accuracy | complex  | double   | LP64       | 1D  | yes     | sizeone   | inplace   | forward   | C       | MT sizeone forward accuracy
    "C2C_MT_CT_1D ./aocl_fftz_bench 210:09:09 -b a -p f -r i -o 3 --dir b -n 2"                               # accuracy | complex  | float    | LP64       | 1D  | no      | CT        | inplace   | backward  | AVX512  | MT CT backward accuracy
    "C2C_MT_BATCHED_CT ./aocl_fftz_bench 12v28:10:10 -b a -p f -r i -o 1 --dir b -n 4"                        # accuracy | complex  | float    | LP64       | 1D  | yes     | CT        | inplace   | backward  | AVX128  | MT batched CT backward accuracy
    "C2C_MT_BLUESTEIN_IN_PLACE_BWD ./aocl_fftz_bench 59:01:01 -b a -m i -r i --dir b -n 6"                    # accuracy | complex  | double   | ILP64      | 1D  | no      | bluestein | inplace   | backward  | C       | MT Bluestein backward accuracy
    "C2C_MT_DIRECT_1D_IN_PLACE_BWD ./aocl_fftz_bench 2:01:01 -b a -r i -o 2 --dir b -n 8"                     # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | inplace   | backward  | AVX256  | MT direct backward accuracy
    "C2C_MT_SIZEONE_IN_PLACE_BWD ./aocl_fftz_bench 1v1:5:5 -b a -r i --dir b -n 10"                           # accuracy | complex  | double   | LP64       | 1D  | yes     | sizeone   | inplace   | backward  | C       | MT sizeone backward accuracy
    "C2C_MT_3D_IN_PLACE_BWD ./aocl_fftz_bench 4x18x32:3:3 -b a -p f -r i -o 3 --dir b -n 12"                  # accuracy | complex  | float    | LP64       | 3D  | no      | NDim      | inplace   | backward  | AVX512  | MT 3D direct backward accuracy
    "C2C_MT_BATCHED_DIRECT ./aocl_fftz_bench 8v16:8:4 -b a -n 8"                                              # accuracy | complex  | double   | LP64       | 1D  | yes     | direct    | outplace  | forward   | C       | MT batched direct forward accuracy
    "C2C_MT_3D_DIRECT ./aocl_fftz_bench 4x8x16:8:4 -b a -p f -o 3 -n 4"                                       # accuracy | complex  | float    | LP64       | 3D  | no      | NDim      | outplace  | forward   | AVX512  | MT 3D direct forward accuracy
    "C2C_MT_2D_CT ./aocl_fftz_bench 8x16:8:4 -b a -o 1 -n 14"                                                 # accuracy | complex  | double   | LP64       | 2D  | no      | CT        | outplace  | forward   | AVX128  | MT 2D CT forward accuracy
    "C2C_MT_DIRECT_1D ./aocl_fftz_bench 16:08:04 -b a -o 2 -n 15"                                             # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | outplace  | forward   | AVX256  | MT direct forward accuracy
    "C2C_MT_BLUESTEIN_FWD ./aocl_fftz_bench 29:01:01 -b a -m i -o 2 -n 2"                                     # accuracy | complex  | double   | ILP64      | 1D  | no      | bluestein | outplace  | forward   | AVX256  | MT Bluestein forward accuracy
    "C2C_MT_SIZEONE ./aocl_fftz_bench 1v1 -b a -n 9"                                                          # accuracy | complex  | double   | LP64       | 1D  | yes     | sizeone   | outplace  | forward   | C       | MT sizeone forward accuracy
    "C2C_MT_CT_1D_BWD ./aocl_fftz_bench 210:10:07 -b a -p f -o 3 --dir b -n 4"                                # accuracy | complex  | float    | LP64       | 1D  | no      | CT        | outplace  | backward  | AVX512  | MT CT backward accuracy
    "C2C_MT_BATCHED_CT_BWD ./aocl_fftz_bench 12v24:10:7 -b a -p f -o 1 --dir b -n 4"                          # accuracy | complex  | float    | LP64       | 1D  | yes     | CT        | outplace  | backward  | AVX128  | MT batched CT backward accuracy
    "C2C_MT_BLUESTEIN ./aocl_fftz_bench 59:01:06 -b a -m i --dir b -n 16"                                     # accuracy | complex  | double   | ILP64      | 1D  | no      | bluestein | outplace  | backward  | C       | MT Bluestein backward accuracy
    "C2C_MT_DIRECT_1D_BWD ./aocl_fftz_bench 10:02:04 -b a -o 2 --dir b -n 11"                                 # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | outplace  | backward  | AVX256  | MT direct backward accuracy
    "C2C_MT_SIZEONE_BWD ./aocl_fftz_bench 1v1:5:4 -b a --dir b -n 10"                                         # accuracy | complex  | double   | LP64       | 1D  | yes     | sizeone   | outplace  | backward  | C       | MT sizeone backward accuracy
    "C2C_MT_3D_DIRECT_BWD ./aocl_fftz_bench 7x4x32x18:8:4 -b a -p f -o 3 --dir b -n 2"                        # accuracy | complex  | float    | LP64       | 4D  | no      | NDim      | outplace  | backward  | AVX512  | MT 4D direct backward accuracy
    "C2AVX512_5D_CTBS_BATCHED ./aocl_fftz_bench 3v6x3x3x18x18:4:4 -b a -p d -o 3 --dir b"                     # accuracy | complex  | double   | LP64       | 5D  | yes     | CT+BS     | outplace  | backward  | AVX512  | 5D batched CT+BS, double, outplace, backward
    "AVX512_MT_3D ./aocl_fftz_bench 30x40x50 -b a -n 8"                                                       # accuracy | complex  | double   | LP64       | 3D  |  no     | MT+NDim   | outplace  | forward   | AVX512  | AVX512 MT 3D outofplace accuracy
    "AVX512_MT_3D_IP ./aocl_fftz_bench 48x49x50 -b a -n 8 --dir b"                                            # accuracy | complex  | double   | LP64       | 3D  |  no     | MT+NDim   | inplace   | backward  | AVX512  | AVX512 MT 3D inplace accuracy
    "R2C_C_DIRECT_ODD ./aocl_fftz_bench 7:3:3 -f r2c -b a -p f"                                               # accuracy | real     | float    | LP64       | 1D  | no      | direct    | outplace  | forward   | C       | C real 1D Direct
    "R2C_C_DIRECT_ODD_IP ./aocl_fftz_bench 7:3:3 -f r2c -b a -p f -r i"                                       # accuracy | real     | float    | LP64       | 1D  | no      | direct    | inplace   | forward   | C       | C real 1D Direct, inplace
    "R2C_C_BATCHED_DIRECT_ODD ./aocl_fftz_bench 7v5:3:3 -f r2c -b a"                                          # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | forward   | C       | C real 1D Batched-direct
    "R2C_AVX128_DIRECT_EVEN ./aocl_fftz_bench 6:3:3 -f r2c -b a -p f -o 1"                                    # accuracy | real     | float    | LP64       | 1D  | no      | direct    | outplace  | forward   | AVX128  | AVX128 C real 1D Direct
    "R2C_AVX128_BATCHED_DIRECT_ODD_IP ./aocl_fftz_bench 7v5:3:3 -f r2c -b a -r i --dir b"                     # accuracy | real     | double   | LP64       | 1D  | yes     | direct    | inplace   | backward  | AVX128  | AVX128 real 1D Batched-direct
    "R2C_C_CT_ODD_IP ./aocl_fftz_bench 2205:7:7 -f r2c -b a -p f -r i"                                        # accuracy | real     | float    | LP64       | 1D  | no      | CT        | inplace   | forward   | C       | C real 1D real CT, inplace
    "R2C_C_BATCHED_CT_ODD ./aocl_fftz_bench 15v315:4:4 -f r2c -b a"                                           # accuracy | real     | double   | LP64       | 1D  | yes     | CT        | outplace  | forward   | C       | C real Batched-CT
    "R2C_C_BATCHED_CT_ODD_IP ./aocl_fftz_bench 15v315:4:4 -f r2c -b a -r i"                                   # accuracy | real     | double   | LP64       | 1D  | yes     | CT        | inplace   | forward   | C       | C real Batched-CT, inplace
    "R2C_C_DIRECT_EVEN_IP ./aocl_fftz_bench 6:3:3 -f r2c -b a -p f -r i"                                      # accuracy | real     | float    | LP64       | 1D  | no      | direct    | inplace   | forward   | C       | C real 1D Direct, inplace
    "R2C_C_BATCHED_DIRECT_EVEN ./aocl_fftz_bench 7v6:3:3 -f r2c -b a --dir b"                                 # accuracy | real     | double   | LP64       | 1D  | yes     | direct    | outplace  | backward  | C       | C real 1D Batched-direct
    "R2C_C_BATCHED_DIRECT_EVEN_IP ./aocl_fftz_bench 7v6:3:3 -f r2c -b a -r i --dir b"                         # accuracy | real     | double   | LP64       | 1D  | yes     | direct    | inplace   | backward  | C       | C real 1D Batched-direct, inplace
    "R2C_AVX256_CT_EVEN ./aocl_fftz_bench 5v1000:10:10 -f r2c -b a -p f -o 2"                                 # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | outplace  | forward   | AVX256  | AVX256 real 1D CT
    "R2C_C_CT_EVEN_IP ./aocl_fftz_bench 1000:10:10 -f r2c -b a -p f  -r i"                                    # accuracy | real     | float    | LP64       | 1D  | no      | CT        | inplace   | forward   | C       | C real 1D CT, inplace
    "R2C_C_BATCHED_CT_EVEN ./aocl_fftz_bench 15v200:4:4 -f r2c -b a"                                          # accuracy | real     | double   | LP64       | 1D  | yes     | CT        | outplace  | forward   | C       | C 1D Batched CT
    "R2C_C_BATCHED_CT_EVEN_IP ./aocl_fftz_bench 15v200:4:4 -f r2c -b a -r i -o 2"                             # accuracy | real     | double   | LP64       | 1D  | yes     | CT        | inplace   | forward   | AVX256  | C 1D Batched CT, inplace
    "R2C_AVX128_BATCHED_DIRECT_INPLACE ./aocl_fftz_bench 8v5:3:3 -b a -r i -f r2c -o 1"                       # accuracy | real     | double   | LP64       | 1D  | yes     | direct    | inplace   | forward   | AVX128  | AVX128 inplace batched forward accuracy
    "R2C_AVX128_BATCHED_DIRECT_IN_PLACE_BWD ./aocl_fftz_bench 144:1:1 -b a -r i -f r2c -o 1 --dir b"          # accuracy | real     | double   | LP64       | 1D  | yes     | direct    | inplace   | backward  | AVX128  | AVX128 inplace batched backward accuracy
    "R2C_AVX256_BATCHED_CT_EVEN_OUTPLACE_BWD ./aocl_fftz_bench 5v1000:10:10 -f r2c -b a --dir b -p f -o 2"    # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | outplace  | backward  | AVX256  | AVX256 1D batched even CT real outplace backward
    "R2C_AVX256_CT_WITH_SAME_STRIDE_F ./aocl_fftz_bench 210:2:2 -b a -r i -p f -f r2c -o 2 --dir b"           # accuracy | real     | float    | LP64       | 1D  | no      | CT        | outplace  | backward  | AVX256  | AVX256 Inplace CT backward accuracy
    "R2C_AVX512_BATCHED_CT ./aocl_fftz_bench 15v81 -b a -r i -p f -f r2c -o 3"                                # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | inplace   | forward   | AVX512  | AVX512 inplace CT accuracy
    "C2R_AVX512_BATCHED_CT ./aocl_fftz_bench 15v24 -b a -r i -p f -f r2c -o 3 --dir b"                        # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | inplace   | backward  | AVX512  | AVX512 Inplace batched ct accuracy
    "R2C_AVX512_CT_ODD ./aocl_fftz_bench 2205:7:7 -f r2c -b a -p f -o 3 --dir b"                              # accuracy | real     | float    | LP64       | 1D  | no      | CT        | outplace  | backward  | AVX512  | AVX512 1D CT accuracy
    "R2C_AVX512_CT_ODD_BWD ./aocl_fftz_bench 2205:7:7 -f r2c -b a -p f -o 3"                                  # accuracy | real     | float    | LP64       | 1D  | no      | CT        | outplace  | forward   | AVX512  | AVX512 BWD 1D Batched-CT
    "R2C_AVX256_CT_EVEN_BWD ./aocl_fftz_bench 5v1000:10:10 -f r2c -b a -p f -o 2 --dir b"                     # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | outplace  | backward  | AVX256  | AVX256 1D CT accuracy
    "C2R_C_DIRECT ./aocl_fftz_bench 8:4:2 -f r2c -b a -p f --dir b"                                           # accuracy | real     | float    | LP64       | 1D  | no      | direct    | outplace  | backward  | C       | C Bwd real Direct accuracy
    "C2R_C_BATCHED_DIRECT ./aocl_fftz_bench 7v5:4:3 -f r2c -b a --dir b"                                      # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | backward  | C       | C Bwd real Batched-direct
    "C2R_C_CT ./aocl_fftz_bench 60:3:5 -f r2c -b a -p f --dir b"                                              # accuracy | real     | float    | LP64       | 1D  |  no     | CT        | outplace  | backward  | C       | C Bwd real Batched-CT
    "C2R_C_CT_IP ./aocl_fftz_bench 25000 -b a -f c2r -r i"                                                    # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | inplace   | backward  | C       | C Bwd real CT, inplace
    "C2R_C_BATCHED_CT ./aocl_fftz_bench 10v24:5:5 -f r2c -b a -p d --dir b -r i"                              # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | backward  | C       | C Bwd real Batched-CT
    "C2R_C_MT_BATCHED_DIRECT_1 ./aocl_fftz_bench 13v6 -f r2c -b a --dir b -n 8"                               # accuracy | real     | double   | LP64       | 1D  |  no     | direct    | outplace  | backward  | C       | C MT Bwd real Batched direct
    "R2C_C_MT_BATCHED_DIRECT_WITH_STRIDE_1 ./aocl_fftz_bench 17v5:4:5 -f r2c -b a --dir f -n 3"               # accuracy | real     | double   | LP64       | 1D  |  no     | direct    | outplace  | forward   | C       | C MT real Batched direct
    "C2R_C_MT_BATCHED_DIRECT_2 ./aocl_fftz_bench 16v5 -f r2c -b a --dir b -n 15"                              # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | backward  | C       | C MT Bwd real Batched direct
    "R2C_C_MT_BATCHED_DIRECT_WITH_STRIDE_2 ./aocl_fftz_bench 24v6:3:5 -f r2c -b a --dir f -n 24"              # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | forward   | C       | C MT real Batched direct with strides
    "C2R_C_MT_DIRECT_SAME_STRIDE_1 ./aocl_fftz_bench 13v7:4:4 -f r2c -b a --dir b -r i -n 7"                  # accuracy | real     | double   | LP64       | 1D  |  no     | direct    | inplace   | backward  | C       | C MT Bwd real Batched direct with strides, inplace
    "R2C_C_MT_DIRECT_SAME_STRIDE_F_1 ./aocl_fftz_bench 16v3:5:5 -f r2c -b a -p f --dir f -r i -n 6"           # accuracy | real     | float    | LP64       | 1D  |  no     | direct    | inplace   | forward   | C       | C MT real direct, inplace
    "C2R_C_MT_BATCHED_DIRECT_SAME_STRIDE ./aocl_fftz_bench 16v7:6:6 -f r2c -b a --dir b -r i -n 8"            # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | inplace   | backward  | C       | C MT Bwd real Batched direct, inplace
    "R2C_C_MT_BATCHED_DIRECT_SAME_STRIDE_F ./aocl_fftz_bench 30v6:7:7 -f r2c -b a -p f --dir f -r i -n 10"    # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | C       | C MT real Batched direct, inplace
    "R2C_C_MT_BATCHED_CT_WITH_STRIDE ./aocl_fftz_bench 24v60:3:5 -f r2c -b a --dir f -n 24"                   # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | outplace  | forward   | C       | C MT real Batched CT, with strides
    "C2R_C_MT_BATCHED_CT_WITH_STRIDE ./aocl_fftz_bench 16v96:7:9 -f r2c -b a --dir b -n 15"                   # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | outplace  | backward  | C       | C MT Bwd real Batched CT, with strides
    "R2C_C_MT_BATCHED_CT_WITH_STRIDE_IP ./aocl_fftz_bench 32v56:5:5 -f r2c -b a --dir f -r i -n 24"           # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | forward   | C       | C MT real Batched CT, inplace, with strides
    "C2R_C_MT_BATCHED_CT_WITH_STRIDE_IP ./aocl_fftz_bench 12v108:9:9 -f r2c -b a --dir b -r i -n 15"          # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | backward  | C       | C MT Bwd real Batched CT, inplace, with strides
    "R2C_C_MT_CT ./aocl_fftz_bench 54 -f r2c -b a --dir f -n 3"                                               # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | outplace  | forward   | C       | C MT Batched CT
    "C2R_C_MT_CT_IP ./aocl_fftz_bench 72 -f r2c -b a --dir b -r i -n 7"                                       # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | inplace   | backward  | C       | C MT Bwd real CT, inplace
    "R2C_C_MT_CT_WITH_STRIDE ./aocl_fftz_bench 64:3:5 -f r2c -b a --dir f -n 24"                              # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | outplace  | forward   | C       | C MT real CT
    "C2R_C_MT_CT_WITH_STRIDE_IP ./aocl_fftz_bench 96:9:9 -f r2c -b a --dir b -r i -n 15"                      # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | backward  | C       | C MT Bwd real CT, inplace
    "R2C_C_MT_BATCHED_CT ./aocl_fftz_bench 4v54 -f r2c -b a --dir f -n 13"                                    # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | outplace  | forward   | C       | C MT real Batched CT
    "C2R_C_MT_BATCHED_CT_IP ./aocl_fftz_bench 15v120 -f r2c -b a --dir b -r i -n 17"                          # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | inplace   | backward  | C       | C MT Bwd real Batched CT, inplace
    "R2C_AVX512_CT_EVEN_IP ./aocl_fftz_bench 4000 -f r2c -b a -p d --dir f -o 3 -r i"                         # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | inplace   | forward   | AVX512  | AVX512 real CT, inplace, even size
    "C2R_AVX512_CT_ODD_IP ./aocl_fftz_bench 4725 -f r2c -b a -p f --dir f -o 3 -r i"                          # accuracy | real     | float    | LP64       | 1D  |  no     | CT        | inplace   | backward  | AVX512  | AVX512 Bwd real CT, inplace, odd size
    "R2C_AVX128_MT_BATCHED_DIRECT_UNIT_STRIDE_F ./aocl_fftz_bench 57v5 -f r2c -b a -r i -n 10 -o 1 -p f"      # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX128  | AVX128 MT real Batched-direct, inplace
    "C2R_AVX128_MT_BATCHED_DIRECT ./aocl_fftz_bench 58v5 -f c2r -b a -r i -n 10 -o 1"                         # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | inplace   | backward  | AVX128  | AVX128 MT Bwd real Batched-direct, inplace
    "R2C_AVX128_MT_BATCHED_DIRECT_F ./aocl_fftz_bench 121v5:2:2 -f r2c -b a -r i -n 10 -o 1 -p f"             # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX128  | AVX128 MT real Batched-direct, float, inplace
    "C2R_AVX128_MT_BATCHED_DIRECT_WITH_STRIDE_OP ./aocl_fftz_bench 122v5:2:8 -f c2r -b a -r o -n 10 -o 1"     # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | backward  | AVX128  | AVX128 MT Bwd real Batched-direct, with strides
    "R2C_AVX256_MT_BATCHED_DIRECT_UNIT_STRIDE_F ./aocl_fftz_bench 61v8 -f r2c -b a -r i -o 2 -n 6 -p f"       # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX256  | AVX256 MT real Batched-direct, inplace
    "C2R_AVX256_MT_BATCHED_DIRECT ./aocl_fftz_bench 62v8 -f c2r -b a -r i -o 2 -n 7"                          # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | inplace   | backward  | AVX256  | AVX256 MT Bwd real Batched-direct, inplace
    "R2C_AVX256_MT_BATCHED_DIRECT_WITH_STRIDE_F ./aocl_fftz_bench 122v10:4:4 -f r2c -b a -r i -o 2 -n 8 -p f" # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX256  | AVX256 MT real Batched-direct, with strides, inplace
    "C2R_AVX256_MT_BATCHED_DIRECT_OP ./aocl_fftz_bench 124v10:10:4 -f c2r -b a -r o -o 2 -n 9"                # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | backward  | AVX256  | AVX256 MT Bwd real Batched-direct
    "C2R_AVX512_MT_BATCHED_DIRECT_F ./aocl_fftz_bench 242v6 -f c2r -b a -r i -o 3 -n 14 -p f"                 # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | backward  | AVX512  | AVX512 MT Bwd real Batched-direct, inplace
    "R2C_AVX512_MT_BATCHED_DIRECT_F ./aocl_fftz_bench 243v6:7:7 -f r2c -b a -r i -o 3 -n 17 -p f"             # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX512  | AVX512 MT real Batched-direct, inplace
    "C2R_AVX512_MT_BATCHED_DIRECT ./aocl_fftz_bench 247v6:7:7 -f c2r -b a -r i -o 3 -n 18"                    # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | inplace   | backward  | AVX512  | AVX512 MT Bwd real Batched-direct, inplace
    "R2C_AVX512_MT_BATCHED_DIRECT_OP ./aocl_fftz_bench 248v6:7:9 -f r2c -b a -r o -o 3 -n 19"                 # accuracy | real     | double   | LP64       | 1D  |  yes    | direct    | outplace  | forward   | AVX512  | AVX512 MT real Batched-direct
    "R2C_AVX512_MT_BATCHED_DIRECT_UNIT_STRIDE_F ./aocl_fftz_bench 241v6 -f r2c -b a -r i -o 3 -n 13 -p f"     # accuracy | real     | float    | LP64       | 1D  |  yes    | direct    | inplace   | forward   | AVX512  | AVX512 MT real Batched-direct, inplace
    "C2R_AVX128_MT_CT ./aocl_fftz_bench 80:2:2 -f c2r -b a -r i -o 1 -n 3"                                    # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | backward  | AVX128  | AVX128 MT Bwd real CT , inplace
    "R2C_AVX128_MT_CT_WITH_STRIDE_F ./aocl_fftz_bench 35:6:6 -f r2c -b a -r i -o 1 -n 12 -p f"                # accuracy | real     | float    | LP64       | 1D  |  yes    | CT        | inplace   | forward   | AVX128  | AVX128 MT real CT , inplace
    "R2C_AVX256_MT_CT_F ./aocl_fftz_bench 100:4:4 -f r2c -b a -r i -o 2 -n 12 -p f"                           # accuracy | real     | float    | LP64       | 1D  |  yes    | CT        | inplace   | forward   | AVX256  | AVX256 MT real CT, inplace
    "C2R_AVX256_MT_CT_WITH_STRIDE ./aocl_fftz_bench 320:4:4 -f c2r -b a -r i -o 2 -n 13"                      # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | backward  | AVX256  | AVX256 Bwd MT real CT, inplace
    "R2C_AVX256_MT_CT_OP_F ./aocl_fftz_bench 1v63:5:3 -f r2c -b a -r o -o 2 -n 20 -p f"                       # accuracy | real     | float    | LP64       | 1D  |  yes    | CT        | outplace  | forward   | AVX256  | AVX256 MT real CT
    "C2R_AVX512_MT_CT_OP_F ./aocl_fftz_bench 300:5:9 -f r2c -b a -r o -o 3 -n 17 -p f"                        # accuracy | real     | float    | LP64       | 1D  | yes     | CT        | outplace  | backward  | AVX512  | AVX512 Bwd MT real CT
    "R2C_AVX512_MT_CT ./aocl_fftz_bench 200:8:8 -f r2c -b a -r i -o 3 -n 20"                                  # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | forward   | AVX512  | AVX512 MT real CT, inplace
    "R2C_AVX512_MT_CT_WITH_STRIDE_F ./aocl_fftz_bench 180:6:6 -f r2c -b a -r i -o 3 -n 31"                    # accuracy | real     | double   | LP64       | 1D  |  yes    | CT        | inplace   | forward   | AVX512  | AVX512 MT real CT, with strides, inplace
    "C2R_AVX512_MT_CT_WITH_STRIDE ./aocl_fftz_bench 196:7:7 -f r2c -b a -r i -o 3 -n 31 -p f"                 # accuracy | real     | float    | LP64       | 1D  |  yes    | CT        | inplace   | backward  | AVX512  | AVX512 Bwd MT real CT, with strides, inplace
    "R2C_AVX512_MT_CT_EVEN_IP ./aocl_fftz_bench 4000:4:4 -f r2c -b a -p d --dir f -r i -o 3 -n 8"             # accuracy | real     | double   | LP64       | 1D  |  no     | CT        | inplace   | forward   | AVX512  | AVX512 MT real CT, inplace
    "C_COLMAJOR_CTBS_PRIME ./aocl_fftz_bench 317:1:1v73:317:317 -b a"                                         # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | forward   | C       | C column major CT+BS prime
    "AVX128_COLMAJOR_CTBS_PRIME ./aocl_fftz_bench 509:1:1v129:509:509 -b a -o 1"                              # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | forward   | AVX128  | AVX128 column major CT+BS prime
    "AVX256_COLMAJOR_CTBS_PRIME ./aocl_fftz_bench 251:1:1v314:251:251 -b a -o 2 --dir b"                      # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | backward  | AVX256  | AVX256 column major CT+BS prime
    "AVX512_COLMAJOR_CTBS_PRIME ./aocl_fftz_bench 127:1:1v179:127:127 -b a -o 3 --dir b"                      # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | backward  | AVX512  | AVX512 column major CT+BS prime
    "C_COLMAJOR_CTBS ./aocl_fftz_bench 317:1:1v74:317:317 -b a"                                               # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | forward   | C       | C column major CT+BS
    "AVX128_COLMAJOR ./aocl_fftz_bench 509:1:1v94:509:509 -b a -o 1"                                          # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | forward   | AVX128  | AVX128 column major CT+BS
    "AVX256_COLMAJOR_CTBS ./aocl_fftz_bench 251:1:1v62:251:251 -b a -o 2 --dir b"                             # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | backward  | AVX256  | AVX256 column major CT+BS
    "AVX512_COLMAJOR_CTBS ./aocl_fftz_bench 127:1:1v34:127:127 -b a -o 3 --dir b"                             # accuracy | complex  | double   | LP64       | 1D  | yes     | CT+BS     | outplace  | backward  | AVX512  | AVX512 column major CT+BS
    )

    set(EXHAUSTIVE_TEST_SUITE
    "EXT_C_LARGE_PRIME_IP ./aocl_fftz_bench 10007:1:1 -b a --dir f -p f -r i"                                 # accuracy | complex  | float    | LP64       | 1D  |  no     | bluestein | inplace   | forward   | C       | C Large Prime
    "EXT_C_1D_LARGE_CT_INPLACE_F ./aocl_fftz_bench 32768 -b a -p f -r i --dir f"                              # accuracy | complex  | float    | LP64       | 1D  | no      | CT        | inplace   | forward   | C       | Large 1D CT inplace float
    "EXT_C_2D_LARGE_BATCHED_CTBS_OUTPLACE_D ./aocl_fftz_bench 128x128v64:8:8 -b a -p d --dir b"               # accuracy | complex  | double   | LP64       | 2D  | yes     | CT+BS     | outplace  | backward  | C       | Large 2D batched CT+BS outplace double
    "EXT_C_3D_LARGE_BATCHED_DIRECT_INPLACE_F ./aocl_fftz_bench 32x32x32v128:4:4 -b a -p f -r i --dir f"       # accuracy | complex  | float    | LP64       | 3D  | yes     | direct    | inplace   | forward   | C       | Large 3D batched direct inplace float
    "EXT_AVX128_1D_LARGE_DIRECT_BATCHED_F ./aocl_fftz_bench 65536v128 -b a -p f -o 1"                         # accuracy | complex  | float    | LP64       | 1D  | yes     | direct    | outplace  | forward   | AVX128  | Large 1D batched direct float AVX128
    "EXT_C_ND_CTBS ./aocl_fftz_bench 4x6x8x10x12x14:4:4 -b a --dir f -p f"                                    # accuracy | complex  | float    | LP64       | 6D  | no      | CT+BS     | outplace  | forward   | C       | 6D CT+BS, float, outplace, forward
    "EXT_AVX256_5D_CTBS_BATCHED ./aocl_fftz_bench 4v4x8x12x17x20:4:4 -b a -p f -o 2 --dir f"                  # accuracy | complex  | float    | LP64       | 5D  | yes     | CT+BS     | outplace  | forward   | AVX256  | 5D batched CT+BS, float, outplace, forward
    "EXT_AVX512_CT_IP ./aocl_fftz_bench 100000:10:10 -b a --dir f -p f -m i -o 3 -r i"                        # accuracy | complex  | float    | ILP64      | 1D  |  no     | CT        | inplace   | forward   | AVX512  | AVX512 large CT, inplace
    "EXT_AVX512_3D_IP ./aocl_fftz_bench 40x50x60:2:2 -b a --dir f -p f -o 3 -r i"                             # accuracy | complex  | float    | LP64       | 3D  |  no     | NDim      | inplace   | forward   | AVX512  | AVX512 3D, inplace
    "EXT_AVX256_2D_LARGE_CT_INPLACE_D ./aocl_fftz_bench 256x256:16:16 -b a -p d -r i -o 2"                    # accuracy | complex  | double   | LP64       | 2D  | no      | CT        | inplace   | forward   | AVX256  | Large 2D CT inplace double AVX256
    "EXT_C_4D_LARGE_CT_OUTPLACE_D_ILP64 ./aocl_fftz_bench 16x16x16x16:8:8 -b a -p d -m i --dir f"             # accuracy | complex  | double   | ILP64      | 4D  | no      | CT        | outplace  | forward   | C       | Large 4D CT outplace double ILP64
    "EXT_C_1D_LARGE_DIRECT_OUTPLACE_D ./aocl_fftz_bench 65536 -b a -p d --dir f"                              # accuracy | complex  | double   | LP64       | 1D  | no      | direct    | outplace  | forward   | C       | Large 1D direct outplace double
    "EXT_AVX128_5D_CTBS ./aocl_fftz_bench 4x4x8x17x18:4:4 -b a --dir b -p d -o 1"                             # accuracy | complex  | double   | LP64       | 5D  | no      | CT+BS     | outplace  | backward  | AVX128  | 5D CT+BS, double, outplace, backward
    # TODO: Add more test cases with different combinations. This list is expected to be larger than TEST_SUITE.
    )

# Determine which test suites to use based on AOCL_TEST_COVERAGE
if(AOCL_TEST_COVERAGE STREQUAL "EXHAUSTIVE")
    set(TEST_SUITE ${STANDARD_TEST_SUITE} ${EXHAUSTIVE_TEST_SUITE})
else()
    set(TEST_SUITE ${STANDARD_TEST_SUITE})
endif()

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
