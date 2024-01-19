# only run if code coverage is enabled
if (CODE_COVERAGE)
    find_program(GCOV gcov)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if (NOT (GCOV AND LCOV AND GENHTML))
        message(WARNING "Required tools not found: gcov,lcov,genhtml. Code coverage can not be performed")
    endif()

    # set lcov variable to exclude paths during code coverage
    set(LCOV_FILTERS "'/usr/*';'/*/_deps/*'")

    # add custom target for running ctest and generating code coverage report
    add_custom_target(code-coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory coverage/
        COMMAND ctest -j
        COMMAND ${LCOV} --rc branch_coverage=1 --capture --directory . --output-file coverage/coverage.info
        COMMAND ${LCOV} --rc branch_coverage=1 --remove  coverage/coverage.info ${LCOV_FILTERS} --output-file coverage/coverage_filtered.info
        COMMAND ${LCOV} --list coverage/coverage_filtered.info
        COMMAND ${GENHTML} --rc branch_coverage=1 coverage/coverage_filtered.info --output-directory coverage/html_report
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Code Coverage Report (LCOV) after tests.."
    )
endif()

if (VALGRIND)
    set(MEMORYCHECK_COMMAND_OPTIONS "-q --trace-children=yes --child-silent-after-fork=yes --leak-check=full --show-leak-kinds=all --track-origins=yes")
    # To run all registered tests with valgrind use "ctest -T memcheck"
    include(CTest) # Default to using valgrind
endif()

# Add testcases to the list as 'testcase_name testcase_command'
set(TEST_SUITE
    "DIRECT_PERFORMANCE ./aocl_fftz_bench 7"                            # performance
    "BATCHED_CT_WITH_SAME_STRIDE ./aocl_fftz_bench 7v18:2:2 -b a"       # accuracy
    "MULTILEVEL_CT_WITH_STRIDES ./aocl_fftz_bench 128:3:2 -b a -m i"    # ILP64 data model
    "CT_WITH_BLUESTEIN ./aocl_fftz_bench 38:4:4 -b a -r i"              # Inplace
    "BLUESTEIN_WITH_STRIDES ./aocl_fftz_bench 23:5:7 -b a -p f"         # Float
    "DIRECT_AVX_WITH_SAME_STRIDE ./aocl_fftz_bench 15:5:5 -b a -o 2"    # AVX128
    "DIRECT_AVX2_WITH_SAME_STRIDE ./aocl_fftz_bench 8:2:2 -b a -o 3"    # AVX256
    "2D_BATCH_3D_DIM ./aocl_fftz_bench 3x2v17x19x8 -b a --dir b"        # Inverse FFT
    "3D_WITH_STRIDE_GREATER_THAN_DIM ./aocl_fftz_bench 2x3:4:5x3 -b a"
    "4D_WITH_STRIDE ./aocl_fftz_bench 2x72x37x13:2:3 -b a"
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
