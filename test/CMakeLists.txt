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
