Running tests with CTest
~~~~~~~~~~~~~~~~~~~~~~~~

Use the AOCL_TEST_COVERAGE option to enable testing with CTest:

- ``OFF``: Disables all tests (default)

- ``STANDARD``: Enables standard test suite

- ``EXHAUSTIVE``: Enables both standard and exhaustive test suite (exhaustive suite contains a much larger set of test cases for comprehensive coverage)

Note: ``ACCURACY_WITH_DFT`` enables running tests with DFT as an additional validation method. This adds another verification mechanism to the set of accuracy tests that the test bench uses for verification.

Here are a few sample commands that can be executed within the build directory to run test cases with CTest.
 
To run all the tests
    
    ctest

To run only TestBench

 Linux  :
        
        ctest -R TESTBENCH

 Windows :
        
        ctest -C <Release/Debug> -R TESTBENCH

 To run GTest test cases for a specific test case
 
       ctest -R <TEST CASE>