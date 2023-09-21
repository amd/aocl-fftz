AOCL-FFTZ
=========

AOCL-FFTZ is a high performance Fast Fourier Transform (FFT) library developed
by AMD supporting advanced optimizations for AMD’s "Zen"-based CPUs.
The library computes FFTs of real and complex data of any size and dimension in
both forward and backward directions. The important kernels in this library are
vectorized to speed-up the single-threaded core performance. The library
supports the computations of parallel FFTs by taking advantage of
(i) shared-memory parallelism using openMP threads, and (ii) distributed-memory
parallelism using MPI.
AOCL-FFTZ introduces a generic and unified API set for supporting any
precision types (single-precision and double-precision), and both the
single-threaded and multi-threaded execution modes.
AOCL-FFTZ supports distributed FFTs with a separate API set that makes use of
an underlying MPI framework for communication.
The library uses a dynamic dispatcher feature to run efficiently and portably
across different x86 based systems.
A test bench is supported for performance and functional tests including the
accuracy tests. GTest based unit testing framework is also supported by the
library.

PreRequisites
-------------
1. CMake - Version 3.10 or above
2. GCC compiler - Version 7.1 or above

Building on Linux
-----------------
1. Clone the repo using the following command :
   ```
   git clone "ssh://gerritgit/cpulibraries/er/aocl-fftz" && cd aocl-fftz/
   ```

2. Run the following command in order to generate and configure build system.
   ```
   cmake -B <build directory> <CMakeLists.txt filepath>
   ```
   Additional options that can be specified for build configuration are:
   ```
   cmake -B <build directory> <CMakeLists.txt filepath>
   -DAOCL_TEST_COVERAGE=<ON or OFF>
   -DCMAKE_INSTALL_PREFIX=<install path>
   -DCMAKE_BUILD_TYPE=<Debug or Release>
   -DENABLE_STRICT_WARNINGS=<ON or OFF>
   <Additional Library Build Options>
   ```

3. Compile using the following command:
   ```
   cmake --build <build directory> --target install -j
   ```
   The library is generated in "lib" directory. <br>
   The test bench executable is generated in "build". <br>
   The additional option `--target install` will install the library, binary, and <br>
   interface header files in the installation path as specified with <br>
   `-DCMAKE_INSTALL_PREFIX` option or in the local system path. <br>
   The option `-j` will run the compilation process using multiple cores.

4. To uninstall the installed files, run the following custom command:
   ```
   cmake --build <build directory> --target uninstall
   ```

   To uninstall and then install the build package, run the following command:
   ```
   cmake --build <build directory> --target uninstall --target install -j -v
   ```

5. To clear or delete the build folder or files, manually remove the build directory or its files.

Building on Windows
-------------------
--TBD--

Additional Library Build Options
--------------------------------
Use the following additional options to configure your build:

Option                              |  Description
------------------------------------|----------------------------------------------------------------------------------------
AOCL_TEST_COVERAGE                  |  Enables GTest and AOCL test bench based CTest suite (Disabled by default)
ENABLE_STRICT_WARNINGS              |  Enables strict warnings (Enabled by default)

Running Test Bench On Linux
---------------------------
The AOCL-FFTZ test bench supports multiple options in order to compute, validate & benchmark FFT.<br>
Following are a few sample commands to use and test with the test bench :

* The test bench can be run by using the following syntax : <br>
   `aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE`

* Use the following command to set the precision for FFT :<br>
  `aocl_fftz_bench -p <d/f>`

* Use the following command to set the data model for FFT:<br>
  `./aocl_fftz_bench -m <l/i>`

* Use the following command to run the test bench with the requested bench type:<br>
  `aocl_fftz_bench -b <p/a>`

* Use the following command to run the test bench with the requested FFT type:<br>
  `aocl_fftz_bench -f <c2c>`

* To check other options for test bench use the following command:<br>
  `aocl_fftz_bench -h`

Running Test Bench On Windows
-----------------------------
--TBD--

Running tests with CTest
------------------------
Use the AOCL_TEST_COVERAGE option to enable testing with CTest.

Here are a few sample commands that can be executed within the build directory to run test cases with CTest.

 To run all the tests<br>
 `ctest`

 To run GTest test cases for a specific test case<br>
 `ctest -R <TEST CASE>`

Running Performance Benchmark test
----------------------------------
--TBD--

Generating Documentation
------------------------
--TBD--

CONTACTS
--------
AOCL-FFTZ is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.