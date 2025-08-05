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

Prerequisites
-------------
1. CMake - Version 3.26 or above
2. Linux :
        GCC compiler - Version 7.1 or above (or)
        AOCC compiler - Version 2.0 or above
3. Windows :
        Visual Studio with Clang 12 or above

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
   To use clang compiler for the build, specify `-DCMAKE_C_COMPILER=clang` as the option.

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
As a prerequisite, make Microsoft Visual Studio® available along with
Desktop development with C++ toolset that includes the Clang compiler.

Building with Visual Studio IDE (GUI)
-------------------------------------
1. Launch CMake GUI and set the locations for source package and build output
2. Click **Configure** option and select:
    * **Generator** as the Installed Microsoft Visual Studio Version
    * **Platform** as **x64**
    * **Optional toolset** as **ClangCl**
3. Select additional library config and build options.
4. Configure CMAKE_INSTALL_PREFIX appropriately.
5. Click **Generate**. Microsoft Visual Studio project is generated.
6. Click **Open Project**. Microsoft Visual Studio project for the source package **is launched**.

Building with Visual Studio IDE (Command Line)
----------------------------------------------
1. Go to AOCL-FFTZ source package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build the library & test bench executable.

```
cmake .. -T ClangCl -G <installed Visual Studio version> && cmake --build . --config Release --target INSTALL
```

Additional Library Build Options
--------------------------------
Use the following additional options to configure your build:

Option                              |  Description
------------------------------------|----------------------------------------------------------------------------------------
AOCL_TEST_COVERAGE                  |  Enables GTest and AOCL test bench based CTest suite (Disabled by default)
AOCL_ENABLE_LOG                     |  Enables logging support within library (Disabled by default)
ENABLE_STRICT_WARNINGS              |  Enables strict warnings (Enabled by default)
ENABLE_AVX128                       |  Compiles library with AVX 128-bit kernels support (Disabled by default)
ENABLE_AVX256                       |  Compiles library with AVX 256-bit kernels support (Disabled by default)
BUILD_STATIC_LIBS                   |  Build static library (Default build type is shared library)
BUILD_DOC                           |  Build documentation for aocl-fftz (Disabled by default)
CODE_COVERAGE                       |  Enables source code coverage and generates coverage report. Only supported on Linux with GCC compiler (Disabled by default)
ASAN                                |  Enables address sanitizer checks. Supported only on Linux Debug build (Disabled by default)
VALGRIND                            |  Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)
ENABLE_MULTI_THREADING              |  Compile with multi-threading support using OpenMP (Disabled by default)


Note : Enabling ENABLE_AVX256 turns on ENABLE_AVX128 implicitly.

Running Test Bench On Linux & Windows
-------------------------------------
The AOCL-FFTZ test bench supports multiple options in order to compute, validate & benchmark FFT.<br>
Following are a few sample commands to use and test with the test bench:

* The test bench can be run by using the following syntax: <br>
  `aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE`

* Use the following command to set the precision for FFT: <br>
  `aocl_fftz_bench -p/--precision <d/f>`

* Use the following command to set the data model for FFT: <br>
  `./aocl_fftz_bench -m/--data-model <l/i>`

* Use the following command to run the test bench with the requested bench type: <br>
  `aocl_fftz_bench -b/--bench-type <p/a>`

* Use the following command to run the test bench with the requested FFT type:<br>
  `aocl_fftz_bench -f/--fft-type <c2c>`

* To check other options for test bench use the following command:<br>
  `aocl_fftz_bench -h/--help`

Running tests with CTest
------------------------
Use the AOCL_TEST_COVERAGE option to enable testing with CTest.

Here are a few sample commands that can be executed within the build directory to run test cases with CTest.

 To run all the tests<br>
 `ctest`

 To run only TestBench<br>

 Linux  : `ctest -R TESTBENCH`<br>
 Windows : `ctest -C <Release/Debug> -R TESTBENCH`

 To run GTest test cases for a specific test case<br>
 `ctest -R <TEST CASE>`

Running source code coverage using GCOV
---------------------------------------

**Prerequisites :** <br>
1. gcov
2. lcov
3. genhtml

To measure source code coverage, set `CODE_COVERAGE=ON` while configuring the CMake build.<br>
Build with the custom target option 'code-coverage' to execute tests and generate code coverage data.
The code coverage reports are generated in the build directory under subdirectory called 'coverage/html_report'. Open the HTML files in browser to view the coverage information.

Sample command to obtain code coverage report :
```
cmake --build <build directory> --target install code-coverage
```

Running Valgrind and ASAN memory checks using CTest
---------------------------------------------------

To perform memory checks using Valgrind/ASAN, enable the relevant build options `VALGRIND` or `ASAN` while configuring CMake.<br>
Please note that Valgrind and ASAN options cannot be enabled together and they are supported only in **Linux Debug build** mode.

Sample commands for Valgrind :

Build :
```
cmake -B <build directory> <CMakeList.txt filepath> -DCMAKE_BUILD_TYPE=Debug -DVALGRIND=ON
```

Run :
```
ctest -T memcheck
```

Sample commands for ASAN :

Build :
```
cmake -B <build directory> <CMakeList.txt filepath> -DCMAKE_BUILD_TYPE=Debug -DASAN=ON
```

Run :
```
ctest
```


Running Performance Benchmark test
----------------------------------
--TBD--

Generating Documentation
------------------------
- To generate documentation, specify the `-DBUILD_DOC=ON` option while building.
- Documents will be generated in HTML format in the folder <b>docs/html</b>. Open the <b>index.html</b> file in any browser to view the documentation.
- CMake will use the existing Doxygen if available. Else, it will prompt to install doxygen (version 1.10.0 or above) and try again.

CONTACTS
--------
AOCL-FFTZ is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.