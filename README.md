AOCL-FFTZ
=========

AOCL-FFTZ is a high performance Fast Fourier Transform (FFT) library developed
by AMD supporting advanced optimizations for AMD’s "Zen"-based CPUs.
The library computes FFTs of both (i) complex and (ii) real data of any size
and dimension, in both forward and backward directions, with support for
in-place and out-of-place result placements.

The kernels in this library are vectorized to speed-up the single-threaded core
performance. The library supports the computations of parallel FFTs by taking
advantage of shared-memory parallelism using OpenMP threads.

AOCL-FFTZ introduces a generic and unified API set for supporting
precision types (single-precision and double-precision), and both the
single-threaded and multi-threaded execution modes.
The library uses a dynamic dispatcher feature to run efficiently and portably
across different x86 based systems.
A test bench is supported for performance and functional tests including the
accuracy tests. GTest-based unit testing framework is also supported by the
library.

Prerequisites
-------------
1. CMake - Version 3.26 or above
2. Linux :
        GCC compiler - Version 8.0 or above (or)
        Clang/AOCC compiler - Version 8.0 or above (AOCC 2.0 or above)
3. Windows :
        Visual Studio with Clang 12 or above (ClangCl toolset)

Building on Linux
-----------------
1. Clone the repo using the following command :
   ```
   git clone "https://github.com/amd/aocl-fftz" && cd aocl-fftz/
   ```

2. Run the following command in order to generate and configure build system.
   ```
   cmake -B <build directory> <CMakeLists.txt directory>
   ```
   Additional options that can be specified for build configuration are:
   ```
   cmake -B <build directory> <CMakeLists.txt directory> \
   -DAOCL_TEST_COVERAGE=<OFF/STANDARD/EXHAUSTIVE> \
   -DCMAKE_INSTALL_PREFIX=<install path> \
   -DCMAKE_BUILD_TYPE=<Debug or Release> \
   -DENABLE_STRICT_WARNINGS=<ON or OFF> \
   <Additional Library Build Options>
   ```
   To use clang compiler for the build, specify `-DCMAKE_C_COMPILER=clang` as the option.

3. Compile using the following command:
   ```
   cmake --build <build directory> --target install -j
   ```
   The library is generated in "lib" directory. <br>
   The test bench executable is generated in "build". <br>
   The additional option `--target install` will install the library,<br>
   interface header files, and the CMake package configuration and pkg-config files in the <br>
   installation path as specified with `-DCMAKE_INSTALL_PREFIX` option or in the local system path. <br>
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
------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------
AOCL_ENABLE_LOG                     |  Enables logging support within the library (Disabled by default)
AOCL_TEST_COVERAGE                  |  Enables GTest and AOCL test bench based CTest suite (OFF / STANDARD / EXHAUSTIVE, default: OFF)
ACCURACY_WITH_DFT                   |  Enables accuracy mode to run with DFT (Disabled by default)
ASAN                                |  Enables address sanitizer checks. Supported only on Linux builds (Disabled by default)
BUILD_DOC                           |  Builds documentation for library (Disabled by default)
BUILD_STATIC_LIBS                   |  Builds static library (Default build type is shared library)
BUILD_THIRD_PARTY_WRAPPERS          |  Builds FFTZ third party wrappers along with the aocl_fftz library (Disabled by default)
CODE_COVERAGE                       |  Enables source code coverage and generates coverage report. Supported only on Linux with GCC compiler (Disabled by default)
CODE_COVERAGE_FOR_ATG               |  Enables source code coverage instrumentation for running coverage and parsing tools for use by AI Test case Generation (ATG) later. Supported only on Linux with GCC compiler (Disabled by default)
ENABLE_APP_INFO_LOGS                |  Enables info logging for FFT problems used by the application (Independent of AOCL_ENABLE_LOG, Disabled by default)
ENABLE_INSTRUCTIONS_UPTO            |  Specifies maximum AVX instruction set to compile (None / AVX128 / AVX256 / AVX512, default: AVX512)
ENABLE_FMA                          |  Enables -ffp-contract=fast (forces FMA generation). Required for Clang/AOCC, implied by GCC at -O3 (Enabled by default)
SELECT_REAL_FFT_EXECUTION_ORDER     |  Selects the Real FFT Cooley-Tukey execution order (all modes are numerically identical): ITERATIVE, PARTIAL_RECURSION, TRUE_RECURSION. Default: TRUE_RECURSION
ENABLE_MULTI_THREADING              |  Compiles library with multi-threading support using OpenMP (Disabled by default)
ENABLE_STRICT_WARNINGS              |  Enables compiler flags to treat all warnings as errors (Enabled by default)
FUZZTEST                            |  Enables Compilation of fuzz test with fuzzing mode. Supported only on Linux Debug build with Clang compiler (Disabled by default)
VALGRIND                            |  Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)
OpenMP_libomp_LIBRARY               |  Path to the custom OpenMP library (System OpenMP is used if not provided)


CPU Architecture Support and FMA Requirements
---------------------------------------------
AOCL-FFTZ leverages advanced CPU features for optimal performance:

**FMA (Fused Multiply-Add) Support:**
- The library uses FMA3 instructions in SIMD kernels when `ENABLE_FMA` is ON (default)
- `ENABLE_FMA` is forced OFF regardless of the option when `ENABLE_INSTRUCTIONS_UPTO` is `None` or `AVX128` only.
- On Linux, when FMA is enabled and `ENABLE_INSTRUCTIONS_UPTO` is `AVX256` or `AVX512`, AVX128 and AVX256 kernels are compiled with `-mfma -ffp-contract=fast`, and AVX512 kernels are compiled with `-ffp-contract=fast`.
- On Windows, FMA intrinsics in AVX128/AVX256 kernels are gated off at compile time regardless of `ENABLE_FMA`

**Runtime Behavior:**
- The library automatically detects CPU capabilities at runtime and selects the highest usable SIMD tier (AVX512 → AVX256 → AVX128 → scalar C).
- On Linux with `ENABLE_FMA` enabled (default), AVX128 and AVX256 kernels require FMA at runtime; if FMA is unavailable, the library uses scalar C kernels.
- On Windows, or when `ENABLE_FMA` is disabled, AVX128 kernels require only AVX support.
- If AVX is not supported, the library executes using the standard C (scalar) implementation.

**SIMD ISA Support:**
- The library uses x86 SIMD AVX128, AVX256 and AVX512 instructions when available
- Library uses dynamic dispatcher to automatically detect the CPU capabilities and
  dispatch the optimal ISA kernels based on selector model


Multi-threading with OpenMP
---------------------------

AOCL-FFTZ currently supports Multi-threading through OpenMP. To enable it, turn on the CMake option `ENABLE_MULTI_THREADING`.
Additionally, you can also provide a custom OpenMP library through the `OpenMP_libomp_LIBRARY` option to override system OpenMP library.

Note: `aoclfftz_execute_io` is safe to call concurrently on a shared handle.
`aoclfftz_execute` is not; call it from a single thread per handle.

Running Test Bench On Linux & Windows
-------------------------------------
The AOCL-FFTZ test bench supports multiple options in order to compute, validate & benchmark FFT.<br>
The test bench is a development tool and is not installed; run it from the build directory.<br>
Following are a few sample commands to use and test with the test bench:

* The test bench can be run by using the following syntax: <br>
  `./aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE`

* Use the following command to set the precision for FFT: <br>
  `./aocl_fftz_bench -p/--precision <d/f>`

* Use the following command to set the data model for FFT: <br>
  `./aocl_fftz_bench -m/--data-model <l/i>`

* Use the following command to run the test bench with the requested bench type: <br>
  `./aocl_fftz_bench -b/--bench-type <p/a>`

* Use the following command to run the test bench with the requested FFT type:<br>
  `./aocl_fftz_bench -f/--fft-type <c2c>`

* To check other options for test bench use the following command:<br>
  `./aocl_fftz_bench -h/--help`

Running tests with CTest
------------------------
Use the AOCL_TEST_COVERAGE option to enable testing with CTest:
- `OFF`: Disables all tests (default)
- `STANDARD`: Enables standard test suite
- `EXHAUSTIVE`: Enables both standard and exhaustive test suite (exhaustive suite contains a much larger set of test cases for comprehensive coverage)

Note: `ACCURACY_WITH_DFT` enables running tests with DFT as an additional validation method.
      This adds another verification mechanism to the set of accuracy tests that the test bench uses for verification.

Here are a few sample commands that can be executed within the build directory to run test cases with CTest.

To run all the tests<br>
`ctest`

To run only TestBench<br>

Linux  : `ctest -R TESTBENCH`<br>
Windows : `ctest -C <Release/Debug> -R TESTBENCH`

To run GTest test cases for a specific test case<br>
`ctest -R <TEST CASE>`

FFTW wrapper tests
-------------------------------
Build the wrapper and its GTest suite with strict warnings enabled:
```
cmake -B <build directory> <CMakeLists.txt directory> \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOCL_TEST_COVERAGE=STANDARD \
  -DBUILD_STATIC_LIBS=ON \
  -DBUILD_THIRD_PARTY_WRAPPERS=ON \
  -DENABLE_STRICT_WARNINGS=ON
cmake --build <build directory> --parallel
```

The test names use `KNOWN_DIVERGENCE` for explicit AOCL-vs-FFTW differences
and `PRODUCTION_BLOCKED` for disabled regressions that depend on production
fixes. The wisdom and miscellaneous suites also pin documented wrapper stubs.
These groups can be selected directly by test-name regular expressions:
```
ctest --test-dir <build directory> --output-on-failure \
  -R "^FftwWrapper" \
  -E "KNOWN_DIVERGENCE|FftwWrapperWisdomTest|FftwWrapperMiscTest|PRODUCTION_BLOCKED"
ctest --test-dir <build directory> --output-on-failure \
  -R "KNOWN_DIVERGENCE|FftwWrapperWisdomTest|FftwWrapperMiscTest"
ctest --test-dir <build directory> -N -R PRODUCTION_BLOCKED
```

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

Running Valgrind and sanitizer checks using CTest
--------------------------------------------------

To perform memory checks using Valgrind or ASAN, enable the relevant build options on Linux while configuring CMake.
`VALGRIND` and `ASAN` are supported only on Linux.
Please note that `VALGRIND` requires `-DCMAKE_BUILD_TYPE=Debug` and cannot be combined with `ASAN`.

Sample commands for Valgrind :

Build :
```
cmake -B <build directory> <CMakeLists.txt directory> -DCMAKE_BUILD_TYPE=Debug -DVALGRIND=ON
```

Run :
```
ctest -T memcheck
```

Sample commands for ASAN :

Build :
```
cmake -B <build directory> <CMakeLists.txt directory> -DASAN=ON
```

Run :
```
ctest
```

Generating Documentation
------------------------
- To generate documentation, specify the `-DBUILD_DOC=ON` option while building.
- Documents will be generated in HTML format in the folder __docs/sphinx/html__ .
  Open index.html file from the folder in any browser to view the documentation.
- The following packages are expected before running CMake with `-DBUILD_DOC=ON` option:
   1. Doxygen.
   2. Python packages:
      - Sphinx
      - rocm_docs
      - breathe
      - myst_parser
- CMake halts if required packages are missing by providing directives for installing the absent packages.

CONTACTS
--------
AOCL-FFTZ is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.
