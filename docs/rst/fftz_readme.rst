..  Copyright Advanced Micro Devices, Inc.
..  SPDX-License-Identifier: BSD-3-Clause

Building on Linux
~~~~~~~~~~~~~~~~~

1. Clone the repository

2. Generate and configure build system using:

.. code-block:: bash

    cmake -B <build directory> <CMakeLists.txt directory>

Additional options that can be specified for build configuration:

.. code-block:: bash

    cmake -B <build directory> <CMakeLists.txt directory> \
            -DAOCL_TEST_COVERAGE=<OFF/STANDARD/EXHAUSTIVE> \
            -DCMAKE_INSTALL_PREFIX=<install path> \
            -DCMAKE_BUILD_TYPE=<Debug or Release> \
            -DENABLE_STRICT_WARNINGS=<ON or OFF> \
            <Additional Library Build Options>

To use clang compiler for the build, specify ``-DCMAKE_C_COMPILER=clang`` as an option.

3. Compile using the following command:

.. code-block:: bash

    cmake --build <build directory> --target install -j

**Output locations:**

- The library is generated in the "lib" directory
- The test bench executable is generated in the "build" directory
- The additional option ``--target install`` will install the library, interface
  header files, and the CMake package configuration and pkg-config files in the
  installation path as specified with ``-DCMAKE_INSTALL_PREFIX`` option or in
  the local system path.


Building on Windows
~~~~~~~~~~~~~~~~~~~

As a prerequisite, make Microsoft Visual Studio (R) available along with
Desktop development with C++ toolset that includes the Clang compiler.

Building with Visual Studio IDE (GUI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Launch CMake GUI and set the locations for source package and build output.
2. Click **Configure** option and select:

   - **Generator** as the Installed Microsoft Visual Studio Version
   - **Platform** as **x64**
   - **Optional toolset** as **ClangCl**

3. Select additional library config and build options.
4. Configure CMAKE_INSTALL_PREFIX appropriately.
5. Click **Generate**. Microsoft Visual Studio project is generated.
6. Click **Open Project**. Microsoft Visual Studio project for the source package is launched.

Building with Visual Studio IDE (Command Line)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to AOCL-FFTZ source package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build the library & test bench executable.

.. code-block:: bash

    cmake .. -T ClangCl -G <installed Visual Studio version> && cmake --build . --config Release --target INSTALL


Additional Library Build Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following additional options to configure your build:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - AOCL_ENABLE_LOG
     - Enables logging support within the library (Disabled by default)
   * - AOCL_TEST_COVERAGE
     - Enables GTest and AOCL test bench based CTest suite (OFF / STANDARD / EXHAUSTIVE, default: OFF)
   * - ACCURACY_WITH_DFT
     - Enables accuracy mode to run with DFT (Disabled by default)
   * - ASAN
     - Enables address sanitizer checks. Supported only on Linux builds (Disabled by default)
   * - BUILD_DOC
     - Builds documentation for library (Disabled by default)
   * - BUILD_STATIC_LIBS
     - Builds static library (Default build type is shared library)
   * - BUILD_THIRD_PARTY_WRAPPERS
     - Builds FFTZ third party wrapper libraries along with the aocl_fftz library (Disabled by default)
   * - CODE_COVERAGE
     - Enables source code coverage and generates coverage report. Supported only on Linux with GCC compiler (Disabled by default)
   * - CODE_COVERAGE_FOR_ATG
     - Enables source code coverage instrumentation for running coverage and parsing tools for use by AI Test case Generation (ATG) later. Supported only on Linux with GCC compiler (Disabled by default)
   * - ENABLE_APP_INFO_LOGS
     - Enables info logging for FFT problems used by the application (Independent of AOCL_ENABLE_LOG, Disabled by default)
   * - ENABLE_INSTRUCTIONS_UPTO
     - Specifies maximum AVX instruction set to compile (None / AVX128 / AVX256 / AVX512, default: AVX512)
   * - ENABLE_FMA
     - Enables -ffp-contract=fast (forces FMA generation). Required for Clang/AOCC, implied by GCC at -O3 (Enabled by default)
   * - SELECT_REAL_FFT_EXECUTION_ORDER
     - Selects the Real FFT Cooley-Tukey execution order (all modes are numerically identical): ITERATIVE, PARTIAL_RECURSION, TRUE_RECURSION. Default: TRUE_RECURSION
   * - ENABLE_MULTI_THREADING
     - Compiles library with multi-threading support using OpenMP (Disabled by default)
   * - ENABLE_STRICT_WARNINGS
     - Enables compiler flags to treat all warnings as errors (Enabled by default)
   * - FUZZTEST
     - Enables Compilation of fuzz test with fuzzing mode. Supported only on Linux Debug build with Clang compiler (Disabled by default)
   * - VALGRIND
     - Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)
   * - OpenMP_libomp_LIBRARY
     - Path to the custom OpenMP library (System OpenMP is used if not provided)


CPU Architecture Support and FMA Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AOCL-FFTZ leverages advanced CPU features for optimal performance:

**FMA (Fused Multiply-Add) Support:**

- The library uses FMA3 instructions in SIMD kernels when ``ENABLE_FMA`` is ON (default)
- ``ENABLE_FMA`` is forced OFF regardless of the option when ``ENABLE_INSTRUCTIONS_UPTO`` is ``None`` or ``AVX128`` only.
- On Linux, when FMA is enabled and ``ENABLE_INSTRUCTIONS_UPTO`` is ``AVX256`` or ``AVX512``, AVX128 and AVX256 kernels are compiled with ``-mfma -ffp-contract=fast``, and AVX512 kernels are compiled with ``-ffp-contract=fast``.
- On Windows, FMA intrinsics in AVX128/AVX256 kernels are gated off at compile time regardless of ``ENABLE_FMA``

**Runtime Behavior:**

- The library automatically detects CPU capabilities at runtime and selects the highest usable SIMD tier (AVX512 → AVX256 → AVX128 → scalar C).
- On Linux with ``ENABLE_FMA`` enabled (default), AVX128 and AVX256 kernels require FMA at runtime; if FMA is unavailable, the library uses scalar C kernels.
- On Windows, or when ``ENABLE_FMA`` is disabled, AVX128 kernels require only AVX support.
- If AVX is not supported, the library executes using the standard C (scalar) implementation.

**SIMD ISA Support:**

- The library uses x86 SIMD AVX128, AVX256 and AVX512 instructions when available
- Library uses dynamic dispatcher to automatically detect the CPU capabilities and dispatch the optimal ISA kernels based on selector model


Multi-threading with OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AOCL-FFTZ currently supports Multi-threading through OpenMP. To enable it, turn on the CMake option ``ENABLE_MULTI_THREADING``.
Additionally, you can also provide a custom OpenMP library through the ``OpenMP_libomp_LIBRARY`` option to override system OpenMP library.

Note: ``aoclfftz_execute_io`` is safe to call concurrently on a shared handle.
``aoclfftz_execute`` is not; call it from a single thread per handle.

Testing and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

Running Test Bench On Linux & Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The AOCL-FFTZ test bench supports multiple options in order to compute, validate, and benchmark FFT.
The test bench is a development tool and is not installed; run it from the build directory.

Following are a few sample commands to use and test with the test bench:

* The test bench can be run by using the following syntax:

.. code-block:: bash

    ./aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE

* Use the following command to set the precision for FFT:

.. code-block:: bash

    ./aocl_fftz_bench -p/--precision <d/f>

* Use the following command to set the data model for FFT:

.. code-block:: bash

    ./aocl_fftz_bench -m/--data-model <l/i>

* Use the following command to run the test bench with the requested bench type:

.. code-block:: bash

    ./aocl_fftz_bench -b/--bench-type <p/a>

* Use the following command to run the test bench with the requested FFT type:

.. code-block:: bash

    ./aocl_fftz_bench -f/--fft-type <c2c>

* Use the following command to view other options available for the test bench:

.. code-block:: bash

    ./aocl_fftz_bench -h/--help


Running tests with CTest
^^^^^^^^^^^^^^^^^^^^^^^^

Use the AOCL_TEST_COVERAGE option to enable testing with CTest:

- ``OFF``: Disables all tests (default)

- ``STANDARD``: Enables standard test suite

- ``EXHAUSTIVE``: Enables both standard and exhaustive test suite. To provide comprehensive coverage, the exhaustive suite contains a larger set of test cases.

Note: ``ACCURACY_WITH_DFT`` enables running tests with DFT as an additional validation method. This adds another verification mechanism to the set of accuracy tests that the test bench uses for verification.

Here are a few sample commands that can be executed within the build directory to run test cases with CTest.

* To run all the tests:

.. code-block:: bash

    ctest

* To run only TestBench on Linux:

.. code-block:: bash

    ctest -R TESTBENCH

* To run only TestBench on Windows:

.. code-block:: bash

    ctest -C <Release/Debug> -R TESTBENCH

* To run GTest test cases for a specific test case:

.. code-block:: bash

    ctest -R <TEST CASE>


FFTW wrapper tests
~~~~~~~~~~~~~~~~~~~

Build the wrapper and its GTest suite with strict warnings enabled:

.. code-block:: bash

    cmake -B <build directory> <CMakeLists.txt directory> \
      -DCMAKE_BUILD_TYPE=Release \
      -DAOCL_TEST_COVERAGE=STANDARD \
      -DBUILD_STATIC_LIBS=ON \
      -DBUILD_THIRD_PARTY_WRAPPERS=ON \
      -DENABLE_STRICT_WARNINGS=ON
    cmake --build <build directory> --parallel

The test names use ``KNOWN_DIVERGENCE`` for explicit AOCL-vs-FFTW differences
and ``PRODUCTION_BLOCKED`` for disabled regressions that depend on production
fixes. The wisdom and miscellaneous suites also pin documented wrapper stubs.
These groups can be selected directly by test-name regular expressions:

.. code-block:: bash

    ctest --test-dir <build directory> --output-on-failure \
      -R "^FftwWrapper" \
      -E "KNOWN_DIVERGENCE|FftwWrapperWisdomTest|FftwWrapperMiscTest|PRODUCTION_BLOCKED"
    ctest --test-dir <build directory> --output-on-failure \
      -R "KNOWN_DIVERGENCE|FftwWrapperWisdomTest|FftwWrapperMiscTest"
    ctest --test-dir <build directory> -N -R PRODUCTION_BLOCKED

Running source code coverage using GCOV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Prerequisites:**

- gcov

- lcov

- genhtml

To measure source code coverage, set ``CODE_COVERAGE=ON`` while configuring the CMake build.

Build with the custom target option 'code-coverage' to execute tests and generate code coverage data.

The code coverage reports are generated in the build directory's 'coverage/html_report' subdirectory. Open the HTML files in a web browser to view the coverage information

Sample command to obtain code coverage report:

.. code-block:: bash

    cmake --build <build directory> --target install code-coverage


Running Valgrind and sanitizer checks using CTest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform memory checks using Valgrind or ASAN, enable the relevant build options on Linux while configuring CMake.
``VALGRIND`` and ``ASAN`` are supported only on Linux.
Please note that ``VALGRIND`` requires ``-DCMAKE_BUILD_TYPE=Debug`` and cannot be combined with ``ASAN``.

Sample commands for Valgrind:

.. code-block:: bash

    # Build
    cmake -B <build directory> <CMakeLists.txt directory> -DCMAKE_BUILD_TYPE=Debug -DVALGRIND=ON

    # Run
    ctest -T memcheck


Sample commands for ASAN:

.. code-block:: bash

    # Build
    cmake -B <build directory> <CMakeLists.txt directory> -DASAN=ON

    # Run
    ctest


Generating Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

- To generate documentation, specify the ``-DBUILD_DOC=ON`` option while building.
- Documents will be generated in HTML format in the folder ``docs/sphinx/html`` .
  Open index.html file from the folder in any browser to view the documentation.
- The following packages are expected before running CMake with ``-DBUILD_DOC=ON`` option:

  1. Doxygen.
  2. Python packages:

     - Sphinx

     - rocm_docs

     - breathe

     - myst_parser

- CMake halts if required packages are missing by providing directives for installing the absent packages.
