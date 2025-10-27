Additional Library Build Options
````````````````````````````````

Use the following additional options to configure your build:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - AOCL_ENABLE_LOG                     
     - Enable logging support within the library (Disabled by default)
   * - AOCL_TEST_COVERAGE                  
     - Enables GTest and AOCL test bench based CTest suite (OFF / STANDARD / EXHAUSTIVE, default: OFF)
   * - ACCURACY_WITH_DFT                   
     - Enables accuracy mode to run with DFT (Disabled by default)
   * - ASAN                                
     - Enables address sanitizer checks. Supported only on Linux Debug build (Disabled by default)
   * - BUILD_DOC                           
     - Build documentation for library (Disabled by default)
   * - BUILD_STATIC_LIBS                   
     - Build static library (Default build type is shared library)
   * - BUILD_THIRD_PARTY_WRAPPERS          
     - Build all the supported FFTZ third party wrappers (Disabled by default)
   * - CODE_COVERAGE                       
     - Enables source code coverage and generates coverage report. Supported only on Linux with GCC compiler (Disabled by default)
   * - ENABLE_INSTRUCTIONS_UPTO            
     - Specify maximum AVX instruction set to compile (None / AVX128 / AVX256 / AVX512, default: AVX512)
   * - ENABLE_FMA                          
     - Enable -ffp-contract=fast (forces FMA generation). Required for Clang/AOCC, implied by GCC at -O3 (Enabled by default)
   * - ENABLE_MULTI_THREADING              
     - Compiles library with multi-threading support using OpenMP (Disabled by default)
   * - ENABLE_STRICT_WARNINGS              
     - Enable compiler flags to treat all warnings as errors (Enabled by default)
   * - FUZZTEST                            
     - Enable Compilation of fuzz test with fuzzing mode. Supported only on Linux Debug build with Clang compiler (Disabled by default)
   * - VALGRIND                            
     - Enables memory checks using Valgrind. Supported only on Linux Debug build. Incompatible with ASAN=ON (Disabled by default)