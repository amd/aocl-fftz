Building on Linux
`````````````````

1. **Clone the repository**

2. **Generate and configure build system** using::

      cmake -B <build directory> <CMakeLists.txt filepath>

   Additional options that can be specified for build configuration::

      cmake -B <build directory> <CMakeLists.txt filepath> \
            -DAOCL_TEST_COVERAGE=<OFF/STANDARD/EXHAUSTIVE> \
            -DCMAKE_INSTALL_PREFIX=<install path> \
            -DCMAKE_BUILD_TYPE=<Debug or Release> \
            -DENABLE_STRICT_WARNINGS=<ON or OFF> \
            <Additional Library Build Options>

   To use clang compiler for the build, specify ``-DCMAKE_C_COMPILER=clang`` as an option.

3. **Compile** using the following command::

      cmake --build <build directory> --target install -j

   **Output locations:**
   
   - The library is generated in the "lib" directory
   - The test bench executable is generated in the "build" directory
   - The additional option ``--target install`` will install the library, binary, and 
     interface header files in the installation path as specified with 
     ``-DCMAKE_INSTALL_PREFIX`` option or in the local system path
