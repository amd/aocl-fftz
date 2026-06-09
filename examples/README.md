AOCL-FFTZ Examples
==================

This directory contains sample source files showing usage of AOCL-FFTZ Library
functions. Use the provided cmake script "CMakeLists.txt" to compile and
run the programs. Same script may be used on Linux and Windows platforms.

Note: The examples in this directory assume that AOCL-FFTZ has been built
       with multi-threading enabled (`ENABLE_MULTI_THREADING=ON`).

Building on Linux
-----------------

1. To create a build directory and configure the build system in it, run the following:
   ```
    cmake -B <build directory> <CMakeLists.txt filepath>
   ```
   Additional options that can be specified for build configuration are:
   ```
   cmake -B <build directory> <CMakeLists.txt filepath>
      -DAOCL_FFTZ_INSTALL_PATH=<Installed path of AOCL-FFTZ Library>
      -DCMAKE_INSTALL_PREFIX=<install path for examples binaries>
   ```

2. Compile using the following command:
   ```
   cmake --build <build directory> --target install -j
   ```
   The example executables are generated in "build". <br>
   The additional option `--target install` will install the test binaries <br>
   to the installation path as specified with the `-DCMAKE_INSTALL_PREFIX` <br>
   option or in "examples/bin". <br>
   The option `-j` will run the compilation process using multiple cores.

3. To uninstall the installed files, run the following custom command:
   ```
   cmake --build <build directory> --target uninstall
   ```

   To uninstall and then install the build package, run the following command:
   ```
   cmake --build <build directory> --target uninstall --target install -j -v
   ```

4. To clear or delete the build folder or files, manually remove the build directory or its files.

Building on Windows
-------------------
As a prerequisite, make Microsoft Visual Studio® available along with <br>
__Desktop development with C++__ toolset that includes the Clang compiler.

Building with Visual Studio IDE (GUI)
-------------------------------------
1. Launch CMake GUI and set the locations for source package and build output.
2.  Click __Configure__ option and select:
      - __Generator__ as the Installed Microsoft Visual Studio Version
      - __Platform__ as __x64__
      - __Optional toolset__ as __ClangCl__
3. Select additional library config and build options.
4. Click __Generate__.
   Microsoft Visual Studio project is generated.
5. Click __Open Project__.
   Microsoft Visual Studio project for the source package __is launched__.
6. Build the entire solution or the required projects.

Building with Visual Studio IDE (command line)
----------------------------------------------
1. Go to examples/ folder in AOCL-FFTZ install package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build example executables.
```
cmake .. -T ClangCl -G <installed Visual Studio version> -DAOCL_FFTZ_INSTALL_PATH=<Installed path of AOCL-FFTZ Library> && cmake --build . --config Release
```

Manual Compilation
------------------
Use the following command to compile manually without CMake:

gcc:
```
gcc -I<path to aoclfftz.h directory> -L<path to libaocl_fftz.so directory> example_one_dim_complex.c -laocl_fftz -lm -o example_one_dim_complex
```

clang:
```
clang -I<path to aoclfftz.h directory> -L<path to libaocl_fftz.so directory> example_one_dim_complex.c -laocl_fftz -lm -o example_one_dim_complex
```

CONTACTS
--------
AOCL-FFTZ is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.
