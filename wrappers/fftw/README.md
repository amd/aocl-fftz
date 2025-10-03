FFTW WRAPPERS
=============

The FFTW Wrapper offers a streamlined interface for the APIs of libraries like
FFTW or AMD-FFTW, while internally leveraging the FFTZ library for enhanced
performance and flexibility in computing discrete Fourier transforms (DFTs).
Supports both the single-threaded and multi-threaded execution modes.

Prerequisites
-------------
1. CMake - Version 3.26 or above
2. Linux :
        GCC compiler - Version 7.1 or above  (or)
        AOCC compiler - Version 2.0 or above
3. Windows :
        Visual Studio with Clang 12 or above
4. AOCL-FFTZ library and header file

Setting up Dependencies
-----------------------
1. Clone, build and install AOCL-FFTZ by following the steps given in aocl-fftz/README.md

Building on Linux
-----------------
1. Run the following command in order to generate and configure fftw wrappers build system.
   ```
   cmake -B <build directory> <CMakeLists.txt filepath>
   ```
   Additional options that can be specified for build configuration are:
   ```
   cmake -B <build directory> <CMakeLists.txt filepath>
   -DFFTZ_LIB_PATH=<FFTZ library directory>
   <Additional Library Build Options>
   ```

2. Compile using the following command:
   ```
   cmake --build <build directory> --target install -j
   ```
   The library is generated in "lib" directory. <br>
   The additional option `--target install` will install the library, binary, and <br>
   interface header files in the installation path as specified with <br>
   `-DCMAKE_INSTALL_PREFIX` option or in the local system path. <br>
   The option `-j` will run the compilation process using multiple cores.

3. To uninstall the installed files, run the following custom command:
   ```
   cmake --build <build directory> --target uninstall
   ```

   To uninstall and then install the build package, run the following command:
   ```
   cmake --build <build directory> --target uninstall --target install -j -v
   ```

4. To clear the build folder or files, run the following custom command:
   ```
   cmake --build <build directory> --target clean
   ```

5. To delete the build folder or files, manually remove the build directory or its files.

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
1. Go to wrappers/fftw source package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build the library

```
cmake .. -T ClangCl -G <installed Visual Studio version> && cmake --build . --config Release --target INSTALL
```

Additional Library Build Options
--------------------------------
Use the following additional options to configure your build:

| Option                 | Description                              | Default        |
| ---------------------- | ---------------------------------------- | -------------- |
| `FFTZ_LIB_PATH`        | Path to FFTZ library directory           | Auto-detect    |
| `BUILD_STATIC_LIBS`    | Build static library instead of shared   | OFF            |
| `CMAKE_INSTALL_PREFIX` | Installation directory                   | System default |
