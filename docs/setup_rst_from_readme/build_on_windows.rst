Building on Windows
```````````````````

As a prerequisite, make Microsoft Visual Studio® available along with
Desktop development with C++ toolset that includes the Clang compiler.

Building with Visual Studio IDE (GUI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Go to AOCL-FFTZ source package and create a folder named build.
2. Go to the build folder.
3. Use the following command to configure and build the library & test bench executable.

    cmake .. -T ClangCl -G <installed Visual Studio version> && cmake --build . --config Release --target INSTALL
