Running Valgrind and ASAN memory checks using CTest
```````````````````````````````````````````````````

To perform memory checks using Valgrind/ASAN, enable the relevant build options ``VALGRIND`` or ``ASAN`` while configuring CMake.
Please note that Valgrind and ASAN options cannot be enabled together and they are supported only in **Linux Debug build** mode.

Sample commands for Valgrind :

Build :

    cmake -B <build directory> <CMakeList.txt filepath> -DCMAKE_BUILD_TYPE=Debug -DVALGRIND=ON

Run :

    ctest -T memcheck


Sample commands for ASAN :

Build :
  
    cmake -B <build directory> <CMakeList.txt filepath> -DCMAKE_BUILD_TYPE=Debug -DASAN=ON


Run :

    ctest
