Running source code coverage using GCOV
```````````````````````````````````````

**Prerequisites :**

- gcov

- lcov
    
- genhtml

To measure source code coverage, set ``CODE_COVERAGE=ON`` while configuring the CMake build.

Build with the custom target option 'code-coverage' to execute tests and generate code coverage data.

The code coverage reports are generated in the build directory under subdirectory called 'coverage/html_report'. Open the HTML files in browser to view the coverage information.

Sample command to obtain code coverage report :

    cmake --build <build directory> --target install code-coverage
