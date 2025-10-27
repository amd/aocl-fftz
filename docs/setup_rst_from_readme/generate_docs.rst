Generating Documentation
````````````````````````

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