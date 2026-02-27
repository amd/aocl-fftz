..  Copyright (C) 2025, Advanced Micro Devices. All rights reserved.

..  Redistribution and use in source and binary forms, with or without
..  modification, are permitted provided that the following conditions are met:

..  1. Redistributions of source code must retain the above copyright notice,
..  this list of conditions and the following disclaimer.
..  2. Redistributions in binary form must reproduce the above copyright notice,
..  this list of conditions and the following disclaimer in the documentation
..  and/or other materials provided with the distribution.
..  3. Neither the name of the copyright holder nor the names of its
..  contributors may be used to endorse or promote products derived from this
..  software without specific prior written permission.

..  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
..  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
..  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
..  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
..  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
..  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
..  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
..  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
..  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
..  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
..  POSSIBILITY OF SUCH DAMAGE.

Example Test Programs
=====================

.. list-table::
   :header-rows: 1

   * - Example
     - Description
   * - `One Dimensional Complex`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one dimensional FFT computation for complex input to complex output using double-precision on LP64 architecture systems.
   * - `N Dimensional Complex`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for complex input to complex output using double-precision on ILP64 architecture systems.
   * - `One Dimensional Real Forward`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one dimensional FFT computation for real input to complex output(Forward FFT) using single-precision on LP64 architecture systems.
   * - `One Dimensional Real Backward`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one-dimensional FFT computation for complex input to real output(Backward FFT) using single-precision on ILP64 architecture systems.
   * - `N Dimensional Real Forward`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for real input to complex output (Forward FFT) using double-precision on ILP64 architecture systems.
   * - `N Dimensional Real Backward`_
     - Sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for complex input to real output (Backward FFT) using double-precision on ILP64 architecture systems.

One Dimensional Complex
-----------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one dimensional FFT computation for complex input to complex output using double-precision on LP64 architecture systems:

.. literalinclude:: ../../examples/example_one_dim_complex.c
   :language: c
   :linenos:
   :lines: 44-159

To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_complex.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_complex.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


N Dimensional Complex
---------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for complex input to complex output using double-precision on ILP64 architecture systems:

.. literalinclude:: ../../examples/example_n_dim_complex.c
   :language: c
   :linenos:
   :lines: 44-160

To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_complex.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_complex.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


One Dimensional Real Forward
----------------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one dimensional FFT computation for real input to complex output(Forward FFT) using single-precision precision on LP64 architecture systems:

.. literalinclude:: ../../examples/example_one_dim_real_forward.c
   :language: c
   :linenos:
   :lines: 44-167

To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_real_forward.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_real_forward.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


One Dimensional Real Backward
-----------------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded one-dimensional FFT computation for complex input to real output(Backward FFT) using single-precision precision on ILP64 architecture systems:

.. literalinclude:: ../../examples/example_one_dim_real_backward.c
   :language: c
   :linenos:
   :lines: 44-167


To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_real_backward.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_one_dim_real_backward.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


N Dimensional Real Forward
--------------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for real input to complex output (Forward FFT) using double-precision on ILP64 architecture systems:

.. literalinclude:: ../../examples/example_n_dim_real_forward.c
   :language: c
   :linenos:
   :lines: 44-177

To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_real_forward.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_real_forward.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


N Dimensional Real Backward
---------------------------

The following test program shows the sample usage and calling sequence of AOCL-FFTZ APIs for single-threaded N-dimensional FFT computation for complex input to real output (Backward FFT) using double-precision on ILP64 architecture systems:

.. literalinclude:: ../../examples/example_n_dim_real_backward.c
   :language: c
   :linenos:
   :lines: 44-178

To build this example test program on a Linux system using GCC or AOCC, you must specify
path to aoclfftz.h header file and link with libaocl_fftz.so file as follows:

GCC : ``gcc -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_real_backward.c -laocl_fftz -lm``

Clang : ``clang -I<aoclfftz.h file directory> -L<libaocl_fftz.so file directory> example_n_dim_real_backward.c -laocl_fftz -lm``

Before running the example program, ensure it points to the right library dependencies for OpenMP for Multithreading.


Helpers
-------

The following file contains helper routines for AOCL-FFTZ examples, including macros/functions for
preparing input data, setting default strides for dimensions and vectors, calculating buffer sizes and handling memory allocation.

.. literalinclude:: ../../examples/helpers.h
   :language: c
   :linenos:
   :lines: 38-270
