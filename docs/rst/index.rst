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

=========
AOCL-FFTZ
=========

AOCL-FFTZ is a high performance Fast Fourier Transform (FFT) library developed
by AMD supporting advanced optimizations for AMD’s "Zen"-based CPUs.
The library computes FFTs of (i) complex data of any size and dimension in
both forward and backward directions, and (ii) real data of any
size and dimension, excluding prime sizes greater than 7 and their multiples,
in both forward and backward directions with support for in-place and
out-of-place result placements.

The kernels in this library are vectorized to speed-up the single-threaded core
performance. The library supports the computations of parallel FFTs by taking
advantage of shared-memory parallelism using OpenMP threads.

AOCL-FFTZ introduces a generic and unified API set for supporting any precision
types (single-precision and double-precision), and both the single-threaded and
multi-threaded execution modes. The library uses a dynamic dispatcher feature to
run efficiently and portably across different x86 based systems.
A test bench is supported for performance and functional tests including the
accuracy tests. GTest-based unit testing framework is also supported by the
library.


:doc:`APIs <interface_api>`

AOCL-FFTZ provides a comprehensive set of interface APIs for computing
both forward (FFT) and backward (IFFT) transforms for real and complex
data in the supported data models. The library offers flexible interfaces
that support both single-threaded and multi-threaded execution modes
to optimize for performance across various workloads.

.. toctree::
   :hidden:
   :maxdepth: 1

   interface_api

:doc:`Data Structures <data_structures_index>`

Data structures and conventions to describe the FFT problems,
guides on problem descriptor construction and explains the data layout conventions
followed through the AOCL-FFTZ library.

.. toctree::
   :hidden:
   :maxdepth: 2

   data_structures_index

:doc:`Datatype Definitions <std_types>`

Type Definitions of Standard Datatypes for AOCL-FFTZ Library.

.. toctree::
   :hidden:
   :maxdepth: 1

   std_types

:doc:`Examples <test_programs>`

Example programs illustrating how to use AOCL-FFTZ APIs are presented here.

.. toctree::
   :hidden:
   :maxdepth: 1

   test_programs

**Contacts**

AOCL-FFTZ is developed and maintained by AMD.
For support, send an email to toolchainsupport@amd.com.
