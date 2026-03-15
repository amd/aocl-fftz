..  Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
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

.. _ref-data_structures_index:

Data Structures
===============

:doc:`Defined Structures <typedefs>`

Data structures and their typedef definitions associated with the APIs to support
both single-precision (`FLOAT`) and double-precision (`DOUBLE`) computations.
These structures encapsulate configuration parameters, execution contexts,
and other essential parameters and flags required for efficient FFT operations.

:doc:`Problem Descriptor Construction <problem_descriptor>`


The problem descriptor defines all parameters required to set up and execute an FFT operation. Constructing a valid problem descriptor is essential for successful FFT computation. This section provides guidance about each member of the descriptor, explains valid values, and provides practical examples for building a correct descriptor.

:doc:`Data Layout Conventions <data_layout_conventions>`

Data layout conventions define how FFT input and output buffers are organized in memory for the AOCL-FFTZ APIs: dimension ordering (innermost to outermost), buffer formats for complex and real transforms (interleaved, half-complex), and stride settings for in-place and out-of-place operations. This section documents these conventions and how they apply when using the library.

.. toctree::
   :hidden:
   :maxdepth: 1

   typedefs
   problem_descriptor
   data_layout_conventions
