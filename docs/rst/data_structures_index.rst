..  Copyright Advanced Micro Devices, Inc.
..  SPDX-License-Identifier: BSD-3-Clause

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
