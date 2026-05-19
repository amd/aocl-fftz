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

.. _ref-data_layout_conventions:

Data Layout Conventions
-----------------------

Dimension Ordering
~~~~~~~~~~~~~~~~~~

AOCL-FFTZ uses a specific convention for dimension ordering in multi-dimensional FFTs. The innermost dimension is always indexed as Dim 0. This means that elements along Dim 0 are stored contiguously in memory, and this is the fastest changing dimension.

For higher-dimensional problems, Dim 1 is the next outer dimension, followed by Dim 2, and so on. In a 3D FFT, for example, Dim 0 is the innermost, Dim 1 is the middle, and Dim 2 is the outermost dimension. The outermost dimension changes the slowest in memory layout.

This ordering is important for the correctness of FFT. When specifying the dimensions in the descriptor, always list them from innermost (Dim 0) to outermost (highest Dim index).

Example: 3D matrix of shape 2x4x3

- Dim 0 (size 3): innermost, elements are contiguous in memory.
- Dim 1 (size 4): middle dimension.
- Dim 2 (size 2): outermost, changes slowest.

Input & Output Data Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~

AOCL-FFTZ operates with the following data layouts for input and output buffers:

- For Complex C2C Transforms:

    - Input & Output - Interleaved format: real and imaginary parts alternate in memory, e.g., [R0, I0, R1, I1, ..., RN-1, IN-1]
        - Each complex value is stored as two consecutive elements: real part followed by imaginary part.
        - Total length is 2*N for N complex points.

- For Real R2C Transforms:

    - Input - Standard format - e.g., [X0, X1, X2, ..., XN-1]
        - Input is real values.
        - Length is N.
    - Output - Half-Complex format - e.g., [R0, R1, I1, ..., RN/2] when N is even & [R0, R1, I1, ..., RN/2, IN/2] when N is odd
        - Only N/2+1 complex values are stored.
        - Imaginary parts of DC and Nyquist (when N is even) terms are always zero.

- For Real C2R Transforms:

    - Input - Half-Complex format - e.g., [R0, R1, I1, ..., RN/2] when N is even
        - Only N/2+1 complex values are stored.
        - Imaginary parts of DC and Nyquist terms are always zero.
    - Output - Standard format - e.g., [X0, X1, X2, ..., XN-1]
        - Output is real values.
        - Length is N.

.. note::
    Half-Complex format is a data layout that stores only half of the complex points, leveraging the Hermitian symmetry property of FFT on real values. The other half can be inferred by taking the complex conjugate of the stored values.

.. _stride-setting:

Stride Setting
~~~~~~~~~~~~~~

The Complex C2C transforms support arbitrary strides for both input and output data. The only requirement is that for in-place transforms, the input and output strides must be equal.

However, for real transforms (R2C and C2R), special stride rules must be followed.

Row-Major Layout
^^^^^^^^^^^^^^^^

In a row-major layout, the elements are contiguous in memory and the batches are strided apart.

This is a 2D representation of the layout:

.. code-block:: text

   B1 -> [ 0 ][ 1 ][ 2 ] ... [ N-1 ]

   B2 -> [ 0 ][ 1 ][ 2 ] ... [ N-1 ]

   ...

   Bk -> [ 0 ][ 1 ][ 2 ] ... [ N-1 ]

where **B** is Batch, **k** is the number of batches, and **N** is the size of FFT.

The stride rules are as follows:

- Inplace Problems: Since the same buffer is to be used for both input and output,
    - For R2C, The input stride for batches should account for the expanded output size of half-complex. Likewise for C2R's output stride.
    - For example, if you have 4 batches and 50 FFT points (4v50),
            * The output for R2C will have (N/2 + 1) complex points, i.e. 26 complex values for N=50. Since each complex value consists of 2 elements (real and imaginary), the input batch stride should be set to 52 (26 x 2), not 50.
            * The output batch stride would be 26 i.e. (N/2 + 1).
    - Similarly, for C2R, the input batch stride for batches should be 26 and output stride 52.
    - This ensures that each batch's data does not overlap in memory and matches the expected layout for in-place transforms.

    - **Example:**

      For an input problem of 4v50, the correct vec stride settings for ``dims[0].in_stride = dims[0].out_stride = 1`` would be:

      - R2C in-place: ``vecs[0].in_stride = 52``, ``vecs[0].out_stride = 26``
      - C2R in-place: ``vecs[0].in_stride = 26``, ``vecs[0].out_stride = 52``

- Out-of-place Problems: As the input and output buffers are separate, the batch strides can be set independently based on the actual data layout in memory.
    - For R2C, the input batch stride can be set to the actual spacing of real valued input data, while the output batch stride should account for the half-complex format.
    - For C2R, the input batch stride should account for the half-complex format, while the output batch stride can be set to the actual spacing of real valued output data.

    - **Example:**

      For an input problem of 4v50, the correct vec stride settings for ``dims[0].in_stride = dims[0].out_stride = 1`` would be:

      - R2C out-of-place: ``vecs[0].in_stride = 50``, ``vecs[0].out_stride = 26``
      - C2R out-of-place: ``vecs[0].in_stride = 26``, ``vecs[0].out_stride = 50``

Column-Major Layout
^^^^^^^^^^^^^^^^^^^

For a column-major layout, the batches are contiguous in memory and the elements are strided apart.

Consider this 2D representation:

.. code-block:: text

                 B1      B2      B3    ...   Bk

                [ 0 ]   [ 0 ]   [ 0 ]  ...  [ 0 ]

                [ 1 ]   [ 1 ]   [ 1 ]  ...  [ 1 ]

                [ 2 ]   [ 2 ]   [ 2 ]  ...  [ 2 ]

                ...

                [ N-1 ] [ N-1 ] [ N-1 ] ... [ N-1 ]


This layout requires careful stride setting as follows:

- Inplace Problems:
    - For R2C, since batches are contiguous in memory and the input is real, for a unit-strided problem, one might expect the transform's elements to be strided by number of batches.

      However, the output is complex where each real value is transformed in-place into a complex value (real and imaginary) that are stored consecutively, as illustrated below:


    .. code-block:: text

                         B1      B2     ...   Bk                   B1          B2     ...     Bk

                        [ 0 ]   [ 0 ]   ...  [ 0 ]            [ (0r,0i) ] [ (0r,0i) ] ... [ (0r,0i) ]

                        [ 1 ]   [ 1 ]   ...  [ 1 ]      ->    [ (1r,1i) ] [ (1r,1i) ] ... [ (1r,1i) ]

                        [ 2 ]   [ 2 ]   ...  [ 2 ]            [ (2r,2i) ] [ (2r,2i) ] ... [ (2r,2i) ]

                        ...                                   ...

                        [ N-1 ] [ N-1 ] ... [ N-1 ]           [ (Nr,Ni) ] [ (Nr,Ni) ] ... [ (Nr,Ni) ]

    - Therefore, the input elemental stride should be set to (Number of batches * 2) , to account for both real and imaginary parts in the output.
      The output elemental stride can be set to number of batches since the output is complex.

    - The same rule applies while setting the batch strides. In a unit-strided problem, one may expect the input batch stride to be 1 since they are contiguous in memory. But in order to account for the complex output, the input batch stride has to be set to 2, while the output batch stride can remain 1.
    - For C2R, the input & output strides are interchanged.

    - **Example:**

      For an input problem of 4v50, the correct stride setting for a unit-strided problem would be:

      - R2C in-place:

        - ``dims[0].in_stride = 8``, ``dims[0].out_stride = 4``
        - ``vecs[0].in_stride = 2``, ``vecs[0].out_stride = 1``

      - C2R in-place:

        - ``dims[0].in_stride = 4``, ``dims[0].out_stride = 8``
        - ``vecs[0].in_stride = 1``, ``vecs[0].out_stride = 2``

- Out-of-place Problems: Similar to row-major layout, as the input and output buffers are separate, the strides can be set independently based on the actual data layout in memory.
    - For R2C, the input & output elemental strides should be set to the number of batches multiplied by the batch strides and the batch strides should be set to the actual spacing of batches in memory.
    - For C2R, it's the same as R2C.

    - **Example:**

      For an input problem of 4v50, the correct stride setting for a unit-strided problem would be:

      - R2C out-of-place:

        - ``dims[0].in_stride = 4``, ``dims[0].out_stride = 4``
        - ``vecs[0].in_stride = 1``, ``vecs[0].out_stride = 1``

      - C2R out-of-place:

        - ``dims[0].in_stride = 4``, ``dims[0].out_stride = 4``
        - ``vecs[0].in_stride = 1``, ``vecs[0].out_stride = 1``
