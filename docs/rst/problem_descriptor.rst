..  Copyright Advanced Micro Devices, Inc.
..  SPDX-License-Identifier: BSD-3-Clause

.. _ref-problem_descriptor:

Problem Descriptor Construction
-------------------------------

The library supports 4 different problem descriptors:

-  :ref:`aoclfftz_prob_desc_f <aoclfftz_prob_desc_f>` - for single-precision FFTs on LP64 systems
-  :ref:`aoclfftz_prob_desc_d <aoclfftz_prob_desc_d>` - for double-precision FFTs on LP64 systems
-  :ref:`aoclfftz_prob_desc_f_64_ <aoclfftz_prob_desc_f_64_>` - for single-precision FFTs on ILP64 systems
-  :ref:`aoclfftz_prob_desc_d_64_ <aoclfftz_prob_desc_d_64_>` - for double-precision FFTs on ILP64 systems

Structure Reference
~~~~~~~~~~~~~~~~~~~

Consider an example for single-precision FFTs on LP64 systems.

The main structure for the problem descriptor is as follows:

.. code-block:: c

    typedef struct
    {
        FFTZ_FLOAT *in;
        FFTZ_FLOAT *out;
        FFTZ_INT32 vec_rank;
        FFTZ_INT32 dim_rank;
        aoclfftz_dim_t *dims;
        aoclfftz_dim_t *vecs;
        aoclfftz_flags_t flags;
        aoclfftz_smp_pfft_t pthr_fft;
        aoclfftz_cntrl_params_t cntrl_params;
    } aoclfftz_prob_desc_f;

Member-by-Member Guide
~~~~~~~~~~~~~~~~~~~~~~

**1. Buffers (``in``, ``out``)**

- *in*: Points to the input data (signal for forward FFT, frequency for backward FFT).
- *out*: Points to the output data (frequency for forward FFT, signal for backward FFT).
- **Constraints**:

  - May be ``NULL`` during setup. A handle whose descriptor leaves ``in`` or ``out`` as ``NULL`` must be run with :cpp:func:`aoclfftz_execute_io`; otherwise, :cpp:func:`aoclfftz_execute` returns ``AOCLFFTZ_EXECUTION_FAILURE``.
  - When supplied, must be allocated with sufficient memory to accommodate the dimensions and strides specified by ``dims`` and ``vecs``. Insufficient allocation may result in out-of-bounds memory access.
  - For in-place transforms (``flags.fft_placement == 0``), ``in`` and ``out`` must be the same pointer.
  - For out-of-place transforms (``flags.fft_placement == 1``), ``in`` and ``out`` must be different pointers.
  - ``NULL`` input and output pointers represent an in-place transform; exactly one ``NULL`` pointer represents an out-of-place transform.
  - Buffers later passed to :cpp:func:`aoclfftz_execute_io` must preserve the setup-time placement (in-place/out-of-place); they are not re-checked against the flag.

**2. Ranks (``vec_rank``, ``dim_rank``)**

- *vec_rank*: Number of batch dimensions (length of ``vecs`` array). Must be :math:`\geq 1`.
- *dim_rank*: Number of signal/frequency dimensions (length of ``dims`` array). Must be :math:`\geq 1`.

**3. Dimension Arrays (``dims``, ``vecs``)**

- *dims*: Array of ``aoclfftz_dim_t`` describing each signal/frequency dimension.
- *vecs*: Array of ``aoclfftz_dim_t`` describing each batch dimension.

- **Constraints for each element**:

  - ``dims`` - ``n`` (size), ``in_stride`` and ``out_stride``: Must be > 0.
  - ``vecs`` - ``in_stride`` and ``out_stride``: Must be > 0.
  - ``vecs`` - ``n`` (batch size): Must be :math:`\geq 0`. A batch size of 0 is treated as a no-op. Negative values are rejected.
  - For in-place transforms, ``in_stride`` must equal ``out_stride`` for each dimension.
  - For real transforms (R2C/C2R), special stride rules must be followed (see :ref:`Stride Setting <stride-setting>`).


**4. Flags (``flags``)**

- *fft_type*: Complex (0) or Real (1).
- *fft_direction*: Forward (0) or Backward (1).
- *storage_order*: Must be 0 (in-order only; out-of-order not supported).
- *fft_placement*: In-place (0) or Out-of-place (1).
- *transpose_mode*: Must be 0 (standalone transpose not supported).
- *bit_reproducibility*: 0 (disable bit reproducibility mode, default) or 1 (enable bit reproducibility mode).


**5. Parallel FFT (``pthr_fft``)**

- *num_threads*: Number of threads (:math:`\geq 1`). If greater than available CPUs, defaults to max CPUs. If less than 1, defaults to single-threaded execution.
- *dynamic_load_model*: 0 (fixed threads - *num_threads* is spent as given) or 1 (dynamic threads - *num_threads* remains the maximum, and fewer may be used when the problem is too small to keep them busy; default). Invalid values default to 1.

**6. Control Parameters (``cntrl_params``)**

- *opt_level*: 0-3 (optimization level). Out-of-range values default to maximum opt level - 3.
- *opt_off*: 0 (optimizations enabled) or 1 (disabled).
- *logger_mode*: 0-3. Out-of-range values default to 0. Error logs are always enabled despite logger_mode setting.
- *measure_stats*: Must be 0 (not supported).

.. note::

   - opt-off takes precedence over opt-level.
   - Setting opt_level to 3 on a non-AVX512 CPU will put dynamic dispatcher in action, where the next highest possible ISA will be chosen.

Validation Checklist
~~~~~~~~~~~~~~~~~~~~

Before calling setup function, :ref:`aoclfftz_setup_f <aoclfftz_setup_f>` in this case, ensure:

- ``dims`` and ``vecs`` are non-NULL.
- If ``in`` or ``out`` is ``NULL`` during setup, use :ref:`aoclfftz_execute_io <aoclfftz_execute_io>` with both buffers allocated and initialized before execution.
- All ranks (``vec_rank``, ``dim_rank``) are :math:`\geq 1`.
- All ``dims`` sizes and strides are > 0, all ``vecs`` strides are > 0, and all ``vecs`` batch sizes are non-negative.
- Stride rules for in-place/out-of-place and real/complex transforms are satisfied.
- Flags are set to supported values.
- Parallel and control parameters are within valid ranges.

.. note::

   Setting out-of-range values for Parallel and Control parameters will not cause Setup failure, but defaults will be applied and Setup will proceed.

Common Pitfalls
~~~~~~~~~~~~~~~

- Using out-of-order storage or standalone transpose (not supported in this release).
- Mismatched strides for in-place transforms.
- Null ``dims`` or ``vecs`` arrays.
- Calling :cpp:func:`aoclfftz_execute` on a handle set up with ``in`` or ``out`` buffer set to ``NULL``. That returns ``AOCLFFTZ_EXECUTION_FAILURE``; use :cpp:func:`aoclfftz_execute_io` with both pointers non-NULL instead.
- Invalid thread or logger values (defaults will be applied).

Example: Minimal Valid Descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

    aoclfftz_dim_t dims[1] = { {1024, 1, 1} }; // 1D FFT of length 1024 with unit elemental stride for input and output data
    aoclfftz_dim_t vecs[1] = { {10, 1024, 1024} }; // Batch of 10; each batch is separated by one full 1024-point transform in input and output

    aoclfftz_prob_desc_f problem =
    {
        .in = input_buffer,
        .out = output_buffer,
        .vec_rank = 1,
        .dim_rank = 1,
        .dims = dims,
        .vecs = vecs,
        .flags = {0, 0, 0, 1, 0, 0}, // Complex, Forward, In-order, Out-of-place, No transpose, Bit reproducibility disabled
        .pthr_fft = {1, 1},
        .cntrl_params = {3, 0, 0, 0}
    };
