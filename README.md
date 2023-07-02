AOCL-FFTZ
=========

AOCL-FFTZ is a high performance Fast Fourier Transform (FFT) library developed
by AMD supporting advanced optimizations for AMD’s "Zen"-based CPUs. 
The library computes FFTs of real and complex data of any size and dimension in
both forward and backward directions. The important kernels in this library are
vectorized to speed-up the single-threaded core performance. The library 
supports the computations of parallel FFTs by taking advantage of 
(i) shared-memory parallelism using openMP threads, and (ii) distributed-memory
parallelism using MPI.
AOCL-FFTZ introduces a generic and unified API set for supporting any 
precision types (single-precision and double-precision), and both the
single-threaded and multi-threaded execution modes.
AOCL-FFTZ supports distributed FFTs with a separate API set that makes use of
an underlying MPI framework for communication.
The library uses a dynamic dispatcher feature to run efficiently and portably
across different x86 based systems.
A test bench is supported for performance and functional tests including the
accuracy tests. GTest based unit testing framework is also supported by the 
library.



Installation
------------


Building on Linux
-----------------


Building on Windows
-------------------


Running Test Bench On Linux
---------------------------


Running Test Bench On Windows
-----------------------------


Running tests with CTest
------------------------


Running Performance Benchmark test
----------------------------------


Generating Documentation
------------------------


CONTACTS
--------
AOCL-FFTZ is developed and maintained by AMD.<br>
For support, send an email to toolchainsupport@amd.com.