Running Test Bench On Linux & Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AOCL-FFTZ test bench supports multiple options in order to compute, validate & benchmark FFT.

Following are a few sample commands to use and test with the test bench:

* The test bench can be run by using the following syntax:
  
    aocl_fftz_bench [OPTIONS]... PROBLEM_SIZE

* Use the following command to set the precision for FFT:
  
    aocl_fftz_bench -p/--precision <d/f>

* Use the following command to set the data model for FFT:
    
    ./aocl_fftz_bench -m/--data-model <l/i>

* Use the following command to run the test bench with the requested bench type:
    
    aocl_fftz_bench -b/--bench-type <p/a>

* Use the following command to run the test bench with the requested FFT type:
    
    aocl_fftz_bench -f/--fft-type <c2c>

* To check other options for test bench use the following command:
    
    aocl_fftz_bench -h/--help
