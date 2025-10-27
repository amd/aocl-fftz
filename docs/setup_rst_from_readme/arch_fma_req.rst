CPU Architecture Support and FMA Requirements
`````````````````````````````````````````````

AOCL-FFTZ leverages advanced CPU features for optimal performance:

**FMA (Fused Multiply-Add) Support:**
- The library uses FMA3 instructions when available
- The FMA compiler flag is added only when compiling AVX512 and AVX256 optimized kernels
- The FMA compiler flag is not added for AVX128 during compilation

**Runtime Behavior:**
- Library automatically detects CPU capabilities at runtime
- If FMA is not supported by the system, the library falls back to AVX128 kernels
- If AVX is not supported, the library executes using standard C implementation

**SIMD ISA Support:**
- The library uses x86 SIMD AVX128, AVX256 and AVX512 instructions when available
- Library uses dynamic dispatcher to automatically detect the CPU capabilities and dispatch the optimal ISA kernels based on selector model