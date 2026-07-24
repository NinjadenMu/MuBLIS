### What's MuBLIS?
MuBLIS is a fast thread-safe library for optimized linear algebra routines, and a "meta" library that makes building new specialized linear algebra libraries easier (more on that in the next paragraphs).

More software than ever can benefit from high performance implementations of linear algebra routines (think machine learning).  Because optimizing linear algebra routines is hard and highly dependent on hardware, software like NumPy and PyTorch usually link against BLAS (basic linear algebra subroutines) libraries provided by hardware vendors like Nvidia (cuBLAS) and Intel (MKL).

Implementing a full BLAS library (which you may want to do if you're working with obscure/new hardware) can be time consuming and difficult to get right.  MuBLIS splits the work of implementing a L3 BLAS library into a larger generic portion that requires no hardware-specific optimizations and small "micro-kernels" which should be hand optimized to the micro-architecture.  

It then serves as two libraries in one.  By providing an implementation of the large generic frame, users can quickly build new L3 BLAS libraries targeted towards custom hardware by writing ~200 lines of hardware-specific code (which explains MuBLIS's name: BLIS = BLAS-like Library Instantiation Software).  I've also written and included a few optimized micro-kernels for common ARM and x86 micro-architectures, so MuBLIS can also be used out of the box as an efficient L3 BLAS implementation for many existing CPUs.

MuBLIS can also be used to produce "fat binaries" at compile-time which support entire families of hardware, only specializing to a specific hardware target at run-time.

### L3 BLAS
The BLAS interface has 3 levels: L1 for scalar and vector operations, L2 for matrix-vector operations, and L3 for matrix-matrix operations.  L3 operations benefit the most from optimization, since memory loads grow in $O(n^2)$ while computation grows in $O(n^3)$.  Because of this, cache and register optimizations that allow for more computation to be done for a single load (usually by achieving better reuse) can lead to dramatically faster (think 2 orders of magnitude!) implementations.  L1 and L2 operations have comparatively less headroom for optimization since they do at most $O(n^2)$ computation work, and are easier to implement.  Therefore, MuBLIS currently only implements functionality for L3 BLAS.  [BLIS](https://github.com/flame/blis), which this project is heavily inspired by, does in fact support the full BLAS interface.

BLAS exposes the following L3 routines for single and double precision real matrices:
With "op" denoting an optional transpose, 
- **GEMM** (General Matrix Multiply):  
  Computes $C \leftarrow \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta C$
- **SYMM** (Symmetric Matrix Multiply):  
  Computes $C \leftarrow \alpha \cdot A \cdot B + \beta C$ (left sided) or $C \leftarrow \alpha \cdot B \cdot A + \beta C$ (right sided) with $A$ as a symmetric matrix (optionally, only one triangle of A needs to be stored)
- **SYRK** (Symmetric Rank-k Update):  
    $C \leftarrow \alpha \cdot A \cdot A^T + \beta \cdot C$ (left sided) or $C \leftarrow \alpha \cdot A^T \cdot A + \beta \cdot C$ (right sided), with $C$ as a symmetric matrix (this routine will only read and update one triangle, since $C$ remains symmetric)
- **SYR2K** (Symmetric Rank-2k Update):
$C \leftarrow \alpha \cdot A \cdot B^T + \alpha \cdot B \cdot A^T + \beta \cdot C$ (left sided) or $C \leftarrow \alpha \cdot A^T \cdot B + \alpha \cdot B^T \cdot A + \beta \cdot C$ (right sided), with $C$ as a symmetric matrix (this routine will only read and update one triangle, since $C$ remains symmetric).
- **TRMM** (Triangular Matrix Multiply):  
Overwrites $B$ in place with $B \leftarrow \alpha \cdot \text{op}(A) \cdot B$ (left sided) or $B \leftarrow \alpha \cdot B \cdot \text{op}(A)$ (right sided), with $A$ as a triangular matrix (optionally unit triangular).
- **TRSM** (Triangular Solve with Multiple Right-Hand Sides):  
Solves $\text{op}(A) \cdot X = \alpha B$ (left sided) or $X \cdot \text{op}(A) = \alpha B$ (right sided), with $A$ as a triangular matrix (optionally unit triangular), and overwrites $B$ in place with the solution $X$.
  
MuBLIS implements the out-of-place BLAS operations (GEMM, SYMM, SYRK, SYR2K) as shims to a general **L3 driver**.  The L3 driver is incredibly powerful, since it can handle arbitrary combinations of input and output matrix structures by allowing callers to specify its iteration space (fundamentally, matrix multiplies are 2 spatial and 1 reduction loops - think i, j, k).

Because this L3 driver can be so general, MuBLIS exposes its interface as well as the standard L3 (C)BLAS interface.

### Micro-kernels
The generic framework MuBLIS provides does cache-level optimizations like packing and tiling (since the "shape" of these optimizations is shared across all modern CPUs with multi-level caches), and breaks down a complex BLAS operation into a small unit of computation handled by a hardware-specific micro-kernel.  To optimize for hardware not supported out of the box by MuBLIS, users will have to implement new micro-kernels.  

Common optimizations include:
- Register blocking
- Vectorization
- Software prefetch
- Considerations made for instruction level parallelism

More information is included in the README in `targets/`, as well as example micro-kernels implementing the aforementioned optimizations with C intrinsics.
