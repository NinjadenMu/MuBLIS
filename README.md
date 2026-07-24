### MuBLIS
More software than ever can benefit from high performance implementations of linear algebra routines (think machine learning).  Because optimizing linear algebra routines is hard and highly dependent on hardware, software like NumPy and PyTorch usually link against BLAS (basic linear algebra subroutines) implementations provided by hardware vendors like Nvidia (cuBLAS) and Intel (MKL).

Implementing a full BLAS library (which you may want to do if you're working with obscure/new hardware) can be time consuming and difficult to get right.  MuBLIS splits the work of implementing a L3 BLAS library into a larger generic portion that requires no hardware-specific optimizations and small "micro-kernels" which should be hand optimized to the micro-architecture.  It then serves as two libraries in one.  By providing an implementation of the large generic frame, users can build new L3 BLAS libraries targeted towards custom hardware by writing ~200 lines of hardware-specific code (which explains MuBLIS's name: BLIS = BLAS-like Library Instantiation Software).  By also providing optimized micro-kernels for some common ARM and x86 micro-architectures, it can also be used out of the box as an efficient L3 BLAS implementation for many existing CPUs.  An unoptimized reference micro-kernel is provided for testing purposes, but it also allows MuBLIS to function as a L3 BLAS implementation for almost any micro-architecture.

The BLAS interface has 3 levels: L1 for scalar and vector operations, L2 for matrix-vector operations, and L3 for matrix-matrix operations.  L3 operations benefit the most from optimization, since memory loads grow in $O(n^2)$ while computation grows in $O(n^3)$.  Because of this, cache and register optimizations that allow for more computation to be done for a single load (usually by achieving better reuse) can lead to dramatically faster (think 2 orders of magnitude!) implementations.  L1 and L2 operations have comparatively less headroom for optimization since they do at most $O(n^2)$ computation work, and are easier to implement.  Therefore, MuBLIS currently only implements functionality for L3 BLAS.  While this is a common decision BLAS-like libraries make (NVBLAS focuses on L3 routines since L1 and L2 often aren't worth the time it takes to send data to the GPU), [BLIS](https://github.com/flame/blis), which this project is heavily inspired by, does in fact support the full BLAS interface.

BLAS exposes the following L3 routines for single and double precision real matrices:
With "op" denoting an optional transpose, 
- GEMM (General Matrix Multiply):  
  Computes $C \leftarrow \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta C$
- SYMM (Symmetric Matrix Multiply):  
  Computes $C \leftarrow \alpha \cdot A \cdot B + \beta C$ (left sided) or $C \leftarrow \alpha \cdot B \cdot A + \beta C$ (right sided) with $A$ as a symmetric matrix (optionally, only one triangle of A needs to be stored)
- SYRK (Symmetric Rank-k Update):  
    $C \leftarrow \alpha \cdot A \cdot A^T + \beta \cdot C$ (left sided) or $C \leftarrow \alpha \cdot A^T \cdot A + \beta \cdot C$ (right sided), with $C$ as a symmetric matrix (this routine will only read and update one triangle, since $C$ remains symmetric)
- SYR2K (Symmetric Rank-2k Update):
$C \leftarrow \alpha \cdot A \cdot B^T + \alpha \cdot B \cdot A^T + \beta \cdot C$ (left sided) or $C \leftarrow \alpha \cdot A^T \cdot B + \alpha \cdot B^T \cdot A + \beta \cdot C$ (right sided), with $C$ as a symmetric matrix (this routine will only read and update one triangle, since $C$ remains symmetric).
- TRMM (Triangular Matrix Multiply):  
Overwrites $B$ in place with $B \leftarrow \alpha \cdot \text{op}(A) \cdot B$ (left sided) or $B \leftarrow \alpha \cdot B \cdot \text{op}(A)$ (right sided), with $A$ as a triangular matrix (optionally unit triangular).
- TRSM (Triangular Solve with Multiple Right-Hand Sides):  
Solves $\text{op}(A) \cdot X = \alpha B$ (left sided) or $X \cdot \text{op}(A) = \alpha B$ (right sided), with $A$ as a triangular matrix (optionally unit triangular), and overwrites $B$ in place with the solution $X$.
  
MuBLIS implements the out-of-place operations (GEMM, SYMM, SYRK, SYR2K) as shims over a general L3 driver.  The L3 driver is incredibly powerful, since it represents operations via their iteration space (fundamentally, matrix multiplies are 2 spatial and 1 reduction loop, often over i-j-k), allowing it to handle arbitrary combinations of input and output matrix structures (dense, symmetric, and triangular).

Because this L3 driver can be so general, MuBLIS exposes its interface as well as the standard L3 (C)BLAS interface.
