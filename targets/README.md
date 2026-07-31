MuBLIS's generic frame breaks down a complex BLAS routine into small units of computation to be done by small hardware-optimized micro-kernels.  MuBLIS takes care of cache-tiling for you, but requires that the target specifies cache blocking parameters.  

A target contains the micro-kernels, parameters, and build flags needed for MuBLIS to specialize to hardwdare.

The `reference` target is the simplest example.  `aarch64`, `firestorm`, and `avx2` demonstrate reasonably well-optimized micro-kernel implementations with register blocking, vectorization, and prefetching (although MuBLIS's purpose is not providing the best possible micro-kernels.)

### Target Code
`include/mublis_instantiate.h` is the primary source of information you'll need for implementation.

Create `targets/{target name}/`.  Target names should be Make-compatible identifiers, such as `avx2` or `cortex_a`.

A target must provide a context object containing all the information needed by the MuBLIS frame for specialization (see `mublis_context_t` in `include/mublis_instantiate.h`).

The context object should include:
- Function pointers to single and double-precision micro-kernels for:
  - GEMM
  - Lower-triangular fused GEMM-TRSM
  - Upper-triangular fused GEMM-TRSM
- Register block sizes:
  - S_MR - rows per single precision A micro-panel
  - S_NR - columns per single precision B micro-panel
  - D_MR - rows per double precision A micro-panel
  - D_NR - columns per double precision B micro-panel
- Cache block sizes (see `mublis_scontext_t` in `include/mublis_instantiate.h` for divisibility requirements):
  - S_MC - rows per single precision A panel
  - S_NC - columns per single precision B panel
  - D_MC - rows per double precision A panel
  - D_NC - columns per double precision B panel

See headers for micro-kernel functions in `include/mublis_instantiate.h`.  MuBLIS supports implementing micro-kernels in assembly.  If you choose to do so, C function headers may be easily created using the `MUBLIS_GEMM_UKR_PROT_STAMP` and `MUBLIS_GEMMTRSM_UKR_PROT_STAMP` macros in `include/mublis_instantiate.h`.

The context object may easily be created using the `MUBLIS_CONTEXT_STAMP` macro defined in `include/mublis_instantiate.h`.

Once the context object is created, register it by name in `targets/target_registry.h`.  The context object is now discoverable and usable by configs(and consequently usable by MuBLIS).  The context name must be unique.

## Target Build Rules
Every target also needs `targets/{target name}/target.mk`.  The suffix in each variable must exactly match the target directory and name used by `CONFIG_TARGETS` in `config/{target user}/config.mk`.

At least one C or assembly source is required.  Available variables are:
- `TARGET_{target name}_C_SRCS`: C source files.
- `TARGET_{target name}_S_SRCS`: preprocessed assembly source files.

Build flags applied specifically to the target may be specified using:
- `TARGET_{target name}_FLAGS`: flags applied to both C and assembly.
- `TARGET_{target name}_CPPFLAGS`: preprocessor flags applied to both.
- `TARGET_{target name}_CFLAGS`: additional C-only flags.
- `TARGET_{target name}_ASFLAGS`: additional assembly-only flags.
Config and frame files will still be built generically.
