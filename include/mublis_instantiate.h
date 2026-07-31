/**
 * @file mublis_instantiate.h
 * @brief Interface and stamping macros for user-provided targets and configs
 * 
 * A target implements the micro-kernels and tuning parameters used by the
 * hardware-independent MuBLIS frame.  A config implements
 * `mublis_get_context`, which selects one of the targets linked into the
 * library.
 *
 * Target authors should provide single- and double-precision GEMM, lower
 * GEMMTRSM, and upper GEMMTRSM micro-kernels.  They should then collect those
 * kernels and their cache and register block sizes in a `mublis_context_t`.
 * 
 * Interfaces, typedefs, and stamping macros for user-provided functions and 
 * objects are defined here.
 */

#ifndef MUBLIS_INSTANTIATE_H
#define MUBLIS_INSTANTIATE_H

/**
 * @brief Optional information supplied to a micro-kernel
 *
 * `c_next` points to the output tile that the frame expects to visit next.
 * Optimized kernels may use it as a write-prefetch hint. 
 * 
 * Check both `aux` and `aux->c_next` before using the hint, since both 
 * may be `NULL`.
 */
typedef struct {
  const void *c_next;
} mublis_auxinfo_t;

/**
 * @brief Defines a GEMM micro-kernel function-pointer type
 *
 * A GEMM micro-kernel computes a complete MR-by-NR tile:
 *  C := alpha * A * B + beta * C
 * 
 * `A` and `B` are packed by the MuBLIS frame. For `0 <= p < k`,
 * `0 <= i < MR`, and `0 <= j < NR`, their elements are stored as:
 *   A(i, p) = a[p * MR + i]
 *   B(p, j) = b[p * NR + j]
 *
 * The output element C(i, j) is stored at
 * `c[i * rs_c + j * cs_c]`.  Kernels must therefore support arbitrary output
 * row and column strides, not only a particular matrix layout.
 *
 * The frame handles edge tiles through a temporary full-size tile, so the
 * kernel always processes exactly MR rows and NR columns.
 * If `beta` is zero, the kernel should not read C.
 */
#define MUBLIS_GEMM_UKR_TYPE(ctype, name)                                      \
  typedef void (*name)(                                                        \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a,                                                   \
    const ctype *restrict b,                                                   \
    ctype beta,                                                                \
    ctype *restrict c,                                                         \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  )

MUBLIS_GEMM_UKR_TYPE(float, mublis_sgemm_ukr_ft);
MUBLIS_GEMM_UKR_TYPE(double, mublis_dgemm_ukr_ft);

#undef MUBLIS_GEMM_UKR_TYPE

/**
 * @brief Defines a fused GEMM and triangular-solve micro-kernel type
 *
 * A GEMMTRSM micro-kernel first forms the MR-by-NR residual
 *  R := alpha * B11 - A1x * Bx1
 * and then solves either a lower- or upper-triangular MR-by-MR system
 *  A11 * X = R.
 *
 * `a1x` is a packed MR-by-k panel with element A1x(i, p) stored at
 * `a1x[p * MR + i]`.  `bx1` is a packed k-by-NR panel with element Bx1(p, j)
 * stored at `bx1[p * NR + j]`.
 *
 * `a11` is a packed MR-by-MR column-major triangular block.  Its diagonal has
 * already been inverted by the frame.  `b11` is an MR-by-NR row-major packed
 * block and must be overwritten with X so later solve steps can reuse the
 * result.
 *
 * The kernel must also write X to `c11`, where element C11(i, j) is stored at
 * `c11[i * rs_c + j * cs_c]`.  The frame handles edge tiles through a
 * temporary full-size tile, so the kernel always processes MR-by-NR values.
 * `k` may be zero.
 *
 * Lower and upper kernels share this interface but should traverse `a11` in 
 * forward and reverse substitution order respectively.
 */
#define MUBLIS_GEMMTRSM_UKR_TYPE(ctype, name)                                  \
  typedef void (*name)(                                                        \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a1x,                                                 \
    const ctype *restrict a11,                                                 \
    const ctype *restrict bx1,                                                 \
    ctype *restrict b11,                                                       \
    ctype *restrict c11,                                                       \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  )

MUBLIS_GEMMTRSM_UKR_TYPE(float, mublis_sgemmtrsml_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(float, mublis_sgemmtrsmu_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(double, mublis_dgemmtrsml_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(double, mublis_dgemmtrsmu_ukr_ft);

#undef MUBLIS_GEMMTRSM_UKR_TYPE

/**
 * @brief Single-precision kernels and block sizes for a target
 *
 * `mr` and `nr` are the register tile dimensions consumed by every
 * single-precision micro-kernel in the context.  `mc`, `kc`, and `nc` are the
 * cache-level block sizes for the m, k, and n dimensions, respectively.
 *
 * All sizes must be positive.  `mc` must be a multiple of `mr`, `nc` must be a
 * multiple of `nr`, and `kc` must be a multiple of both `mr` and `nr`.
 */
typedef struct {
  mublis_sgemm_ukr_ft gemm_ukr;
  mublis_sgemmtrsml_ukr_ft gemmtrsml_ukr;
  mublis_sgemmtrsmu_ukr_ft gemmtrsmu_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_scontext_t;

/**
 * @brief Double-precision kernels and block sizes for a target
 * 
 * See the header for `mublis_scontext_t`
 */
typedef struct {
  mublis_dgemm_ukr_ft gemm_ukr;
  mublis_dgemmtrsml_ukr_ft gemmtrsml_ukr;
  mublis_dgemmtrsmu_ukr_ft gemmtrsmu_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_dcontext_t;

/**
 * @brief Complete specialization selected by a configuration
 */
typedef struct {
  mublis_scontext_t s;
  mublis_dcontext_t d;
} mublis_context_t;

/**
 * @brief Declares a GEMM micro-kernel with the required signature
 *
 * This is useful for kernels implemented outside the source file that creates
 * their context, such as kernels implemented in assembly.
 */
#define MUBLIS_GEMM_UKR_PROT_STAMP(ctype, name)                                \
  void name(                                                                   \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a,                                                   \
    const ctype *restrict b,                                                   \
    ctype beta,                                                                \
    ctype *restrict c,                                                         \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  );

/**
 * @brief Declares a GEMMTRSM micro-kernel with the required signature
 *
 * This is useful for kernels implemented outside the source file that creates
 * their context, such as kernels implemented in assembly.
 */
#define MUBLIS_GEMMTRSM_UKR_PROT_STAMP(ctype, name)                            \
  void name(                                                                   \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a1x,                                                 \
    const ctype *restrict a11,                                                 \
    const ctype *restrict bx1,                                                 \
    ctype *restrict b11,                                                       \
    ctype *restrict c11,                                                       \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  );

/**
 * @brief Defines a constant target context
 *
 * The first group of arguments supplies the single-precision kernels, register
 * tile sizes, and cache block sizes.  The second group supplies the equivalent
 * double-precision values.
 *
 * `context_name` must match the name registered in `targets/target_registry.h`.
 */
#define MUBLIS_CONTEXT_STAMP(                                                  \
  context_name,                                                                \
  sgemm_ukr_value,                                                             \
  sgemmtrsml_ukr_value,                                                        \
  sgemmtrsmu_ukr_value,                                                        \
  s_mr_value, s_nr_value,                                                      \
  s_mc_value, s_kc_value, s_nc_value,                                          \
  dgemm_ukr_value,                                                             \
  dgemmtrsml_ukr_value,                                                        \
  dgemmtrsmu_ukr_value,                                                        \
  d_mr_value, d_nr_value,                                                      \
  d_mc_value, d_kc_value, d_nc_value                                           \
)                                                                              \
  const mublis_context_t context_name = {                                      \
    .s = {                                                                     \
      .gemm_ukr = (sgemm_ukr_value),                                           \
      .gemmtrsml_ukr = (sgemmtrsml_ukr_value),                                 \
      .gemmtrsmu_ukr = (sgemmtrsmu_ukr_value),                                 \
      .mr = (s_mr_value),                                                      \
      .nr = (s_nr_value),                                                      \
      .mc = (s_mc_value),                                                      \
      .kc = (s_kc_value),                                                      \
      .nc = (s_nc_value)                                                       \
    },                                                                         \
    .d = {                                                                     \
      .gemm_ukr = (dgemm_ukr_value),                                           \
      .gemmtrsml_ukr = (dgemmtrsml_ukr_value),                                 \
      .gemmtrsmu_ukr = (dgemmtrsmu_ukr_value),                                 \
      .mr = (d_mr_value),                                                      \
      .nr = (d_nr_value),                                                      \
      .mc = (d_mc_value),                                                      \
      .kc = (d_kc_value),                                                      \
      .nc = (d_nc_value)                                                       \
    }                                                                          \
  };

/**
 * @brief Selects the target context used by the library
 * @param context Destination for the selected context
 * @return 0 on success and a nonzero value on failure
 *
 * Every configuration must define this function.  On success, it must write a
 * complete context to `context`.  A configuration may inspect the running
 * hardware and select any target listed in its `CONFIG_TARGETS`.
 */
int mublis_get_context(mublis_context_t *context);

#endif
