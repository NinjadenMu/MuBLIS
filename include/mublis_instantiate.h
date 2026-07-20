#ifndef MUBLIS_INSTANTIATE_H
#define MUBLIS_INSTANTIATE_H

typedef struct {
  const void *a_next;
  const void *b_next;
} mublis_auxinfo_t;

typedef void (*mublis_sgemm_ukr_ft)(
  int k,
  float alpha,
  const float *restrict a,
  const float *restrict b,
  const float beta,
  float *restrict c,
  int rs_c, int cs_c,
  const mublis_auxinfo_t *aux
);

typedef void (*mublis_dgemm_ukr_ft)(
  int k,
  double alpha,
  const double *restrict a,
  const double *restrict b,
  const double beta,
  double *restrict c,
  int rs_c, int cs_c,
  const mublis_auxinfo_t *aux
);

/**
 * @brief hardware context for single precision datatype
 * 
 * Contains pointer to micro-kernel function and register and cache 
 * block dimensions.
 * 
 * Context is separated by precision because double precision block dimensions 
 * are typically half of those for single precision, and because  they require 
 * different micro-kernels.
 */
typedef struct {
  // ukr = micro-kernel, with the mu symbol collapsing to a u
  mublis_sgemm_ukr_ft gemm_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_scontext_t;

/**
 * @brief hardware context for double precision datatype
 */
typedef struct {
  mublis_dgemm_ukr_ft gemm_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_dcontext_t;

/**
 * @brief full hardware context
 * 
 * Contains a context object for single and double precision.
 */
typedef struct {
  mublis_scontext_t s;
  mublis_dcontext_t d;
} mublis_context_t;

#define MUBLIS_GEMM_UKR_PROT_STAMP(ctype, name) \
  void name(                                    \
    int k,                                      \
    ctype alpha,                                \
    const ctype *restrict a,                    \
    const ctype *restrict b,                    \
    ctype beta,                                 \
    ctype *restrict c,                          \
    int rs_c, int cs_c,                         \
    const mublis_auxinfo_t *aux                 \
  );

#define MUBLIS_CONTEXT_STAMP(                                             \
  context_name,                                                           \
  sgemm_ukr_value, s_mr_value, s_nr_value,                                \
  s_mc_value, s_kc_value, s_nc_value,                                     \
  dgemm_ukr_value, d_mr_value, d_nr_value,                                \
  d_mc_value, d_kc_value, d_nc_value                                      \
)                                                                         \
  mublis_context_t const context_name = {                                 \
    .s = {                                                                \
      .gemm_ukr = (sgemm_ukr_value),                                      \
      .mr = (s_mr_value),                                                 \
      .nr = (s_nr_value),                                                 \
      .mc = (s_mc_value),                                                 \
      .kc = (s_kc_value),                                                 \
      .nc = (s_nc_value)                                                  \
    },                                                                    \
    .d = {                                                                \
      .gemm_ukr = (dgemm_ukr_value),                                      \
      .mr = (d_mr_value),                                                 \
      .nr = (d_nr_value),                                                 \
      .mc = (d_mc_value),                                                 \
      .kc = (d_kc_value),                                                 \
      .nc = (d_nc_value)                                                  \
    }                                                                     \
  };

/**
 * @brief Gets context object for runtime hardware specialization
 * @return 0 on success, nonzero integer otherwise.
 * 
 * Must be defined by user at compile-time in config/<config_name>/context.c.
 * MuBLIS uses this at runtime to get the hardware context used to dispatch 
 * to optimized implementations for the hardware.
 */
int mublis_get_context(mublis_context_t *context);

#endif
