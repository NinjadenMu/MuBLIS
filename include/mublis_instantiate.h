#ifndef MUBLIS_INSTANTIATE_H
#define MUBLIS_INSTANTIATE_H

typedef struct {
  const void *c_next;
} mublis_auxinfo_t;

#define MUBLIS_GEMM_UKR_TYPE(ctype, name)                                     \
  typedef void (*name)(                                                       \
    int k,                                                                    \
    ctype alpha,                                                              \
    const ctype *restrict a,                                                  \
    const ctype *restrict b,                                                  \
    ctype beta,                                                               \
    ctype *restrict c,                                                        \
    int rs_c, int cs_c,                                                       \
    const mublis_auxinfo_t *aux                                               \
  )

MUBLIS_GEMM_UKR_TYPE(float, mublis_sgemm_ukr_ft);
MUBLIS_GEMM_UKR_TYPE(double, mublis_dgemm_ukr_ft);

#undef MUBLIS_GEMM_UKR_TYPE

#define MUBLIS_GEMMTRSM_UKR_TYPE(ctype, name)                                 \
  typedef void (*name)(                                                       \
    int k,                                                                    \
    ctype alpha,                                                              \
    const ctype *restrict a1x,                                                \
    const ctype *restrict a11,                                                \
    const ctype *restrict bx1,                                                \
    ctype *restrict b11,                                                      \
    ctype *restrict c11,                                                      \
    int rs_c, int cs_c,                                                       \
    const mublis_auxinfo_t *aux                                               \
  )

MUBLIS_GEMMTRSM_UKR_TYPE(float, mublis_sgemmtrsml_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(float, mublis_sgemmtrsmu_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(double, mublis_dgemmtrsml_ukr_ft);
MUBLIS_GEMMTRSM_UKR_TYPE(double, mublis_dgemmtrsmu_ukr_ft);

#undef MUBLIS_GEMMTRSM_UKR_TYPE

typedef struct {
  mublis_sgemm_ukr_ft gemm_ukr;
  mublis_sgemmtrsml_ukr_ft gemmtrsml_ukr;
  mublis_sgemmtrsmu_ukr_ft gemmtrsmu_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_scontext_t;

typedef struct {
  mublis_dgemm_ukr_ft gemm_ukr;
  mublis_dgemmtrsml_ukr_ft gemmtrsml_ukr;
  mublis_dgemmtrsmu_ukr_ft gemmtrsmu_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_dcontext_t;

typedef struct {
  mublis_scontext_t s;
  mublis_dcontext_t d;
} mublis_context_t;

#define MUBLIS_GEMM_UKR_PROT_STAMP(ctype, name)                               \
  void name(                                                                  \
    int k,                                                                    \
    ctype alpha,                                                              \
    const ctype *restrict a,                                                  \
    const ctype *restrict b,                                                  \
    ctype beta,                                                               \
    ctype *restrict c,                                                        \
    int rs_c, int cs_c,                                                       \
    const mublis_auxinfo_t *aux                                               \
  );

#define MUBLIS_GEMMTRSM_UKR_PROT_STAMP(ctype, name)                           \
  void name(                                                                  \
    int k,                                                                    \
    ctype alpha,                                                              \
    const ctype *restrict a1x,                                                \
    const ctype *restrict a11,                                                \
    const ctype *restrict bx1,                                                \
    ctype *restrict b11,                                                      \
    ctype *restrict c11,                                                      \
    int rs_c, int cs_c,                                                       \
    const mublis_auxinfo_t *aux                                               \
  );

#define MUBLIS_CONTEXT_STAMP(                                                 \
  context_name,                                                              \
  sgemm_ukr_value,                                                           \
  sgemmtrsml_ukr_value,                                                      \
  sgemmtrsmu_ukr_value,                                                      \
  s_mr_value, s_nr_value,                                                    \
  s_mc_value, s_kc_value, s_nc_value,                                        \
  dgemm_ukr_value,                                                           \
  dgemmtrsml_ukr_value,                                                      \
  dgemmtrsmu_ukr_value,                                                      \
  d_mr_value, d_nr_value,                                                    \
  d_mc_value, d_kc_value, d_nc_value                                         \
)                                                                            \
  const mublis_context_t context_name = {                                    \
    .s = {                                                                   \
      .gemm_ukr = (sgemm_ukr_value),                                         \
      .gemmtrsml_ukr = (sgemmtrsml_ukr_value),                               \
      .gemmtrsmu_ukr = (sgemmtrsmu_ukr_value),                               \
      .mr = (s_mr_value),                                                    \
      .nr = (s_nr_value),                                                    \
      .mc = (s_mc_value),                                                    \
      .kc = (s_kc_value),                                                    \
      .nc = (s_nc_value)                                                     \
    },                                                                       \
    .d = {                                                                   \
      .gemm_ukr = (dgemm_ukr_value),                                         \
      .gemmtrsml_ukr = (dgemmtrsml_ukr_value),                               \
      .gemmtrsmu_ukr = (dgemmtrsmu_ukr_value),                               \
      .mr = (d_mr_value),                                                    \
      .nr = (d_nr_value),                                                    \
      .mc = (d_mc_value),                                                    \
      .kc = (d_kc_value),                                                    \
      .nc = (d_nc_value)                                                     \
    }                                                                        \
  };

int mublis_get_context(mublis_context_t *context);

#endif
