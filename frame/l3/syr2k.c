#include "l3.h"

#define MUBLIS_SYR2K_IMPL(ctype, function_name, driver_name)                   \
  int function_name(                                                          \
    mublis_uplo_t uplo, mublis_trans_t trans,                                 \
    int n, int k,                                                              \
    ctype alpha,                                                               \
    const ctype *a, int rs_a, int cs_a,                                       \
    const ctype *b, int rs_b, int cs_b,                                       \
    ctype beta,                                                                \
    ctype *c, int rs_c, int cs_c                                              \
  ) {                                                                          \
    if (trans == MUBLIS_TRANSPOSE) {                                           \
      int tmp = rs_a;                                                          \
      rs_a = cs_a;                                                             \
      cs_a = tmp;                                                              \
                                                                               \
      tmp = rs_b;                                                              \
      rs_b = cs_b;                                                             \
      cs_b = tmp;                                                              \
    }                                                                          \
                                                                               \
    mublis_l3_relation_t ji = MUBLIS_L3_ALL;                                   \
    if (uplo == MUBLIS_LOWER)                                                  \
      ji = MUBLIS_L3_LOWER;                                                    \
    else if (uplo == MUBLIS_UPPER)                                             \
      ji = MUBLIS_L3_UPPER;                                                    \
                                                                               \
    const mublis_l3_product_t product = {                                      \
      .domain = {                                                              \
        .jp = MUBLIS_L3_ALL,                                                   \
        .ji = ji,                                                              \
        .pi = MUBLIS_L3_ALL                                                    \
      },                                                                       \
      .a = {                                                                   \
        .struc = MUBLIS_STRUC_GENERAL,                                         \
        .uplo = MUBLIS_DENSE,                                                  \
        .diag = MUBLIS_PACKM_DIAG_NONUNIT                                      \
      },                                                                       \
      .b = {                                                                   \
        .struc = MUBLIS_STRUC_GENERAL,                                         \
        .uplo = MUBLIS_DENSE,                                                  \
        .diag = MUBLIS_PACKM_DIAG_NONUNIT                                      \
      }                                                                        \
    };                                                                         \
                                                                               \
    int error_code = driver_name(                                              \
      n, n, k,                                                                 \
      alpha,                                                                   \
      a, rs_a, cs_a,                                                           \
      b, cs_b, rs_b,                                                           \
      beta,                                                                    \
      c, rs_c, cs_c,                                                           \
      &product                                                                 \
    );                                                                         \
                                                                               \
    if (error_code)                                                            \
      return error_code;                                                       \
                                                                               \
    return driver_name(                                                        \
      n, n, k,                                                                 \
      alpha,                                                                   \
      b, rs_b, cs_b,                                                           \
      a, cs_a, rs_a,                                                           \
      1,                                                                       \
      c, rs_c, cs_c,                                                           \
      &product                                                                 \
    );                                                                         \
  }

MUBLIS_SYR2K_IMPL(float, mublis_ssyr2k, mublis_l3_sdriver)
MUBLIS_SYR2K_IMPL(double, mublis_dsyr2k, mublis_l3_ddriver)

#undef MUBLIS_SYR2K_IMPL
