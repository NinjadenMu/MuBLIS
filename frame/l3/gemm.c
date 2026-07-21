#include "l3.h"
#include "types.h"

#define MUBLIS_GEMM_IMPL(ctype, function_name, driver_name)                    \
  int function_name(                                                           \
    mublis_trans_t trans_a, mublis_trans_t trans_b,                            \
    int m, int n, int k,                                                       \
    ctype alpha,                                                               \
    const ctype *a, int rs_a, int cs_a,                                        \
    const ctype *b, int rs_b, int cs_b,                                        \
    ctype beta,                                                                \
    ctype *c, int rs_c, int cs_c                                               \
  ) {                                                                          \
    if (trans_a == MUBLIS_TRANSPOSE) {                                         \
      int tmp = rs_a;                                                          \
      rs_a = cs_a;                                                             \
      cs_a = tmp;                                                              \
    }                                                                          \
                                                                               \
    if (trans_b == MUBLIS_TRANSPOSE) {                                         \
      int tmp = rs_b;                                                          \
      rs_b = cs_b;                                                             \
      cs_b = tmp;                                                              \
    }                                                                          \
                                                                               \
    const mublis_l3_product_t product = {                                      \
      .domain = {                                                              \
        .jp = MUBLIS_L3_ALL,                                                   \
        .ji = MUBLIS_L3_ALL,                                                   \
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
    return driver_name(                                                        \
      m, n, k,                                                                 \
      alpha,                                                                   \
      a, rs_a, cs_a,                                                           \
      b, rs_b, cs_b,                                                           \
      beta,                                                                    \
      c, rs_c, cs_c,                                                           \
      &product                                                                 \
    );                                                                         \
  }

MUBLIS_GEMM_IMPL(float, mublis_sgemm, mublis_l3_sdriver)
MUBLIS_GEMM_IMPL(double, mublis_dgemm, mublis_l3_ddriver)
