#include "l3.h"
#include "types.h"

#define MUBLIS_SYMM_IMPL(ctype, function_name, driver_name)                    \
  int function_name(                                                           \
    mublis_side_t side, mublis_uplo_t uplo,                                    \
    int m, int n,                                                              \
    ctype alpha,                                                               \
    const ctype *a, int rs_a, int cs_a,                                        \
    const ctype *b, int rs_b, int cs_b,                                        \
    ctype beta,                                                                \
    ctype *c, int rs_c, int cs_c                                               \
  ) {                                                                          \
    const mublis_l3_product_t product = {                                      \
      .domain = {                                                              \
        .jp = MUBLIS_L3_ALL,                                                   \
        .ji = MUBLIS_L3_ALL,                                                   \
        .pi = MUBLIS_L3_ALL                                                    \
      },                                                                       \
      .a = {                                                                   \
        .struc = side == MUBLIS_LEFT                                           \
          ? MUBLIS_STRUC_SYMMETRIC                                             \
          : MUBLIS_STRUC_GENERAL,                                              \
        .uplo = side == MUBLIS_LEFT ? uplo : MUBLIS_DENSE,                     \
        .diag = MUBLIS_PACKM_DIAG_NONUNIT                                      \
      },                                                                       \
      .b = {                                                                   \
        .struc = side == MUBLIS_RIGHT                                          \
          ? MUBLIS_STRUC_SYMMETRIC                                             \
          : MUBLIS_STRUC_GENERAL,                                              \
        .uplo = side == MUBLIS_RIGHT ? uplo : MUBLIS_DENSE,                    \
        .diag = MUBLIS_PACKM_DIAG_NONUNIT                                      \
      }                                                                        \
    };                                                                         \
                                                                               \
    if (side == MUBLIS_LEFT) {                                                 \
      return driver_name(                                                      \
        m, n, m,                                                               \
        alpha,                                                                 \
        a, rs_a, cs_a,                                                         \
        b, rs_b, cs_b,                                                         \
        beta,                                                                  \
        c, rs_c, cs_c,                                                         \
        &product                                                               \
      );                                                                       \
    }                                                                          \
                                                                               \
    return driver_name(                                                        \
      m, n, n,                                                                 \
      alpha,                                                                   \
      b, rs_b, cs_b,                                                           \
      a, rs_a, cs_a,                                                           \
      beta,                                                                    \
      c, rs_c, cs_c,                                                           \
      &product                                                                 \
    );                                                                         \
  }

MUBLIS_SYMM_IMPL(float, mublis_ssymm, mublis_l3_sdriver)
MUBLIS_SYMM_IMPL(double, mublis_dsymm, mublis_l3_ddriver)
