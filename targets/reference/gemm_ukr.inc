#include "mublis_instantiate.h"
#include "reference.h"

#define MUBLIS_GEMM_UKR_REFERENCE_IMPL(ctype, name, mr, nr)                    \
  static void name(                                                            \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a,                                                   \
    const ctype *restrict b,                                                   \
    ctype beta,                                                                \
    ctype *restrict c,                                                         \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  ) {                                                                          \
    (void)aux;                                                                 \
                                                                               \
    if (k <= 0 || alpha == 0) {                                                \
      for (int j = 0; j < (nr); j++) {                                         \
        for (int i = 0; i < (mr); i++) {                                       \
          ctype *cij = c + i * rs_c + j * cs_c;                                \
                                                                               \
          if (beta == 0)                                                       \
            *cij = 0;                                                          \
          else                                                                 \
            *cij *= beta;                                                      \
        }                                                                      \
      }                                                                        \
                                                                               \
      return;                                                                  \
    }                                                                          \
    for (int j = 0; j < nr; j++) {                                             \
      for (int i = 0; i < mr; i++) {                                           \
        ctype r = 0;                                                           \
                                                                               \
        for (int p = 0; p < k; p++)                                            \
          r += a[p * mr + i] * b[p * nr + j];                                  \
                                                                               \
        ctype *cij = c + i * rs_c + j * cs_c;                                  \
        ctype product = alpha * r;                                             \
                                                                               \
        if (beta == 0)                                                         \
          *cij = product;                                                      \
        else if (beta == 1)                                                    \
          *cij += product;                                                     \
        else                                                                   \
          *cij = product + beta * *cij;                                        \
      }                                                                        \
    }                                                                          \
  }

MUBLIS_GEMM_UKR_REFERENCE_IMPL(float, mublis_sgemm_ukr_reference, S_MR,
                               S_NR)
MUBLIS_GEMM_UKR_REFERENCE_IMPL(double, mublis_dgemm_ukr_reference, D_MR,
                               D_NR)

#undef MUBLIS_GEMM_UKR_REFERENCE_IMPL
