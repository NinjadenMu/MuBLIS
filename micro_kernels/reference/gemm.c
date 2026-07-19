#include "kernels.h"

/*
 * These dimensions must match mr and nr in every context that selects the
 * corresponding reference kernel.
 */
#define SGEMM_MR 4
#define SGEMM_NR 4
#define DGEMM_MR 4
#define DGEMM_NR 4

#define MUBLIS_GEMM_UKR_REFERENCE_IMPL(ctype, name, mr, nr)                    \
  void name(                                                                   \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *a,                                                            \
    const ctype *b,                                                            \
    ctype beta,                                                                \
    ctype *c,                                                                  \
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

MUBLIS_GEMM_UKR_REFERENCE_IMPL(float, mublis_sgemm_ukr_reference, SGEMM_MR,
                               SGEMM_NR)

MUBLIS_GEMM_UKR_REFERENCE_IMPL(double, mublis_dgemm_ukr_reference, DGEMM_MR,
                               DGEMM_NR)

#undef MUBLIS_GEMM_UKR_REFERENCE_IMPL
