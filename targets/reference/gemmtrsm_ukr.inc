// clang-format off
#include "mublis_instantiate.h"
#include "reference.h"

#define MUBLIS_GEMMTRSM_L_REFERENCE_IMPL(ctype, name, mr, nr)                  \
  static void name(                                                            \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a10,                                                 \
    const ctype *restrict a11,                                                 \
    const ctype *restrict b01,                                                 \
    ctype *restrict b11,                                                       \
    ctype *restrict c11,                                                       \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  ) {                                                                          \
    (void)aux;                                                                 \
                                                                               \
    for (int i = 0; i < mr; i++) {                                             \
      for (int j = 0; j < nr; j++) {                                           \
        ctype value = alpha * b11[i * nr + j];                                 \
                                                                               \
        for (int p = 0; p < k; p++)                                            \
          value -= a10[p * mr + i] * b01[p * nr + j];                          \
                                                                               \
        b11[i * nr + j] = value;                                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (int i = 0; i < mr; i++) {                                             \
      for (int p = 0; p < i; p++) {                                            \
        ctype aip = a11[p * mr + i];                                           \
                                                                               \
        for (int j = 0; j < nr; j++)                                           \
          b11[i * nr + j] -= aip * b11[p * nr + j];                            \
      }                                                                        \
                                                                               \
      ctype inv_aii = a11[i * mr + i];                                         \
      for (int j = 0; j < nr; j++)                                             \
        b11[i * nr + j] *= inv_aii;                                            \
    }                                                                          \
                                                                               \
    for (int j = 0; j < nr; j++)                                               \
      for (int i = 0; i < mr; i++)                                             \
        c11[i * rs_c + j * cs_c] = b11[i * nr + j];                            \
  }

#define MUBLIS_GEMMTRSM_U_REFERENCE_IMPL(ctype, name, mr, nr)                  \
  static void name(                                                            \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a12,                                                 \
    const ctype *restrict a11,                                                 \
    const ctype *restrict b21,                                                 \
    ctype *restrict b11,                                                       \
    ctype *restrict c11,                                                       \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  ) {                                                                          \
    (void)aux;                                                                 \
                                                                               \
    for (int i = 0; i < mr; i++) {                                             \
      for (int j = 0; j < nr; j++) {                                           \
        ctype value = alpha * b11[i * nr + j];                                 \
                                                                               \
        for (int p = 0; p < k; p++)                                            \
          value -= a12[p * mr + i] * b21[p * nr + j];                          \
                                                                               \
        b11[i * nr + j] = value;                                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (int i = mr - 1; i >= 0; i--) {                                        \
      for (int p = i + 1; p < mr; p++) {                                       \
        ctype aip = a11[p * mr + i];                                           \
                                                                               \
        for (int j = 0; j < nr; j++)                                           \
          b11[i * nr + j] -= aip * b11[p * nr + j];                            \
      }                                                                        \
                                                                               \
      ctype inv_aii = a11[i * mr   + i];                                       \
      for (int j = 0; j < nr; j++)                                             \
        b11[i * nr + j] *= inv_aii;                                            \
    }                                                                          \
                                                                               \
    for (int j = 0; j < nr; j++)                                               \
      for (int i = 0; i < mr; i++)                                             \
        c11[i * rs_c + j * cs_c] = b11[i * nr + j];                            \
  }

MUBLIS_GEMMTRSM_L_REFERENCE_IMPL(float, mublis_sgemmtrsml_ukr_reference, S_MR,
                                 S_NR)
MUBLIS_GEMMTRSM_U_REFERENCE_IMPL(float, mublis_sgemmtrsmu_ukr_reference, S_MR,
                                 S_NR)
MUBLIS_GEMMTRSM_L_REFERENCE_IMPL(double, mublis_dgemmtrsml_ukr_reference, D_MR,
                                 D_NR)
MUBLIS_GEMMTRSM_U_REFERENCE_IMPL(double, mublis_dgemmtrsmu_ukr_reference, D_MR,
                                 D_NR)

/* clang-format on */
