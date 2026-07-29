#include <arm_neon.h>

#include "aarch64.h"
#include "mublis_instantiate.h"

static inline void mublis_sgemmtrsm_transpose_store_firestorm(
  const float32x4_t acc[S_MR][S_NV],
  float *restrict c11,
  int cs_c
) {
  UNROLL
  for (int i = 0; i < S_MR; i += 4) {
    UNROLL
    for (int j = 0; j < S_NR; j += 4) {
      float32x4_t r0 = acc[i + 0][j / 4];
      float32x4_t r1 = acc[i + 1][j / 4];
      float32x4_t r2 = acc[i + 2][j / 4];
      float32x4_t r3 = acc[i + 3][j / 4];

      float32x4x2_t t01 = vtrnq_f32(r0, r1);
      float32x4x2_t t23 = vtrnq_f32(r2, r3);

      float32x4_t c0 = vcombine_f32(
        vget_low_f32(t01.val[0]),
        vget_low_f32(t23.val[0])
      );
      float32x4_t c1 = vcombine_f32(
        vget_low_f32(t01.val[1]),
        vget_low_f32(t23.val[1])
      );
      float32x4_t c2 = vcombine_f32(
        vget_high_f32(t01.val[0]),
        vget_high_f32(t23.val[0])
      );
      float32x4_t c3 = vcombine_f32(
        vget_high_f32(t01.val[1]),
        vget_high_f32(t23.val[1])
      );

      vst1q_f32(c11 + i + (j + 0) * cs_c, c0);
      vst1q_f32(c11 + i + (j + 1) * cs_c, c1);
      vst1q_f32(c11 + i + (j + 2) * cs_c, c2);
      vst1q_f32(c11 + i + (j + 3) * cs_c, c3);
    }
  }
}

static inline void mublis_dgemmtrsm_transpose_store_firestorm(
  const float64x2_t acc[D_MR][D_NV],
  double *restrict c11,
  int cs_c
) {
  UNROLL
  for (int i = 0; i < D_MR; i += 2) {
    UNROLL
    for (int j = 0; j < D_NR; j += 2) {
      float64x2_t r0 = acc[i + 0][j / 2];
      float64x2_t r1 = acc[i + 1][j / 2];
      float64x2_t c0 = vtrn1q_f64(r0, r1);
      float64x2_t c1 = vtrn2q_f64(r0, r1);

      vst1q_f64(c11 + i + (j + 0) * cs_c, c0);
      vst1q_f64(c11 + i + (j + 1) * cs_c, c1);
    }
  }
}

#define MUBLIS_GEMMTRSM_FIRESTORM_IMPL(                                        \
  ctype, vtype, name, mr, nr, nv, width, is_lower,                             \
  vld, vst, vmul, vfms, store_column_major                                     \
)                                                                              \
  static void name(                                                            \
    int k,                                                                     \
    ctype alpha,                                                               \
    const ctype *restrict a1x,                                                 \
    const ctype *restrict a11,                                                 \
    const ctype *restrict bx1,                                                 \
    ctype *restrict b11,                                                       \
    ctype *restrict c11,                                                       \
    int rs_c, int cs_c,                                                        \
    const mublis_auxinfo_t *aux                                                \
  ) {                                                                          \
    vtype acc[mr][nv];                                                         \
                                                                               \
    if (aux && aux->c_next)                                                    \
      __builtin_prefetch(aux->c_next, 1, 3);                                   \
                                                                               \
    if (alpha == 1) {                                                          \
      UNROLL                                                                   \
      for (int i = 0; i < mr; i++) {                                           \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++)                                        \
          acc[i][vi] = vld(b11 + i * nr + vi * width);                         \
      }                                                                        \
    } else {                                                                   \
      UNROLL                                                                   \
      for (int i = 0; i < mr; i++) {                                           \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++) {                                      \
          vtype value = vld(b11 + i * nr + vi * width);                        \
          acc[i][vi] = vmul(value, alpha);                                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (int p = 0; p < k; p++) {                                              \
      const ctype *bx_row = bx1 + p * nr;                                      \
      vtype vectorized_bx_row[nv];                                             \
                                                                               \
      UNROLL                                                                   \
      for (int vi = 0; vi < nv; vi++)                                          \
        vectorized_bx_row[vi] = vld(bx_row + vi * width);                      \
                                                                               \
      UNROLL                                                                   \
      for (int i = 0; i < mr; i++) {                                           \
        ctype a_value = a1x[p * mr + i];                                       \
                                                                               \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++)                                        \
          acc[i][vi] = vfms(                                                   \
            acc[i][vi], vectorized_bx_row[vi], a_value                         \
          );                                                                   \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (is_lower) {                                                            \
      UNROLL                                                                   \
      for (int i = 0; i < mr; i++) {                                           \
        UNROLL                                                                 \
        for (int p = 0; p < i; p++) {                                          \
          ctype a_value = a11[p * mr + i];                                     \
                                                                               \
          UNROLL                                                               \
          for (int vi = 0; vi < nv; vi++)                                      \
            acc[i][vi] = vfms(acc[i][vi], acc[p][vi], a_value);                \
        }                                                                      \
                                                                               \
        ctype inv_aii = a11[i * mr + i];                                       \
                                                                               \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++)                                        \
          acc[i][vi] = vmul(acc[i][vi], inv_aii);                              \
      }                                                                        \
    } else {                                                                   \
      UNROLL                                                                   \
      for (int i = mr - 1; i >= 0; i--) {                                      \
        UNROLL                                                                 \
        for (int p = i + 1; p < mr; p++) {                                     \
          ctype a_value = a11[p * mr + i];                                     \
                                                                               \
          UNROLL                                                               \
          for (int vi = 0; vi < nv; vi++)                                      \
            acc[i][vi] = vfms(acc[i][vi], acc[p][vi], a_value);                \
        }                                                                      \
                                                                               \
        ctype inv_aii = a11[i * mr + i];                                       \
                                                                               \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++)                                        \
          acc[i][vi] = vmul(acc[i][vi], inv_aii);                              \
      }                                                                        \
    }                                                                          \
                                                                               \
    UNROLL                                                                     \
    for (int i = 0; i < mr; i++) {                                             \
      UNROLL                                                                   \
      for (int vi = 0; vi < nv; vi++)                                          \
        vst(b11 + i * nr + vi * width, acc[i][vi]);                            \
    }                                                                          \
                                                                               \
    if (rs_c == 1) {                                                           \
      store_column_major(acc, c11, cs_c);                                      \
    } else if (cs_c == 1) {                                                    \
      UNROLL                                                                   \
      for (int i = 0; i < mr; i++) {                                           \
        UNROLL                                                                 \
        for (int vi = 0; vi < nv; vi++)                                        \
          vst(c11 + i * rs_c + vi * width, acc[i][vi]);                        \
      }                                                                        \
    } else {                                                                   \
      UNROLL                                                                   \
      for (int j = 0; j < nr; j++) {                                           \
        UNROLL                                                                 \
        for (int i = 0; i < mr; i++)                                           \
          c11[i * rs_c + j * cs_c] = b11[i * nr + j];                          \
      }                                                                        \
    }                                                                          \
  }

MUBLIS_GEMMTRSM_FIRESTORM_IMPL(
  float, float32x4_t, mublis_sgemmtrsml_ukr_firestorm,
  S_MR, S_NR, S_NV, 4, 1,
  vld1q_f32, vst1q_f32, vmulq_n_f32, vfmsq_n_f32,
  mublis_sgemmtrsm_transpose_store_firestorm
)

MUBLIS_GEMMTRSM_FIRESTORM_IMPL(
  float, float32x4_t, mublis_sgemmtrsmu_ukr_firestorm,
  S_MR, S_NR, S_NV, 4, 0,
  vld1q_f32, vst1q_f32, vmulq_n_f32, vfmsq_n_f32,
  mublis_sgemmtrsm_transpose_store_firestorm
)

MUBLIS_GEMMTRSM_FIRESTORM_IMPL(
  double, float64x2_t, mublis_dgemmtrsml_ukr_firestorm,
  D_MR, D_NR, D_NV, 2, 1,
  vld1q_f64, vst1q_f64, vmulq_n_f64, vfmsq_n_f64,
  mublis_dgemmtrsm_transpose_store_firestorm
)

MUBLIS_GEMMTRSM_FIRESTORM_IMPL(
  double, float64x2_t, mublis_dgemmtrsmu_ukr_firestorm,
  D_MR, D_NR, D_NV, 2, 0,
  vld1q_f64, vst1q_f64, vmulq_n_f64, vfmsq_n_f64,
  mublis_dgemmtrsm_transpose_store_firestorm
)

#undef MUBLIS_GEMMTRSM_FIRESTORM_IMPL
