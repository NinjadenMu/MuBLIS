#include <immintrin.h>

#include "avx2.h"
#include "mublis_instantiate.h"

static inline void mublis_sgemmtrsm_transpose_store_avx2(
  const __m256 acc[S_MR][S_NV],
  float *restrict c11,
  int cs_c
) {
  UNROLL
  for (int i = 0; i < S_MR; i += 8) {
    UNROLL
    for (int j = 0; j < S_NR; j += 8) {
      __m256 r0 = acc[i + 0][j / 8];
      __m256 r1 = acc[i + 1][j / 8];
      __m256 r2 = acc[i + 2][j / 8];
      __m256 r3 = acc[i + 3][j / 8];
      __m256 r4 = acc[i + 4][j / 8];
      __m256 r5 = acc[i + 5][j / 8];
      __m256 r6 = acc[i + 6][j / 8];
      __m256 r7 = acc[i + 7][j / 8];

      __m256 t0 = _mm256_unpacklo_ps(r0, r1);
      __m256 t1 = _mm256_unpackhi_ps(r0, r1);
      __m256 t2 = _mm256_unpacklo_ps(r2, r3);
      __m256 t3 = _mm256_unpackhi_ps(r2, r3);
      __m256 t4 = _mm256_unpacklo_ps(r4, r5);
      __m256 t5 = _mm256_unpackhi_ps(r4, r5);
      __m256 t6 = _mm256_unpacklo_ps(r6, r7);
      __m256 t7 = _mm256_unpackhi_ps(r6, r7);

      __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
      __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
      __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

      __m256 c0 = _mm256_permute2f128_ps(s0, s4, 0x20);
      __m256 c1 = _mm256_permute2f128_ps(s1, s5, 0x20);
      __m256 c2 = _mm256_permute2f128_ps(s2, s6, 0x20);
      __m256 c3 = _mm256_permute2f128_ps(s3, s7, 0x20);
      __m256 c4 = _mm256_permute2f128_ps(s0, s4, 0x31);
      __m256 c5 = _mm256_permute2f128_ps(s1, s5, 0x31);
      __m256 c6 = _mm256_permute2f128_ps(s2, s6, 0x31);
      __m256 c7 = _mm256_permute2f128_ps(s3, s7, 0x31);

      _mm256_storeu_ps(c11 + i + (j + 0) * cs_c, c0);
      _mm256_storeu_ps(c11 + i + (j + 1) * cs_c, c1);
      _mm256_storeu_ps(c11 + i + (j + 2) * cs_c, c2);
      _mm256_storeu_ps(c11 + i + (j + 3) * cs_c, c3);
      _mm256_storeu_ps(c11 + i + (j + 4) * cs_c, c4);
      _mm256_storeu_ps(c11 + i + (j + 5) * cs_c, c5);
      _mm256_storeu_ps(c11 + i + (j + 6) * cs_c, c6);
      _mm256_storeu_ps(c11 + i + (j + 7) * cs_c, c7);
    }
  }
}

static inline void mublis_dgemmtrsm_transpose_store_avx2(
  const __m256d acc[D_MR][D_NV],
  double *restrict c11,
  int cs_c
) {
  UNROLL
  for (int i = 0; i < D_MR; i += 4) {
    UNROLL
    for (int j = 0; j < D_NR; j += 4) {
      __m256d r0 = acc[i + 0][j / 4];
      __m256d r1 = acc[i + 1][j / 4];
      __m256d r2 = acc[i + 2][j / 4];
      __m256d r3 = acc[i + 3][j / 4];

      __m256d t0 = _mm256_unpacklo_pd(r0, r1);
      __m256d t1 = _mm256_unpackhi_pd(r0, r1);
      __m256d t2 = _mm256_unpacklo_pd(r2, r3);
      __m256d t3 = _mm256_unpackhi_pd(r2, r3);

      __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
      __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
      __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
      __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

      _mm256_storeu_pd(c11 + i + (j + 0) * cs_c, c0);
      _mm256_storeu_pd(c11 + i + (j + 1) * cs_c, c1);
      _mm256_storeu_pd(c11 + i + (j + 2) * cs_c, c2);
      _mm256_storeu_pd(c11 + i + (j + 3) * cs_c, c3);
    }
  }
}

#define MUBLIS_GEMMTRSM_AVX2_IMPL(                                             \
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

MUBLIS_GEMMTRSM_AVX2_IMPL(
  float, __m256, mublis_sgemmtrsml_ukr_avx2,
  S_MR, S_NR, S_NV, 8, 1,
  _mm256_loadu_ps, _mm256_storeu_ps,
  mublis_smul_avx2, mublis_sfnmadd_avx2,
  mublis_sgemmtrsm_transpose_store_avx2
)

MUBLIS_GEMMTRSM_AVX2_IMPL(
  float, __m256, mublis_sgemmtrsmu_ukr_avx2,
  S_MR, S_NR, S_NV, 8, 0,
  _mm256_loadu_ps, _mm256_storeu_ps,
  mublis_smul_avx2, mublis_sfnmadd_avx2,
  mublis_sgemmtrsm_transpose_store_avx2
)

MUBLIS_GEMMTRSM_AVX2_IMPL(
  double, __m256d, mublis_dgemmtrsml_ukr_avx2,
  D_MR, D_NR, D_NV, 4, 1,
  _mm256_loadu_pd, _mm256_storeu_pd,
  mublis_dmul_avx2, mublis_dfnmadd_avx2,
  mublis_dgemmtrsm_transpose_store_avx2
)

MUBLIS_GEMMTRSM_AVX2_IMPL(
  double, __m256d, mublis_dgemmtrsmu_ukr_avx2,
  D_MR, D_NR, D_NV, 4, 0,
  _mm256_loadu_pd, _mm256_storeu_pd,
  mublis_dmul_avx2, mublis_dfnmadd_avx2,
  mublis_dgemmtrsm_transpose_store_avx2
)

#undef MUBLIS_GEMMTRSM_AVX2_IMPL
