#include <immintrin.h>

#include "avx2.h"
#include "mublis_instantiate.h"

#define MUBLIS_GEMM_UKR_AVX2_IMPL(                                             \
  ctype, vtype, name, mr, nr, mv, width,                                       \
  vdup, vld, vst, vfma, vmul, vadd                                             \
)                                                                              \
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
    const int has_product = k > 0 && alpha != 0;                               \
    vtype acc[nr][mv];                                                         \
                                                                               \
    UNROLL                                                                     \
    for (int j = 0; j < nr; j++) {                                             \
      UNROLL                                                                   \
      for (int vi = 0; vi < mv; vi++)                                          \
        acc[j][vi] = vdup(0);                                                  \
    }                                                                          \
                                                                               \
    if (aux && aux->c_next)                                                    \
      __builtin_prefetch(aux->c_next, 1, 3);                                   \
                                                                               \
    if (has_product) {                                                         \
      for (int p = 0; p < k; p++) {                                            \
        const ctype *a_col = a + p * mr;                                       \
        const ctype *b_row = b + p * nr;                                       \
        vtype vectorized_a_col[mv];                                            \
                                                                               \
        UNROLL                                                                 \
        for (int vi = 0; vi < mv; vi++)                                        \
          vectorized_a_col[vi] = vld(a_col + vi * width);                      \
                                                                               \
        UNROLL                                                                 \
        for (int j = 0; j < nr; j++) {                                         \
          ctype b_value = b_row[j];                                            \
                                                                               \
          UNROLL                                                               \
          for (int vi = 0; vi < mv; vi++)                                      \
            acc[j][vi] = vfma(                                                 \
              acc[j][vi], vectorized_a_col[vi], b_value                        \
            );                                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (rs_c == 1) {                                                           \
      UNROLL                                                                   \
      for (int j = 0; j < nr; j++) {                                           \
        UNROLL                                                                 \
        for (int vi = 0; vi < mv; vi++) {                                      \
          ctype *cij = c + j * cs_c + vi * width;                              \
          vtype value;                                                         \
                                                                               \
          if (!has_product)                                                    \
            value = vdup(0);                                                   \
          else if (alpha == 1)                                                 \
            value = acc[j][vi];                                                \
          else                                                                 \
            value = vmul(acc[j][vi], alpha);                                   \
                                                                               \
          if (beta != 0) {                                                     \
            vtype c_value = vld(cij);                                          \
                                                                               \
            if (beta == 1)                                                     \
              value = vadd(value, c_value);                                    \
            else                                                               \
              value = vfma(value, c_value, beta);                              \
          }                                                                    \
                                                                               \
          vst(cij, value);                                                     \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      ctype product[mr * nr];                                                  \
                                                                               \
      UNROLL                                                                   \
      for (int j = 0; j < nr; j++) {                                           \
        UNROLL                                                                 \
        for (int vi = 0; vi < mv; vi++) {                                      \
          vtype value;                                                         \
                                                                               \
          if (!has_product)                                                    \
            value = vdup(0);                                                   \
          else if (alpha == 1)                                                 \
            value = acc[j][vi];                                                \
          else                                                                 \
            value = vmul(acc[j][vi], alpha);                                   \
                                                                               \
          vst(product + j * mr + vi * width, value);                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      UNROLL                                                                   \
      for (int j = 0; j < nr; j++) {                                           \
        UNROLL                                                                 \
        for (int i = 0; i < mr; i++) {                                         \
          ctype *cij = c + i * rs_c + j * cs_c;                                \
          ctype value = product[i + j * mr];                                   \
                                                                               \
          if (beta == 0)                                                       \
            *cij = value;                                                      \
          else if (beta == 1)                                                  \
            *cij += value;                                                     \
          else                                                                 \
            *cij = value + beta * *cij;                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

MUBLIS_GEMM_UKR_AVX2_IMPL(
  float, __m256, mublis_sgemm_ukr_avx2,
  S_MR, S_NR, S_MV, 8,
  _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps,
  mublis_sfmadd_avx2, mublis_smul_avx2, _mm256_add_ps
)

MUBLIS_GEMM_UKR_AVX2_IMPL(
  double, __m256d, mublis_dgemm_ukr_avx2,
  D_MR, D_NR, D_MV, 4,
  _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd,
  mublis_dfmadd_avx2, mublis_dmul_avx2, _mm256_add_pd
)

#undef MUBLIS_GEMM_UKR_AVX2_IMPL
