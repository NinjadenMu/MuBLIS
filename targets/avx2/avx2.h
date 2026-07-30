#ifndef AVX2_H
#define AVX2_H

#include <immintrin.h>

#if defined(__clang__)
#define UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define UNROLL _Pragma("GCC unroll 16")
#else
#define UNROLL
#endif

#define S_MR 8
#define S_NR 8
#define D_MR 12
#define D_NR 4

#define S_MV (S_MR / 8)
#define S_NV (S_NR / 8)
#define D_MV (D_MR / 4)
#define D_NV (D_NR / 4)

static inline __m256 mublis_sfmadd_avx2(
  __m256 acc,
  __m256 value,
  float scalar
) {
  return _mm256_fmadd_ps(value, _mm256_set1_ps(scalar), acc);
}

static inline __m256d mublis_dfmadd_avx2(
  __m256d acc,
  __m256d value,
  double scalar
) {
  return _mm256_fmadd_pd(value, _mm256_set1_pd(scalar), acc);
}

static inline __m256 mublis_sfnmadd_avx2(
  __m256 acc,
  __m256 value,
  float scalar
) {
  return _mm256_fnmadd_ps(value, _mm256_set1_ps(scalar), acc);
}

static inline __m256d mublis_dfnmadd_avx2(
  __m256d acc,
  __m256d value,
  double scalar
) {
  return _mm256_fnmadd_pd(value, _mm256_set1_pd(scalar), acc);
}

static inline __m256 mublis_smul_avx2(__m256 value, float scalar) {
  return _mm256_mul_ps(value, _mm256_set1_ps(scalar));
}

static inline __m256d mublis_dmul_avx2(__m256d value, double scalar) {
  return _mm256_mul_pd(value, _mm256_set1_pd(scalar));
}

#endif
