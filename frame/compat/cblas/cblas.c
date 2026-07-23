#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "cblas.h"
#include "mublis.h"

#if defined(__GNUC__) || defined(__clang__)
#define MUBLIS_WEAK __attribute__((weak))
#else
#define MUBLIS_WEAK
#endif

void MUBLIS_WEAK cblas_xerbla(
  int p,
  const char *rout,
  const char *form,
  ...
) {
  va_list args;

  if (p > 0)
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", p, rout);
  else
    fprintf(stderr, "Routine %s failed\n", rout);

  va_start(args, form);
  vfprintf(stderr, form, args);
  va_end(args);

  exit(EXIT_FAILURE);
}

static int cblas_max1(int value) {
  return value > 1 ? value : 1;
}

static int cblas_valid_order(enum CBLAS_ORDER value) {
  return value == CblasRowMajor || value == CblasColMajor;
}

static int cblas_valid_trans(enum CBLAS_TRANSPOSE value) {
  return value == CblasNoTrans || value == CblasTrans;
}

static int cblas_valid_uplo(enum CBLAS_UPLO value) {
  return value == CblasUpper || value == CblasLower;
}

static int cblas_valid_diag(enum CBLAS_DIAG value) {
  return value == CblasNonUnit || value == CblasUnit;
}

static int cblas_valid_side(enum CBLAS_SIDE value) {
  return value == CblasLeft || value == CblasRight;
}

static mublis_trans_t cblas_trans(enum CBLAS_TRANSPOSE value) {
  return value == CblasNoTrans
    ? MUBLIS_NO_TRANSPOSE
    : MUBLIS_TRANSPOSE;
}

static mublis_trans_t cblas_flip_trans(enum CBLAS_TRANSPOSE value) {
  return value == CblasNoTrans
    ? MUBLIS_TRANSPOSE
    : MUBLIS_NO_TRANSPOSE;
}

static mublis_uplo_t cblas_uplo(enum CBLAS_UPLO value) {
  return value == CblasUpper ? MUBLIS_UPPER : MUBLIS_LOWER;
}

static mublis_uplo_t cblas_flip_uplo(enum CBLAS_UPLO value) {
  return value == CblasUpper ? MUBLIS_LOWER : MUBLIS_UPPER;
}

static mublis_diag_t cblas_diag(enum CBLAS_DIAG value) {
  return value == CblasUnit
    ? MUBLIS_UNIT_DIAG
    : MUBLIS_NON_UNIT_DIAG;
}

static mublis_side_t cblas_side(enum CBLAS_SIDE value) {
  return value == CblasLeft ? MUBLIS_LEFT : MUBLIS_RIGHT;
}

static mublis_side_t cblas_flip_side(enum CBLAS_SIDE value) {
  return value == CblasLeft ? MUBLIS_RIGHT : MUBLIS_LEFT;
}

#define CBLAS_CHECK(condition, parameter, routine, ...)                       \
  do {                                                                        \
    if (!(condition)) {                                                       \
      cblas_xerbla((parameter), (routine), __VA_ARGS__);                      \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define CBLAS_RETURN_STATUS(call, routine)                                    \
  do {                                                                        \
    int cblas_status = (call);                                                \
    if (cblas_status != 0) {                                                  \
      cblas_xerbla(                                                           \
        0,                                                                    \
        (routine),                                                            \
        "MuBLIS backend returned status %d\n",                                \
        cblas_status                                                          \
      );                                                                      \
    }                                                                         \
    return;                                                                   \
  } while (0)

#define CBLAS_GEMM_IMPL(prefix, ctype)                                        \
  void cblas_##prefix##gemm(                                                  \
    const enum CBLAS_ORDER Order,                                             \
    const enum CBLAS_TRANSPOSE TransA,                                        \
    const enum CBLAS_TRANSPOSE TransB,                                        \
    const int M, const int N, const int K,                                    \
    const ctype alpha,                                                        \
    const ctype *A, const int lda,                                            \
    const ctype *B, const int ldb,                                            \
    const ctype beta,                                                         \
    ctype *C, const int ldc                                                   \
  ) {                                                                         \
    const char *routine = "cblas_" #prefix "gemm";                            \
    int min_lda;                                                              \
    int min_ldb;                                                              \
                                                                               \
    CBLAS_CHECK(                                                              \
      cblas_valid_order(Order), 1, routine,                                   \
      "Illegal Order setting, %d\n", Order                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_trans(TransA), 2, routine,                                  \
      "Illegal TransA setting, %d\n", TransA                                  \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_trans(TransB), 3, routine,                                  \
      "Illegal TransB setting, %d\n", TransB                                  \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      M >= 0, 4, routine, "M must be nonnegative, %d\n", M                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      N >= 0, 5, routine, "N must be nonnegative, %d\n", N                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      K >= 0, 6, routine, "K must be nonnegative, %d\n", K                    \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      min_lda = cblas_max1(TransA == CblasNoTrans ? M : K);                   \
      min_ldb = cblas_max1(TransB == CblasNoTrans ? K : N);                   \
    } else {                                                                  \
      min_lda = cblas_max1(TransA == CblasNoTrans ? K : M);                   \
      min_ldb = cblas_max1(TransB == CblasNoTrans ? N : K);                   \
    }                                                                         \
                                                                               \
    CBLAS_CHECK(                                                              \
      lda >= min_lda, 9, routine,                                             \
      "lda must be at least %d, %d\n", min_lda, lda                           \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldb >= min_ldb, 11, routine,                                            \
      "ldb must be at least %d, %d\n", min_ldb, ldb                           \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldc >= cblas_max1(Order == CblasColMajor ? M : N),                      \
      14, routine, "ldc is too small, %d\n", ldc                              \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      CBLAS_RETURN_STATUS(                                                    \
        mublis_##prefix##gemm(                                                \
          cblas_trans(TransA), cblas_trans(TransB),                           \
          M, N, K,                                                            \
          alpha,                                                              \
          A, 1, lda,                                                          \
          B, 1, ldb,                                                          \
          beta,                                                               \
          C, 1, ldc                                                           \
        ),                                                                    \
        routine                                                               \
      );                                                                      \
    }                                                                         \
                                                                               \
    /* C^T = op(B)^T op(A)^T. */                                              \
    CBLAS_RETURN_STATUS(                                                      \
      mublis_##prefix##gemm(                                                  \
        cblas_trans(TransB), cblas_trans(TransA),                             \
        N, M, K,                                                              \
        alpha,                                                                \
        B, 1, ldb,                                                            \
        A, 1, lda,                                                            \
        beta,                                                                 \
        C, 1, ldc                                                             \
      ),                                                                      \
      routine                                                                 \
    );                                                                        \
  }

#define CBLAS_SYMM_IMPL(prefix, ctype)                                        \
  void cblas_##prefix##symm(                                                  \
    const enum CBLAS_ORDER Order,                                             \
    const enum CBLAS_SIDE Side,                                               \
    const enum CBLAS_UPLO Uplo,                                               \
    const int M, const int N,                                                 \
    const ctype alpha,                                                        \
    const ctype *A, const int lda,                                            \
    const ctype *B, const int ldb,                                            \
    const ctype beta,                                                         \
    ctype *C, const int ldc                                                   \
  ) {                                                                         \
    const char *routine = "cblas_" #prefix "symm";                            \
    int min_ldbc;                                                             \
                                                                               \
    CBLAS_CHECK(                                                              \
      cblas_valid_order(Order), 1, routine,                                   \
      "Illegal Order setting, %d\n", Order                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_side(Side), 2, routine,                                     \
      "Illegal Side setting, %d\n", Side                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_uplo(Uplo), 3, routine,                                     \
      "Illegal Uplo setting, %d\n", Uplo                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      M >= 0, 4, routine, "M must be nonnegative, %d\n", M                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      N >= 0, 5, routine, "N must be nonnegative, %d\n", N                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      lda >= cblas_max1(Side == CblasLeft ? M : N),                           \
      8, routine, "lda is too small, %d\n", lda                               \
    );                                                                        \
                                                                               \
    min_ldbc = cblas_max1(Order == CblasColMajor ? M : N);                    \
    CBLAS_CHECK(                                                              \
      ldb >= min_ldbc, 10, routine,                                           \
      "ldb must be at least %d, %d\n", min_ldbc, ldb                          \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldc >= min_ldbc, 13, routine,                                           \
      "ldc must be at least %d, %d\n", min_ldbc, ldc                          \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      CBLAS_RETURN_STATUS(                                                    \
        mublis_##prefix##symm(                                                \
          cblas_side(Side), cblas_uplo(Uplo),                                 \
          M, N,                                                               \
          alpha,                                                              \
          A, 1, lda,                                                          \
          B, 1, ldb,                                                          \
          beta,                                                               \
          C, 1, ldc                                                           \
        ),                                                                    \
        routine                                                               \
      );                                                                      \
    }                                                                         \
                                                                               \
    CBLAS_RETURN_STATUS(                                                      \
      mublis_##prefix##symm(                                                  \
        cblas_flip_side(Side), cblas_flip_uplo(Uplo),                         \
        N, M,                                                                 \
        alpha,                                                                \
        A, 1, lda,                                                            \
        B, 1, ldb,                                                            \
        beta,                                                                 \
        C, 1, ldc                                                             \
      ),                                                                      \
      routine                                                                 \
    );                                                                        \
  }

#define CBLAS_SYRK_IMPL(prefix, ctype)                                        \
  void cblas_##prefix##syrk(                                                  \
    const enum CBLAS_ORDER Order,                                             \
    const enum CBLAS_UPLO Uplo,                                               \
    const enum CBLAS_TRANSPOSE Trans,                                         \
    const int N, const int K,                                                 \
    const ctype alpha,                                                        \
    const ctype *A, const int lda,                                            \
    const ctype beta,                                                         \
    ctype *C, const int ldc                                                   \
  ) {                                                                         \
    const char *routine = "cblas_" #prefix "syrk";                            \
    int min_lda;                                                              \
                                                                               \
    CBLAS_CHECK(                                                              \
      cblas_valid_order(Order), 1, routine,                                   \
      "Illegal Order setting, %d\n", Order                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_uplo(Uplo), 2, routine,                                     \
      "Illegal Uplo setting, %d\n", Uplo                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_trans(Trans), 3, routine,                                   \
      "Illegal Trans setting, %d\n", Trans                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      N >= 0, 4, routine, "N must be nonnegative, %d\n", N                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      K >= 0, 5, routine, "K must be nonnegative, %d\n", K                    \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor)                                               \
      min_lda = cblas_max1(Trans == CblasNoTrans ? N : K);                    \
    else                                                                      \
      min_lda = cblas_max1(Trans == CblasNoTrans ? K : N);                    \
                                                                               \
    CBLAS_CHECK(                                                              \
      lda >= min_lda, 8, routine,                                             \
      "lda must be at least %d, %d\n", min_lda, lda                           \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldc >= cblas_max1(N), 11, routine,                                      \
      "ldc is too small, %d\n", ldc                                           \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      CBLAS_RETURN_STATUS(                                                    \
        mublis_##prefix##syrk(                                                \
          cblas_uplo(Uplo), cblas_trans(Trans),                               \
          N, K,                                                               \
          alpha,                                                              \
          A, 1, lda,                                                          \
          beta,                                                               \
          C, 1, ldc                                                           \
        ),                                                                    \
        routine                                                               \
      );                                                                      \
    }                                                                         \
                                                                               \
    CBLAS_RETURN_STATUS(                                                      \
      mublis_##prefix##syrk(                                                  \
        cblas_flip_uplo(Uplo), cblas_flip_trans(Trans),                       \
        N, K,                                                                 \
        alpha,                                                                \
        A, 1, lda,                                                            \
        beta,                                                                 \
        C, 1, ldc                                                             \
      ),                                                                      \
      routine                                                                 \
    );                                                                        \
  }

#define CBLAS_SYR2K_IMPL(prefix, ctype)                                       \
  void cblas_##prefix##syr2k(                                                 \
    const enum CBLAS_ORDER Order,                                             \
    const enum CBLAS_UPLO Uplo,                                               \
    const enum CBLAS_TRANSPOSE Trans,                                         \
    const int N, const int K,                                                 \
    const ctype alpha,                                                        \
    const ctype *A, const int lda,                                            \
    const ctype *B, const int ldb,                                            \
    const ctype beta,                                                         \
    ctype *C, const int ldc                                                   \
  ) {                                                                         \
    const char *routine = "cblas_" #prefix "syr2k";                           \
    int min_ldab;                                                             \
                                                                               \
    CBLAS_CHECK(                                                              \
      cblas_valid_order(Order), 1, routine,                                   \
      "Illegal Order setting, %d\n", Order                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_uplo(Uplo), 2, routine,                                     \
      "Illegal Uplo setting, %d\n", Uplo                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_trans(Trans), 3, routine,                                   \
      "Illegal Trans setting, %d\n", Trans                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      N >= 0, 4, routine, "N must be nonnegative, %d\n", N                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      K >= 0, 5, routine, "K must be nonnegative, %d\n", K                    \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor)                                               \
      min_ldab = cblas_max1(Trans == CblasNoTrans ? N : K);                   \
    else                                                                      \
      min_ldab = cblas_max1(Trans == CblasNoTrans ? K : N);                   \
                                                                               \
    CBLAS_CHECK(                                                              \
      lda >= min_ldab, 8, routine,                                            \
      "lda must be at least %d, %d\n", min_ldab, lda                          \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldb >= min_ldab, 10, routine,                                           \
      "ldb must be at least %d, %d\n", min_ldab, ldb                          \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      ldc >= cblas_max1(N), 13, routine,                                      \
      "ldc is too small, %d\n", ldc                                           \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      CBLAS_RETURN_STATUS(                                                    \
        mublis_##prefix##syr2k(                                               \
          cblas_uplo(Uplo), cblas_trans(Trans),                               \
          N, K,                                                               \
          alpha,                                                              \
          A, 1, lda,                                                          \
          B, 1, ldb,                                                          \
          beta,                                                               \
          C, 1, ldc                                                           \
        ),                                                                    \
        routine                                                               \
      );                                                                      \
    }                                                                         \
                                                                               \
    CBLAS_RETURN_STATUS(                                                      \
      mublis_##prefix##syr2k(                                                 \
        cblas_flip_uplo(Uplo), cblas_flip_trans(Trans),                       \
        N, K,                                                                 \
        alpha,                                                                \
        A, 1, lda,                                                            \
        B, 1, ldb,                                                            \
        beta,                                                                 \
        C, 1, ldc                                                             \
      ),                                                                      \
      routine                                                                 \
    );                                                                        \
  }

#define CBLAS_TR_IMPL(prefix, suffix, ctype)                                  \
  void cblas_##prefix##suffix(                                                \
    const enum CBLAS_ORDER Order,                                             \
    const enum CBLAS_SIDE Side,                                               \
    const enum CBLAS_UPLO Uplo,                                               \
    const enum CBLAS_TRANSPOSE TransA,                                        \
    const enum CBLAS_DIAG Diag,                                               \
    const int M, const int N,                                                 \
    const ctype alpha,                                                        \
    const ctype *A, const int lda,                                            \
    ctype *B, const int ldb                                                   \
  ) {                                                                         \
    const char *routine = "cblas_" #prefix #suffix;                           \
    int min_ldb;                                                              \
                                                                               \
    CBLAS_CHECK(                                                              \
      cblas_valid_order(Order), 1, routine,                                   \
      "Illegal Order setting, %d\n", Order                                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_side(Side), 2, routine,                                     \
      "Illegal Side setting, %d\n", Side                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_uplo(Uplo), 3, routine,                                     \
      "Illegal Uplo setting, %d\n", Uplo                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_trans(TransA), 4, routine,                                  \
      "Illegal TransA setting, %d\n", TransA                                  \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      cblas_valid_diag(Diag), 5, routine,                                     \
      "Illegal Diag setting, %d\n", Diag                                      \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      M >= 0, 6, routine, "M must be nonnegative, %d\n", M                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      N >= 0, 7, routine, "N must be nonnegative, %d\n", N                    \
    );                                                                        \
    CBLAS_CHECK(                                                              \
      lda >= cblas_max1(Side == CblasLeft ? M : N),                           \
      10, routine, "lda is too small, %d\n", lda                              \
    );                                                                        \
                                                                               \
    min_ldb = cblas_max1(Order == CblasColMajor ? M : N);                     \
    CBLAS_CHECK(                                                              \
      ldb >= min_ldb, 12, routine,                                            \
      "ldb must be at least %d, %d\n", min_ldb, ldb                           \
    );                                                                        \
                                                                               \
    if (Order == CblasColMajor) {                                             \
      CBLAS_RETURN_STATUS(                                                    \
        mublis_##prefix##suffix(                                              \
          cblas_side(Side),                                                   \
          cblas_uplo(Uplo),                                                   \
          cblas_trans(TransA),                                                \
          cblas_diag(Diag),                                                   \
          M, N,                                                               \
          alpha,                                                              \
          A, 1, lda,                                                          \
          B, 1, ldb                                                           \
        ),                                                                    \
        routine                                                               \
      );                                                                      \
    }                                                                         \
                                                                               \
    CBLAS_RETURN_STATUS(                                                      \
      mublis_##prefix##suffix(                                                \
        cblas_flip_side(Side),                                                \
        cblas_flip_uplo(Uplo),                                                \
        cblas_trans(TransA),                                                  \
        cblas_diag(Diag),                                                     \
        N, M,                                                                 \
        alpha,                                                                \
        A, 1, lda,                                                            \
        B, 1, ldb                                                             \
      ),                                                                      \
      routine                                                                 \
    );                                                                        \
  }

CBLAS_GEMM_IMPL(s, float)
CBLAS_GEMM_IMPL(d, double)

CBLAS_SYMM_IMPL(s, float)
CBLAS_SYMM_IMPL(d, double)

CBLAS_SYRK_IMPL(s, float)
CBLAS_SYRK_IMPL(d, double)

CBLAS_SYR2K_IMPL(s, float)
CBLAS_SYR2K_IMPL(d, double)

CBLAS_TR_IMPL(s, trmm, float)
CBLAS_TR_IMPL(d, trmm, double)

CBLAS_TR_IMPL(s, trsm, float)
CBLAS_TR_IMPL(d, trsm, double)
