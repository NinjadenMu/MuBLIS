#ifndef MUBLIS_L3_H
#define MUBLIS_L3_H

#include "l1m.h"
#include "types.h"

/*
 * Triangular relationships used for specifying loop domain.
 */
typedef enum {
  MUBLIS_L3_ALL = 0,

  // left index <= right index
  MUBLIS_L3_LOWER = 1,

  // left index >= right index
  MUBLIS_L3_UPPER = 2,
} mublis_l3_relation_t;

/*
 * Specifies loop domain of driver, allowing for arbitrary combinations of 
 * input and output shapes of the logical matrix.
 */
typedef struct {
  mublis_l3_relation_t jp;
  mublis_l3_relation_t ji;
  mublis_l3_relation_t pi;
} mublis_l3_domain_t;

/*
 * Used for packing.  `struc` determines the logical view of the matrix, 
 * which informs the output of the packing function.  `uplo` determines 
 * the shape of the stored data, which tells the packing function where 
 * to look for data.
 * 
 * For example, the packing function will look for data in the upper triangle 
 * when asked to pack a panel in the lower triangle of a symmetric matrix if 
 * `uplo` tells it that only the upper triangle is stored.
 */
typedef struct {
  mublis_packm_struc_t struc;
  mublis_uplo_t uplo;
  mublis_packm_diag_t diag;
} mublis_l3_operand_t;


// Stores all information the generic driver needs to define an operation
typedef struct {
  mublis_l3_domain_t domain;

  mublis_l3_operand_t a;
  mublis_l3_operand_t b;
} mublis_l3_product_t;

/**
 * @brief Uniform interface for single precision out-of-place operations
 * 
 * To handle different output shapes, the driver relies on the `ji` 
 * relation in the provided domain to determine which (i, j) coordinates 
 * in C should be written to.  To handle different logical A and B 
 * shapes, the driver relies on the `jp` and `pi` relations 
 * respectively, which determine which (j, p) and (p, i) coordinates to 
 * read.
 * 
 * Note that the symmetric structure is NOT described by a triangular 
 * relation.  While data may actually only be stored in one triangle, 
 * it represents a matrix whose other triangle still has non-zero values.
 * Instead, to handle symmetry in input matrices, the driver relies on the 
 * packing function to always look in the stored triangle.
 * 
 * The driver does not allow for transpose flags.  Instead, the row and column 
 * strides should be swapped, which has the effect of transposing the logical 
 * view of the matrix.
 * 
 * Cache panel sizes, register block sizes, and micro-kernels are determined 
 * by a call to `mublis_get_safe_context`.  Buffer allocations are done using 
 * the pool.
 */
int mublis_l3_sdriver(
  int m, int n, int k,
  float alpha,
  const float *restrict a, int rs_a, int cs_a,
  const float *restrict b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c,
  const mublis_l3_product_t *ctl
);

/**
 * @brief Uniform interface for double precision out-of-place operations
 * 
 * See the header for `mublis_l3_sdriver`
 */
int mublis_l3_ddriver(
  int m, int n, int k,
  double alpha,
  const double *restrict a, int rs_a, int cs_a,
  const double *restrict b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c,
  const mublis_l3_product_t *ctl
);

/**
 * @brief mirror of cblas_sgemm
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_sgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b, 
  int m, int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
/**
 * @brief mirror of cblas_ssymm
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_ssymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
/**
 * @brief mirror of cblas_ssyrk
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_ssyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float beta,
  float *c, int rs_c, int cs_c
);
/**
 * @brief mirror of cblas_ssyr2k
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_ssyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
/**
 * @brief mirror of cblas_strmm
 * 
 * Because this is an in-place operation, it has its own unique implementation.
 * However, like the driver, it still specializes at runtime via 
 * `mublis_get_safe_context`, and makes allocations from the central pool.
 * Unlike the out-of-place operations, this doesn't require a separate output 
 * buffer.  Instead, it outputs directly to B by calculating loop traversal 
 * directions that ensure that entries in B that have been written to are 
 * never read a second time.
 */
int mublis_strmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);
/**
 * @brief mirror of cblas_strsm
 * 
 * Because this is an in-place operation, it has its own unique implementation.
 * However, like the driver, it still specializes at runtime via 
 * `mublis_get_safe_context`, and makes allocations from the central pool.
 * Unlike the out-of-place operations, this doesn't require a separate output 
 * buffer.  Instead, it outputs directly to B by calculating loop traversal 
 * directions that ensure that entries in B that have been written to are 
 * never read a second time.
 */
int mublis_strsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);

/**
 * @brief Mirror of cblas_dgemm
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_dgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b, 
  int m, int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
/**
 * @brief Mirror of cblas_dsymm
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_dsymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
/**
 * @brief Mirror of cblas_dsyrk
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_dsyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double beta,
  double *c, int rs_c, int cs_c
);
/**
 * @brief Mirror of cblas_dsyr2k
 * 
 * Implemented as a shim over the general out-of-place driver.
 */
int mublis_dsyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
/**
 * @brief mirror of cblas_dtrmm
 * 
 * See the header of `mublis_strmm`.
 */
int mublis_dtrmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);
/**
 * @brief mirror of cblas_strsm
 * 
 * See the header of `mublis_strsm`.
 */
int mublis_dtrsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);

#endif
