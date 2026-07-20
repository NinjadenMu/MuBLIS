#ifndef MUBLIS_H
#define MUBLIS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MUBLIS_NO_TRANSPOSE = 0,
  MUBLIS_TRANSPOSE = 1
} mublis_trans_t;

typedef enum {
  MUBLIS_DENSE = 0,
  MUBLIS_LOWER = 1,
  MUBLIS_UPPER = 2
} mublis_uplo_t;

typedef enum {
  MUBLIS_LEFT = 0,
  MUBLIS_RIGHT = 1
} mublis_side_t;

typedef enum {
  MUBLIS_NON_UNIT_DIAG = 0,
  MUBLIS_UNIT_DIAG = 1
} mublis_diag_t;

typedef enum {
  MUBLIS_STRUC_GENERAL = 0,
  MUBLIS_STRUC_SYMMETRIC = 1,
  MUBLIS_STRUC_TRIANGULAR = 2
} mublis_packm_struc_t;

typedef enum {
  MUBLIS_PACKM_DIAG_NONUNIT = 0,
  MUBLIS_PACKM_DIAG_UNIT = 1,
  MUBLIS_PACKM_DIAG_INVERT = 2
} mublis_packm_diag_t;

typedef enum {
  MUBLIS_L3_ALL = 0,
  MUBLIS_L3_LOWER = 1,
  MUBLIS_L3_UPPER = 2
} mublis_l3_relation_t;

typedef struct {
  mublis_l3_relation_t jp;
  mublis_l3_relation_t ji;
  mublis_l3_relation_t pi;
} mublis_l3_domain_t;

typedef struct {
  mublis_packm_struc_t struc;
  mublis_uplo_t uplo;
  mublis_packm_diag_t diag;
} mublis_l3_operand_t;

typedef struct {
  mublis_l3_domain_t domain;
  mublis_l3_operand_t a;
  mublis_l3_operand_t b;
} mublis_l3_product_t;

int mublis_l3_sdriver(
  int m, int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c,
  const mublis_l3_product_t *product
);

int mublis_l3_ddriver(
  int m, int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c,
  const mublis_l3_product_t *product
);

int mublis_sgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b,
  int m, int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);

int mublis_ssymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);

int mublis_ssyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float beta,
  float *c, int rs_c, int cs_c
);

int mublis_ssyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);

int mublis_strmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);

int mublis_strsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);

int mublis_dgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b,
  int m, int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);

int mublis_dsymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);

int mublis_dsyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double beta,
  double *c, int rs_c, int cs_c
);

int mublis_dsyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);

int mublis_dtrmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);

int mublis_dtrsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);

#ifdef __cplusplus
}
#endif

#endif
