#ifndef MUBLIS_L3_H
#define MUBLIS_L3_H

#include "context.h"
#include "l1m.h"
#include "types.h"

typedef enum {
  MUBLIS_L3_ALL = 0,

  // left index <= right index
  MUBLIS_L3_LOWER = 1,

  // left index >= right index
  MUBLIS_L3_UPPER = 2,
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
  const mublis_l3_product_t *ctl,
  const mublis_context_t *context
);

int mublis_l3_ddriver(
  int m, int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c,
  const mublis_l3_product_t *ctl,
  const mublis_context_t *context
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

#endif