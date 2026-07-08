#ifndef MUBLIS_L3_H
#define MUBLIS_L3_H

#include "context.h"
#include "packm.h"
#include "types.h"

typedef enum {
  ALL = 0,
  LOWER = 1,
  UPPER = 2,
} mublis_l3_driver_shape_t;

typedef struct {
  mublis_struc_t struc;
  mublis_uplo_t uplo;
  mublis_packm_diag_t diag;
} mublis_l3_driver_pack_args_t;

typedef struct {
  mublis_l3_driver_pack_args_t pack_a_args;
  mublis_l3_driver_pack_args_t pack_b_args;

  mublis_l3_driver_shape_t driver_shape;
} mublis_l3_ctl_t;

void mublis_sl3_driver(
  int m, int n, int k,
  const float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  const float *beta,
  float *c, int rs_c, int cs_c,
  const mublis_l3_ctl_t *ctl,
  const mublis_context_t *context
);

void mublis_sgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b, 
  int m, int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
void mublis_ssymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
void mublis_ssyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float beta,
  float *c, int rs_c, int cs_c
);
void mublis_ssyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c
);
void mublis_strmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);
void mublis_strsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  float alpha,
  const float *a, int rs_a, int cs_a,
  float *b, int rs_b, int cs_b
);

void mublis_dgemm(
  mublis_trans_t trans_a, mublis_trans_t trans_b, 
  int m, int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
void mublis_dsymm(
  mublis_side_t side, mublis_uplo_t uplo,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
void mublis_dsyrk(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double beta,
  double *c, int rs_c, int cs_c
);
void mublis_dsyr2k(
  mublis_uplo_t uplo, mublis_trans_t trans,
  int n, int k,
  double alpha,
  const double *a, int rs_a, int cs_a,
  const double *b, int rs_b, int cs_b,
  double beta,
  double *c, int rs_c, int cs_c
);
void mublis_dtrmm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);
void mublis_dtrsm(
  mublis_side_t side, mublis_uplo_t uplo,
  mublis_trans_t trans_a, mublis_diag_t diag,
  int m, int n,
  double alpha,
  const double *a, int rs_a, int cs_a,
  double *b, int rs_b, int cs_b
);

#endif