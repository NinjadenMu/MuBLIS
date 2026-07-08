#ifndef MUBLIS_PACKM_H
#define MUBLIS_PACKM_H

#include "types.h"

typedef enum {
  MUBLIS_STRUC_GENERAL = 0,
  MUBLIS_STRUC_SYMMETRIC = 1,
  MUBLIS_STRUC_TRIANGULAR = 2
} mublis_struc_t;

typedef enum {
  MUBLIS_PACKM_DIAG_NONUNIT = 0,
  MUBLIS_PACKM_DIAG_UNIT = 1,
  MUBLIS_PACKM_DIAG_INVERT = 2,
} mublis_packm_diag_t;

void mublis_spackm(
  float* dst,
  const float *src,
  int rs, int cs,
  int m0, int k0, int mr_or_nr,
  mublis_trans_t trans,
  mublis_struc_t struc,
  mublis_uplo_t uplo,
  mublis_packm_diag_t diag
);

void mublis_dpackm(
  float* dst,
  const float *src,
  int rs, int cs,
  int m0, int k0, int mr_or_nr,
  mublis_trans_t trans,
  mublis_struc_t struc,
  mublis_uplo_t uplo,
  mublis_packm_diag_t diag
);

#endif
