#ifndef MUBLIS_1M_H
#define MUBLIS_1M_H

#include "types.h"

/*
 * Represents shape of logical matrix.
 * Note that this is distinct from the shape of the data that's actually 
 * stored, since symmetric matrices may only store one triangle, even 
 * though both triangles have non-zero values.
 */
typedef enum {
  MUBLIS_STRUC_GENERAL = 0,
  MUBLIS_STRUC_SYMMETRIC = 1,
  MUBLIS_STRUC_TRIANGULAR = 2
} mublis_packm_struc_t;

/*
 * Determines the values the packing function places on the diagonal of 
 * panels intersecting with the diagonal.
 * 
 * Note that this is distinct from `mublis_diag_t`, since this is for 
 * use by the packing function, and does not necessarily describe the 
 * actual matrix.  For example, `MUBLIS_PACKM_DIAG_INVERT` is useful for 
 * the implementation of trsm, but BLAS doesn't actually expose an option 
 * to invert the diagonal of a matrix.
 */
typedef enum {
  MUBLIS_PACKM_DIAG_NONUNIT = 0,
  MUBLIS_PACKM_DIAG_UNIT = 1,
  MUBLIS_PACKM_DIAG_INVERT = 2
} mublis_packm_diag_t;

/**
 * @brief Uniform packing interface for single precision A and B
 * 
 * Packs data into contiguous buffer in the order the micro-kernel reads it.
 * Can be used for A directly, and for B by providing the logically transposed 
 * view (swap `rs` and `cs`, swap `m0` and `k0`, and invert `diagoff` and 
 * `uplo`). 
 */
void mublis_spackm(
  float *restrict dst,
  const float *restrict src,
  int rs, int cs,
  int m0, int k0, int mr_or_nr,
  mublis_packm_struc_t struc,
  mublis_uplo_t uplo,
  mublis_packm_diag_t diag,
  int diagoff
);

/**
 * @brief Uniform packing interface for double precision A and B
 * 
 * See header for `mublis_spackm`.
 */
void mublis_dpackm(
  double *restrict dst,
  const double *restrict src,
  int rs, int cs,
  int m0, int k0, int mr_or_nr,
  mublis_packm_struc_t struc,
  mublis_uplo_t uplo,
  mublis_packm_diag_t diag,
  int diagoff
);

#endif
