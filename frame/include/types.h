/**
 * @file types.h
 * 
 * Defines MuBLIS mirrors of BLAS types
 */

#ifndef MUBLIS_TYPES_H
#define MUBLIS_TYPES_H

// Mirror of CBLAS_TRANSPOSE
typedef enum {
  MUBLIS_NO_TRANSPOSE = 0,
  MUBLIS_TRANSPOSE = 1
} mublis_trans_t;

/*
 * Mirror of CBLAS_UPLO
 *
 * Specifies shape of stored data.
 * Note that MuBLIS makes a distinction between the shape of stored data 
 * and the shape of the logical matrix it represents.  For example, a 
 * buffer which should be viewed logically as a full symmetric matrix may 
 * actually only store one triangle (which is specified by `uplo`.)
 * See l3.h for more information.
 * 
 * Adds MUBLIS_DENSE, since the l3 driver always takes an `uplo` argument.
 */
typedef enum {
  MUBLIS_DENSE = 0,
  MUBLIS_LOWER = 1,
  MUBLIS_UPPER = 2
} mublis_uplo_t;

// Mirror of CBLAS_SIDE
typedef enum {
  MUBLIS_LEFT = 0,
  MUBLIS_RIGHT = 1
} mublis_side_t;

// Mirror of CBLAS_DIAG
typedef enum {
  MUBLIS_NON_UNIT_DIAG = 0,
  MUBLIS_UNIT_DIAG = 1
} mublis_diag_t;

#endif
