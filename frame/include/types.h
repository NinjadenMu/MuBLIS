#ifndef MUBLIS_TYPES_H
#define MUBLIS_TYPES_H

typedef enum {
  MUBLIS_NO_TRANSPOSE = 0,
  MUBLIS_TRANSPOSE = 1,
} mublis_trans_t;

typedef enum {
  MYBLIS_LOWER = 0,
  MYBLIS_UPPER = 1,
  MYBLIS_DENSE = 2
} myblis_uplo_t;

typedef enum {
  MUBLIS_LEFT = 0,
  MUBLIS_RIGHT = 1,
} mublis_side_t;

typedef enum {
  MUBLIS_NON_UNIT_DIAG = 0,
  MUBLIS_UNIT_DIAG = 1,
} mublis_diag_t;

#endif
