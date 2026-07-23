#ifndef MUBLIS_L3_UTILS_H
#define MUBLIS_L3_UTILS_H

#include "stdbool.h"

#include "types.h"
#include "l3.h"

#define MIN(x, y) (((x) <= (y)) ? (x) : (y))

static inline mublis_uplo_t flip_uplo(mublis_uplo_t uplo);

static inline bool relation_holds(mublis_l3_relation_t relation, int lhs, int rhs);

static inline bool block_is_outside(
  mublis_l3_relation_t relation,
  int lhs0, int lhs_len,
  int rhs0, int rhs_len
);

static inline bool block_is_inside(
  mublis_l3_relation_t relation,
  int lhs0, int lhs_len,
  int rhs0, int rhs_len
);

static int round_up_to_multiple(int value, int multiple);

#endif
