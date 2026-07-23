#include "stdbool.h"

#include "l3_utils.h"
#include "types.h"
#include "l3.h"

mublis_uplo_t flip_uplo(mublis_uplo_t uplo) {
  switch (uplo) {
    case MUBLIS_DENSE:
      return MUBLIS_DENSE;
    case MUBLIS_UPPER:
      return MUBLIS_LOWER;
    case MUBLIS_LOWER:
      return MUBLIS_UPPER;
  }
}

bool relation_holds(mublis_l3_relation_t relation, int lhs, int rhs) {
  switch (relation) {
    case MUBLIS_L3_ALL:
      return true;
    case MUBLIS_L3_LOWER:
      return lhs <= rhs;
    case MUBLIS_L3_UPPER:
      return lhs >= rhs;
  }
}

bool block_is_outside(
  mublis_l3_relation_t relation,
  int lhs0, int lhs_len,
  int rhs0, int rhs_len
) {
  switch (relation) {
    case MUBLIS_L3_ALL:
      return false;
    
    case MUBLIS_L3_LOWER:
      return lhs0 >= rhs0 + rhs_len;

    case MUBLIS_L3_UPPER:
      return lhs0 + lhs_len <= rhs0;
  }
}

bool block_is_inside(
  mublis_l3_relation_t relation,
  int lhs0, int lhs_len,
  int rhs0, int rhs_len
) {
  switch (relation) {
    case MUBLIS_L3_ALL:
      return true;
    
    case MUBLIS_L3_LOWER:
      return lhs0 + lhs_len - 1 <= rhs0;

    case MUBLIS_L3_UPPER:
      return lhs0 >= rhs0 + rhs_len - 1;
  }
}

int round_up_to_multiple(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}
