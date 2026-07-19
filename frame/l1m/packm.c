#include <stdbool.h>

#include "l1m.h"

static inline int mublis_packm_is_stored(mublis_uplo_t uplo, int diagoff) {
  switch (uplo) {
  case MUBLIS_DENSE:
    return true;
  case MUBLIS_LOWER:
    return diagoff >= 0;
  case MUBLIS_UPPER:
    return diagoff <= 0;
  }

  return false;
}

#define MUBLIS_PACKM_IMPL(ctype, function_name)                                \
  void function(                                                               \
    ctype *restrict dst,                                                       \
    const ctype *restrict src,                                                 \
    int rs, int cs,                                                            \
    int m0, int k0, int mr_or_nr,                                              \
    mublis_packm_struc_t struc,                                                \
    mublis_uplo_t uplo,                                                        \
    mublis_packm_diag_t diag,                                                  \
    int diagoff                                                                \
  ) {                                                                          \
    for (int panel_start = 0; panel_start < m0; panel_start += mr_or_nr) {     \
      int panel_len = m0 - panel_start;                                        \
      if (panel_len > mr_or_nr)                                                \
        panel_len = mr_or_nr;                                                  \
                                                                               \
      ctype *dst_panel = dst + panel_start * k0;                               \
                                                                               \
      for (int column = 0; column < k0; column++) {                            \
        ctype *dst_column = dst_panel + column * mr_or_nr;                     \
                                                                               \
        for (int lane = 0; lane < panel_len; lane++) {                         \
          int row = panel_start + lane;                                        \
          int diag_delta = diagoff + row - column;                             \
          int src_offset = row * rs + column * cs;                             \
          ctype value = 0;                                                     \
                                                                               \
          switch (struc) {                                                     \
          case MUBLIS_STRUC_GENERAL:                                           \
            value = src[src_offset];                                           \
            break;                                                             \
                                                                               \
          case MUBLIS_STRUC_SYMMETRIC:                                         \
            if (mublis_packm_is_stored(uplo, diag_delta)) {                    \
              value = src[src_offset];                                         \
            } else {                                                           \
              int reflected_row = column - diagoff;                            \
              int reflected_column = row + diagoff;                            \
              value = src[reflected_row * rs + reflected_column * cs];         \
            }                                                                  \
            break;                                                             \
                                                                               \
          case MUBLIS_STRUC_TRIANGULAR:                                        \
            if (!mublis_packm_is_stored(uplo, diag_delta)) {                   \
              value = 0;                                                       \
            } else if (diag_delta == 0 && diag == MUBLIS_PACKM_DIAG_UNIT) {    \
              value = 1;                                                       \
            } else {                                                           \
              value = src[src_offset];                                         \
                                                                               \
              if (diag_delta == 0 && diag == MUBLIS_PACKM_DIAG_INVERT)         \
                value = (ctype)1 / value;                                      \
            }                                                                  \
            break;                                                             \
          }                                                                    \
                                                                               \
          dst_column[lane] = value;                                            \
        }                                                                      \
                                                                               \
        for (int lane = panel_len; lane < mr_or_nr; lane++)                    \
          dst_column[lane] = 0;                                                \
      }                                                                        \
    }                                                                          \
  }

MUBLIS_PACKM_IMPL(float, mublis_spackm)
MUBLIS_PACKM_IMPL(double, mublis_dpackm)

#undef MUBLIS_PACKM_IMPL
