#include <stdbool.h>
#include <stdlib.h>

#include "l1m.h"
#include "l3.h"
#include "l3_utils.h"
#include "mublis_instantiate.h"
#include "pool.h"
#include "safe_dispatch.h"
#include "types.h"

#define MUBLIS_TRSM_IMPL(ctype, function_name, context_field, pool_role,       \
                         packm_name, packm_trsm_rhs_name, packm_trsm_a_name)   \
  int function_name(                                                           \
    mublis_side_t side,                                                        \
    mublis_uplo_t uplo,                                                        \
    mublis_trans_t trans_a,                                                    \
    mublis_diag_t diag,                                                        \
    int m, int n,                                                              \
    ctype alpha,                                                               \
    const ctype *a, int rs_a, int cs_a,                                        \
    ctype *b, int rs_b, int cs_b                                               \
  ) {                                                                          \
    if (m == 0 || n == 0)                                                      \
      return 0;                                                                \
                                                                               \
    if (alpha == 0) {                                                          \
      for (int j = 0; j < n; j++)                                              \
        for (int i = 0; i < m; i++)                                            \
          b[i * rs_b + j * cs_b] = 0;                                          \
                                                                               \
      return 0;                                                                \
    }                                                                          \
                                                                               \
    if (side == MUBLIS_RIGHT) {                                                \
      int tmp = m;                                                             \
      m = n;                                                                   \
      n = tmp;                                                                 \
                                                                               \
      tmp = rs_b;                                                              \
      rs_b = cs_b;                                                             \
      cs_b = tmp;                                                              \
                                                                               \
      trans_a = trans_a == MUBLIS_NO_TRANSPOSE                                 \
        ? MUBLIS_TRANSPOSE                                                     \
        : MUBLIS_NO_TRANSPOSE;                                                 \
    }                                                                          \
                                                                               \
    if (trans_a == MUBLIS_TRANSPOSE) {                                         \
      int tmp = rs_a;                                                          \
      rs_a = cs_a;                                                             \
      cs_a = tmp;                                                              \
      uplo = flip_uplo(uplo);                                                  \
    }                                                                          \
                                                                               \
    int error_code;                                                            \
    const mublis_context_t *context;                                           \
                                                                               \
    if ((error_code = mublis_get_safe_context(&context)))                      \
      return error_code;                                                       \
                                                                               \
    if ((error_code = mublis_pool_init(context)))                              \
      return error_code;                                                       \
                                                                               \
    int mr = context->context_field.mr;                                        \
    int nr = context->context_field.nr;                                        \
    int mc = context->context_field.mc;                                        \
    int kc = MIN(context->context_field.kc, mc);                               \
    int nc = context->context_field.nc;                                        \
                                                                               \
    kc -= kc % mr;                                                             \
    if (kc == 0)                                                               \
      return 1;                                                                \
                                                                               \
    mublis_pool_block_t pool_block;                                            \
    if ((error_code = mublis_pool_checkout(pool_role, &pool_block)))           \
      return error_code;                                                       \
                                                                               \
    ctype *a_pack_buf = pool_block.a_pack_buf;                                 \
    ctype *b_pack_buf = pool_block.b_pack_buf;                                 \
    ctype *c_buf = pool_block.c_buf;                                           \
                                                                               \
    mublis_packm_diag_t pack_diag = diag == MUBLIS_UNIT_DIAG                   \
      ? MUBLIS_PACKM_DIAG_UNIT                                                 \
      : MUBLIS_PACKM_DIAG_INVERT;                                              \
                                                                               \
    for (int jc = 0; jc < n; jc += nc) {                                       \
      int nb = MIN(nc, n - jc);                                                \
      bool first = true;                                                       \
                                                                               \
      if (uplo == MUBLIS_LOWER) {                                              \
        for (int pc = 0; pc < m; pc += kc) {                                   \
          int kb = MIN(kc, m - pc);                                            \
          int k_pack = round_up_to_multiple(kb, mr);                           \
          int mb = MIN(mc, m - pc);                                            \
          ctype alpha_eff = first ? alpha : 1;                                 \
          first = false;                                                       \
                                                                               \
          packm_trsm_rhs_name(                                                 \
            b_pack_buf,                                                        \
            &b[pc * rs_b + jc * cs_b],                                         \
            rs_b, cs_b,                                                        \
            kb, nb, k_pack, nr                                                 \
          );                                                                   \
                                                                               \
          packm_trsm_a_name(                                                   \
            a_pack_buf,                                                        \
            &a[pc * rs_a + pc * cs_a],                                         \
            rs_a, cs_a,                                                        \
            mb, kb, k_pack, mr,                                                \
            MUBLIS_LOWER, pack_diag                                            \
          );                                                                   \
                                                                               \
          for (int jr = 0; jr < nb; jr += nr) {                                \
            int nr_cur = MIN(nr, nb - jr);                                     \
            ctype *bj = b_pack_buf + (jr / nr) * k_pack * nr;                  \
            for (int ir = 0; ir < mb; ir += mr) {                              \
              int mr_cur = MIN(mr, mb - ir);                                   \
              const ctype *ai =                                                \
                a_pack_buf + (ir / mr) * mr * k_pack;                          \
              ctype *cij =                                                     \
                &b[(pc + ir) * rs_b + (jc + jr) * cs_b];                       \
                                                                               \
              if (ir < kb) {                                                   \
                ctype *b11 = bj + ir * nr;                                     \
                ctype *c_dst =                                                 \
                  mr_cur == mr && nr_cur == nr ? cij : c_buf;                  \
                int rs_dst =                                                   \
                  mr_cur == mr && nr_cur == nr ? rs_b : 1;                     \
                int cs_dst =                                                   \
                  mr_cur == mr && nr_cur == nr ? cs_b : mr;                    \
                                                                               \
                context->context_field.gemmtrsml_ukr(                          \
                  ir,                                                          \
                  alpha_eff,                                                   \
                  ai,                                                          \
                  ai + ir * mr,                                                \
                  bj,                                                          \
                  b11,                                                         \
                  c_dst,                                                       \
                  rs_dst, cs_dst,                                              \
                  NULL                                                         \
                );                                                             \
                                                                               \
                if (c_dst == c_buf) {                                          \
                  for (int j = 0; j < nr_cur; j++)                             \
                    for (int i = 0; i < mr_cur; i++)                           \
                      cij[i * rs_b + j * cs_b] = c_buf[i + j * mr];            \
                }                                                              \
              } else {                                                         \
                if (mr_cur == mr && nr_cur == nr) {                            \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    alpha_eff,                                                 \
                    cij, rs_b, cs_b,                                           \
                    NULL                                                       \
                  );                                                           \
                } else {                                                       \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    0,                                                         \
                    c_buf, 1, mr,                                              \
                    NULL                                                       \
                  );                                                           \
                                                                               \
                  for (int j = 0; j < nr_cur; j++) {                           \
                    for (int i = 0; i < mr_cur; i++) {                         \
                      ctype *dst = cij + i * rs_b + j * cs_b;                  \
                      ctype value = c_buf[i + j * mr];                         \
                                                                               \
                      if (alpha_eff == 1)                                      \
                        *dst += value;                                         \
                      else                                                     \
                        *dst = value + alpha_eff * *dst;                       \
                    }                                                          \
                  }                                                            \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
                                                                               \
          for (int ic = pc + mb; ic < m; ic += mc) {                           \
            int mb_cur = MIN(mc, m - ic);                                      \
                                                                               \
            packm_name(                                                        \
              a_pack_buf,                                                      \
              &a[ic * rs_a + pc * cs_a],                                       \
              rs_a, cs_a,                                                      \
              mb_cur, kb, mr,                                                  \
              MUBLIS_STRUC_GENERAL,                                            \
              MUBLIS_DENSE,                                                    \
              MUBLIS_PACKM_DIAG_NONUNIT,                                       \
              0                                                                \
            );                                                                 \
                                                                               \
            for (int jr = 0; jr < nb; jr += nr) {                              \
              int nr_cur = MIN(nr, nb - jr);                                   \
              const ctype *bj =                                                \
                b_pack_buf + (jr / nr) * k_pack * nr;                          \
                                                                               \
              for (int ir = 0; ir < mb_cur; ir += mr) {                        \
                int mr_cur = MIN(mr, mb_cur - ir);                             \
                const ctype *ai = a_pack_buf + ir * kb;                        \
                ctype *cij =                                                   \
                  &b[(ic + ir) * rs_b + (jc + jr) * cs_b];                     \
                                                                               \
                if (mr_cur == mr && nr_cur == nr) {                            \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    alpha_eff,                                                 \
                    cij, rs_b, cs_b,                                           \
                    NULL                                                       \
                  );                                                           \
                } else {                                                       \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    0,                                                         \
                    c_buf, 1, mr,                                              \
                    NULL                                                       \
                  );                                                           \
                                                                               \
                  for (int j = 0; j < nr_cur; j++) {                           \
                    for (int i = 0; i < mr_cur; i++) {                         \
                      ctype *dst = cij + i * rs_b + j * cs_b;                  \
                      ctype value = c_buf[i + j * mr];                         \
                                                                               \
                      if (alpha_eff == 1)                                      \
                        *dst += value;                                         \
                      else                                                     \
                        *dst = value + alpha_eff * *dst;                       \
                    }                                                          \
                  }                                                            \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      } else {                                                                 \
        int pc_start = ((m - 1) / kc) * kc;                                    \
                                                                               \
        for (int pc = pc_start; pc >= 0; pc -= kc) {                           \
          int kb = MIN(kc, m - pc);                                            \
          int k_pack = round_up_to_multiple(kb, mr);                           \
          ctype alpha_eff = first ? alpha : 1;                                 \
          first = false;                                                       \
                                                                               \
          packm_trsm_rhs_name(                                                 \
            b_pack_buf,                                                        \
            &b[pc * rs_b + jc * cs_b],                                         \
            rs_b, cs_b,                                                        \
            kb, nb, k_pack, nr                                                 \
          );                                                                   \
                                                                               \
          packm_trsm_a_name(                                                   \
            a_pack_buf,                                                        \
            &a[pc * rs_a + pc * cs_a],                                         \
            rs_a, cs_a,                                                        \
            kb, kb, k_pack, mr,                                                \
            MUBLIS_UPPER, pack_diag                                            \
          );                                                                   \
                                                                               \
          for (int jr = 0; jr < nb; jr += nr) {                                \
            int nr_cur = MIN(nr, nb - jr);                                     \
            ctype *bj = b_pack_buf + (jr / nr) * k_pack * nr;                  \
            int ir_start = ((kb - 1) / mr) * mr;                               \
            for (int ir = ir_start; ir >= 0; ir -= mr) {                       \
              int mr_cur = MIN(mr, kb - ir);                                   \
              int k_right = k_pack - ir - mr;                                  \
              const ctype *ai =                                                \
                a_pack_buf + (ir / mr) * mr * k_pack;                          \
              const ctype *a11 = ai + ir * mr;                                 \
              ctype *b11 = bj + ir * nr;                                       \
              ctype *cij =                                                     \
                &b[(pc + ir) * rs_b + (jc + jr) * cs_b];                       \
              ctype *c_dst =                                                   \
                mr_cur == mr && nr_cur == nr ? cij : c_buf;                    \
              int rs_dst =                                                     \
                mr_cur == mr && nr_cur == nr ? rs_b : 1;                       \
              int cs_dst =                                                     \
                mr_cur == mr && nr_cur == nr ? cs_b : mr;                      \
                                                                               \
              context->context_field.gemmtrsmu_ukr(                            \
                k_right,                                                       \
                alpha_eff,                                                     \
                a11 + mr * mr,                                                 \
                a11,                                                           \
                b11 + mr * nr,                                                 \
                b11,                                                           \
                c_dst,                                                         \
                rs_dst, cs_dst,                                                \
                NULL                                                           \
              );                                                               \
                                                                               \
              if (c_dst == c_buf) {                                            \
                for (int j = 0; j < nr_cur; j++)                               \
                  for (int i = 0; i < mr_cur; i++)                             \
                    cij[i * rs_b + j * cs_b] = c_buf[i + j * mr];              \
              }                                                                \
            }                                                                  \
          }                                                                    \
                                                                               \
          for (int ic = 0; ic < pc; ic += mc) {                                \
            int mb_cur = MIN(mc, pc - ic);                                     \
                                                                               \
            packm_name(                                                        \
              a_pack_buf,                                                      \
              &a[ic * rs_a + pc * cs_a],                                       \
              rs_a, cs_a,                                                      \
              mb_cur, kb, mr,                                                  \
              MUBLIS_STRUC_GENERAL,                                            \
              MUBLIS_DENSE,                                                    \
              MUBLIS_PACKM_DIAG_NONUNIT,                                       \
              0                                                                \
            );                                                                 \
                                                                               \
            for (int jr = 0; jr < nb; jr += nr) {                              \
              int nr_cur = MIN(nr, nb - jr);                                   \
              const ctype *bj =                                                \
                b_pack_buf + (jr / nr) * k_pack * nr;                          \
                                                                               \
              for (int ir = 0; ir < mb_cur; ir += mr) {                        \
                int mr_cur = MIN(mr, mb_cur - ir);                             \
                const ctype *ai = a_pack_buf + ir * kb;                        \
                ctype *cij =                                                   \
                  &b[(ic + ir) * rs_b + (jc + jr) * cs_b];                     \
                                                                               \
                if (mr_cur == mr && nr_cur == nr) {                            \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    alpha_eff,                                                 \
                    cij, rs_b, cs_b,                                           \
                    NULL                                                       \
                  );                                                           \
                } else {                                                       \
                  context->context_field.gemm_ukr(                             \
                    kb,                                                        \
                    -1,                                                        \
                    ai, bj,                                                    \
                    0,                                                         \
                    c_buf, 1, mr,                                              \
                    NULL                                                       \
                  );                                                           \
                                                                               \
                  for (int j = 0; j < nr_cur; j++) {                           \
                    for (int i = 0; i < mr_cur; i++) {                         \
                      ctype *dst = cij + i * rs_b + j * cs_b;                  \
                      ctype value = c_buf[i + j * mr];                         \
                                                                               \
                      if (alpha_eff == 1)                                      \
                        *dst += value;                                         \
                      else                                                     \
                        *dst = value + alpha_eff * *dst;                       \
                    }                                                          \
                  }                                                            \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (mublis_pool_checkin(pool_block)) {                                     \
      free(a_pack_buf);                                                        \
      free(b_pack_buf);                                                        \
      free(c_buf);                                                             \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
  }

MUBLIS_TRSM_IMPL(float, mublis_strsm, s, MUBLIS_POOL_S, mublis_spackm,
                 mublis_spackm_trsm_rhs, mublis_spackm_trsm_a)
MUBLIS_TRSM_IMPL(double, mublis_dtrsm, d, MUBLIS_POOL_D, mublis_dpackm,
                 mublis_dpackm_trsm_rhs, mublis_dpackm_trsm_a)
