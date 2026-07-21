#include "stdbool.h"
#include "stdlib.h"

#include "l1m.h"
#include "l3.h"
#include "l3_utils.h"
#include "pool.h"
#include "types.h"

static int first_update_pc(
  mublis_l3_domain_t domain,
  bool reverse_pc,
  int jr, int ir,
  int kc
) {
  int first_p = 0;

  if ((!reverse_pc && domain.jp == MUBLIS_L3_LOWER) ||
      (reverse_pc && domain.jp == MUBLIS_L3_UPPER))
    first_p = jr;

  if ((!reverse_pc && domain.pi == MUBLIS_L3_UPPER) ||
      (reverse_pc && domain.pi == MUBLIS_L3_LOWER))
    first_p = ir;

  return first_p - first_p % kc;
}

#define MUBLIS_TRMM_IMPL(                                                     \
  ctype, function_name, packm_name, pool_role, context_field                 \
)                                                                            \
  int function_name(                                                         \
    mublis_side_t side, mublis_uplo_t uplo,                                  \
    mublis_trans_t trans_a, mublis_diag_t diag,                              \
    int m, int n,                                                            \
    ctype alpha,                                                             \
    const ctype *a, int rs_a, int cs_a,                                      \
    ctype *b, int rs_b, int cs_b                                             \
  ) {                                                                        \
    if (m == 0 || n == 0)                                                    \
      return 0;                                                              \
                                                                             \
    if (trans_a == MUBLIS_TRANSPOSE) {                                       \
      int tmp = rs_a;                                                        \
      rs_a = cs_a;                                                           \
      cs_a = tmp;                                                            \
      uplo = flip_uplo(uplo);                                                \
    }                                                                        \
                                                                             \
    mublis_l3_relation_t triangle_relation =                                 \
      uplo == MUBLIS_LOWER ? MUBLIS_L3_LOWER : MUBLIS_L3_UPPER;              \
                                                                             \
    const mublis_l3_product_t product = {                                    \
      .domain = {                                                            \
        .jp = side == MUBLIS_RIGHT                                           \
          ? triangle_relation                                                \
          : MUBLIS_L3_ALL,                                                   \
        .ji = MUBLIS_L3_ALL,                                                 \
        .pi = side == MUBLIS_LEFT                                            \
          ? triangle_relation                                                \
          : MUBLIS_L3_ALL                                                    \
      },                                                                     \
      .a = {                                                                 \
        .struc = side == MUBLIS_LEFT                                         \
          ? MUBLIS_STRUC_TRIANGULAR                                          \
          : MUBLIS_STRUC_GENERAL,                                            \
        .uplo = side == MUBLIS_LEFT ? uplo : MUBLIS_DENSE,                  \
        .diag = side == MUBLIS_LEFT && diag == MUBLIS_UNIT_DIAG             \
          ? MUBLIS_PACKM_DIAG_UNIT                                           \
          : MUBLIS_PACKM_DIAG_NONUNIT                                        \
      },                                                                     \
      .b = {                                                                 \
        .struc = side == MUBLIS_RIGHT                                        \
          ? MUBLIS_STRUC_TRIANGULAR                                          \
          : MUBLIS_STRUC_GENERAL,                                            \
        .uplo = side == MUBLIS_RIGHT ? uplo : MUBLIS_DENSE,                 \
        .diag = side == MUBLIS_RIGHT && diag == MUBLIS_UNIT_DIAG            \
          ? MUBLIS_PACKM_DIAG_UNIT                                           \
          : MUBLIS_PACKM_DIAG_NONUNIT                                        \
      }                                                                      \
    };                                                                       \
                                                                             \
    int k;                                                                   \
    const ctype *a_product;                                                  \
    int rs_a_product, cs_a_product;                                          \
    const ctype *b_product;                                                  \
    int rs_b_product, cs_b_product;                                          \
                                                                             \
    if (side == MUBLIS_LEFT) {                                               \
      k = m;                                                                 \
      a_product = a;                                                         \
      rs_a_product = rs_a;                                                   \
      cs_a_product = cs_a;                                                   \
      b_product = b;                                                         \
      rs_b_product = rs_b;                                                   \
      cs_b_product = cs_b;                                                   \
    }                                                                        \
    else {                                                                   \
      k = n;                                                                 \
      a_product = b;                                                         \
      rs_a_product = rs_b;                                                   \
      cs_a_product = cs_b;                                                   \
      b_product = a;                                                         \
      rs_b_product = rs_a;                                                   \
      cs_b_product = cs_a;                                                   \
    }                                                                        \
                                                                             \
    int error_code;                                                          \
    const mublis_context_t *context;                                         \
    if ((error_code = mublis_get_safe_context(&context)))                    \
      return error_code;                                                     \
                                                                             \
    if ((error_code = mublis_pool_init(context)))                            \
      return error_code;                                                     \
                                                                             \
    int mr = context->context_field.mr;                                      \
    int nr = context->context_field.nr;                                      \
    int mc = context->context_field.mc;                                      \
    int kc = context->context_field.kc;                                      \
    int nc = context->context_field.nc;                                      \
                                                                             \
    mublis_pool_block_t pool_block;                                          \
    if ((error_code = mublis_pool_checkout(pool_role, &pool_block)))         \
      return error_code;                                                     \
    ctype *a_pack_buf = pool_block.a_pack_buf;                               \
    ctype *b_pack_buf = pool_block.b_pack_buf;                               \
    ctype *c_buf = pool_block.c_buf;                                         \
                                                                             \
    if (alpha == 0) {                                                        \
      for (int j = 0; j < n; j++)                                            \
        for (int i = 0; i < m; i++)                                          \
          b[i * rs_b + j * cs_b] = 0;                                        \
                                                                             \
      goto exit;                                                             \
    }                                                                        \
                                                                             \
    mublis_l3_domain_t domain = product.domain;                              \
    bool reverse_jc = domain.jp == MUBLIS_L3_UPPER;                          \
    bool reverse_pc = domain.jp == MUBLIS_L3_UPPER ||                        \
                      domain.pi == MUBLIS_L3_LOWER;                          \
    int jc_start = reverse_jc ? (n - 1) / nc * nc : 0;                       \
    int pc_start = reverse_pc ? (k - 1) / kc * kc : 0;                       \
                                                                             \
    for (int jc = jc_start;                                                  \
         reverse_jc ? jc >= 0 : jc < n;                                      \
         jc += reverse_jc ? -nc : nc) {                                      \
      int j_max = MIN(jc + nc, n);                                           \
                                                                             \
      for (int pc = pc_start;                                                \
           reverse_pc ? pc >= 0 : pc < k;                                    \
           pc += reverse_pc ? -kc : kc) {                                    \
        int p_max = MIN(pc + kc, k);                                         \
        if (block_is_outside(                                                \
              domain.jp, jc, j_max - jc, pc, p_max - pc))                    \
          continue;                                                          \
                                                                             \
        packm_name(                                                          \
          b_pack_buf,                                                        \
          &b_product[pc * rs_b_product + jc * cs_b_product],                 \
          cs_b_product, rs_b_product,                                        \
          j_max - jc, p_max - pc, nr,                                        \
          product.b.struc,                                                   \
          flip_uplo(product.b.uplo),                                         \
          product.b.diag,                                                    \
          jc - pc                                                            \
        );                                                                   \
                                                                             \
        for (int ic = 0; ic < m; ic += mc) {                                 \
          int i_max = MIN(ic + mc, m);                                       \
          if (block_is_outside(                                              \
                domain.pi, pc, p_max - pc, ic, i_max - ic))                  \
            continue;                                                        \
                                                                             \
          packm_name(                                                        \
            a_pack_buf,                                                      \
            &a_product[ic * rs_a_product + pc * cs_a_product],               \
            rs_a_product, cs_a_product,                                      \
            i_max - ic, p_max - pc, mr,                                      \
            product.a.struc,                                                 \
            product.a.uplo,                                                  \
            product.a.diag,                                                  \
            ic - pc                                                          \
          );                                                                 \
                                                                             \
          for (int jr = jc; jr < j_max; jr += nr) {                          \
            int j_tile_max = MIN(jr + nr, j_max);                            \
            if (block_is_outside(                                            \
                  domain.jp, jr, j_tile_max - jr, pc, p_max - pc))           \
              continue;                                                      \
                                                                             \
            for (int ir = ic; ir < i_max; ir += mr) {                        \
              int i_tile_max = MIN(ir + mr, i_max);                          \
              if (block_is_outside(                                          \
                    domain.pi, pc, p_max - pc, ir, i_tile_max - ir))         \
                continue;                                                    \
                                                                             \
              bool first_update = pc == first_update_pc(                     \
                domain, reverse_pc, jr, ir, kc                               \
              );                                                             \
                                                                             \
              if (ir + mr <= i_max && jr + nr <= j_max) {                    \
                (context->context_field.gemm_ukr)(                           \
                  p_max - pc,                                                \
                  alpha,                                                     \
                  &a_pack_buf[(ir - ic) * (p_max - pc)],                     \
                  &b_pack_buf[(jr - jc) * (p_max - pc)],                     \
                  first_update ? 0 : 1,                                      \
                  &b[ir * rs_b + jr * cs_b],                                 \
                  rs_b, cs_b,                                                \
                  NULL                                                       \
                );                                                           \
              }                                                              \
              else {                                                         \
                (context->context_field.gemm_ukr)(                           \
                  p_max - pc,                                                \
                  alpha,                                                     \
                  &a_pack_buf[(ir - ic) * (p_max - pc)],                     \
                  &b_pack_buf[(jr - jc) * (p_max - pc)],                     \
                  0,                                                         \
                  c_buf,                                                     \
                  1, mr,                                                     \
                  NULL                                                       \
                );                                                           \
                                                                             \
                for (int j = jr; j < j_tile_max; j++) {                      \
                  for (int i = ir; i < i_tile_max; i++) {                    \
                    ctype *bij = &b[i * rs_b + j * cs_b];                    \
                    ctype value = c_buf[(i - ir) + (j - jr) * mr];           \
                                                                             \
                    if (first_update)                                        \
                      *bij = value;                                          \
                    else                                                     \
                      *bij += value;                                         \
                  }                                                          \
                }                                                            \
              }                                                              \
            }                                                                \
          }                                                                  \
        }                                                                    \
      }                                                                      \
    }                                                                        \
                                                                             \
  exit:                                                                      \
    if (mublis_pool_checkin(pool_block)) {                                   \
      free(a_pack_buf);                                                      \
      free(b_pack_buf);                                                      \
      free(c_buf);                                                           \
    }                                                                        \
                                                                             \
    return 0;                                                                \
  }

MUBLIS_TRMM_IMPL(
  float, mublis_strmm, mublis_spackm, MUBLIS_POOL_S, s
)
MUBLIS_TRMM_IMPL(
  double, mublis_dtrmm, mublis_dpackm, MUBLIS_POOL_D, d
)
