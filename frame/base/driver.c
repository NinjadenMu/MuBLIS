#include "stdbool.h"

#include "kernels.h"
#include "l3.h"
#include "packm.h"
#include "pool.h"

static inline bool relation_holds(mublis_l3_relation_t relation, int lhs, int rhs) {
  switch (relation) {
    case MUBLIS_L3_ALL:
      return true;
    case MUBLIS_L3_LOWER:
      return lhs <= rhs;
    case MUBLIS_L3_UPPER:
      return lhs >= rhs;
  }
}

static bool block_is_outside(
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

static inline mublis_l3_relation_t flip_relation(mublis_l3_relation_t relation) {
  switch (relation) {
    case MUBLIS_L3_ALL:
      return MUBLIS_L3_ALL;
    case MUBLIS_L3_LOWER:
      return MUBLIS_L3_UPPER;
    case MUBLIS_L3_UPPER:
      return MUBLIS_L3_LOWER;
  }
}

int mublis_l3_sdriver(
  int m, int n, int k,
  float alpha,
  const float *a, int rs_a, int cs_a,
  const float *b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c,
  const mublis_l3_product_t *product,
  const mublis_context_t *context
) {
  if (m == 0 || n == 0) {
    return 0;
  }

  int error_code;
  // Once the pool is initialized, this is a no-op
  if (error_code = mublis_pool_init(context)) {
    return error_code;
  }

  int mr = context->s.mr;
  int nr = context->s.nr;
  int mc = context->s.mc;
  int kc = context->s.kc;
  int nc = context->s.nc;

  mublis_l3_domain_t domain = product->domain;

  for (int jc = 0; jc < n; jc += nc) {
    int j_max = jc + nc < n ? jc + nc : n;
    for (int ic = 0; ic < m; ic += mc) {
      int i_max = ic + mc < m ? ic + mc : m;
      if (block_is_outside(domain.ji, jc, j_max - jc, ic, i_max - ic))
        continue;

      for (int j = jc; j < j_max; j++) {
        for (int i = ic; i < i_max; i++) {
          if (relation_holds(domain.ji, j, i)) {
            if (beta == 0)
              c[i * rs_c + j * cs_c] = 0.0f;
            else
              c[i * rs_c + j * cs_c] *= beta;
          }
        }
      }
    }
  }

  mublis_pool_block_t pool_block_a;
  mublis_pool_block_t pool_block_b;
  if (error_code = mublis_pool_checkout(MUBLIS_POOL_SA, &pool_block_a))
    return error_code;
  if (error_code = mublis_pool_checkout(MUBLIS_POOL_SB, &pool_block_b)) {
    mublis_pool_checkin(pool_block_a);
    return error_code;
  }

  for (int jc = 0; jc < n; jc += nc) {
    int j_max = jc + nc < n ? jc + nc : n;
    for (int pc = 0; pc < k; pc += kc) {
      int p_max = pc + kc < k ? pc + kc : n;
      if (block_is_outside(domain.jp, jc, j_max - jc, pc, p_max - pc))
        continue;

      mublis_spackm(
        pool_block_b.buf, 
        &b[pc * rs_b + jc * cs_b],
        cs_b, rs_b,
        j_max - jc, p_max - pc, nr,
        product->b.struc,
        domain.jp,
        product->b.diag,
        jc - pc
      );
      for (int ic = 0; ic < m; ic += mc) {
        int i_max = ic + mc < m ? ic + mc : m;
        if (block_is_outside(domain.ji, jc, j_max - jc, ic, i_max - ic))
          continue;
        if (block_is_outside(domain.pi, pc, p_max - pc, ic, i_max - ic))
            continue;

        mublis_spackm(
          pool_block_a.buf, 
          &a[ic * rs_a + pc * cs_a],
          rs_a, cs_a,
          i_max - ic, p_max - pc, mr,
          product->a.struc,
          flip_relation(domain.pi),
          product->a.diag,
          ic - pc
        );
        for (int jr = jc; jr < j_max; jr += nr) {
          if (block_is_outside(domain.jp, jr, nr, pc, p_max - pc))
            continue;
          if (block_is_outside(domain.ji, jr, nr, ic, i_max - ic))
            continue;

          for (int ir = ic; ir < i_max; ir += mr) {
            if (block_is_outside(domain.ji, jr, nr, ir, mr))
              continue;
            if (block_is_outside(domain.pi, pc, p_max - pc, ir, mr))
              continue;

            if (ir + mr <= i_max && jr + nr <= j_max) {
              ((mublis_sgemm_ukr_ft) context->s.gemm_ukr)(
                p_max - pc,
                alpha,
                &pool_block_a.buf[(ir - ic) * (p_max - pc)],
                &pool_block_b.buf[(jr - jc) * (p_max - pc)],
                1.0f,
                &c[ir * rs_c + jr * cs_c],
                rs_c, cs_c,
                NULL // TODO: pass pointers for prefetching
              );
            }
            else {
              // TODO: write to temp, then do a masked write to C
            }
          }
        }
      }
    }
  }

  mublis_pool_checkin(pool_block_a);
  mublis_pool_checkin(pool_block_b);
  return 0;
}