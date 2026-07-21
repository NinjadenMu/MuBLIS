#include "stdbool.h"
#include "stdlib.h"

#include "l1m.h"
#include "l3.h"
#include "l3_utils.h"
#include "pool.h"
#include "types.h"

static int first_update_pc(
  mublis_l3_domain_t domain,
  int jr, int ir,
  int kc
) {
  int first_p = 0;

  if (domain.jp == MUBLIS_L3_LOWER)
    first_p = jr;

  if (domain.pi == MUBLIS_L3_UPPER && ir > first_p)
    first_p = ir;

  return first_p - first_p % kc;
}

static bool block_has_update(
  mublis_l3_domain_t domain,
  int jr, int ir,
  int nr, int mr,
  int k, int kc
) {
  int pc = first_update_pc(domain, jr, ir, kc);

  if (pc >= k)
    return false;

  int p_max = pc + kc < k ? pc + kc : k;

  if (block_is_outside(domain.jp, jr, nr, pc, p_max - pc))
    return false;

  if (block_is_outside(domain.pi, pc, p_max - pc, ir, mr))
    return false;

  return true;
}

int mublis_l3_sdriver(
  int m, int n, int k,
  float alpha,
  const float *restrict a, int rs_a, int cs_a,
  const float *restrict b, int rs_b, int cs_b,
  float beta,
  float *c, int rs_c, int cs_c,
  const mublis_l3_product_t *product
) {
  if (m == 0 || n == 0) {
    return 0;
  }

  int error_code;
  const mublis_context_t *context;
  if ((error_code = mublis_get_safe_context(&context)))
    return error_code;

  // Once the pool is initialized, this is a no-op
  if ((error_code = mublis_pool_init(context))) {
    return error_code;
  }

  int mr = context->s.mr;
  int nr = context->s.nr;
  int mc = context->s.mc;
  int kc = context->s.kc;
  int nc = context->s.nc;

  mublis_l3_domain_t domain = product->domain;

  mublis_pool_block_t pool_block;
  if (error_code = mublis_pool_checkout(MUBLIS_POOL_S, &pool_block))
    return error_code;
  float *a_pack_buf = pool_block.a_pack_buf;
  float *b_pack_buf = pool_block.b_pack_buf;
  float *c_buf = pool_block.c_buf;

  for (int jc = 0; jc < n; jc += nc) {
    int j_max = jc + nc < n ? jc + nc : n;
    for (int ic = 0; ic < m; ic += mc) {
      int i_max = ic + mc < m ? ic + mc : m;
      if (block_is_outside(domain.ji, jc, j_max - jc, ic, i_max - ic))
        continue;

      for (int jr = jc; jr < j_max; jr += nr) {
        if (block_is_outside(domain.ji, jr, nr, ic, i_max - ic))
          continue;

        for (int ir = ic; ir < i_max; ir += mr) {
          if (block_is_outside(domain.ji, jr, nr, ir, mr))
            continue;

          if (alpha != 0 &&
              block_has_update(domain, jr, ir, nr, mr, k, kc))
            continue;

          for (int j = jr; j < MIN(jr + nr, j_max); j++) {
            for (int i = ir; i < MIN(ir + mr, i_max); i++) {
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
    }
  }

  if (alpha == 0) {
    goto exit;
  }

  for (int jc = 0; jc < n; jc += nc) {
    int j_max = MIN(jc + nc, n);
    for (int pc = 0; pc < k; pc += kc) {
      int p_max = MIN(pc + kc, k);
      if (block_is_outside(domain.jp, jc, j_max - jc, pc, p_max - pc))
        continue;

      mublis_spackm(
        b_pack_buf, 
        &b[pc * rs_b + jc * cs_b],
        cs_b, rs_b,
        j_max - jc, p_max - pc, nr,
        product->b.struc,
        flip_uplo(product->b.uplo),
        product->b.diag,
        jc - pc
      );
      for (int ic = 0; ic < m; ic += mc) {
        int i_max = MIN(ic + mc, m);
        if (block_is_outside(domain.ji, jc, j_max - jc, ic, i_max - ic))
          continue;
        if (block_is_outside(domain.pi, pc, p_max - pc, ic, i_max - ic))
          continue;

        mublis_spackm(
          a_pack_buf, 
          &a[ic * rs_a + pc * cs_a],
          rs_a, cs_a,
          i_max - ic, p_max - pc, mr,
          product->a.struc,
          product->a.uplo,
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

            bool first_update =
              pc == first_update_pc(domain, jr, ir, kc);

            if (block_is_inside(domain.ji, jr, nr, ir, mr) &&
                ir + mr <= i_max && jr + nr <= j_max) {
              (context->s.gemm_ukr)(
                p_max - pc,
                alpha,
                &a_pack_buf[(ir - ic) * (p_max - pc)],
                &b_pack_buf[(jr - jc) * (p_max - pc)],
                first_update ? beta : 1.0f,
                &c[ir * rs_c + jr * cs_c],
                rs_c, cs_c,
                NULL // TODO: pass pointers for prefetching
              );
            }
            else {
              (context->s.gemm_ukr)(
                p_max - pc,
                alpha,
                &a_pack_buf[(ir - ic) * (p_max - pc)],
                &b_pack_buf[(jr - jc) * (p_max - pc)],
                0.0f,
                c_buf,
                1, mr,
                NULL // TODO: pass pointers for prefetching
              );

              for (int j = jr; j < MIN(jr + nr, j_max); j++) {
                for (int i = ir; i < MIN(ir + mr, i_max); i++) {
                  if (relation_holds(domain.ji, j, i)) {
                    float *cij = &c[i * rs_c + j * cs_c];
                    float value = c_buf[(i - ir) + (j - jr) * mr];

                    if (first_update) {
                      if (beta == 0)
                        *cij = value;
                      else
                        *cij = beta * *cij + value;
                    }
                    else {
                      *cij += value;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

exit:
  if (mublis_pool_checkin(pool_block)) {
    free(a_pack_buf);
    free(b_pack_buf);
    free(c_buf);
  }
    
  return 0;
}
