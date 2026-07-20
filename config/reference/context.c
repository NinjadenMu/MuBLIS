#include "context.h"
#include "kernels.h"

int mublis_get_context(mublis_context_t *context) {
  mublis_scontext_t s_context = {
    .gemm_ukr = mublis_sgemm_ukr_reference,
    .mr = 8, .nr = 8, 
    .mc = 64, .kc = 64, .nc = 64
  };
  mublis_dcontext_t d_context = {
    .gemm_ukr = mublis_dgemm_ukr_reference,
    .mr = 4, .nr = 4, 
    .mc = 64, .kc = 64, .nc = 64
  };

  mublis_context_t out = {
    .s = s_context, .d = d_context
  };

  *context = out;

  return 0;
}
