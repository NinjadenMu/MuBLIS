#include "avx2.h"
#include "mublis_instantiate.h"
#include "gemm_ukr.inc"
#include "gemmtrsm_ukr.inc"

MUBLIS_CONTEXT_STAMP(
  avx2_context, // name here should match its name in registry

  mublis_sgemm_ukr_avx2,
  mublis_sgemmtrsml_ukr_avx2,
  mublis_sgemmtrsmu_ukr_avx2,
  S_MR, S_NR,
  144, 256, 512,

  mublis_dgemm_ukr_avx2,
  mublis_dgemmtrsml_ukr_avx2,
  mublis_dgemmtrsmu_ukr_avx2,
  D_MR, D_NR,
  72, 256, 512
)
