#include "mublis_instantiate.h"
#include "gemm_ukr.inc"

MUBLIS_CONTEXT_STAMP(
  reference_context, 
  mublis_sgemm_ukr_reference,
  S_MR, S_NR,
  64, 64, 64,
  mublis_dgemm_ukr_reference,
  D_MR, D_NR,
  32, 32, 32
)
