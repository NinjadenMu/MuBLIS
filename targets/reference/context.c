#include "mublis_instantiate.h"
#include "reference.h"
#include "gemm_ukr.inc"
#include "gemmtrsm_ukr.inc"

MUBLIS_CONTEXT_STAMP(
  reference_context,

  mublis_sgemm_ukr_reference,
  mublis_sgemmtrsml_ukr_reference,
  mublis_sgemmtrsmu_ukr_reference,
  S_MR, S_NR,
  64, 64, 64,

  mublis_dgemm_ukr_reference,
  mublis_dgemmtrsml_ukr_reference,
  mublis_dgemmtrsmu_ukr_reference,
  D_MR, D_NR,
  32, 32, 32
)
