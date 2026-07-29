#include "aarch64.h"
#include "mublis_instantiate.h"
#include "gemm_ukr.inc"
#include "gemmtrsm_ukr.inc"

MUBLIS_CONTEXT_STAMP(
  aarch64_context, // name here should match its name in registry

  mublis_sgemm_ukr_firestorm,
  mublis_sgemmtrsml_ukr_firestorm,
  mublis_sgemmtrsmu_ukr_firestorm,
  S_MR, S_NR,
  720, 960, 640,

  mublis_dgemm_ukr_firestorm,
  mublis_dgemmtrsml_ukr_firestorm,
  mublis_dgemmtrsmu_ukr_firestorm,
  D_MR, D_NR,
  480, 720, 512
)
