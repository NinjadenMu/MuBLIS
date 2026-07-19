#ifndef MUBLIS_KERNELS_H
#define MUBLIS_KERNELS_H

#include "types.h"

typedef struct {
  const void *a_next;
  const void *b_next;
} mublis_auxinfo_t;

typedef void (*mublis_sgemm_ukr_ft)(
  int k,
  float alpha,
  const float *a,
  const float *b,
  const float beta,
  float *c,
  int rs_c, int cs_c,
  const mublis_auxinfo_t *aux
);

typedef void (*mublis_dgemm_ukr_ft)(
  int k,
  double alpha,
  const double *a,
  const double *b,
  const double beta,
  double *c,
  int rs_c, int cs_c,
  const mublis_auxinfo_t *aux
);

#define MUBLIS_GEMM_UKR_PROT(ctype, name) \
  void name(                              \
    int k,                                \
    ctype alpha,                          \
    const ctype *a,                       \
    const ctype *b,                       \
    ctype beta,                           \
    ctype *c,                             \
    int rs_c, int cs_c,                   \
    const mublis_auxinfo_t *aux           \
  );

MUBLIS_GEMM_UKR_PROT(float, mublis_sgemm_ukr_reference)
MUBLIS_GEMM_UKR_PROT(double, mublis_dgemm_ukr_reference)

#endif