#ifndef MUBLIS_CONTEXT_H
#define MUBLIS_CONTEXT_H

#include "kernels.h"

/**
 * @brief hardware context for either single or double precision datatype
 * 
 * Contains pointer to micro-kernel function and register and cache 
 * block dimensions.
 * 
 * Context is separated by precision because double precision block dimensions 
 * are half of those for single precision, and because  they require different 
 * micro-kernels.
 */
typedef struct {
  // ukr = micro-kernel, with the mu symbol collapsing to a u
  mublis_sgemm_ukr_ft gemm_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_scontext_t;
typedef struct {
  // ukr = micro-kernel, with the mu symbol collapsing to a u
  mublis_dgemm_ukr_ft gemm_ukr;

  int mr, nr;
  int mc, kc, nc;
} mublis_dcontext_t;

/**
 * @brief full hardware context
 * 
 * Contains a context object for single and double precision.
 */
typedef struct {
  mublis_scontext_t s;
  mublis_dcontext_t d;
} mublis_context_t;

/**
 * @brief Gets context object for runtime hardware specialization
 * @return 0 on success, nonzero integer otherwise.
 * 
 * Must be defined by user at compile-time in config/<config_name>/context.c.
 * MuBLIS uses this at runtime to get the hardware context used to dispatch 
 * to optimized implementations for the hardware.
 */
int mublis_get_context(mublis_context_t *context);

/**
 * @brief Gets and validates context using user provided `mublis_get_context`
 * @return 1 if context values are invalid, otherwise, passes through the 
 * return value of `mublis_get_context`
 * 
 * Calls on `mublis_get_context` defined by user, and checks if values are 
 * valid.
 * 
 * Creates a guarded persistent context object, so repeated calls (including  
 * from different threads) only need to use `mublis_get_context` once.
 */
int mublis_get_safe_context(const mublis_context_t **context);

#endif
