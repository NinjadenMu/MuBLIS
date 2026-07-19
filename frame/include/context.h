#ifndef MUBLIS_CONTEXT_H
#define MUBLIS_CONTEXT_H

/**
 * @brief hardware context for either single or double precision datatype
 * 
 * Contains void* pointer to micro-kernel function and register and cache 
 * block dimensions.
 * 
 * Context is separated by precision because double precision block dimensions 
 * are half of those for single precision, and because  they require different 
 * micro-kernels.
 */
typedef struct {
  // ukr = micro-kernel, with the mu symbol collapsing to a u
  void *gemm_ukr; // cast to mublis_{s, d}gemm_ukr_ft

  int mr, nr;
  int mc, kc, nc;
} mublis_context_dt_t;

/**
 * @brief full hardware context
 * 
 * Contains a context object for single and double precision.
 */
typedef struct {
  mublis_context_dt_t s;
  mublis_context_dt_t d;
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
 */
int mublis_get_safe_context(mublis_context_t *context);

#endif
