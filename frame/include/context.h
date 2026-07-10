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

/*
 * Must be defined by user in config/<config_name>/context.c.
 * MuBLIS calls this at runtime to get the hardware context used to dispatch 
 * to optimized implementations for the hardware.
 */
const mublis_context_t mublis_get_context(void);

#endif
