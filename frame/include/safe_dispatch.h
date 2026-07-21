#ifndef MUBLIS_CONTEXT_H
#define MUBLIS_CONTEXT_H

#include "mublis_instantiate.h"

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
 * Subsequent calls are efficient and directly read the cached context object.
 */
int mublis_get_safe_context(const mublis_context_t **context);

#endif
