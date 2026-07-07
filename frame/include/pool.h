/**
 * @file pool.h
 * @brief interface for internal allocator responsible for packed buffers
 * 
 * All BLAS routines pack panels of A and B into buffers for the micro-kernel.
 * MuBLIS creates a central mutex-guarded checkout/checkin pool once for these 
 * panels so BLAS calls don't have to allocate their own memory.  
 * It also allows for reuse of allocated memory.  
 * 
 * This is possible since all allocations will be for A/B panels, whose sizes 
 * are fully determined by mc, nc, and kc and as such known immediately at 
 * runtime.  
 */

#ifndef MUBLIS_POOL_H
#define MUBLIS_POOL_Y

#include <stddef.h>
#include "context.h"

// All allocations are either for
typedef enum {
  MYBLIS_POOL_A = 0,
  MYBLIS_POOL_B = 1
} mublis_pool_role_t;

typedef struct {
  void *buf;
  mublis_pool_role_t role;
} mublis_pool_block_t;

void mublis_pool_init(const mublis_context_t *context);
void mublis_pool_destroy(const mublis_context_t *context);

mublis_pool_block_t mublis_pool_checkout(mublis_pool_role_t role);
void mublis_pool_checkin(mublis_pool_block_t block);

#endif
