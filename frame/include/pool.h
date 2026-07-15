/**
 * @file pool.h
 * @brief allocator for packed Level 3 driver workspaces
 *
 * All BLAS routines pack panels of A and B into buffers for the micro-kernel.
 * Routines also usually need a temporary C buffer from which masked writes 
 * can be made when micro-tiles are out-of-bounds.
 * 
 * MuBLIS has a central mutex-guarded checkout/checkin pool for these 
 * buffers so routines don't have to allocate their own memory.  
 * It also allows for reuse of allocated memory.
 * 
 * A "block" contains pointers to the 3 buffers (`a_pack_buf`, `b_pack_buf`, 
 * `c_buf`), so users can simply checkin and checkout blocks that contain 
 * all the bufers they'll need.
 */

#ifndef MUBLIS_POOL_H
#define MUBLIS_POOL_H

#include "context.h"

/*
 * Maximum expected number of threads expected to be concurrently making 
 * calls to BLAS/MuBLIS routines
*/
#define EXPECTED_LIVE_THREADS 8
/* 
 * Minimum alignment in bytes expected for buffers
 * Usually decided by vectorization strategy
*/
#define MIN_BLOCK_ALIGNMENT 64

typedef enum {
  // Use to request blocks for single precision operations
  MUBLIS_POOL_S = 0,

  // Use to request blocks for single precision operations
  MUBLIS_POOL_D,

  MUBLIS_POOL_NUM_ROLES
} mublis_pool_role_t;

/**
 * All BLAS and MuBLIS operations require buffers for packed A, packed B, and 
 * a temp C tile.
 * 
 * These 3 buffers are grouped into a block, so routines simply request a 
 * block from the pool.
 */
typedef struct {
  void *a_pack_buf;
  void *b_pack_buf;
  void *c_buf;

  mublis_pool_role_t role;
} mublis_pool_block_t;

/**
 * @brief Initializes memory pool
 *
 * Returns 0 on success, 1 otherwise.
 * 
 * On success, the pool is ready for checkin/checkout.
 * 
 * Doesn't read the pool's state when called, so an existing pool should 
 * be destroyed to free resources before another init call.
 */
int mublis_pool_init(const mublis_context_t *context);

/**
 * @brief Frees resources used by memory pool
 * 
 * After being called, blocks allocated by the user are detached and must have 
 * their buffers freed manually (don't attempt to check them back in to the 
 * destroyed pool.)
 * 
 * In-flight checkout requests may fail if a destroy request is issued 
 * concurrently.
 */
void mublis_pool_destroy(void);

/**
 * @brief request memory for one BLAS/MuBLIS operation
 * 
 * Writes into block at `out` with buffers for packed A, packed B, and a 
 * temporary C tile.
 * 
 * While checked-out, the buffers are the users responsibility and must be 
 * checked back in or freed manually if the pool is destroyed.
 */
int mublis_pool_checkout(
  mublis_pool_role_t role,
  mublis_pool_block_t *out
);

/**
 * @brief Checks memory back into pool
 * 
 * Can only fail if the pool is destroyed by another thread concurrently 
 * to the checkin request.  In this case, the buffers should be manually freed.
 */
int mublis_pool_checkin(mublis_pool_block_t block);

#endif
