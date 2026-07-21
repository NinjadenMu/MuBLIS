/**
 * @file implementation of `pool.h`
 * 
 * To allow for concurrent requests, all operations are guarded by a central 
 * mutex.  
 * 
 * At init, enough blocks are allocated to simultaneously satisfy 
 * `EXPECTED_LIVE_THREADS`.  Should there not be enough blocks available to 
 * satisfy all live threads, the pool will attempt to double its number of 
 * blocks until all requests can be satisfied.
 * 
 * See pool.h for more details.
 */

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pool.h"

typedef struct freelist {
  mublis_pool_block_t block;
  struct freelist *next;
} freelist_t;

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static bool pool_initialized = false;
// The version of the pool, incremented each time it's destroyed
static int pool_epoch = 0;

static struct {
  /*
   * Number of new blocks to allocate per expansion for each precision.
   * Initialized to EXPECTED_LIVE_THREADS and doubled whenever a freelist
   * is exhausted.
   */
  int num_blocks_to_allocate[MUBLIS_POOL_NUM_ROLES];

  // Number of bytes required per buffer type for each precision
  size_t a_pack_buf_bytes[MUBLIS_POOL_NUM_ROLES];
  size_t b_pack_buf_bytes[MUBLIS_POOL_NUM_ROLES];
  size_t c_buf_bytes[MUBLIS_POOL_NUM_ROLES];

  // Null-terminated freelists containing blocks that store pointers to buffers
  freelist_t *freelists[MUBLIS_POOL_NUM_ROLES];

  /*
   * Freelist nodes that have been consumed are cached here to be re-added to 
   * a freelist when blocks are checked back into the pool.
   * 
   * This means `mublis_pool_checkin` requires no dynamic memory allocation to 
   * add blocks back to a freelist.   
   * 
   * The block attribute of nodes cached here should be treated as garbage, 
   * and the next pointer is repurposed to point to the next empty node.
   */
  freelist_t *empty_node_cache;
} pool;

/**
 * @brief releases only the resources taken by list nodes (not their contents)
 * 
 * Frees memory used for nodes in the freelist, doesn't touch its contents, 
 * meaning buffers stored in the node's block are not freed.
 * 
 * Mostly useful for cleaning up `empty_node_cache`.
 */
static void node_list_destroy(freelist_t *head) {
  while (head) {
    freelist_t *next = head->next;
    free(head);
    head = next;
  }
}

/**
 * @brief frees buffers stored in a block
 */
static void block_destroy(mublis_pool_block_t block) {
  free(block.a_pack_buf);
  free(block.b_pack_buf);
  free(block.c_buf);
}

/**
 * @brief frees all resources associated with freelist, (nodes and buffers)
 * 
 * Frees all freelist nodes and the buffers stored in their blocks
 */
static void freelist_destroy(freelist_t *head) {
  while (head) {
    freelist_t *next = head->next;

    block_destroy(head->block);
    free(head);

    head = next;
  }
}

/**
 * @brief allocates new blocks and prepends them to the freelist
 * @param role the role of the new blocks, used to index freelists
 * @return 0 on success, 1 otherwise
 * 
 * Allocates buffers for `pool.num_blocks_to_allocate[role]` new blocks, 
 * prepends new blocks into role's existing freelist.
 * 
 * This function cleans up after itself on failure, and the original freelist 
 * is left unchanged.
 */
static int freelist_expand(mublis_pool_role_t role) {
  assert(pool.num_blocks_to_allocate[role] > 0);

  freelist_t *new_head = NULL;
  freelist_t *new_tail = NULL;

  for (int i = 0; i < pool.num_blocks_to_allocate[role]; ++i) {
    freelist_t *node = malloc(sizeof(*node));

    if (!node) {
      freelist_destroy(new_head);
      return 1;
    }

    node->block.a_pack_buf = NULL;
    node->block.b_pack_buf = NULL;
    node->block.c_buf = NULL;
    node->block.role = role;
    node->block.epoch = pool_epoch;

    if (
      posix_memalign(
        &node->block.a_pack_buf,
        MIN_BLOCK_ALIGNMENT,
        pool.a_pack_buf_bytes[role]
      ) ||
      posix_memalign(
        &node->block.b_pack_buf,
        MIN_BLOCK_ALIGNMENT,
        pool.b_pack_buf_bytes[role]
      ) ||
      posix_memalign(
        &node->block.c_buf,
        MIN_BLOCK_ALIGNMENT,
        pool.c_buf_bytes[role]
      )
    ) {
      block_destroy(node->block);
      free(node);
      freelist_destroy(new_head);
      return 1;
    }

    node->next = new_head;
    new_head = node;

    if (!new_tail)
      new_tail = node;
  }

  new_tail->next = pool.freelists[role];
  pool.freelists[role] = new_head;

  return 0;
}

/**
 * @brief core logic for `mublis_pool_init`
 * 
 * Relies on `mublis_pool_init` to handle locking.
 * See `mublis_pool_init` header for more info.
 */
static int mublis_pool_init_impl(const mublis_context_t *context) {
  pool.a_pack_buf_bytes[MUBLIS_POOL_S] =
    (size_t)context->s.mc * context->s.kc * sizeof(float);

  pool.b_pack_buf_bytes[MUBLIS_POOL_S] =
    (size_t)context->s.nc * context->s.kc * sizeof(float);

  pool.c_buf_bytes[MUBLIS_POOL_S] =
    (size_t)context->s.mr * context->s.nr * sizeof(float);

  pool.a_pack_buf_bytes[MUBLIS_POOL_D] =
    (size_t)context->d.mc * context->d.kc * sizeof(double);

  pool.b_pack_buf_bytes[MUBLIS_POOL_D] =
    (size_t)context->d.nc * context->d.kc * sizeof(double);

  pool.c_buf_bytes[MUBLIS_POOL_D] =
    (size_t)context->d.mr * context->d.nr * sizeof(double);

  /*
   * lists at init may contain garbage pointers which should not be freed 
   * again or expanded
  */
  for (int role = 0; role < MUBLIS_POOL_NUM_ROLES; role++) {
    pool.freelists[role] = NULL;
  }
  pool.empty_node_cache = NULL;

  bool allocation_failed = false;
  for (int role = 0; role < MUBLIS_POOL_NUM_ROLES; role++) {
    pool.num_blocks_to_allocate[role] = EXPECTED_LIVE_THREADS;

    if (freelist_expand(role)) {
      allocation_failed = true;
      break;
    }
  }

  if (allocation_failed) {
    for (int role = 0; role < MUBLIS_POOL_NUM_ROLES; role++) {
      freelist_destroy(pool.freelists[role]);
      pool.freelists[role] = NULL; 
    }

    return 1;
  }

  pool_initialized = true;
  return 0;
}

int mublis_pool_init(const mublis_context_t *context) {
  pthread_mutex_lock(&pool_lock);

  int error_code = 0;
  if (!pool_initialized) {
    error_code = mublis_pool_init_impl(context);
  }

  pthread_mutex_unlock(&pool_lock);

  return error_code;
}

void mublis_pool_destroy(void) {
  pthread_mutex_lock(&pool_lock);

  if (pool_initialized) {
    for (int role = 0; role < MUBLIS_POOL_NUM_ROLES; role++) {
      freelist_destroy(pool.freelists[role]);
      pool.freelists[role] = NULL;
    }

    node_list_destroy(pool.empty_node_cache);
    pool.empty_node_cache = NULL;

    pool_initialized = false;

    pool_epoch += 1;
  }

  pthread_mutex_unlock(&pool_lock);
}

int mublis_pool_checkout(mublis_pool_role_t role, mublis_pool_block_t *out) {
  pthread_mutex_lock(&pool_lock);

  int error_code = 0;
  if (pool_initialized) {
    // allocate new blocks to fulfill request if all existing blocks are in use
    if (!pool.freelists[role]) {
      error_code = freelist_expand(role);
      if (error_code) {
        pthread_mutex_unlock(&pool_lock);
        return error_code;
      }

      // allocate more blocks next time if we run out again
      pool.num_blocks_to_allocate[role] *= 2;
    }

    freelist_t *node = pool.freelists[role];
    *out = node->block;

    pool.freelists[role] = node->next;
    node->next = pool.empty_node_cache;
    pool.empty_node_cache = node;
  }
  else {
    fprintf(stderr, "Error: pool not initialized.\n");
    error_code = 1;
  }

  pthread_mutex_unlock(&pool_lock);

  return error_code;
}

int mublis_pool_checkin(mublis_pool_block_t block) {
  pthread_mutex_lock(&pool_lock);

  if (pool_initialized && block.epoch == pool_epoch) {
    assert(pool.empty_node_cache != NULL);

    freelist_t *node = pool.empty_node_cache;
    pool.empty_node_cache = node->next;
    node->block = block;
    node->next = pool.freelists[block.role];
    pool.freelists[block.role] = node;

    pthread_mutex_unlock(&pool_lock);
    return 0;
  }

  pthread_mutex_unlock(&pool_lock);
  return 1;
}
