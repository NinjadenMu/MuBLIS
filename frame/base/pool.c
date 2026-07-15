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

static struct {
  // Number of new blocks to allocate per expansion for each request type, 
  // set to `EXPECTED_LIVE_THREADS` at init and doubled each time a freelist 
  // is exhausted
  int num_blocks_to_allocate[MUBLIS_POOL_NUM_ROLES];

  // Number of bytes required to fulfill each request type
  size_t buf_bytes[MUBLIS_POOL_NUM_ROLES];

  // Every possible request has a freelist implemented as a linked 
  // list, with each node storing a block that can fulfill the request.
  freelist_t *freelists[MUBLIS_POOL_NUM_ROLES];

  // Freelist nodes that have been consumed are cached here to be re-added to 
  // a freelist when blocks are checked back into the pool.
  // This means `mublis_pool_checkin` requires no dynamic memory allocation to 
  // add blocks back to a freelist.   
  // The block attribute of nodes cached here should be treated as garbage, 
  // and the next pointer is repurposed to point to the next empty node.
  freelist_t *empty_node_cache;
} pool;

static void node_list_destroy(freelist_t *head) {
  while (head) {
    freelist_t *next = head->next;
    free(head);
    head = next;
  }
}

static void freelist_destroy(freelist_t *head) {
  for (freelist_t *n = head; n; n = n->next)
    free(n->block.buf);

  node_list_destroy(head);
}

static int freelist_expand(mublis_pool_role_t role) {
  assert(pool.num_blocks_to_allocate[role] > 0);

  freelist_t *new_head = NULL;
  freelist_t *new_tail = NULL;
  for (int i = 0; i < pool.num_blocks_to_allocate[role]; i++) {
    freelist_t *node = malloc(sizeof(freelist_t));
    if (!node) {
      freelist_destroy(new_head);
      return 1;
    }

    if (posix_memalign(&(node->block.buf), MIN_BLOCK_ALIGNMENT, pool.buf_bytes[role])) {
      free(node);
      freelist_destroy(new_head);
      return 1;
    }

    node->block.role = role;
    node->next = new_head;
    new_head = node;
    if (!new_tail) {
      new_tail = new_head;
    }
  }

  new_tail->next = pool.freelists[role];

  pool.freelists[role] = new_head;
  return 0;
}

static int mublis_pool_init_impl(const mublis_context_t *context) {
  pool.buf_bytes[MUBLIS_POOL_SA] = (size_t)context->s.mc * context->s.kc * sizeof(float);
  pool.buf_bytes[MUBLIS_POOL_SB] = (size_t)context->s.nc * context->s.kc * sizeof(float);
  pool.buf_bytes[MUBLIS_POOL_DA] = (size_t)context->d.mc * context->d.kc * sizeof(double);
  pool.buf_bytes[MUBLIS_POOL_DB] = (size_t)context->d.nc * context->d.kc * sizeof(double);

  for (int r = 0; r < MUBLIS_POOL_NUM_ROLES; r++) {
    pool.num_blocks_to_allocate[r] = EXPECTED_LIVE_THREADS;
  }

  if (freelist_expand(MUBLIS_POOL_SA) ||
      freelist_expand(MUBLIS_POOL_SB) ||
      freelist_expand(MUBLIS_POOL_DA) ||
      freelist_expand(MUBLIS_POOL_DB)) {
    for (int r = 0; r < MUBLIS_POOL_NUM_ROLES; r++) {
      freelist_destroy(pool.freelists[r]);
      pool.freelists[r] = NULL;
    }

    fprintf(stderr, "Error: pool allocation failed.\n");
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
    for (int r = 0; r < MUBLIS_POOL_NUM_ROLES; r++) {
      freelist_destroy(pool.freelists[r]);
      pool.freelists[r] = NULL;
    }

    node_list_destroy(pool.empty_node_cache);
    pool.empty_node_cache = NULL;

    pool_initialized = false;
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

void mublis_pool_checkin(mublis_pool_block_t block) {
  pthread_mutex_lock(&pool_lock);

  if (pool_initialized) {
    assert(pool.empty_node_cache != NULL);

    freelist_t *node = pool.empty_node_cache;
    pool.empty_node_cache = node->next;
    node->block = block;
    node->next = pool.freelists[block.role];
    pool.freelists[block.role] = node;
  }

  pthread_mutex_unlock(&pool_lock);
}
