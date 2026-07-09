#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "pool.h"

int num_blocks_to_allocate = EXPECTED_LIVE_THREADS;

typedef struct freelist {
  mublis_pool_block_t block;
  struct freelist *next;
} freelist_t;

struct {
  freelist_t *freelists[4];
  pthread_mutex_t lock;
} pool;

static void freelist_destroy(freelist_t *head) {
  while (head) {
    freelist_t *next = head->next;
    free(head->block.buf);
    free(head);
    head = next;
  }
}

static int freelist_build(mublis_pool_role_t role, size_t block_bytes, int num_blocks, freelist_t **out) {
  freelist_t *head = NULL;
  for (int i = 0; i < num_blocks; i++) {
    freelist_t *node = malloc(sizeof(freelist_t));
    if (!node) {
      freelist_destroy(head);
      return 1;
    }

    if (posix_memalign(&(node->block.buf), MIN_BLOCK_ALIGNMENT, block_bytes)) {
      free(node);
      freelist_destroy(head);
      return 1;
    }

    node->block.role = role;
    node->next = head;
    head = node;
  }

  *out = head;
  return 0;
}

int mublis_pool_init(const mublis_context_t *context) {
  if (pthread_mutex_init(&(pool.lock), NULL)) {
    fprintf(stderr, "Error: mutex initialization failed.\n");
    return 1;
  }

  const size_t sbuf_a_bytes = (size_t)context->s.mc * context->s.kc * sizeof(float);
  const size_t sbuf_b_bytes = (size_t)context->s.nc * context->s.kc * sizeof(float);
  const size_t dbuf_a_bytes = (size_t)context->d.mc * context->d.kc * sizeof(double);
  const size_t dbuf_b_bytes = (size_t)context->d.nc * context->d.kc * sizeof(double);

  if (freelist_build(MUBLIS_POOL_SA, sbuf_a_bytes, num_blocks_to_allocate, &(pool.freelists[MUBLIS_POOL_SA])) ||
      freelist_build(MUBLIS_POOL_SB, sbuf_b_bytes, num_blocks_to_allocate, &(pool.freelists[MUBLIS_POOL_SB])) ||
      freelist_build(MUBLIS_POOL_DA, dbuf_a_bytes, num_blocks_to_allocate, &(pool.freelists[MUBLIS_POOL_DA])) ||
      freelist_build(MUBLIS_POOL_DB, dbuf_b_bytes, num_blocks_to_allocate, &(pool.freelists[MUBLIS_POOL_DB]))) {
    freelist_destroy(pool.freelists[MUBLIS_POOL_SA]);
    freelist_destroy(pool.freelists[MUBLIS_POOL_SB]);
    freelist_destroy(pool.freelists[MUBLIS_POOL_DA]);
    freelist_destroy(pool.freelists[MUBLIS_POOL_DB]);

    pthread_mutex_destroy(&pool.lock);
    fprintf(stderr, "Error: pool allocation failed.\n");
    return 1;
  }

  return 0;
}