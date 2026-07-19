#include <pthread.h>
#include <stdbool.h>

#include "context.h"

static pthread_once_t context_once = PTHREAD_ONCE_INIT;
static mublis_context_t cached_context;
static int cached_result;

static bool context_dt_is_valid(const mublis_context_dt_t *context) {
  return
    context->gemm_ukr != NULL &&
    context->mr > 0 &&
    context->nr > 0 &&
    context->mc > 0 &&
    context->kc > 0 &&
    context->nc > 0 &&
    context->mc % context->mr == 0 &&
    context->nc % context->nr == 0;
}

static void initialize_context(void) {
  mublis_context_t candidate = {0};

  cached_result = mublis_get_context(&candidate);
  if (cached_result != 0)
    return;

  if (
    !context_dt_is_valid(&candidate.s) ||
    !context_dt_is_valid(&candidate.d)
  ) {
    cached_result = 1;
    return;
  }

  cached_context = candidate;
}

int mublis_get_safe_context(const mublis_context_t **context) {
  if (context == NULL)
    return 1;

  pthread_once(&context_once, initialize_context);

  if (cached_result == 0)
    *context = &cached_context;

  return cached_result;
}
