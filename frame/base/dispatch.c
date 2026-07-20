#include <pthread.h>
#include <stdbool.h>

#include "context.h"

static pthread_once_t context_once = PTHREAD_ONCE_INIT;
static mublis_context_t cached_context;
static int cached_result;

static bool context_is_valid(const mublis_context_t *context) {
  mublis_scontext_t scontext = context->s;
  mublis_dcontext_t dcontext = context->d;
  bool scontext_valid = scontext.gemm_ukr != NULL &&
                        scontext.mr > 0 &&
                        scontext.nr > 0 &&
                        scontext.mc > 0 &&
                        scontext.kc > 0 &&
                        scontext.nc > 0 &&
                        scontext.mc % scontext.mr == 0 &&
                        scontext.nc % scontext.nr == 0;
  bool dcontext_valid = dcontext.gemm_ukr != NULL &&
                        dcontext.mr > 0 &&
                        dcontext.nr > 0 &&
                        dcontext.mc > 0 &&
                        dcontext.kc > 0 &&
                        dcontext.nc > 0 &&
                        dcontext.mc % dcontext.mr == 0 &&
                        dcontext.nc % dcontext.nr == 0;

  return scontext_valid && dcontext_valid;
}

static void initialize_context(void) {
  mublis_context_t candidate = {0};

  cached_result = mublis_get_context(&candidate);
  if (cached_result != 0)
    return;

  if (!context_is_valid(&candidate)) {
    cached_result = 1;
    return;
  }

  cached_context = candidate;
}

int mublis_get_safe_context(const mublis_context_t **context) {
  if (context == NULL)
    return 1;

  pthread_once(&context_once, initialize_context);

  *context = NULL;
  if (cached_result == 0)
    *context = &cached_context;

  return cached_result;
}
