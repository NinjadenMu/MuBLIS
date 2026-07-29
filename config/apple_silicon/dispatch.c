#include "mublis_instantiate.h"
#include "target_registry.h"

int mublis_get_context(mublis_context_t *context) {
  *context = firestorm_context;

  return 0;
}
