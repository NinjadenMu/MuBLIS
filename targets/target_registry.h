#ifndef MUBLIS_TARGET_REGISTRY_H
#define MUBLIS_TARGET_REGISTRY_H

#include "mublis_instantiate.h"

#define REGISTER_TARGET_CONTEXT(name) extern const mublis_context_t name;

// Register target contexts by name here
REGISTER_TARGET_CONTEXT(reference_context)

#endif
