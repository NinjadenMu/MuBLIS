MuBLIS instantiates BLAS libraries based on a config.  

A configuration defines which targets are linked into a MuBLIS library and which one is selected at runtime. This separation allows one library to contain multiple hardware targets while keeping ISA-specific code isolated in those targets.

Configurations are discovered automatically from `config/{config name}/config.mk`, and may be selected with `make CONFIG={config name}`.

The `reference` configuration is a minimal (and portable) example.

### Config Code
A C source file in `config/{config name}` must define `mublis_get_context` (signature in `include/mublis_instantiate.h`).  That function is responsible for selecting the target context MuBLIS's frame should specialize to at runtime.  

For configs targeting a single micro-architecture or hardware feature, this is a fairly trivial function - just select the context representing that single target.  

For configs targeting a family with multiple hardware features, a more sophisticated dispatch function is more appropriate, for example, selecting from many contexts depending on detected CPU ID, checking for support for ISA extensions, checking cache sizes, etc.

Feature detection must run safely on every platform supported by the configuration.  Note that this is why configs separate on OS - Mac, Linux, Windows, etc. expose different methods for detecting CPU features.

It may be smart to include `reference` as a fallback for broad hardware families.

### Config Build Rules
Create `config/<name>/config.mk` with a dispatch source (defining `mublis_get_context`) and at least one target to be linked into the lbirary (name of specific target subdirectory in `targets`).

This should be done by defining `CONFIG_DISPATCH_SOURCE` and `CONFIG_TARGETS`.  

`CONFIG_TARGET_TRIPLE` is optional and supplies Clang's target triple.  

Optional flags for the dispatch source are:
- `CONFIG_DISPATCH_FLAGS`
- `CONFIG_DISPATCH_CPPFLAGS`
- `CONFIG_DISPATCH_CFLAGS`

Flags should be valid for all platforms included in the config.
