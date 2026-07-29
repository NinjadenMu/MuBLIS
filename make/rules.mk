CC ?= cc
AR ?= ar

ARFLAGS := rcs

CC_TARGET_MODE ?= $(if $(findstring clang,$(shell $(CC) --version 2>/dev/null)),clang,fixed)

ifeq ($(CC_TARGET_MODE),clang)
CC_TARGET_FLAGS ?= \
	$(if $(strip $(TARGET_TRIPLE)),--target=$(strip $(TARGET_TRIPLE)))
else
CC_TARGET_FLAGS ?=
endif

COMMON_TARGET_FLAGS := $(strip $(CC_TARGET_FLAGS))

ifeq ($(TARGET_OS),Darwin)
DYNAMIC_LDFLAGS ?= -dynamiclib
else
DYNAMIC_LDFLAGS ?= -shared
endif

CSTD ?= c17
CFLAGS ?= -O3
WARN_CFLAGS ?= -Wall -Wextra -Wpedantic

# Probably not necessary for static archives, can disable if you want
PIC_CFLAGS ?= -fPIC
THREAD_CFLAGS ?= -pthread
POSIX_CPPFLAGS ?= -D_POSIX_C_SOURCE=200112L # for posix_memalign

COMMON_CPPFLAGS := \
	$(POSIX_CPPFLAGS) \
	-Iinclude \
	-Iframe/include \
	-Iframe/l3 \
	-Itargets

COMMON_CFLAGS := \
	-std=$(CSTD) \
	$(PIC_CFLAGS) \
	$(THREAD_CFLAGS) \
	$(WARN_CFLAGS)

COMMON_ASFLAGS := \
	$(PIC_CFLAGS)

.PHONY: all lib static dynamic clean

all: lib

lib: $(STATIC_LIB) $(DYNAMIC_LIB)

static: $(STATIC_LIB)

dynamic: $(DYNAMIC_LIB)

$(ALL_OBJS): \
	$(BUILD_SYSTEM_MAKEFILES) \
	$(CONFIG_FILE) \
	$(TARGET_MK_FILES)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) \
		$(COMMON_TARGET_FLAGS) \
		$(COMMON_CPPFLAGS) \
		$(CPPFLAGS) \
		$(PRIVATE_CPPFLAGS) \
		$(COMMON_CFLAGS) \
		$(CFLAGS) \
		$(PRIVATE_FLAGS) \
		$(PRIVATE_CFLAGS) \
		-MMD -MP \
		-MF "$(@:.o=.d)" \
		-MT "$@" \
		-c "$<" \
		-o "$@"

$(OBJ_DIR)/%.o: %.S
	@mkdir -p "$(@D)"
	$(CC) \
		$(COMMON_TARGET_FLAGS) \
		$(COMMON_CPPFLAGS) \
		$(CPPFLAGS) \
		$(PRIVATE_CPPFLAGS) \
		$(COMMON_ASFLAGS) \
		$(ASFLAGS) \
		$(PRIVATE_FLAGS) \
		$(PRIVATE_ASFLAGS) \
		-MMD -MP \
		-MF "$(@:.o=.d)" \
		-MT "$@" \
		-c "$<" \
		-o "$@"

$(STATIC_LIB): $(ALL_OBJS)
	@mkdir -p "$(@D)"
	$(RM) "$@"
	$(AR) $(ARFLAGS) "$@" $(ALL_OBJS)

$(DYNAMIC_LIB): $(ALL_OBJS)
	@mkdir -p "$(@D)"
	$(RM) "$@"
	$(CC) \
		$(COMMON_TARGET_FLAGS) \
		$(DYNAMIC_LDFLAGS) \
		$(LDFLAGS) \
		$(THREAD_CFLAGS) \
		-o "$@" \
		$(ALL_OBJS) \
		$(LDLIBS)

-include $(ALL_DEPS)

clean:
	rm -rf $(CONFIG_BUILD_DIR)

