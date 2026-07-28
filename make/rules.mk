CC ?= cc
AR ?= ar

ARFLAGS := rcs

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

.PHONY: all lib clean format

all: lib

lib: $(LIB)

$(DISPATCH_OBJ): $(TARGET_REGISTRY)

$(ALL_OBJS): \
	$(BUILD_SYSTEM_MAKEFILES) \
	$(CONFIG_FILE) \
	$(TARGET_MK_FILES)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) \
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

$(LIB): $(ALL_OBJS)
	@mkdir -p "$(@D)"
	$(RM) "$@"
	$(AR) $(ARFLAGS) "$@" $(ALL_OBJS)

-include $(ALL_DEPS)

clean:
	rm -rf $(CONFIG_BUILD_DIR)

