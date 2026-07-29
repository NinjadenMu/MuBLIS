# All configs must define targets and dispatch source in config.mk
AVAILABLE_CONFIGS := $(sort \
	$(patsubst config/%/config.mk,%,$(wildcard config/*/config.mk)) \
)

CONFIG_FILE := config/$(CONFIG)/config.mk

ifeq ($(wildcard $(CONFIG_FILE)),)
$(error Unknown CONFIG "$(CONFIG)". Available configurations: $(AVAILABLE_CONFIGS))
endif

include $(CONFIG_FILE)

ifeq ($(strip $(CONFIG_DISPATCH_SRC)),)
$(error $(CONFIG_FILE) must set CONFIG_DISPATCH_SRC)
endif

ifeq ($(strip $(CONFIG_TARGETS)),)
$(error $(CONFIG_FILE) must set CONFIG_TARGETS)
endif

# Start processing targets after validating config

# All targets must define target.mk
TARGET_MK_FILES := $(foreach target,$(CONFIG_TARGETS),\
	targets/$(target)/target.mk \
)

MISSING_TARGET_MKS := $(filter-out \
	$(wildcard $(TARGET_MK_FILES)),\
	$(TARGET_MK_FILES) \
)

ifneq ($(strip $(MISSING_TARGET_MKS)),)
$(error Missing target descriptions: $(MISSING_TARGET_MKS))
endif

include $(TARGET_MK_FILES)

# All targets must define at least one C or assembly source
$(foreach target,$(CONFIG_TARGETS),\
	$(if $(strip \
		$(TARGET_$(target)_C_SRCS) \
		$(TARGET_$(target)_S_SRCS) \
	),,\
		$(error Target "$(target)" does not list any sources)))

# Start processing source files after validating targets

CONFIG_BUILD_DIR := $(BUILD_ROOT)/$(CONFIG)
OBJ_DIR := $(CONFIG_BUILD_DIR)/obj
LIB_DIR := $(CONFIG_BUILD_DIR)/lib

HOST_OS ?= $(shell uname -s)

ifeq ($(HOST_OS),Darwin)
DYNAMIC_LIB_SUFFIX ?= dylib
else
DYNAMIC_LIB_SUFFIX ?= so
endif

STATIC_LIB := $(LIB_DIR)/libmublis.a
DYNAMIC_LIB := $(LIB_DIR)/libmublis.$(DYNAMIC_LIB_SUFFIX)

c_to_o = $(patsubst %.c,$(OBJ_DIR)/%.o,$(1))
S_to_o = $(patsubst %.S,$(OBJ_DIR)/%.o,$(1))

FRAME_OBJS := $(call c_to_o,$(FRAME_C_SRCS))
DISPATCH_OBJ := $(call c_to_o,$(CONFIG_DISPATCH_SRC))

TARGET_C_SRCS := $(foreach target,$(CONFIG_TARGETS),\
	$(TARGET_$(target)_C_SRCS) \
)

TARGET_S_SRCS := $(foreach target,$(CONFIG_TARGETS),\
	$(TARGET_$(target)_S_SRCS) \
)

TARGET_C_OBJS := $(call c_to_o,$(TARGET_C_SRCS))
TARGET_S_OBJS := $(call S_to_o,$(TARGET_S_SRCS))

ALL_C_SRCS := \
	$(FRAME_C_SRCS) \
	$(CONFIG_DISPATCH_SRC) \
	$(TARGET_C_SRCS)

ALL_S_SRCS := $(TARGET_S_SRCS)

ALL_SRCS := $(ALL_C_SRCS) $(ALL_S_SRCS)

ALL_OBJS := \
	$(FRAME_OBJS) \
	$(DISPATCH_OBJ) \
	$(TARGET_C_OBJS) \
	$(TARGET_S_OBJS)

ALL_DEPS := $(ALL_OBJS:.o=.d)

MISSING_SRCS := $(filter-out $(wildcard $(ALL_SRCS)),$(ALL_SRCS))

ifneq ($(strip $(MISSING_SRCS)),)
$(error Required sources do not exist: $(MISSING_SRCS))
endif

# TARGET_<name>_FLAGS applies to both C and .S files.
# TARGET_<name>_CFLAGS applies only to C.
# TARGET_<name>_ASFLAGS applies only to .S.
# TARGET_<name>_CPPFLAGS applies to both preprocessing steps.
define apply_target_c_flags
$(call c_to_o,$(TARGET_$(1)_C_SRCS)): PRIVATE_FLAGS += $(TARGET_$(1)_FLAGS)
$(call c_to_o,$(TARGET_$(1)_C_SRCS)): PRIVATE_CPPFLAGS += $(TARGET_$(1)_CPPFLAGS)
$(call c_to_o,$(TARGET_$(1)_C_SRCS)): PRIVATE_CFLAGS += $(TARGET_$(1)_CFLAGS)
endef

define apply_target_S_flags
$(call S_to_o,$(TARGET_$(1)_S_SRCS)): PRIVATE_FLAGS += $(TARGET_$(1)_FLAGS)
$(call S_to_o,$(TARGET_$(1)_S_SRCS)): PRIVATE_CPPFLAGS += $(TARGET_$(1)_CPPFLAGS)
$(call S_to_o,$(TARGET_$(1)_S_SRCS)): PRIVATE_ASFLAGS += $(TARGET_$(1)_ASFLAGS)
endef

$(foreach target,$(CONFIG_TARGETS),\
	$(if $(strip $(TARGET_$(target)_C_SRCS)),\
		$(eval $(call apply_target_c_flags,$(target)))))

$(foreach target,$(CONFIG_TARGETS),\
	$(if $(strip $(TARGET_$(target)_S_SRCS)),\
		$(eval $(call apply_target_S_flags,$(target)))))

$(DISPATCH_OBJ): PRIVATE_FLAGS += $(CONFIG_DISPATCH_FLAGS)
$(DISPATCH_OBJ): PRIVATE_CPPFLAGS += $(CONFIG_DISPATCH_CPPFLAGS)
$(DISPATCH_OBJ): PRIVATE_CFLAGS += $(CONFIG_DISPATCH_CFLAGS)
