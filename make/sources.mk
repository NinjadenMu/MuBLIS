FRAME_C_DIRS := \
	frame/base \
	frame/l1m \
	frame/l3 \
	frame/compat/cblas

FRAME_C_SRCS := $(sort \
	$(foreach dir,$(FRAME_C_DIRS),$(wildcard $(dir)/*.c)) \
)
