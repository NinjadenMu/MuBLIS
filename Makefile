.DEFAULT_GOAL := all

# Select with make CONFIG={config name} ...
CONFIG ?= reference

BUILD_ROOT ?= build

BUILD_SYSTEM_MAKEFILES := \
	Makefile \
	make/sources.mk \
	make/configure.mk \
	make/rules.mk

include make/sources.mk
include make/configure.mk
include make/rules.mk
