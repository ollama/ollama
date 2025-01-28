# Common definitions for the various Makefiles which set musa settings
# No rules are defined here so this is safe to include at the beginning of other makefiles

ifeq ($(OS),linux)
	MUSA_PATH?=/usr/local/musa
	MUSA_1_PATH:=$(shell ls -d $(MUSA_PATH) 2>/dev/null)
	MUSA_1_COMPILER:=$(wildcard $(MUSA_1_PATH)/bin/clang)
	MUSA_1_LIB_DIR=$(strip $(shell ls -d $(MUSA_1_PATH)/lib 2>/dev/null))
	MUSA_1_CGO_EXTRA_LDFLAGS = -L"$(MUSA_1_LIB_DIR)"
endif
