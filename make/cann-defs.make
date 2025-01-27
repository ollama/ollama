# Common definitions for the various Makefiles which set cann settings
# No rules are defined here so this is safe to include at the beginning of other makefiles

ifeq ($(OS),linux)
	CANN_INSTALL_DIR?=$(shell ls -d /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null)
ifeq ($(strip $(CANN_INSTALL_DIR)),)
	CANN_INSTALL_DIR=$(shell ls -d "$(HOME)/Ascend/ascend-toolkit/latest" 2>/dev/null)
endif
endif
