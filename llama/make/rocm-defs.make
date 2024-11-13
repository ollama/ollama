# Common definitions for the various Makefiles which set cuda settings
# No rules are defined here so this is safe to include at the beginning of other makefiles

ifeq ($(OS),windows)
	HIP_COMPILER:=$(wildcard $(HIP_PATH)/bin/hipcc.bin.exe)
else ifeq ($(OS),linux)
	HIP_PATH?=$(shell ls -d /opt/rocm 2>/dev/null)
	HIP_COMPILER:=$(wildcard $(HIP_PATH)/bin/hipcc)
endif
