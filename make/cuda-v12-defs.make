# Common definitions for the various Makefiles which set cuda settings
# No rules are defined here so this is safe to include at the beginning of other makefiles

ifeq ($(OS),windows)
	CUDA_PATH?=$(shell cygpath -m -s "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" 2>/dev/null)unknown
	CUDA_BASE_DIR := $(dir $(shell cygpath -m -s "$(CUDA_PATH)\\.." 2>/dev/null))
	CUDA_12_PATH:=$(shell ls -d $(CUDA_BASE_DIR)/v12.? 2>/dev/null)
	CUDA_12_COMPILER:=$(wildcard $(CUDA_12_PATH)/bin/nvcc.exe)
	CUDA_12_LIB_DIR = $(strip $(shell ls -d $(CUDA_12_PATH)/bin 2>/dev/null))
	CUDA_12_CGO_EXTRA_LDFLAGS = -L"$(CUDA_12_PATH)/lib/x64"
else ifeq ($(OS),linux)
	CUDA_PATH?=/usr/local/cuda
	CUDA_12_PATH:=$(shell ls -d $(CUDA_PATH)-12 2>/dev/null)
	CUDA_12_COMPILER:=$(wildcard $(CUDA_12_PATH)/bin/nvcc)
	CUDA_12_LIB_DIR=$(strip $(shell ls -d $(CUDA_12_PATH)/lib64 2>/dev/null || ls -d $(CUDA_12_PATH)/lib 2>/dev/null))
	CUDA_12_CGO_EXTRA_LDFLAGS = -L"$(CUDA_12_LIB_DIR)" -L"$(CUDA_12_LIB_DIR)/stubs" 
endif
