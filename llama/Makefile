# top level makefile for Go server
include make/common-defs.make

RUNNER_TARGETS := default

# Determine which if any GPU runners we should build
ifeq ($(OS),windows)
	CUDA_PATH?=$(shell cygpath -m -s "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" 2>/dev/null)unknown
	CUDA_BASE_DIR := $(dir $(shell cygpath -m -s "$(CUDA_PATH)\\.." 2>/dev/null))
	CUDA_11:=$(shell ls -d $(CUDA_BASE_DIR)/v11.? 2>/dev/null)
	CUDA_12:=$(shell ls -d $(CUDA_BASE_DIR)/v12.? 2>/dev/null)
	HIP_PATH_83 := $(shell cygpath -m -s "$(subst \,/,$(HIP_PATH))" 2>/dev/null)
	HIP_LIB_DIR := $(shell ls -d $(HIP_PATH_83)/lib 2>/dev/null)
else ifeq ($(OS),linux)
	HIP_PATH?=/opt/rocm
	HIP_LIB_DIR := $(shell ls -d $(HIP_PATH)/lib 2>/dev/null)
	CUDA_PATH?=/usr/local/cuda
	CUDA_11:=$(shell ls -d $(CUDA_PATH)-11 2>/dev/null)
	CUDA_12:=$(shell ls -d $(CUDA_PATH)-12 2>/dev/null)
endif

ifeq ($(OLLAMA_SKIP_CUDA_GENERATE),)
ifneq ($(CUDA_11),)
	RUNNER_TARGETS += cuda_v11
endif
ifneq ($(CUDA_12),)
	RUNNER_TARGETS += cuda_v12
endif
endif
ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIP_LIB_DIR),)
	RUNNER_TARGETS += rocm
endif
endif


all: clean-payload .WAIT runners

runners: $(RUNNER_TARGETS)

$(RUNNER_TARGETS):
	$(MAKE) -f make/Makefile.$@

help-sync apply-patches create-patches sync:
	$(MAKE) -f make/Makefile.sync $@

clean:
	rm -rf $(BUILD_DIR) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)
	go clean -cache

clean-payload:
	rm -rf $(addprefix $(RUNNERS_PAYLOAD_DIR)/, $(RUNNER_TARGETS) metal cpu cpu_avx cpu_avx2)

.PHONY: all runners clean clean-payload $(RUNNER_TARGETS) .WAIT

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
