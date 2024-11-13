# top level makefile for Go server
include make/common-defs.make

RUNNER_TARGETS := default

# Determine which if any GPU runners we should build
include make/cuda-v11-defs.make
include make/cuda-v12-defs.make
include make/rocm-defs.make

# Without CUSTOM_CPU_FLAGS we default to build both v11 and v12 if present
ifeq ($(CUSTOM_CPU_FLAGS),)
ifeq ($(OLLAMA_SKIP_CUDA_GENERATE),)
ifneq ($(CUDA_11_COMPILER),)
	RUNNER_TARGETS += cuda_v11
endif
ifneq ($(CUDA_12_COMPILER),)
	RUNNER_TARGETS += cuda_v12
endif
endif
else # CUSTOM_CPU_FLAGS is set, we'll build only the latest cuda version detected
ifneq ($(CUDA_12),)
	RUNNER_TARGETS += cuda_v12
else ifneq ($(CUDA_11),)
	RUNNER_TARGETS += cuda_v11
endif
endif

ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIP_COMPILER),)
	RUNNER_TARGETS += rocm
endif
endif


all: runners exe

dist: $(addprefix dist_, $(RUNNER_TARGETS)) dist_exe

dist_%:
	@$(MAKE) --no-print-directory -f make/Makefile.$* dist

payload: $(addprefix payload_, $(RUNNER_TARGETS))

payload_%:
	@$(MAKE) --no-print-directory -f make/Makefile.$* payload

runners: $(RUNNER_TARGETS)

$(RUNNER_TARGETS):
	@$(MAKE) --no-print-directory -f make/Makefile.$@

exe dist_exe:
	@$(MAKE) --no-print-directory -f make/Makefile.ollama $@

help-sync apply-patches create-patches sync:
	@$(MAKE) --no-print-directory -f make/Makefile.sync $@

test integration lint:
	@$(MAKE) --no-print-directory -f make/Makefile.test $@

clean: clean-payload
	rm -rf $(BUILD_DIR) $(DIST_LIB_DIR)
	go clean -cache

clean-payload:
	rm -rf $(addprefix $(RUNNERS_PAYLOAD_DIR)/, $(RUNNER_TARGETS) metal cpu cpu_avx cpu_avx2)

help:
	@echo "The following make targets will help you BUILD Ollama"
	@echo ""
	@echo "	make all   		# (default target) Build Ollama llm subprocess runners, and the primary ollama executable"
	@echo "	make runners		# Build Ollama llm subprocess runners; after you may use 'go build .' to build the primary ollama exectuable"
	@echo "	make <runner>		# Build specific runners. Enabled: '$(RUNNER_TARGETS)'"
	@echo "	make payload		# Build the runners as payloads (Linux/Mac only)"
	@echo "	make dist		# Build the runners for distribution and gather dependencies"
	@echo "	make help-sync 		# Help information on vendor update targets"
	@echo "	make help-runners 	# Help information on runner targets"
	@echo ""
	@echo "The following make targets will help you TEST Ollama"
	@echo ""
	@echo "	make test   		# Run unit tests"
	@echo "	make integration	# Run integration tests.  You must 'make all' first"
	@echo "	make lint   		# Run lint and style tests"
	@echo ""
	@echo "For more information see 'docs/development.md'"
	@echo ""


help-runners:
	@echo "The following runners will be built based on discovered GPU libraries: '$(RUNNER_TARGETS)'"
	@echo "(On MacOS arm64 'default' is the metal runner.  For all other platforms 'default' is one or more CPU runners)"
	@echo ""
	@echo "GPU Runner CPU Flags: '$(GPU_RUNNER_CPU_FLAGS)'  (Override with CUSTOM_CPU_FLAGS)"
	@echo ""
	@echo "# CUDA_PATH sets the location where CUDA toolkits are present"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "	CUDA_11_PATH=$(CUDA_11_PATH)"
	@echo "	CUDA_11_COMPILER=$(CUDA_11_COMPILER)"
	@echo "	CUDA_12_PATH=$(CUDA_12_PATH)"
	@echo "	CUDA_12_COMPILER=$(CUDA_12_COMPILER)"
	@echo ""
	@echo "# HIP_PATH sets the location where the ROCm toolkit is present"
	@echo "HIP_PATH=$(HIP_PATH)"
	@echo "	HIP_COMPILER=$(HIP_COMPILER)"

.PHONY: all exe dist payload help help-sync help-runners test integration lint runners clean clean-payload $(RUNNER_TARGETS)

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
