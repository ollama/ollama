# Common definitions for all musa versions

ifndef GPU_RUNNER_VARIANT
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif


GPU_RUNNER_NAME := musa$(GPU_RUNNER_VARIANT)
GPU_RUNNER_GO_TAGS := musa musa$(GPU_RUNNER_VARIANT)
GPU_RUNNER_DRIVER_LIB_LINK := -lmusa
GPU_RUNNER_LIBS_SHORT := mublas musart musa

ifeq ($(OS),linux)
	GPU_COMPILER_EXTRA_FLAGS = -fPIC -Wno-unused-function -std=c++17
	GPU_LIBS = $(sort $(wildcard $(addsuffix *.$(SHARED_EXT)*,$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT)))))
	GPU_COMPILER_CFLAGS = $(CFLAGS) -fPIC -D_GNU_SOURCE
	GPU_COMPILER_CXXFLAGS = $(CXXFLAGS) -fPIC -D_GNU_SOURCE
endif
GPU_DIST_LIB_DEPS= $(sort $(addprefix $(DIST_GPU_RUNNER_DEPS_DIR)/,$(notdir $(GPU_LIBS))))

GPU_RUNNER_ARCH_FLAGS := $(foreach arch,$(subst ;,$(space),$(MUSA_ARCHITECTURES)),--cuda-gpu-arch=mp_$(arch)) \
	-x musa -mtgpu
GPU_COMPILER_CUFLAGS = \
	$(GPU_COMPILER_EXTRA_FLAGS) \
	-DGGML_USE_MUSA=1 \
	-DGGML_CUDA_DMMV_X=32 \
	-DGGML_CUDA_MMV_Y=1 \
	-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
	-DGGML_USE_CUDA=1 \
	-DGGML_SHARED=1 \
	-DGGML_BACKEND_SHARED=1 \
	-DGGML_BUILD=1 \
	-DGGML_BACKEND_BUILD=1 \
	-DGGML_USE_LLAMAFILE \
	-DK_QUANTS_PER_ITERATION=2 \
	-DNDEBUG \
	-D_GNU_SOURCE \
	-D_XOPEN_SOURCE=600 \
	-I./llama/ \
	-O3
