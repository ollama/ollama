# Common definitions for all musa versions

ifndef GPU_RUNNER_VARIANT
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif


GPU_RUNNER_NAME := musa$(GPU_RUNNER_VARIANT)
GPU_RUNNER_GO_TAGS := musa musa$(GPU_RUNNER_VARIANT)
GPU_RUNNER_DRIVER_LIB_LINK := -lmusa
GPU_RUNNER_LIBS_SHORT := mublas musart musa
GPU_LIB_DIR_LINUX = $(GPU_PATH_ROOT_LINUX)/lib
GPU_COMPILER_LINUX = $(GPU_PATH_ROOT_LINUX)/bin/clang
GPU_COMPILER_CFLAGS_LINUX = $(CFLAGS) -fPIC -D_GNU_SOURCE
GPU_COMPILER_CXXFLAGS_LINUX = $(CXXFLAGS) -fPIC -D_GNU_SOURCE
GPU_LIBS = $(sort $(wildcard $(addsuffix *.$(SHARED_EXT)*,$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT)))))
GPU_DIST_DEPS_LIBS= $(sort $(addprefix $(DIST_LIB_DIR)/,$(notdir $(GPU_LIBS))))

ifeq ($(OS),linux)
	MUSA_PATH?=/usr/local/musa
	GPU_COMPILER_FPIC = -fPIC -Wno-unused-function -std=c++11
endif
GPU_RUNNER_ARCH_FLAGS := $(foreach arch,$(subst ;,$(space),$(MUSA_ARCHITECTURES)),--cuda-gpu-arch=mp_$(arch)) \
	-x musa -mtgpu
GPU_COMPILER_CUFLAGS = \
	$(GPU_COMPILER_FPIC) \
	-DGGML_USE_MUSA=1 \
	-DGGML_CUDA_DMMV_X=32 \
	-DGGML_CUDA_MMV_Y=1 \
	-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
	-DGGML_USE_CUDA=1 \
	-DGGML_SHARED=1 \
	-DGGML_BUILD=1 \
	-DGGML_USE_LLAMAFILE \
	-DK_QUANTS_PER_ITERATION=2 \
	-DNDEBUG \
	-D_GNU_SOURCE \
	-D_XOPEN_SOURCE=600 \
	-I. \
	-O3
