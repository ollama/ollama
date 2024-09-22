# Common definitions for all cuda versions

ifndef GPU_RUNNER_VARIANT
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif


GPU_RUNNER_NAME := cuda$(GPU_RUNNER_VARIANT)
GPU_RUNNER_GO_TAGS := cuda cuda$(GPU_RUNNER_VARIANT)
GPU_RUNNER_DRIVER_LIB_LINK := -lcuda
GPU_RUNNER_LIBS_SHORT := cublas cudart cublasLt
GPU_LIB_DIR_WIN = $(GPU_PATH_ROOT_WIN)/bin
GPU_LIB_DIR_LINUX = $(GPU_PATH_ROOT_LINUX)/lib64
CGO_EXTRA_LDFLAGS_WIN = -L"$(GPU_PATH_ROOT_WIN)/lib/x64"
GPU_COMPILER_WIN = $(GPU_PATH_ROOT_WIN)/bin/nvcc
GPU_COMPILER_LINUX = $(GPU_PATH_ROOT_LINUX)/bin/nvcc
GPU_COMPILER_CFLAGS_WIN = $(CFLAGS) -D_WIN32_WINNT=0x602
GPU_COMPILER_CFLAGS_LINUX = $(CFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
GPU_COMPILER_CXXFLAGS_WIN = $(CXXFLAGS) -D_WIN32_WINNT=0x602
GPU_COMPILER_CXXFLAGS_LINUX = $(CXXFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
ifeq ($(OS),linux)
	CUDA_PATH?=/usr/local/cuda
	GPU_COMPILER_FPIC = -fPIC -Wno-unused-function
endif
GPU_RUNNER_ARCH_FLAGS := $(foreach arch,$(subst ;,$(space),$(CUDA_ARCHITECTURES)),--generate-code=arch=compute_$(arch)$(comma)code=[compute_$(arch)$(comma)sm_$(arch)]) \
	-DGGML_CUDA_USE_GRAPHS=on
GPU_COMPILER_CUFLAGS = \
	$(GPU_COMPILER_FPIC) \
	-Xcompiler "$(addprefix $(CPU_FLAG_PREFIX),$(_OS_GPU_RUNNER_CPU_FLAGS))" \
	-t2 \
	-DGGML_CUDA_DMMV_X=32 \
	-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
	-DGGML_USE_CUDA=1 \
	-DGGML_SHARED=1 \
	-DGGML_BUILD=1 \
	-DGGML_USE_LLAMAFILE \
	-DNDEBUG \
	-D_GNU_SOURCE \
	-DCMAKE_POSITION_INDEPENDENT_CODE=on \
	-Wno-deprecated-gpu-targets \
	--forward-unknown-to-host-compiler \
	-use_fast_math \
	-link \
	-shared \
	-I. \
	-O3
