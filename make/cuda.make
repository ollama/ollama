# Common definitions for all cuda versions

ifndef GPU_RUNNER_VARIANT
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif


GPU_RUNNER_NAME := cuda$(GPU_RUNNER_VARIANT)
GPU_RUNNER_GO_TAGS := cuda cuda$(GPU_RUNNER_VARIANT)
GPU_RUNNER_DRIVER_LIB_LINK := -lcuda
GPU_RUNNER_LIBS_SHORT := cublas cudart cublasLt

ifeq ($(OS),windows)
	# On windows, nvcc uses msvc which does not support avx512vbmi avx512vnni avx512bf16, but macros can turn them on
	GPU_VECTOR_FLAGS=$(call uc,$(filter-out avx512bf16,$(filter-out avx512vnni,$(filter-out avx512vbmi,$(GPU_RUNNER_CPU_FLAGS)))))
	GPU_COMPILER_EXTRA_FLAGS=$(if $(filter avx512vbmi,$(GPU_RUNNER_CPU_FLAGS)),-D__AVX512VBMI__)
	GPU_COMPILER_EXTRA_FLAGS+=$(if $(filter avx512vnni,$(GPU_RUNNER_CPU_FLAGS)),-D__AVX512VNNI__)
	GPU_COMPILER_EXTRA_FLAGS+=$(if $(filter avx512bf16,$(GPU_RUNNER_CPU_FLAGS)),-D__AVX512BF16__)
	GPU_LIBS = $(sort $(wildcard $(addsuffix *.$(SHARED_EXT),$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT)))))
	GPU_COMPILER_CFLAGS = $(CFLAGS) -D_WIN32_WINNT=0x602
	GPU_COMPILER_CXXFLAGS = $(CXXFLAGS) -D_WIN32_WINNT=0x602
else ifeq ($(OS),linux)
	# On linux, nvcc requires avx512 -> -mavx512f -mavx512dq -mavx512bw
	GPU_VECTOR_FLAGS=$(if $(filter avx512,$(GPU_RUNNER_CPU_FLAGS)),avx512f avx512dq avx512bw) $(filter-out avx512,$(GPU_RUNNER_CPU_FLAGS))
	GPU_COMPILER_EXTRA_FLAGS = -fPIC -Wno-unused-function -std=c++17
	GPU_LIBS = $(sort $(wildcard $(addsuffix *.$(SHARED_EXT).*,$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT)))))
	GPU_COMPILER_CFLAGS = $(CFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
	GPU_COMPILER_CXXFLAGS = $(CXXFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
endif
GPU_DIST_LIB_DEPS= $(sort $(addprefix $(DIST_GPU_RUNNER_DEPS_DIR)/,$(notdir $(GPU_LIBS))))

GPU_RUNNER_ARCH_FLAGS := $(foreach arch,$(subst ;,$(space),$(CUDA_ARCHITECTURES)),--generate-code=arch=compute_$(arch)$(comma)code=[compute_$(arch)$(comma)sm_$(arch)]) \
	-DGGML_CUDA_USE_GRAPHS=1
GPU_COMPILER_CUFLAGS = \
	$(GPU_COMPILER_EXTRA_FLAGS) \
	-Xcompiler "$(addprefix $(CPU_FLAG_PREFIX),$(GPU_VECTOR_FLAGS))" \
	-t2 \
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
	-Wno-deprecated-gpu-targets \
	--forward-unknown-to-host-compiler \
	-use_fast_math \
	-I./llama/  \
	-O3
