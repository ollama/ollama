OS := $(shell uname -s)
ARCH := $(or $(ARCH), $(shell uname -m))
ifeq ($(ARCH),x86_64)
	ARCH := amd64
endif
ifneq (,$(findstring MINGW,$(OS))$(findstring MSYS,$(OS)))
	OS := windows
else ifeq ($(OS),Linux)
	OS := linux
else ifeq ($(OS),Darwin)
	OS := darwin
endif

export CGO_CFLAGS_ALLOW = -mfma|-mf16c
export CGO_CXXFLAGS_ALLOW = -mfma|-mf16c
export HIP_PLATFORM = amd

SRC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(SRC_DIR)/build/$(OS)-$(ARCH)/
DIST_DIR := $(abspath $(SRC_DIR)/../dist/$(OS)-$(ARCH))/
RUNNERS_DIR := $(DIST_DIR)ollama_runners/
ifeq ($(OS),darwin)
DEFAULT_RUNNER_DIR := $(RUNNERS_DIR)metal/
else
DEFAULT_RUNNER_DIR := $(RUNNERS_DIR)cpu/
endif
CPU_AVX_RUNNER_DIR := $(RUNNERS_DIR)cpu_avx/
CPU_AVX2_RUNNER_DIR := $(RUNNERS_DIR)cpu_avx2/
CUDA_RUNNER_DIR := $(RUNNERS_DIR)cuda/
ROCM_RUNNER_DIR := $(RUNNERS_DIR)rocm/

ifeq ($(OS),windows)
	OBJ_EXT := obj
	SHARED_EXT := dll
	EXE_EXT := .exe
	SHARED_PREFIX := ""
	CUDA_LIB := $(shell cygpath -w -s "$(CUDA_PATH)\lib\x64")
	HIP_LIB := $(shell cygpath -w -s "$(HIP_PATH)\lib")
	NVCC := nvcc
	# If HIP_PATH has spaces, hipcc trips over them when subprocessing
	HIP_PATH := $(shell cygpath -m -s "$(HIP_PATH)\")
	export HIP_PATH
	HIPCC := $(HIP_PATH)bin/hipcc.bin.exe
else ifeq ($(OS),linux)
	OBJ_EXT := o
	SHARED_EXT := so
	SHARED_PREFIX := lib
	CUDA_PATH?=/usr/local/cuda
	HIP_PATH?=/opt/rocm
	CUDA_LIB := "$(CUDA_PATH)/lib64"
	HIP_LIB := "$(HIP_PATH)/lib"
	NVCC := nvcc
	HIPCC := hipcc
else
	OBJ_EXT := o
	SHARED_EXT := so
endif

CUDA_SRCS := \
	ggml-cuda.cu \
	$(wildcard ggml-cuda/*.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-wmma*.cu) \
	$(wildcard ggml-cuda/template-instances/mmq*.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*f16-f16.cu) \
	ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp

CUDA_OBJS := $(CUDA_SRCS:.cu=.cuda.$(OBJ_EXT))
CUDA_OBJS := $(CUDA_OBJS:.c=.cuda.$(OBJ_EXT))
CUDA_OBJS := $(addprefix $(BUILD_DIR),$(CUDA_OBJS:.cpp=.cuda.$(OBJ_EXT)))

HIP_OBJS := $(CUDA_SRCS:.cu=.hip.$(OBJ_EXT))
HIP_OBJS := $(HIP_OBJS:.c=.hip.$(OBJ_EXT))
HIP_OBJS := $(addprefix $(BUILD_DIR),$(HIP_OBJS:.cpp=.hip.$(OBJ_EXT)))

CUDA_FLAGS := \
	--generate-code=arch=compute_50,code=[compute_50,sm_50] \
	--generate-code=arch=compute_52,code=[compute_52,sm_52] \
	--generate-code=arch=compute_61,code=[compute_61,sm_61] \
	--generate-code=arch=compute_70,code=[compute_70,sm_70] \
	--generate-code=arch=compute_75,code=[compute_75,sm_75] \
	--generate-code=arch=compute_80,code=[compute_80,sm_80] \
	-DGGML_CUDA_DMMV_X=32 \
	-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
	-DGGML_USE_CUDA=1 \
	-DGGML_SHARED=1 \
	-DGGML_BUILD=1 \
	-DGGML_USE_LLAMAFILE \
	-D_GNU_SOURCE \
	-DCMAKE_POSITION_INDEPENDENT_CODE=on \
	-Wno-deprecated-gpu-targets \
	--forward-unknown-to-host-compiler \
	-use_fast_math \
	-link \
	-shared \
	-I. \
	-O3

HIP_ARCHS := gfx900 gfx940 gfx941 gfx942 gfx1010 gfx1012 gfx1030 gfx1100 gfx1101 gfx1102
LINUX_HIP_ARCHS := gfx906:xnack- gfx908:xnack- gfx90a:xnack+ gfx90a:xnack-

HIP_FLAGS := \
	-c \
	-O3 \
	-DGGML_USE_CUDA \
	-DGGML_BUILD=1 \
	-DGGML_SHARED=1 \
	-DGGML_CUDA_DMMV_X=32 \
	-DGGML_CUDA_MMV_Y=1 \
	-DGGML_SCHED_MAX_COPIES=4 \
	-DGGML_USE_HIPBLAS \
	-DGGML_USE_LLAMAFILE \
	-DHIP_FAST_MATH \
	-DNDEBUG \
	-DK_QUANTS_PER_ITERATION=2 \
	-D_CRT_SECURE_NO_WARNINGS \
	-DCMAKE_POSITION_INDEPENDENT_CODE=on \
	-D_GNU_SOURCE \
	-Wno-expansion-to-defined \
	-Wno-invalid-noreturn \
	-Wno-ignored-attributes \
	-Wno-pass-failed \
	-Wno-deprecated-declarations \
	-Wno-unused-result \
	-I. \
	$(foreach arch, $(HIP_ARCHS), --offload-arch=$(arch))

ifeq ($(OS),linux)
	HIP_FLAGS += $(foreach arch, $(LINUX_HIP_ARCHS), --offload-arch=$(arch)) -fPIC
	CUDA_FLAGS += -fPIC
	CFLAGS += -Xcompiler -fPIC -D_GNU_SOURCE
	CXXFLAGS += -Xcompiler -fPIC -D_GNU_SOURCE
else ifeq ($(OS),windows)
	HIP_FLAGS += -Xclang --dependent-lib=msvcrt
endif

RUNNERS := $(DEFAULT_RUNNER_DIR)ollama_runner$(EXE_EXT)
ifeq ($(ARCH),amd64)
	RUNNERS += $(CPU_AVX_RUNNER_DIR)ollama_runner$(EXE_EXT) $(CPU_AVX2_RUNNER_DIR)ollama_runner$(EXE_EXT)
endif
ifneq ($(NVCC),)
	RUNNERS += $(CUDA_RUNNER_DIR)ollama_runner$(EXE_EXT)
endif
ifneq ($(HIPCC),)
	RUNNERS += $(ROCM_RUNNER_DIR)ollama_runner$(EXE_EXT)
endif

runners: $(RUNNERS)

$(BUILD_DIR)%.cuda.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(NVCC) -c $(CUDA_FLAGS) -o $@ $<

$(BUILD_DIR)%.cuda.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(NVCC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)%.cuda.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(NVCC) -c $(CXXFLAGS) -o $@ $<

$(CUDA_RUNNER_DIR)$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT): $(BUILD_DIR)$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT)
	@-mkdir -p $(dir $@)
	cp $< $@

$(BUILD_DIR)$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT): $(CUDA_OBJS)
	@-mkdir -p $(dir $@)
	$(NVCC) --shared -lcuda -lcublas -lcudart -lcublasLt $(CUDA_FLAGS) $(CUDA_OBJS) -o $@

$(BUILD_DIR)%.hip.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(HIP_FLAGS) -o $@ $<

$(BUILD_DIR)%.hip.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)%.hip.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(CXXFLAGS) -o $@ $<

$(ROCM_RUNNER_DIR)$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT): $(BUILD_DIR)$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT)
	@-mkdir -p $(dir $@)
	cp $< $@

$(BUILD_DIR)$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT): $(HIP_OBJS)
	@-mkdir -p $(dir $@)
	$(HIPCC) --shared -lhipblas -lamdhip64 -lrocblas $(HIP_OBJS) -o $@

$(DEFAULT_RUNNER_DIR)ollama_runner$(EXE_EXT):
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -o $@ ./runner

$(CPU_AVX_RUNNER_DIR)ollama_runner$(EXE_EXT):
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -tags avx -o $@ ./runner

$(CPU_AVX2_RUNNER_DIR)ollama_runner$(EXE_EXT):
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -tags avx,avx2 -o $@ ./runner

$(CUDA_RUNNER_DIR)ollama_runner$(EXE_EXT): $(CUDA_RUNNER_DIR)$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT)
	CGO_ENABLED=1 GOARCH=$(ARCH) CGO_LDFLAGS=-L"$(CUDA_LIB)" go build -ldflags "-s -w" -tags avx,cuda -o $@ ./runner

$(ROCM_RUNNER_DIR)ollama_runner$(EXE_EXT): $(ROCM_RUNNER_DIR)$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT)
	CGO_ENABLED=1 GOARCH=$(ARCH) CGO_LDFLAGS=-L"$(HIP_LIB)" go build -ldflags "-s -w" -tags avx,rocm -o $@ ./runner

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)

.PHONY: runners clean

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
