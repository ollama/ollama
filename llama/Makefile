OS := $(shell uname -s)
ARCH ?= $(subst aarch64,arm64,$(subst x86_64,amd64,$(shell uname -m)))
ifneq (,$(findstring MINGW,$(OS))$(findstring MSYS,$(OS)))
	OS := windows
	ARCH = $(shell systeminfo 2>/dev/null | grep "System Type" | grep ARM64 > /dev/null && echo "arm64" || echo "amd64" )
else ifeq ($(OS),Linux)
	OS := linux
else ifeq ($(OS),Darwin)
	OS := darwin
endif
comma:= ,
empty:=
space:= $(empty) $(empty)
uc = $(subst a,A,$(subst b,B,$(subst c,C,$(subst d,D,$(subst e,E,$(subst f,F,$(subst g,G,$(subst h,H,$(subst i,I,$(subst j,J,$(subst k,K,$(subst l,L,$(subst m,M,$(subst n,N,$(subst o,O,$(subst p,P,$(subst q,Q,$(subst r,R,$(subst s,S,$(subst t,T,$(subst u,U,$(subst v,V,$(subst w,W,$(subst x,X,$(subst y,Y,$(subst z,Z,$1))))))))))))))))))))))))))


export CGO_CFLAGS_ALLOW = -mfma|-mf16c
export CGO_CXXFLAGS_ALLOW = -mfma|-mf16c
export HIP_PLATFORM = amd
export CGO_ENABLED=1

SRC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR = $(SRC_DIR)build/$(OS)-$(ARCH)
DIST_BASE = $(abspath $(SRC_DIR)/../dist/$(OS)-$(ARCH))
DIST_LIB_DIR = $(DIST_BASE)/lib/ollama
RUNNERS_DIST_DIR = $(DIST_LIB_DIR)/runners
RUNNERS_PAYLOAD_DIR = $(abspath $(SRC_DIR)/../build/$(OS)/$(ARCH))
RUNNERS_BUILD_DIR = $(BUILD_DIR)/runners
DEFAULT_RUNNER := $(if $(and $(filter darwin,$(OS)),$(filter arm64,$(ARCH))),metal,cpu)
GZIP:=$(shell command -v pigz 2>/dev/null || echo "gzip")
ifneq ($(OS),windows)
	CCACHE:=$(shell command -v ccache 2>/dev/null || echo "")
endif
VERSION?=$(shell git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")

# Conditionally enable ccache for cgo builds too
ifneq ($(CCACHE),)
	CC=$(CCACHE) gcc
	CXX=$(CCACHE) g++
	export CC
	export CXX
endif


CUDA_LIBS_SHORT := cublas cudart cublasLt
ROCM_LIBS_SHORT := hipblas rocblas

# Override in environment space separated to tune GPU runner CPU vector flags
ifeq ($(ARCH),amd64)
# TODO may need a bit more work - setting 'GPU_RUNNER_CPU_FLAGS="avx avx2 avx512f avx512bw"' doesn't yield
# a system_info showing 'AVX512 = 1' so there may be additional macros that are needed in GGML
	GPU_RUNNER_CPU_FLAGS ?= avx
endif

GPU_GOFLAGS="-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$(VERSION)\" \"-X=github.com/ollama/ollama/llama.CpuFeatures=$(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS))\" $(TARGET_LDFLAGS)"
CPU_GOFLAGS="-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$(VERSION)\" \"-X=github.com/ollama/ollama/llama.CpuFeatures=$(subst $(space),$(comma),$(TARGET_CPU_FLAGS))\" $(TARGET_LDFLAGS)"

ifeq ($(OS),windows)
	SRC_DIR := $(shell cygpath -m -s "$(SRC_DIR)")
	OBJ_EXT := obj
	SHARED_EXT := dll
	EXE_EXT := .exe
	SHARED_PREFIX := 
	CPU_FLAG_PREFIX := /arch:
	CUDA_BASE_DIR := $(dir $(shell cygpath -m -s "$(CUDA_PATH)\.."))
	CUDA_11=$(shell ls -d $(CUDA_BASE_DIR)/v11.? 2>/dev/null)
	CUDA_12=$(shell ls -d $(CUDA_BASE_DIR)/v12.? 2>/dev/null)
	CUDA_11_LIB_DIR := $(CUDA_11)/bin
	CUDA_12_LIB_DIR := $(CUDA_12)/bin
	CUDA_11_STATIC_LIB_DIR := $(CUDA_11)/lib/x64
	CUDA_12_STATIC_LIB_DIR := $(CUDA_12)/lib/x64
	NVCC := $(shell X=$$(which nvcc 2>/dev/null) && cygpath -m -s "$$X")
	ifneq ($(HIP_PATH),)
		HIP_LIB_DIR := $(shell cygpath -m -s "$(HIP_PATH)\bin")
		# If HIP_PATH has spaces, hipcc trips over them when subprocessing
		HIP_PATH := $(shell cygpath -m -s "$(HIP_PATH)\")
		HIP_STATIC_LIB_DIR := $(shell cygpath -m -s "$(HIP_PATH)\lib")
		export HIP_PATH
		HIPCC := $(HIP_PATH)bin/hipcc.bin.exe
	endif
	CP := cp
	CUDA_LIBS = $(wildcard $(addsuffix 64*.$(SHARED_EXT),$(addprefix $(CUDA_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
	_OS_GPU_RUNNER_CPU_FLAGS=$(call uc,$(GPU_RUNNER_CPU_FLAGS))
else ifeq ($(OS),linux)
	CP := cp -af
	OBJ_EXT := o
	SHARED_EXT := so
	SHARED_PREFIX := lib
	CPU_FLAG_PREFIX := -m
	HIP_PATH?=/opt/rocm
	HIP_LIB_DIR := $(HIP_PATH)/lib
	HIPCC := $(shell X=$$(which hipcc 2>/dev/null) && echo $$X)
	CUDA_PATH?=/usr/local/cuda
	CUDA_11=$(shell ls -d $(CUDA_PATH)-11 2>/dev/null)
	CUDA_12=$(shell ls -d $(CUDA_PATH)-12 2>/dev/null)
	CUDA_11_LIB_DIR := $(CUDA_11)/lib64
	CUDA_12_LIB_DIR := $(CUDA_12)/lib64
	_OS_GPU_RUNNER_CPU_FLAGS=$(GPU_RUNNER_CPU_FLAGS)
else
	OBJ_EXT := o
	SHARED_EXT := so
	CPU_FLAG_PREFIX := -m
	CP := cp -af
	_OS_GPU_RUNNER_CPU_FLAGS=$(GPU_RUNNER_CPU_FLAGS)
endif

CUDA_11_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(CUDA_11_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
CUDA_12_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(CUDA_12_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
NVCC_11 = $(CUDA_11)/bin/nvcc
NVCC_12 = $(CUDA_12)/bin/nvcc

CUDA_DEPS_DIR = $(DIST_LIB_DIR)
# Too large for unified bundle, so keep isolated
ROCM_DEPS_DIR = $(abspath $(SRC_DIR)/../dist/$(OS)-$(ARCH)-rocm)/lib/ollama

ifeq ($(OLLAMA_SKIP_CUDA_GENERATE),)
ifneq ($(CUDA_11),)
	CUDA_11_VARIANT= _v11
	CUDA_11_LIB_DEPS = $(addprefix $(CUDA_DEPS_DIR)/,$(notdir $(CUDA_11_LIBS)))
endif
ifneq ($(CUDA_12),)
	CUDA_12_VARIANT= _v12
	CUDA_12_LIB_DEPS = $(addprefix $(CUDA_DEPS_DIR)/,$(notdir $(CUDA_12_LIBS)))
endif
endif
ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIPCC),)
	ROCM_VERSION := $(subst $(space),.,$(wordlist 1,2,$(subst .,$(space),$(word 3,$(subst -,$(space),$(filter HIP version: %,$(shell $(HIPCC) --version)))))))
    ifneq (,$(ROCM_VERSION))
		ROCM_VARIANT = _v$(ROCM_VERSION)
	endif
	ROCM_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT),$(addprefix $(HIP_LIB_DIR)/$(SHARED_PREFIX),$(ROCM_LIBS_SHORT))))
	ifeq ($(OS),linux)
		ROCM_TRANSITIVE_LIBS = $(shell ldd $(ROCM_LIBS) | grep "=>" | cut -f2 -d= | cut -f2 -d' '  | grep -e rocm -e amdgpu -e libtinfo -e libnuma -e libelf | sort -u )
	endif
	ROCM_LIB_DEPS = $(addprefix $(ROCM_DEPS_DIR)/,$(notdir $(ROCM_LIBS)) $(notdir $(ROCM_TRANSITIVE_LIBS)))
	ROCBLAS_DEP_MANIFEST = $(ROCM_DEPS_DIR)/rocblas/library/TensileManifest.txt
endif
endif

COMMON_SRCS := \
	$(wildcard *.c) \
	$(wildcard *.cpp)
COMMON_HDRS := \
	$(wildcard *.h) \
	$(wildcard *.hpp)

CUDA_SRCS := \
	ggml-cuda.cu \
	$(filter-out $(wildcard ggml-cuda/fattn*.cu),$(wildcard ggml-cuda/*.cu)) \
	$(wildcard ggml-cuda/template-instances/mmq*.cu) \
	ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp ggml-aarch64.c
CUDA_HDRS := \
	$(wildcard ggml-cuda/*.cuh)

CUDA_FLAGS := \
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

# Conditional flags and components to speed up developer builds
ifneq ($(OLLAMA_FAST_BUILD),)
	CUDA_FLAGS += 	\
		-DGGML_DISABLE_FLASH_ATTN
else
	CUDA_SRCS += \
		$(wildcard ggml-cuda/fattn*.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-wmma*.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*f16-f16.cu)
endif

CUDA_11_OBJS := $(CUDA_SRCS:.cu=.cuda.$(OBJ_EXT))
CUDA_11_OBJS := $(CUDA_11_OBJS:.c=.cuda.$(OBJ_EXT))
CUDA_11_OBJS := $(addprefix $(BUILD_DIR)/cuda_v11/,$(CUDA_11_OBJS:.cpp=.cuda.$(OBJ_EXT)))
CUDA_12_OBJS := $(CUDA_SRCS:.cu=.cuda.$(OBJ_EXT))
CUDA_12_OBJS := $(CUDA_12_OBJS:.c=.cuda.$(OBJ_EXT))
CUDA_12_OBJS := $(addprefix $(BUILD_DIR)/cuda_v12/,$(CUDA_12_OBJS:.cpp=.cuda.$(OBJ_EXT)))

HIP_OBJS := $(CUDA_SRCS:.cu=.hip.$(OBJ_EXT))
HIP_OBJS := $(HIP_OBJS:.c=.hip.$(OBJ_EXT))
HIP_OBJS := $(addprefix $(BUILD_DIR)/,$(HIP_OBJS:.cpp=.hip.$(OBJ_EXT)))

CUDA_V11_ARCHITECTURES?=50;52;53;60;61;62;70;72;75;80;86
CUDA_11_FLAGS :=$(foreach arch,$(subst ;,$(space),$(CUDA_V11_ARCHITECTURES)),--generate-code=arch=compute_$(arch)$(comma)code=[compute_$(arch)$(comma)sm_$(arch)])
CUDA_V12_ARCHITECTURES?=60;61;62;70;72;75;80;86;87;89;90;90a
CUDA_12_FLAGS := $(foreach arch,$(subst ;,$(space),$(CUDA_V12_ARCHITECTURES)),--generate-code=arch=compute_$(arch)$(comma)code=[compute_$(arch)$(comma)sm_$(arch)]) \
	-DGGML_CUDA_USE_GRAPHS=on

# TODO allow override with 
# AMDGPU_TARGETS=gfx900;gfx940
HIP_ARCHS := gfx900 gfx940 gfx941 gfx942 gfx1010 gfx1012 gfx1030 gfx1100 gfx1101 gfx1102
LINUX_HIP_ARCHS := gfx906:xnack- gfx908:xnack- gfx90a:xnack+ gfx90a:xnack-

HIP_FLAGS := \
	$(addprefix -m,$(GPU_RUNNER_CPU_FLAGS)) \
	-parallel-jobs=2 \
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
	HIP_FLAGS += $(foreach arch, $(LINUX_HIP_ARCHS), --offload-arch=$(arch)) -fPIC -Wno-unused-function
	CUDA_FLAGS += -fPIC -Wno-unused-function
	NVCC_CFLAGS = $(CFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
	NVCC_CXXFLAGS = $(CXXFLAGS) -Xcompiler -fPIC -D_GNU_SOURCE
	HIPCC_CFLAGS = $(CFLAGS) -fPIC -D_GNU_SOURCE
	HIPCC_CXXFLAGS = $(CXXFLAGS) -fPIC -D_GNU_SOURCE
else ifeq ($(OS),windows)
	HIP_FLAGS += -Xclang --dependent-lib=msvcrt
	CFLAGS += -D_WIN32_WINNT=0x602
	CXXFLAGS += -D_WIN32_WINNT=0x602
	NVCC_CFLAGS = $(CFLAGS)
	NVCC_CXXFLAGS = $(CXXFLAGS)
	HIPCC_CFLAGS = $(CFLAGS)
	HIPCC_CXXFLAGS = $(CXXFLAGS)
endif

ifeq ($(OLLAMA_SKIP_CPU_GENERATE),)
RUNNERS := $(DEFAULT_RUNNER)
ifeq ($(ARCH),amd64)
	RUNNERS += cpu_avx cpu_avx2
endif
endif
ifeq ($(OLLAMA_SKIP_CUDA_GENERATE),)
ifeq ($(OLLAMA_SKIP_CUDA_11_GENERATE),)
ifneq ($(CUDA_11),)
	RUNNERS += cuda_v11
endif
endif
ifeq ($(OLLAMA_SKIP_CUDA_12_GENERATE),)
ifneq ($(CUDA_12),)
	RUNNERS += cuda_v12
endif
endif
endif
ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIPCC),)
	RUNNERS += rocm$(ROCM_VARIANT)
endif
endif

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(RUNNERS)))
PAYLOAD_RUNNERS = $(addprefix $(RUNNERS_PAYLOAD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT).gz,$(RUNNERS)))
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(RUNNERS)))

all: clean-payload dist payload 

dist: $(DIST_RUNNERS) $(ROCBLAS_DEP_MANIFEST)

ifeq ($(OS),windows)
# Unused on windows as we don't cary the payloads in the go binary
payload:
else
payload: $(PAYLOAD_RUNNERS)
endif

runners: $(BUILD_RUNNERS)

$(BUILD_DIR)/cuda_v11/%.cuda.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_11) -c $(CUDA_FLAGS) $(CUDA_11_FLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v11/%.cuda.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_11) -c $(NVCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v11/%.cuda.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_11) -c $(NVCC_CXXFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_12) -c $(CUDA_FLAGS) $(CUDA_12_FLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_12) -c $(NVCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_12) -c $(NVCC_CXXFLAGS) -o $@ $<

$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(dir $@)
	cp $< $@

$(RUNNERS_DIST_DIR)/cuda_v11/ollama_llama_server$(EXE_EXT): $(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_cuda_v11.$(SHARED_EXT)
$(RUNNERS_DIST_DIR)/cuda_v12/ollama_llama_server$(EXE_EXT): $(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_cuda_v12.$(SHARED_EXT)

$(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_cuda_v11.$(SHARED_EXT): $(RUNNERS_BUILD_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda_v11.$(SHARED_EXT)
	cp $< $@
$(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_cuda_v12.$(SHARED_EXT): $(RUNNERS_BUILD_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda_v12.$(SHARED_EXT)
	cp $< $@

$(RUNNERS_BUILD_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda_v11.$(SHARED_EXT): $(CUDA_11_OBJS) $(CUDA_11_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_11) --shared -lcuda -L${CUDA_DEPS_DIR} $(foreach lib, $(CUDA_LIBS_SHORT), -l$(lib)) $(CUDA_FLAGS) $(CUDA_11_FLAGS) $(CUDA_11_OBJS) -o $@

$(RUNNERS_BUILD_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda_v12.$(SHARED_EXT): $(CUDA_12_OBJS) $(CUDA_12_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(NVCC_12) --shared -lcuda -L${CUDA_DEPS_DIR} $(foreach lib, $(CUDA_LIBS_SHORT), -l$(lib)) $(CUDA_FLAGS) $(CUDA_12_FLAGS) $(CUDA_12_OBJS) -o $@

$(CUDA_11_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(CUDA_11_LIB_DIR)/$(notdir $@)* $(dir $@)

$(CUDA_12_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(CUDA_12_LIB_DIR)/$(notdir $@)* $(dir $@)

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(HIPCC) -c $(HIP_FLAGS) -o $@ $<

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(HIPCC) -c $(HIPCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(HIPCC) -c $(HIPCC_CXXFLAGS) -o $@ $<

$(RUNNERS_DIST_DIR)/rocm$(ROCM_VARIANT)/ollama_llama_server$(EXE_EXT): $(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT)

$(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT): $(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT)
	@-mkdir -p $(dir $@)
	cp $< $@

$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT): $(HIP_OBJS) $(ROCM_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(HIPCC) --shared -lamdhip64 -L${ROCM_DEPS_DIR} $(foreach lib, $(ROCM_LIBS_SHORT), -l$(lib)) $(HIP_OBJS) -o $@

$(ROCM_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(dir $(filter %$(notdir $@),$(ROCM_LIBS) $(ROCM_TRANSITIVE_LIBS)))/$(notdir $@)* $(dir $@)

$(RUNNERS_BUILD_DIR)/$(DEFAULT_RUNNER)/ollama_llama_server$(EXE_EXT): TARGET_CPU_FLAGS=
$(RUNNERS_BUILD_DIR)/$(DEFAULT_RUNNER)/ollama_llama_server$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) go build $(CPU_GOFLAGS) -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cpu_avx/ollama_llama_server$(EXE_EXT): TARGET_CPU_FLAGS="avx"
$(RUNNERS_BUILD_DIR)/cpu_avx/ollama_llama_server$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) go build $(CPU_GOFLAGS) -tags $(subst $(space),$(comma),$(TARGET_CPU_FLAGS)) -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cpu_avx2/ollama_llama_server$(EXE_EXT): TARGET_CPU_FLAGS="avx avx2"
$(RUNNERS_BUILD_DIR)/cpu_avx2/ollama_llama_server$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) go build $(CPU_GOFLAGS) -tags $(subst $(space),$(comma),$(TARGET_CPU_FLAGS)) -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cuda_v11/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = -L"$(RUNNERS_BUILD_DIR)/cuda_v11/" -L"$(CUDA_11_STATIC_LIB_DIR)"
$(RUNNERS_BUILD_DIR)/cuda_v11/ollama_llama_server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda_v11.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build $(GPU_GOFLAGS) -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) cuda cuda11) -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cuda_v12/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = -L"$(RUNNERS_BUILD_DIR)/cuda_v12/" -L"$(CUDA_12_STATIC_LIB_DIR)"
$(RUNNERS_BUILD_DIR)/cuda_v12/ollama_llama_server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda_v12.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build $(GPU_GOFLAGS) -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) cuda cuda12) -o $@ ./runner

$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = -L"$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/" -L"$(HIP_STATIC_LIB_DIR)"
$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/ollama_llama_server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build $(GPU_GOFLAGS) -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) rocm) -o $@ ./runner


$(ROCBLAS_DEP_MANIFEST):
	@-mkdir -p $(dir $@)
	@echo "Copying rocblas library..."
	cd $(HIP_LIB_DIR)/rocblas/library/ && tar cf - . | (cd $(dir $@) && tar xf - )
	@echo "rocblas library copy complete"

$(RUNNERS_PAYLOAD_DIR)/%/ollama_llama_server.gz: $(RUNNERS_BUILD_DIR)/%/ollama_llama_server 
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/cuda_v11/%.gz: $(RUNNERS_BUILD_DIR)/cuda_v11/%
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/cuda_v12/%.gz: $(RUNNERS_BUILD_DIR)/cuda_v12/%
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/rocm$(ROCM_VARIANT)/%.gz: $(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/%
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@

clean:
	rm -rf $(BUILD_DIR) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS) $(RUNNERS_PAYLOAD_DIR)

clean-payload:
	rm -rf $(addprefix $(RUNNERS_PAYLOAD_DIR)/, $(RUNNERS))
	mkdir -p $(addprefix $(RUNNERS_PAYLOAD_DIR)/, $(RUNNERS))

.PHONY: all dist payload runners clean clean-payload $(RUNNERS)

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
