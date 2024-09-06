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
comma:= ,
empty:=
space:= $(empty) $(empty)

export CGO_CFLAGS_ALLOW = -mfma|-mf16c
export CGO_CXXFLAGS_ALLOW = -mfma|-mf16c
export HIP_PLATFORM = amd

SRC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR = $(SRC_DIR)build/$(OS)-$(ARCH)
DIST_BASE = $(abspath $(SRC_DIR)/../dist/$(OS)-$(ARCH))
RUNNERS_DIST_DIR = $(DIST_BASE)/ollama_runners
RUNNERS_PAYLOAD_DIR = $(abspath $(SRC_DIR)/../llm/build/$(OS)/$(patsubst amd64,x86_64,$(ARCH)))
RUNNERS_BUILD_DIR = $(BUILD_DIR)/ollama_runners
DEFAULT_RUNNER := $(if $(and $(filter darwin,$(OS)),$(filter arm64,$(ARCH))),metal,cpu)

CUDA_LIBS_SHORT := cublas cudart cublasLt
ROCM_LIBS_SHORT := hipblas rocblas

ifeq ($(OS),windows)
	SRC_DIR := $(shell cygpath -m -s "$(SRC_DIR)")
	OBJ_EXT := obj
	SHARED_EXT := dll
	EXE_EXT := .exe
	SHARED_PREFIX := 
	
	# TODO needs work for multiple cuda versions on windows

	CUDA_BASE_DIR := $(dir $(shell cygpath -m -s "$(CUDA_PATH)\.."))
	CUDA_11=$(shell ls -d $(CUDA_BASE_DIR)/v11.? 2>/dev/null)
	CUDA_12=$(shell ls -d $(CUDA_BASE_DIR)/v12.? 2>/dev/null)
	CUDA_11_LIB_DIR := $(CUDA_11)/bin
	CUDA_12_LIB_DIR := $(CUDA_12)/bin


	NVCC := $(shell X=$$(which nvcc 2>/dev/null) && cygpath -m -s "$$X")
	ifneq ($(HIP_PATH),)
		HIP_LIB_DIR := $(shell cygpath -m -s "$(HIP_PATH)\bin")
		# If HIP_PATH has spaces, hipcc trips over them when subprocessing
		HIP_PATH := $(shell cygpath -m -s "$(HIP_PATH)\")
		export HIP_PATH
		HIPCC := $(HIP_PATH)bin/hipcc.bin.exe
	endif
	CP := cp
	CUDA_LIBS = $(wildcard $(addsuffix 64*.$(SHARED_EXT),$(addprefix $(CUDA_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
else ifeq ($(OS),linux)
	CP := cp -a
	OBJ_EXT := o
	SHARED_EXT := so
	SHARED_PREFIX := lib
	HIP_PATH?=/opt/rocm
	HIP_LIB_DIR := $(HIP_PATH)/lib
	HIPCC := $(shell X=$$(which hipcc 2>/dev/null) && echo $$X)
	CUDA_PATH?=/usr/local/cuda
	CUDA_11=$(shell ls -d $(CUDA_PATH)-11 2>/dev/null)
	CUDA_12=$(shell ls -d $(CUDA_PATH)-12 2>/dev/null)
	CUDA_11_LIB_DIR := $(CUDA_11)/lib64
	CUDA_12_LIB_DIR := $(CUDA_12)/lib64
else
	OBJ_EXT := o
	SHARED_EXT := so
	CP := cp -a
endif

CUDA_11_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(CUDA_11_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
CUDA_12_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(CUDA_12_LIB_DIR)/$(SHARED_PREFIX),$(CUDA_LIBS_SHORT))))
NVCC_11 = $(CUDA_11)/bin/nvcc
NVCC_12 = $(CUDA_12)/bin/nvcc

CUDA_DEPS_DIR = $(DIST_BASE)cuda/
ROCM_DEPS_DIR = $(DIST_BASE)rocm/

ifneq ($(CUDA_11),)
	CUDA_11_VARIANT= _v11
	CUDA_11_LIB_DEPS = $(addprefix $(CUDA_DEPS_DIR),$(notdir $(CUDA_11_LIBS)))
endif
ifneq ($(CUDA_12),)
	CUDA_12_VARIANT= _v12
	CUDA_12_LIB_DEPS = $(addprefix $(CUDA_DEPS_DIR),$(notdir $(CUDA_12_LIBS)))
endif
ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIPCC),)
	ROCM_VERSION := $(subst $(space),.,$(wordlist 1,2,$(subst .,$(space),$(word 3,$(subst -,$(space),$(filter HIP version: %,$(shell $(HIPCC) --version)))))))
    ifneq (,$(ROCM_VERSION))
		ROCM_VARIANT = _v$(ROCM_VERSION)
	endif
	ROCM_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT),$(addprefix $(HIP_LIB_DIR)/$(SHARED_PREFIX),$(ROCM_LIBS_SHORT))))
	ROCM_LIB_DEPS = $(addprefix $(ROCM_DEPS_DIR),$(notdir $(ROCM_LIBS)))
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
	$(wildcard ggml-cuda/*.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-wmma*.cu) \
	$(wildcard ggml-cuda/template-instances/mmq*.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu) \
	$(wildcard ggml-cuda/template-instances/fattn-vec*f16-f16.cu) \
	ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp
CUDA_HDRS := \
	$(wildcard ggml-cuda/*.cuh)

CUDA_11_OBJS := $(CUDA_SRCS:.cu=.cuda.$(OBJ_EXT))
CUDA_11_OBJS := $(CUDA_11_OBJS:.c=.cuda.$(OBJ_EXT))
CUDA_11_OBJS := $(addprefix $(BUILD_DIR)/cuda_v11/,$(CUDA_11_OBJS:.cpp=.cuda.$(OBJ_EXT)))
CUDA_12_OBJS := $(CUDA_SRCS:.cu=.cuda.$(OBJ_EXT))
CUDA_12_OBJS := $(CUDA_12_OBJS:.c=.cuda.$(OBJ_EXT))
CUDA_12_OBJS := $(addprefix $(BUILD_DIR)/cuda_v12/,$(CUDA_12_OBJS:.cpp=.cuda.$(OBJ_EXT)))

HIP_OBJS := $(CUDA_SRCS:.cu=.hip.$(OBJ_EXT))
HIP_OBJS := $(HIP_OBJS:.c=.hip.$(OBJ_EXT))
HIP_OBJS := $(addprefix $(BUILD_DIR)/,$(HIP_OBJS:.cpp=.hip.$(OBJ_EXT)))

CUDA_FLAGS := \
	-t4 \
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

CUDA_11_FLAGS := \
	--generate-code=arch=compute_50,code=[compute_50,sm_50] \
	--generate-code=arch=compute_52,code=[compute_52,sm_52] \
	--generate-code=arch=compute_53,code=[compute_53,sm_53] \
	--generate-code=arch=compute_60,code=[compute_60,sm_60] \
	--generate-code=arch=compute_61,code=[compute_61,sm_61] \
	--generate-code=arch=compute_62,code=[compute_62,sm_62] \
	--generate-code=arch=compute_70,code=[compute_70,sm_70] \
	--generate-code=arch=compute_72,code=[compute_72,sm_72] \
	--generate-code=arch=compute_75,code=[compute_75,sm_75] \
	--generate-code=arch=compute_80,code=[compute_80,sm_80] \
	--generate-code=arch=compute_86,code=[compute_86,sm_86]

CUDA_12_FLAGS := \
	--generate-code=arch=compute_60,code=[compute_60,sm_60] \
	--generate-code=arch=compute_61,code=[compute_61,sm_61] \
	--generate-code=arch=compute_62,code=[compute_62,sm_62] \
	--generate-code=arch=compute_70,code=[compute_70,sm_70] \
	--generate-code=arch=compute_72,code=[compute_72,sm_72] \
	--generate-code=arch=compute_75,code=[compute_75,sm_75] \
	--generate-code=arch=compute_80,code=[compute_80,sm_80] \
	--generate-code=arch=compute_86,code=[compute_86,sm_86] \
	--generate-code=arch=compute_87,code=[compute_87,sm_87] \
	--generate-code=arch=compute_89,code=[compute_89,sm_89] \
	--generate-code=arch=compute_90,code=[compute_90,sm_90] \
	--generate-code=arch=compute_90a,code=[compute_90a,sm_90a] \
	-DGGML_CUDA_USE_GRAPHS=on

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
ifneq ($(CUDA_11),)
	RUNNERS += cuda_v11
endif
ifneq ($(CUDA_12),)
	RUNNERS += cuda_v12
endif
endif
ifeq ($(OLLAMA_SKIP_ROCM_GENERATE),)
ifneq ($(HIPCC),)
	RUNNERS += rocm$(ROCM_VARIANT)
endif
endif

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_runner$(EXE_EXT),$(RUNNERS)))
PAYLOAD_RUNNERS = $(addprefix $(RUNNERS_PAYLOAD_DIR)/,$(addsuffix /ollama_runner$(EXE_EXT).gz,$(addsuffix /bin,$(RUNNERS))))
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_runner$(EXE_EXT),$(RUNNERS)))

all: dist payload 

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
	$(NVCC_11) -c $(CUDA_FLAGS) $(CUDA_11_FLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v11/%.cuda.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(NVCC_11) -c $(NVCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v11/%.cuda.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(NVCC_11) -c $(NVCC_CXXFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(NVCC_12) -c $(CUDA_FLAGS) $(CUDA_12_FLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(NVCC_12) -c $(NVCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/cuda_v12/%.cuda.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(NVCC_12) -c $(NVCC_CXXFLAGS) -o $@ $<

$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(dir $@)
	cp $< $@

$(RUNNERS_DIST_DIR)/cuda_v11/ollama_runner$(EXE_EXT): $(RUNNERS_DIST_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT)
$(RUNNERS_PAYLOAD_DIR)/cuda_v11/bin/ollama_runner$(EXE_EXT).gz: $(RUNNERS_PAYLOAD_DIR)/cuda_v11/bin/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT).gz
$(RUNNERS_DIST_DIR)/cuda_v12/ollama_runner$(EXE_EXT): $(RUNNERS_DIST_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT)
$(RUNNERS_PAYLOAD_DIR)/cuda_v12/bin/ollama_runner$(EXE_EXT).gz: $(RUNNERS_PAYLOAD_DIR)/cuda_v12/bin/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT).gz

$(RUNNERS_BUILD_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT): $(CUDA_11_OBJS) $(CUDA_11_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(NVCC_11) --shared -lcuda -L${CUDA_DEPS_DIR} $(foreach lib, $(CUDA_LIBS_SHORT), -l$(lib)) $(CUDA_FLAGS) $(CUDA_11_FLAGS) $(CUDA_11_OBJS) -o $@

$(RUNNERS_BUILD_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT): $(CUDA_12_OBJS) $(CUDA_12_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(NVCC_12) --shared -lcuda -L${CUDA_DEPS_DIR} $(foreach lib, $(CUDA_LIBS_SHORT), -l$(lib)) $(CUDA_FLAGS) $(CUDA_12_FLAGS) $(CUDA_12_OBJS) -o $@

$(CUDA_11_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(CUDA_11_LIB_DIR)/$(notdir $@)* $(dir $@)

$(CUDA_12_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(CUDA_12_LIB_DIR)/$(notdir $@)* $(dir $@)

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(HIP_FLAGS) -o $@ $<

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(HIPCC_CFLAGS) -o $@ $<

$(BUILD_DIR)/%.hip.$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(HIPCC) -c $(HIPCC_CXXFLAGS) -o $@ $<

$(RUNNERS_DIST_DIR)/rocm$(ROCM_VARIANT)/ollama_runner$(EXE_EXT): $(RUNNERS_DIST_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT)
$(RUNNERS_PAYLOAD_DIR)/rocm$(ROCM_VARIANT)/bin/ollama_runner$(EXE_EXT).gz: $(RUNNERS_PAYLOAD_DIR)/rocm$(ROCM_VARIANT)/bin/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT).gz

$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT): $(HIP_OBJS) $(ROCM_LIB_DEPS) $(COMMON_HDRS) $(CUDA_HDRS)
	@-mkdir -p $(dir $@)
	$(HIPCC) --shared -lamdhip64 -L${ROCM_DEPS_DIR} $(foreach lib, $(ROCM_LIBS_SHORT), -l$(lib)) $(HIP_OBJS) -o $@

$(ROCM_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(HIP_LIB_DIR)/$(notdir $@)* $(dir $@)

$(RUNNERS_BUILD_DIR)/$(DEFAULT_RUNNER)/ollama_runner$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cpu_avx/ollama_runner$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -tags avx -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cpu_avx2/ollama_runner$(EXE_EXT): *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) go build -ldflags "-s -w" -tags avx,avx2 -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cuda_v11/ollama_runner$(EXE_EXT): $(RUNNERS_BUILD_DIR)/cuda_v11/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) CGO_LDFLAGS=-L"$(RUNNERS_BUILD_DIR)/cuda_v11/" go build -ldflags "-s -w" -tags avx,cuda -o $@ ./runner

$(RUNNERS_BUILD_DIR)/cuda_v12/ollama_runner$(EXE_EXT): $(RUNNERS_BUILD_DIR)/cuda_v12/$(SHARED_PREFIX)ggml_cuda.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) CGO_LDFLAGS=-L"$(RUNNERS_BUILD_DIR)/cuda_v12/" go build -ldflags "-s -w" -tags avx,cuda -o $@ ./runner

$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/ollama_runner$(EXE_EXT): $(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/$(SHARED_PREFIX)ggml_hipblas.$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	CGO_ENABLED=1 GOARCH=$(ARCH) CGO_LDFLAGS=-L"$(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/" go build -ldflags "-s -w" -tags avx,rocm -o $@ ./runner


$(ROCBLAS_DEP_MANIFEST):
	@-mkdir -p $(dir $@)
	@echo "Copying rocblas library..."
	cd $(HIP_LIB_DIR)/rocblas/library/ && tar cf - . | (cd $(dir $@) && tar xf - )
	@echo "rocblas library copy complete"


$(RUNNERS_PAYLOAD_DIR)/%/bin/ollama_runner.gz: $(RUNNERS_BUILD_DIR)/%/ollama_runner
	@-mkdir -p $(dir $@)
	gzip --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/cuda_v11/bin/%.gz: $(RUNNERS_BUILD_DIR)/cuda_v11/%
	@-mkdir -p $(dir $@)
	gzip --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/cuda_v12/bin/%.gz: $(RUNNERS_BUILD_DIR)/cuda_v12/%
	@-mkdir -p $(dir $@)
	gzip --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/rocm$(ROCM_VARIANT)/bin/%.gz: $(RUNNERS_BUILD_DIR)/rocm$(ROCM_VARIANT)/%
	@-mkdir -p $(dir $@)
	gzip --best -c $< > $@

clean:
	rm -rf $(BUILD_DIR) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)

.PHONY: all dist payload runners clean $(RUNNERS)

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
