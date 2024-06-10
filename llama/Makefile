OS := $(shell uname -s)
NVCC := nvcc
HIPCC := "$(HIP_PATH)/bin/hipcc"

# Determine object file extension based on OS
ifeq ($(OS),Windows_NT)
    OBJ_EXT := obj
else
    OBJ_EXT := o
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
CUDA_OBJS := $(CUDA_OBJS:.cpp=.cuda.$(OBJ_EXT))

HIP_OBJS := $(CUDA_SRCS:.cu=.hip.$(OBJ_EXT))
HIP_OBJS := $(HIP_OBJS:.c=.hip.$(OBJ_EXT))
HIP_OBJS := $(HIP_OBJS:.cpp=.hip.$(OBJ_EXT))

# TODO (jmorganca): shared flags for cuda/hip
CUDA_FLAGS := \
    --generate-code=arch=compute_50,code=[compute_50,sm_50] \
    --generate-code=arch=compute_52,code=[compute_52,sm_52] \
    --generate-code=arch=compute_61,code=[compute_61,sm_61] \
    --generate-code=arch=compute_70,code=[compute_70,sm_70] \
    --generate-code=arch=compute_75,code=[compute_75,sm_75] \
    --generate-code=arch=compute_80,code=[compute_80,sm_80] \
    -DGGML_CUDA_DMMV_X=32 \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DGGML_CUDA_MMV_Y=1 \
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

ifeq ($(UNAME_S), Linux)
    HIP_FLAGS += $(foreach arch, $(LINUX_HIP_ARCHS), --offload-arch=$(arch))
endif

all: ollama_runner_cpu ollama_runner_cpu_avx ollama_runner_cpu_avx2 ollama_runner_cuda ollama_runner_rocm

%.cuda.$(OBJ_EXT): %.cu
	nvcc -c $(CUDA_FLAGS) -o $@ $<

%.cuda.$(OBJ_EXT): %.c
	nvcc -c $(CUDA_FLAGS) -o $@ $<

%.cuda.$(OBJ_EXT): %.cpp
	nvcc -c $(CUDA_FLAGS) -o $@ $<

%.hip.$(OBJ_EXT): %.cu
	$(HIPCC) -c $(HIP_FLAGS) -o $@ $<

%.hip.$(OBJ_EXT): %.c
	$(HIPCC) -c $(HIP_FLAGS) -o $@ $<

%.hip.$(OBJ_EXT): %.cpp
	$(HIPCC) -c $(HIP_FLAGS) -o $@ $<

ggml_cuda.dll: $(CUDA_OBJS)
	nvcc --shared -lcuda -lcublas -lcudart -lcublasLt $(CUDA_FLAGS) $(CUDA_OBJS) -o $@

ggml_hipblas.dll: $(HIP_OBJS)
	$(HIPCC) --shared -lhipblas -lamdhip64 -lrocblas $(HIP_OBJS) -o $@

ollama_runner_cpu:
	go build -ldflags "-s -w" -o $@ ./runner 

ollama_runner_cpu_avx:
	go build -ldflags "-s -w" -tags avx -o $@ ./runner

ollama_runner_cpu_avx2:
	go build -ldflags "-s -w" -tags avx,avx2 -o $@ ./runner

ollama_runner_cuda: ggml_cuda.dll
	go build -ldflags "-s -w" -tags avx,cuda -o $@ ./runner

ollama_runner_rocm: ggml_hipblas.dll
	go build -ldflags "-s -w" -tags avx,rocm -o $@ ./runner

clean:
	rm -f $(CUDA_OBJS) $(HIP_OBJS) ggml_cuda.dll ggml_cuda.exp ggml_cuda.lib ggml_hipblas.dll ggml_hipblas.lib ggml_hipblas.exp ollama_runner_cpu ollama_runner_cpu_avx ollama_runner_cpu_avx2 ollama_runner_cuda ollama_runner_rocm

.PHONY: all clean ollama_runner_cpu ollama_runner_cpu_avx ollama_runner_cpu_avx2 ollama_runner_cuda ollama_runner_rocm
