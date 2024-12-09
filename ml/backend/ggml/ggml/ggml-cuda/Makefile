NVCC?=nvcc

NVCC_PREPEND_FLAGS= \
	-t 2 \
	-split-compile 0 \
	-std=c++17 \
	-I.. \
	-I../include \
	$(foreach ARCH,$(subst ;, ,$(CUDA_ARCHS)),--generate-code=arch=compute_$(ARCH),code=sm_$(ARCH)) \

NVCC_APPEND_FLAGS= \
	# -DGGML_CUDA_USE_GRAPHS=1 \

ALL_SOURCES=$(wildcard *.cu)
FATTN_SOURCES=$(wildcard fattn*.cu)

SOURCES= \
	$(filter-out $(FATTN_SOURCES),$(ALL_SOURCES)) \
	$(wildcard template-instances/mmq*.cu) \

ifneq ($(OLLAMA_FAST_BUILD),)
NVCC_APPEND_FLAGS+=-DGGML_DISABLE_FLASH_ATTN
else
SOURCES+= \
	$(FATTN_SOURCES) \
	$(wildcard template-instances/fattn-wmma*.cu) \
	$(wildcard template-instances/fattn-vec*q4_0-q4_0.cu) \
	$(wildcard template-instances/fattn-vec*q8_0-q8_0.cu) \
	$(wildcard template-instances/fattn-vec*f16-f16.cu)
endif

all: cuda_v11 cuda_v12

cuda_v11: CUDA_ARCHS?=50;52;53;60;61;62;70;72;75;80;86
cuda_v11: OBJECTS=$(patsubst %.cu,%.v11.o,$(SOURCES))

cuda_v12: CUDA_ARCHS?=60;61;62;70;72;75;80;86;87;89;90;90a
cuda_v12: OBJECTS=$(patsubst %.cu,%.v12.o,$(SOURCES))

rocm: CPPFLAGS+=-DGGML_USE_HIP
rocm: OBJECTS=$(patsubst %.cu,%.rocm.o,$(SOURCES))

ifeq ($(OS),Windows_NT)
cuda_v11: libggml_cuda_v11.dll
cuda_v12: libggml_cuda_v12.dll
rocm: libggml_rocm.dll
else
cuda_v11: libggml_cuda_v11.a
cuda_v12: libggml_cuda_v12.a
rocm: libggml_rocm.a
endif

clean:
	$(RM) *.a *.o template-instances/*.o

%.v11.o %.v12.o %.rocm.o: %.cu
	$(NVCC) $(NVCC_PREPEND_FLAGS) -c $< -o $@ $(NVCC_APPEND_FLAGS)

.SECONDEXPANSION:
%.a: $$(OBJECTS)
	$(AR) rcs $@ $^

%.dll: $$(OBJECTS)
	$(NVCC) -shared -o $@ $^
