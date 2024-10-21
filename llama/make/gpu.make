# Generalized GPU runner build

ifndef GPU_RUNNER_NAME
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif

ifeq ($(OS),windows)
	GPU_COMPILER:=$(GPU_COMPILER_WIN)
	GPU_LIB_DIR:=$(GPU_LIB_DIR_WIN)
	CGO_EXTRA_LDFLAGS:=$(CGO_EXTRA_LDFLAGS_WIN)
	GPU_COMPILER_CFLAGS = $(GPU_COMPILER_CFLAGS_WIN)
	GPU_COMPILER_CXXFLAGS = $(GPU_COMPILER_CXXFLAGS_WIN)
else ifeq ($(OS),linux)
	GPU_COMPILER:=$(GPU_COMPILER_LINUX)
	GPU_LIB_DIR:=$(GPU_LIB_DIR_LINUX)
	CGO_EXTRA_LDFLAGS:=$(CGO_EXTRA_LDFLAGS_LINUX)
	GPU_COMPILER_CFLAGS = $(GPU_COMPILER_CFLAGS_LINUX)
	GPU_COMPILER_CXXFLAGS = $(GPU_COMPILER_CXXFLAGS_LINUX)
endif

GPU_GOFLAGS="-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$(VERSION)\" \"-X=github.com/ollama/ollama/llama.CpuFeatures=$(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS))\" $(TARGET_LDFLAGS)"

# TODO Unify how we handle dependencies in the dist/packaging and install flow
# today, cuda is bundled, but rocm is split out.  Should split them each out by runner
DIST_GPU_RUNNER_DEPS_DIR = $(DIST_LIB_DIR)

ifeq ($(OS),windows)
	_OS_GPU_RUNNER_CPU_FLAGS=$(call uc,$(GPU_RUNNER_CPU_FLAGS))
else ifeq ($(OS),linux)
	_OS_GPU_RUNNER_CPU_FLAGS=$(GPU_RUNNER_CPU_FLAGS)
endif

GPU_RUNNER_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT))))
DIST_GPU_RUNNER_LIB_DEPS = $(addprefix $(DIST_GPU_RUNNER_DEPS_DIR)/,$(notdir $(GPU_RUNNER_LIBS)))

COMMON_SRCS := \
	$(wildcard *.c) \
	$(wildcard *.cpp)
COMMON_HDRS := \
	$(wildcard *.h) \
	$(wildcard *.hpp)

GPU_RUNNER_SRCS := \
	ggml-cuda.cu \
	$(filter-out $(wildcard ggml-cuda/fattn*.cu),$(wildcard ggml-cuda/*.cu)) \
	$(wildcard ggml-cuda/template-instances/mmq*.cu) \
	ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp ggml-aarch64.c
GPU_RUNNER_HDRS := \
	$(wildcard ggml-cuda/*.cuh)


# Conditional flags and components to speed up developer builds
ifneq ($(OLLAMA_FAST_BUILD),)
	GPU_COMPILER_CUFLAGS += 	\
		-DGGML_DISABLE_FLASH_ATTN
else
	GPU_RUNNER_SRCS += \
		$(wildcard ggml-cuda/fattn*.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-wmma*.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu) \
		$(wildcard ggml-cuda/template-instances/fattn-vec*f16-f16.cu)
endif

GPU_RUNNER_OBJS := $(GPU_RUNNER_SRCS:.cu=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(GPU_RUNNER_OBJS:.c=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(addprefix $(BUILD_DIR)/,$(GPU_RUNNER_OBJS:.cpp=.$(GPU_RUNNER_NAME).$(OBJ_EXT)))

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)))
ifneq ($(OS),windows)
PAYLOAD_RUNNERS = $(addprefix $(RUNNERS_PAYLOAD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT).gz,$(GPU_RUNNER_NAME)))
endif
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)))


$(GPU_RUNNER_NAME): $(BUILD_RUNNERS) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)

# Build targets
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CUFLAGS) $(GPU_RUNNER_ARCH_FLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CFLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CXXFLAGS) -o $@ $<
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = -L"$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/" $(CGO_EXTRA_LDFLAGS)
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/ollama_llama_server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT) *.go ./runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build -buildmode=pie  $(GPU_GOFLAGS) -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) $(GPU_RUNNER_GO_TAGS)) -o $@ ./runner
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT): $(GPU_RUNNER_OBJS) $(DIST_GPU_RUNNER_LIB_DEPS) $(COMMON_HDRS) $(GPU_RUNNER_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) --shared $(GPU_RUNNER_DRIVER_LIB_LINK) -L${DIST_GPU_RUNNER_DEPS_DIR} $(foreach lib, $(GPU_RUNNER_LIBS_SHORT), -l$(lib)) $(GPU_RUNNER_OBJS) -o $@

# Distribution targets
$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(dir $@)
	cp $< $@
$(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)/ollama_llama_server$(EXE_EXT): $(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT)
$(DIST_LIB_DIR)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT): $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT)
	@-mkdir -p $(dir $@)
	cp $< $@
$(DIST_GPU_RUNNER_LIB_DEPS): 
	@-mkdir -p $(dir $@)
	$(CP) $(GPU_LIB_DIR)/$(notdir $@)* $(dir $@)

# Payload targets
$(RUNNERS_PAYLOAD_DIR)/%/ollama_llama_server.gz: $(RUNNERS_BUILD_DIR)/%/ollama_llama_server 
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/$(GPU_RUNNER_NAME)/%.gz: $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/%
	@-mkdir -p $(dir $@)
	${GZIP} --best -c $< > $@

clean: 
	rm -f $(GPU_RUNNER_OBJS) $(BUILD_RUNNERS) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)

.PHONY: clean $(GPU_RUNNER_NAME)


# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'

