# Generalized GPU runner build

ifndef GPU_RUNNER_NAME
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif

GPU_GOFLAGS="-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$(VERSION)\" $(EXTRA_GOLDFLAGS) $(TARGET_LDFLAGS)"

# TODO Unify how we handle dependencies in the dist/packaging and install flow
# today, cuda is bundled, but rocm is split out.  Should split them each out by runner
DIST_GPU_RUNNER_DEPS_DIR = $(DIST_LIB_DIR)


GPU_RUNNER_LIBS = $(wildcard $(addsuffix .$(SHARED_EXT).*,$(addprefix $(GPU_LIB_DIR)/$(SHARED_PREFIX),$(GPU_RUNNER_LIBS_SHORT))))

GPU_RUNNER_SRCS := \
	$(wildcard llama/ggml-cuda/*.cu) \
	$(wildcard llama/ggml-cuda/template-instances/*.cu) \
	llama/ggml.c llama/ggml-backend.cpp llama/ggml-alloc.c llama/ggml-quants.c llama/sgemm.cpp llama/ggml-threading.cpp
GPU_RUNNER_HDRS := \
	$(wildcard llama/ggml-cuda/*.cuh)

GPU_RUNNER_OBJS := $(GPU_RUNNER_SRCS:.cu=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(GPU_RUNNER_OBJS:.c=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(addprefix $(BUILD_DIR)/,$(GPU_RUNNER_OBJS:.cpp=.$(GPU_RUNNER_NAME).$(OBJ_EXT)))

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)))
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)))


$(GPU_RUNNER_NAME): $(BUILD_RUNNERS) 

dist: $(DIST_RUNNERS)

# Build targets
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CFLAGS) $(GPU_COMPILER_CUFLAGS) $(GPU_RUNNER_ARCH_FLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CFLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CXXFLAGS) -o $@ $<
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = $(CGO_EXTRA_LDFLAGS) -L"$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/"
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT) ./llama/*.go ./llama/runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build -buildmode=pie $(GPU_GOFLAGS) -trimpath -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) $(GPU_RUNNER_GO_TAGS)) -o $@ ./cmd/runner
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT): $(GPU_RUNNER_OBJS) $(COMMON_HDRS) $(GPU_RUNNER_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) --shared -L$(GPU_LIB_DIR) $(GPU_RUNNER_DRIVER_LIB_LINK) -L${DIST_GPU_RUNNER_DEPS_DIR} $(foreach lib, $(GPU_RUNNER_LIBS_SHORT), -l$(lib)) $(GPU_RUNNER_OBJS) -o $@

# Distribution targets
$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(dir $@)
	$(CP) $< $@
$(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): $(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT) $(GPU_DIST_LIB_DEPS)
$(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT): $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT)
	@-mkdir -p $(dir $@)
	$(CP) $< $@
$(GPU_DIST_LIB_DEPS):
	@-mkdir -p $(dir $@)
	$(CP) $(GPU_LIB_DIR)/$(notdir $@) $(dir $@)

clean: 
	rm -f $(GPU_RUNNER_OBJS) $(BUILD_RUNNERS) $(DIST_RUNNERS)

.PHONY: clean $(GPU_RUNNER_NAME)


# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'

