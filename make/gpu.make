# Generalized GPU runner build

ifndef GPU_RUNNER_NAME
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
endif

GPU_GOFLAGS="-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$(VERSION)\" $(EXTRA_GOLDFLAGS) $(TARGET_LDFLAGS)"

# TODO Unify how we handle dependencies in the dist/packaging and install flow
# today, cuda is bundled, but rocm is split out.  Should split them each out by runner
DIST_GPU_RUNNER_DEPS_DIR = $(DIST_LIB_DIR)

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)))
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)))

$(GPU_RUNNER_NAME): $(BUILD_RUNNERS)

dist: $(DIST_RUNNERS)

# Build targets
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = $(CGO_EXTRA_LDFLAGS) -L"$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/"
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): ./llama/*.go ./llama/runner/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(@D)
	$(MAKE) -C ml/backend/ggml/ggml/ggml-cuda $(GPU_RUNNER_NAME) CXX=$(GPU_COMPILER)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build -buildmode=pie $(GPU_GOFLAGS) -trimpath -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) $(GPU_RUNNER_GO_TAGS)) -o $@ ./cmd/runner

# Distribution targets
$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(@D)
	$(CP) $< $@
$(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/ollama_llama_server$(EXE_EXT): $(RUNNERS_DIST_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT) $(GPU_DIST_LIB_DEPS)
$(GPU_DIST_LIB_DEPS):
	@-mkdir -p $(@D)
	$(CP) $(GPU_LIB_DIR)/$(@F) $(@D)

clean:
	$(RM) $(BUILD_RUNNERS) $(DIST_RUNNERS)

.PHONY: clean $(GPU_RUNNER_NAME)


# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'

