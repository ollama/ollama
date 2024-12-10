# Generalized GPU runner build

ifndef GPU_RUNNER_NAME
dummy:
	$(error This makefile is not meant to build directly, but instead included in other Makefiles that set required variables)
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

DIST_RUNNERS = $(addprefix $(RUNNERS_DIST_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)))
ifneq ($(OS),windows)
PAYLOAD_RUNNERS = $(addprefix $(RUNNERS_PAYLOAD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT).gz,$(GPU_RUNNER_NAME)))
endif
BUILD_RUNNERS = $(addprefix $(RUNNERS_BUILD_DIR)/,$(addsuffix /ollama_llama_server$(EXE_EXT),$(GPU_RUNNER_NAME)))

$(GPU_RUNNER_NAME): $(BUILD_RUNNERS) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)

# Build targets
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/ollama_llama_server$(EXE_EXT): TARGET_CGO_LDFLAGS = -L"$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/" $(CGO_EXTRA_LDFLAGS)
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/ollama_llama_server$(EXE_EXT):
	@-mkdir -p $(@D)
	$(MAKE) -C ../ml/backend/ggml/ggml/ggml-cuda $(GPU_RUNNER_NAME)
	GOARCH=$(ARCH) CGO_LDFLAGS="$(TARGET_CGO_LDFLAGS)" go build -buildmode=pie  $(GPU_GOFLAGS) -trimpath -tags $(subst $(space),$(comma),$(GPU_RUNNER_CPU_FLAGS) $(GPU_RUNNER_GO_TAGS)) -o $@ ./runner

# Distribution targets
$(RUNNERS_DIST_DIR)/%: $(RUNNERS_BUILD_DIR)/%
	@-mkdir -p $(@D)
	$(CP) $< $@
$(DIST_GPU_RUNNER_LIB_DEPS):
	@-mkdir -p $(@D)
	$(CP) $(GPU_LIB_DIR)/$(@F) $(@D)
$(GPU_DIST_DEPS_LIBS):
	@-mkdir -p $(@D)
	$(CP) $(dir $(filter %$(@F),$(GPU_LIBS) $(GPU_TRANSITIVE_LIBS)))/$(@F) $(@D)

# Payload targets
$(RUNNERS_PAYLOAD_DIR)/%/ollama_llama_server.gz: $(RUNNERS_BUILD_DIR)/%/ollama_llama_server
	@-mkdir -p $(@D)
	${GZIP} --best -c $< > $@
$(RUNNERS_PAYLOAD_DIR)/$(GPU_RUNNER_NAME)/%.gz: $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)/%
	@-mkdir -p $(@D)
	${GZIP} --best -c $< > $@

clean:
	rm -f $(GPU_RUNNER_OBJS) $(BUILD_RUNNERS) $(DIST_RUNNERS) $(PAYLOAD_RUNNERS)

.PHONY: clean $(GPU_RUNNER_NAME)


# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'

