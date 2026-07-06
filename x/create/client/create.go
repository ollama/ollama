// Package client provides client-side model creation for safetensors-based models.
//
// This package is in x/ because the safetensors model storage format is under development.
// It also exists to break an import cycle: server imports x/create, so x/create
// cannot import server. This sub-package can import server because server doesn't
// import it.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	modelparsers "github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	imagemanifest "github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/quant"
)

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.19.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template   string
	System     string
	License    string
	Draft      string
	Parser     string
	Renderer   string
	Requires   string
	Parameters map[string]any
}

var ignoredModelfileParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

// ConfigFromModelfile extracts the model directory and x/create-specific
// Modelfile configuration from a parsed Modelfile.
func ConfigFromModelfile(modelfile *parser.Modelfile) (string, *ModelfileConfig, error) {
	var modelDir string
	mfConfig := &ModelfileConfig{}

	for _, cmd := range modelfile.Commands {
		switch cmd.Name {
		case "model":
			modelDir = cmd.Args
		case "template":
			mfConfig.Template = cmd.Args
		case "system":
			mfConfig.System = cmd.Args
		case "license":
			mfConfig.License = cmd.Args
		case "draft":
			mfConfig.Draft = cmd.Args
		case "parser":
			mfConfig.Parser = cmd.Args
		case "renderer":
			mfConfig.Renderer = cmd.Args
		case "requires":
			requires := cmd.Args
			if !strings.HasPrefix(requires, "v") {
				requires = "v" + requires
			}
			if !semver.IsValid(requires) {
				return "", nil, fmt.Errorf("requires must be a valid semver (e.g. 0.14.0)")
			}
			minVersion := "v" + MinOllamaVersion
			if semver.Compare(requires, minVersion) < 0 {
				return "", nil, fmt.Errorf("requires %s is below the minimum supported version %s for safetensors models", strings.TrimPrefix(requires, "v"), MinOllamaVersion)
			}
			mfConfig.Requires = strings.TrimPrefix(requires, "v")
		case "adapter", "message":
			continue
		default:
			if slices.Contains(ignoredModelfileParameters, cmd.Name) {
				continue
			}

			ps, err := api.FormatParams(map[string][]string{cmd.Name: {cmd.Args}})
			if err != nil {
				return "", nil, err
			}

			if mfConfig.Parameters == nil {
				mfConfig.Parameters = make(map[string]any)
			}

			for k, v := range ps {
				if ks, ok := mfConfig.Parameters[k].([]string); ok {
					mfConfig.Parameters[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					mfConfig.Parameters[k] = vs
				} else {
					mfConfig.Parameters[k] = v
				}
			}
		}
	}

	if modelDir == "" {
		modelDir = "."
	}

	return modelDir, mfConfig, nil
}

// CreateOptions holds all options for model creation.
type CreateOptions struct {
	ModelName     string
	ModelDir      string
	Quantize      string           // "int4", "int8", "nvfp4", "mxfp4", or "mxfp8" for quantization
	DraftQuantize string           // optional quantization level for draft model tensors
	Modelfile     *ModelfileConfig // template/system/license/parser/renderer/parameters from Modelfile
	BaseConfig    *model.ConfigV2
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects model type (safetensors LLM vs image gen) and routes accordingly.
func CreateModel(opts CreateOptions, p *progress.Progress) error {
	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	isBaseModelWithDraft := hasDraft && !isSafetensors && create.IsSafetensorsLLMModel(opts.ModelDir)
	if opts.DraftQuantize != "" && !hasDraft {
		return fmt.Errorf("--draft-quantize requires a DRAFT model")
	}
	if opts.Quantize != "" && quant.Canonical(opts.Quantize) == "" {
		return fmt.Errorf("unsupported --quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.Quantize)
	}
	if opts.DraftQuantize != "" && quant.Canonical(opts.DraftQuantize) == "" {
		return fmt.Errorf("unsupported --draft-quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.DraftQuantize)
	}

	if !isSafetensors && !isBaseModelWithDraft {
		return fmt.Errorf("%s is not a supported safetensors model directory (needs config.json + *.safetensors)", opts.ModelDir)
	}

	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}

	modelType := "safetensors model"
	spinnerKey := "create"
	var capabilities []string
	var parserName, rendererName string
	if isSafetensors {
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
		capabilities = inferSafetensorsCapabilities(opts.ModelDir, resolveParserName(opts.Modelfile, parserName))
	}

	// Set up progress spinner
	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)

	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	var draftLayers []create.LayerInfo
	var err error
	if hasDraft {
		draftLayers, err = create.CreateDraftLayers(
			opts.Modelfile.Draft,
			"draft.",
			"draft/",
			opts.DraftQuantize,
			create.StoreFromLayerCreator(newLayerCreator()),
			progressFn,
		)
		if err != nil {
			spinner.Stop()
			return err
		}
	}

	if isBaseModelWithDraft {
		err = createModelFromBaseWithDraft(opts, draftLayers, progressFn)
		spinner.Stop()
		if err != nil {
			return err
		}
		fmt.Printf("Created safetensors model '%s'\n", opts.ModelName)
		return nil
	}

	// Create the model through the x/create pipeline (read → classify → plan
	// → write), supplying blob storage and manifest assembly.
	writer := newManifestWriter(opts, capabilities, parserName, rendererName)
	if len(draftLayers) > 0 {
		writer = appendLayersManifestWriter(writer, draftLayers)
	}
	err = create.Create(
		opts.ModelName, opts.ModelDir, opts.Quantize,
		create.StoreFromLayerCreator(newLayerCreator()),
		writer,
		progressFn,
	)

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

func appendLayersManifestWriter(next create.ManifestWriter, extra []create.LayerInfo) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo) error {
		layers = append(layers, extra...)
		return next(modelName, config, layers)
	}
}

func draftMetadata(draftDir string) (*model.Draft, error) {
	configPath := filepath.Join(draftDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read draft config %s: %w", configPath, err)
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse draft config %s: %w", configPath, err)
	}

	arch := ""
	if len(cfg.Architectures) > 0 {
		arch = cfg.Architectures[0]
	}
	if arch == "" {
		arch = cfg.ModelType
	}
	if arch == "" {
		return nil, fmt.Errorf("draft architecture not found in %s", configPath)
	}

	return &model.Draft{
		ModelFormat:  "safetensors",
		Architecture: arch,
		TensorPrefix: "draft.",
		Config:       "draft/config.json",
	}, nil
}

func createModelFromBaseWithDraft(opts CreateOptions, draftLayers []create.LayerInfo, progressFn func(string)) error {
	progressFn(fmt.Sprintf("loading base model %s", opts.ModelDir))
	baseManifest, err := imagemanifest.LoadManifest(opts.ModelDir)
	if err != nil {
		return err
	}

	baseConfig, err := readConfigV2(baseManifest)
	if err != nil {
		return err
	}
	opts.BaseConfig = baseConfig

	configLayer := baseManifest.GetConfigLayer("config.json")
	if configLayer == nil {
		return fmt.Errorf("base model %s does not contain config.json", opts.ModelDir)
	}

	layers := make([]create.LayerInfo, 0, len(baseManifest.Manifest.Layers)+len(draftLayers))
	for _, layer := range baseManifest.Manifest.Layers {
		layers = append(layers, create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      layer.Name,
		})
	}
	layers = append(layers, draftLayers...)

	progressFn(fmt.Sprintf("writing manifest for %s", opts.ModelName))
	return newManifestWriter(opts, baseConfig.Capabilities, baseConfig.Parser, baseConfig.Renderer)(
		opts.ModelName,
		create.LayerInfo{
			Digest:    configLayer.Digest,
			Size:      configLayer.Size,
			MediaType: configLayer.MediaType,
			Name:      configLayer.Name,
		},
		layers,
	)
}

func readConfigV2(m *imagemanifest.ModelManifest) (*model.ConfigV2, error) {
	data, err := os.ReadFile(m.BlobPath(m.Manifest.Config.Digest))
	if err != nil {
		return nil, fmt.Errorf("failed to read base config: %w", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse base config: %w", err)
	}
	return &cfg, nil
}

func inferSafetensorsCapabilities(modelDir, parserName string) []string {
	capabilities := []string{"completion"}

	caps := detectCapabilities(modelDir)
	if caps.vision {
		capabilities = append(capabilities, "vision")
	}

	if caps.audio {
		capabilities = append(capabilities, "audio")
	}

	var builtinParser modelparsers.Parser
	if parserName != "" {
		builtinParser = modelparsers.ParserForName(parserName)
	}

	if builtinParser != nil && builtinParser.HasToolSupport() {
		capabilities = append(capabilities, "tools")
	}

	if caps.thinking || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
}

// newLayerCreator returns a LayerCreator callback for creating config/JSON layers.
func newLayerCreator() create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		layer, err := manifest.NewLayer(r, mediaType)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}
}

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions, capabilities []string, parserName, rendererName string) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// TODO: find a better way to detect image input support
		// For now, hardcode Flux2KleinPipeline as supporting vision (image input)
		caps := capabilities
		modelIndex := filepath.Join(opts.ModelDir, "model_index.json")
		if data, err := os.ReadFile(modelIndex); err == nil {
			var cfg struct {
				ClassName string `json:"_class_name"`
			}
			if json.Unmarshal(data, &cfg) == nil && cfg.ClassName == "Flux2KleinPipeline" {
				caps = append(caps, "vision")
			}
		}

		// Create config blob with version requirement.
		configData := model.ConfigV2{}
		if opts.BaseConfig != nil {
			configData = *opts.BaseConfig
		}
		configData.ModelFormat = "safetensors"
		if opts.Quantize != "" || configData.FileType == "" {
			configData.FileType = strings.ToLower(strings.TrimSpace(opts.Quantize))
		}
		configData.Capabilities = caps
		configData.Requires = MinOllamaVersion
		if opts.Modelfile != nil && opts.Modelfile.Requires != "" {
			configData.Requires = opts.Modelfile.Requires
		}
		configData.Parser = resolveParserName(opts.Modelfile, parserName)
		configData.Renderer = resolveRendererName(opts.Modelfile, rendererName)
		if opts.Modelfile != nil && opts.Modelfile.Draft != "" {
			draft, err := draftMetadata(opts.Modelfile.Draft)
			if err != nil {
				return err
			}
			configData.Draft = draft
		}
		configJSON, err := json.Marshal(configData)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		// Create config layer blob
		configLayer, err := manifest.NewLayer(bytes.NewReader(configJSON), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			return fmt.Errorf("failed to create config layer: %w", err)
		}

		// Convert LayerInfo to manifest.Layer
		manifestLayers := make([]manifest.Layer, 0, len(layers))
		for _, l := range layers {
			manifestLayers = append(manifestLayers, manifest.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				Name:      l.Name,
			})
		}

		// Add Modelfile layers if present
		if opts.Modelfile != nil {
			modelfileLayers, err := createModelfileLayers(opts.Modelfile)
			if err != nil {
				return err
			}
			manifestLayers = append(manifestLayers, modelfileLayers...)
		}

		return manifest.WriteManifest(name, configLayer, manifestLayers)
	}
}

func resolveParserName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Parser != "" {
		return mf.Parser
	}

	return inferred
}

func resolveRendererName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Renderer != "" {
		return mf.Renderer
	}

	return inferred
}

// createModelfileLayers creates layers for template, system, and license from Modelfile config.
func createModelfileLayers(mf *ModelfileConfig) ([]manifest.Layer, error) {
	var layers []manifest.Layer

	if mf.Template != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.Template)), "application/vnd.ollama.image.template")
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.System != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.System)), "application/vnd.ollama.image.system")
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.License != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.License)), "application/vnd.ollama.image.license")
		if err != nil {
			return nil, fmt.Errorf("failed to create license layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if len(mf.Parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(mf.Parameters); err != nil {
			return nil, fmt.Errorf("failed to encode parameters: %w", err)
		}

		layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return nil, fmt.Errorf("failed to create params layer: %w", err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// modelCapabilities holds the input-modality and reasoning capabilities a model
// advertises, inferred from its source metadata.
type modelCapabilities struct {
	vision   bool
	audio    bool
	thinking bool
}

// detectCapabilities reads the model directory once and reports the vision,
// audio, and thinking capabilities it can infer.
func detectCapabilities(modelDir string) modelCapabilities {
	var cfg struct {
		Architectures []string        `json:"architectures"`
		ModelType     string          `json:"model_type"`
		VisionConfig  *map[string]any `json:"vision_config"`
		AudioConfig   *map[string]any `json:"audio_config"`
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "config.json")); err == nil {
		_ = json.Unmarshal(data, &cfg)
	}

	return modelCapabilities{
		vision: cfg.VisionConfig != nil,
		audio:  cfg.AudioConfig != nil,
		thinking: chatTemplateHasThinkingSupport(readChatTemplate(modelDir)) ||
			alwaysSupportsThinking(cfg.Architectures, cfg.ModelType),
	}
}

// readChatTemplate returns the model's chat template, preferring the
// chat_template field of tokenizer_config.json and falling back to a standalone
// chat_template.jinja. It returns "" when neither is present.
func readChatTemplate(modelDir string) string {
	if data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.ChatTemplate != "" {
			return cfg.ChatTemplate
		}
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		return string(data)
	}
	return ""
}

// chatTemplateHasThinkingSupport reports whether a chat template emits thinking
// blocks. Copied from server.chatTemplateHasThinkingSupport so this package need
// not depend on the server package for an eight-line string check.
func chatTemplateHasThinkingSupport(chatTemplate string) bool {
	if strings.Contains(chatTemplate, "<think>") && strings.Contains(chatTemplate, "</think>") {
		return true
	}

	// Some Qwen/DeepSeek templates strip prior reasoning by splitting assistant
	// content at </think>; llama.cpp can still extract reasoning from them.
	return (strings.Contains(chatTemplate, "content.split('</think>')") ||
		strings.Contains(chatTemplate, `content.split("</think>")`)) &&
		!strings.Contains(chatTemplate, "reasoning_content") &&
		!strings.Contains(chatTemplate, "<SPECIAL_12>")
}

func alwaysSupportsThinking(architectures []string, modelType string) bool {
	if isQwen35Family(modelType) {
		return true
	}
	for _, arch := range architectures {
		if isQwen35Family(arch) {
			return true
		}
	}
	return false
}

func isQwen35Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen3_5") || strings.Contains(s, "qwen3next")
}

// getParserName returns the parser name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate parser.
func getParserName(modelDir string) string {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	// Check architectures for known parsers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "laguna") {
			return "laguna"
		}
		if strings.Contains(archLower, "cohere2moe") || strings.Contains(archLower, "cohere2_moe") {
			return "cohere"
		}
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "gemma4") {
			return "gemma4"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "laguna") {
			return "laguna"
		}
		if strings.Contains(typeLower, "cohere2_moe") {
			return "cohere"
		}
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "gemma4") {
			return "gemma4"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3"
		}
	}

	return ""
}

// getRendererName returns the renderer name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate renderer.
func getRendererName(modelDir string) string {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	// Check architectures for known renderers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "laguna") {
			return "laguna"
		}
		if strings.Contains(archLower, "cohere2moe") || strings.Contains(archLower, "cohere2_moe") {
			return "cohere"
		}
		if strings.Contains(archLower, "gemma4") {
			return "gemma4"
		}
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "laguna") {
			return "laguna"
		}
		if strings.Contains(typeLower, "cohere2_moe") {
			return "cohere"
		}
		if strings.Contains(typeLower, "gemma4") {
			return "gemma4"
		}
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	return ""
}
