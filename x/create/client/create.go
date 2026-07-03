// Package client provides local and server-backed model creation for
// safetensors-based models.
package client

import (
	"context"
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
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	imagemanifest "github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/quant"
)

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
			minVersion := "v" + create.SafetensorsMinOllamaVersion
			if semver.Compare(requires, minVersion) < 0 {
				return "", nil, fmt.Errorf("requires %s is below the minimum supported version %s for safetensors models", strings.TrimPrefix(requires, "v"), create.SafetensorsMinOllamaVersion)
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
// Automatically detects safetensors source imports and existing safetensors base models.
func CreateModel(ctx context.Context, opts CreateOptions, p *progress.Progress) error {
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
	if isBaseModelWithDraft && opts.Quantize != "" {
		return fmt.Errorf("--quantize is only supported when importing a safetensors source directory")
	}

	if !isSafetensors && !isBaseModelWithDraft {
		return fmt.Errorf("%s is not a supported safetensors model directory (needs config.json + *.safetensors)", opts.ModelDir)
	}

	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	modelType := "safetensors model"
	spinnerKey := "create"
	var metadata sourceMetadata
	if isSafetensors {
		var err error
		metadata, err = readSourceMetadata(opts.ModelDir, opts.Modelfile)
		if err != nil {
			return err
		}
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
			ctx,
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
		err = createModelFromBaseWithDraft(ctx, opts, draftLayers, progressFn)
		spinner.Stop()
		if err != nil {
			return err
		}
		fmt.Printf("Created safetensors model '%s'\n", opts.ModelName)
		return nil
	}

	// Create the model through the x/create pipeline (read → classify → plan
	// → write), supplying blob storage and manifest assembly.
	writer := newManifestWriter(opts, metadata.capabilities, metadata.parserName, metadata.rendererName)
	if len(draftLayers) > 0 {
		writer = appendLayersManifestWriter(writer, draftLayers)
	}
	err = create.Create(
		ctx,
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
	return func(modelName string, info create.ManifestInfo) error {
		info.Layers = append(info.Layers, extra...)
		return next(modelName, info)
	}
}

func createModelFromBaseWithDraft(ctx context.Context, opts CreateOptions, draftLayers []create.LayerInfo, progressFn func(string)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	progressFn(fmt.Sprintf("loading base model %s", opts.ModelDir))
	baseManifest, err := imagemanifest.LoadManifest(opts.ModelDir)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	baseConfig, err := readConfigV2(baseManifest)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
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
	if err := ctx.Err(); err != nil {
		return err
	}
	return newManifestWriter(opts, baseConfig.Capabilities, baseConfig.Parser, baseConfig.Renderer)(
		opts.ModelName,
		create.ManifestInfo{
			Config: create.LayerInfo{
				Digest:    configLayer.Digest,
				Size:      configLayer.Size,
				MediaType: configLayer.MediaType,
				Name:      configLayer.Name,
			},
			Layers: layers,
		},
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

type sourceConfig struct {
	Architectures []string        `json:"architectures"`
	ModelType     string          `json:"model_type"`
	VisionConfig  *map[string]any `json:"vision_config"`
	AudioConfig   *map[string]any `json:"audio_config"`
	HasVision     bool            `json:"has_vision"`
	SoundConfig   *map[string]any `json:"sound_config"`
	LLMConfig     struct {
		ModelType string `json:"model_type"`
	} `json:"llm_config"`
}

type sourceMetadata struct {
	parserName   string
	rendererName string
	capabilities []string
}

func readSourceMetadata(modelDir string, mf *ModelfileConfig) (sourceMetadata, error) {
	cfg, chatTemplate, err := readSourceMetadataInputs(modelDir)
	if err != nil {
		return sourceMetadata{}, err
	}
	parserName, err := parserNameForConfig(modelDir, cfg, chatTemplate)
	if err != nil {
		return sourceMetadata{}, err
	}
	rendererName, err := rendererNameForConfig(modelDir, cfg, chatTemplate)
	if err != nil {
		return sourceMetadata{}, err
	}

	resolvedParser := resolveParserName(mf, parserName)
	return sourceMetadata{
		parserName:   parserName,
		rendererName: rendererName,
		capabilities: inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, resolvedParser),
	}, nil
}

func readSourceMetadataInputs(modelDir string) (sourceConfig, string, error) {
	cfg, err := readSourceConfig(modelDir)
	if err != nil {
		return sourceConfig{}, "", err
	}
	chatTemplate, err := readChatTemplateStrict(modelDir)
	if err != nil {
		return sourceConfig{}, "", err
	}
	return cfg, chatTemplate, nil
}

func readSourceConfig(modelDir string) (sourceConfig, error) {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return sourceConfig{}, fmt.Errorf("read %s: %w", configPath, err)
	}

	var cfg sourceConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return sourceConfig{}, fmt.Errorf("parse %s: %w", configPath, err)
	}
	return cfg, nil
}

func readChatTemplateStrict(modelDir string) (string, error) {
	tokenizerConfig := filepath.Join(modelDir, "tokenizer_config.json")
	if data, err := os.ReadFile(tokenizerConfig); err == nil {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return "", fmt.Errorf("parse %s: %w", tokenizerConfig, err)
		}
		if cfg.ChatTemplate != "" {
			return cfg.ChatTemplate, nil
		}
	} else if !os.IsNotExist(err) {
		return "", fmt.Errorf("read %s: %w", tokenizerConfig, err)
	}

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil {
		return string(data), nil
	}
	if os.IsNotExist(err) {
		return "", nil
	}
	return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
}

func inferSafetensorsCapabilitiesFromConfig(cfg sourceConfig, chatTemplate, parserName string) []string {
	capabilities := []string{"completion"}

	caps := detectCapabilitiesFromConfig(cfg, chatTemplate)
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
	var template, system, license, requires, draftDir string
	var parameters map[string]any
	if opts.Modelfile != nil {
		template = opts.Modelfile.Template
		system = opts.Modelfile.System
		license = opts.Modelfile.License
		requires = opts.Modelfile.Requires
		draftDir = opts.Modelfile.Draft
		parameters = opts.Modelfile.Parameters
	}
	return create.NewSafetensorsManifestWriter(create.SafetensorsManifestOptions{
		ModelDir:     opts.ModelDir,
		BaseConfig:   opts.BaseConfig,
		Capabilities: capabilities,
		MinVersion:   create.SafetensorsMinOllamaVersion,
		Requires:     requires,
		Parser:       resolveParserName(opts.Modelfile, parserName),
		Renderer:     resolveRendererName(opts.Modelfile, rendererName),
		DraftDir:     draftDir,
		Template:     template,
		System:       system,
		License:      license,
		Parameters:   parameters,
		ExtraLayers:  nil,
	})
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

// modelCapabilities holds the input-modality and reasoning capabilities a model
// advertises, inferred from its source metadata.
type modelCapabilities struct {
	vision   bool
	audio    bool
	thinking bool
}

func detectCapabilitiesFromConfig(cfg sourceConfig, chatTemplate string) modelCapabilities {
	return modelCapabilities{
		vision: cfg.VisionConfig != nil || cfg.HasVision,
		audio:  cfg.AudioConfig != nil || cfg.SoundConfig != nil,
		thinking: thinking.TemplateSupportsThinking(chatTemplate) ||
			alwaysSupportsThinking(cfg.Architectures, cfg.ModelType),
	}
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

func qwen35RendererNameFromTemplate(chatTemplate string) string {
	if strings.Contains(chatTemplate, "resolved_reasoning_effort") &&
		strings.Contains(chatTemplate, "preserve_thinking") {
		return "qwen3.8"
	}

	return "qwen3.5"
}

func lagunaRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const poolsideV1Marker = "laguna_glm_thinking_v8"

	if strings.Contains(chatTemplate, poolsideV1Marker) {
		return "poolside-v1", nil
	}

	// Poolside's tokenizer config includes the standalone template by name
	// rather than embedding it, so inspect that file as well.
	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), poolsideV1Marker) {
		return "poolside-v1", nil
	}
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}

	return "laguna", nil
}

func nemotronRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const v35Marker = "{reasoning effort: efficient}"

	// Nemotron 3.5 publishes its updated template as a standalone file while
	// tokenizer_config.json can retain the older template, so inspect both.
	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), v35Marker) {
		return "nemotron-3.5-nano", nil
	}
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}
	if strings.Contains(chatTemplate, v35Marker) {
		return "nemotron-3.5-nano", nil
	}

	return "nemotron-3-nano", nil
}

func sourceConfigIdentifiers(cfg sourceConfig) []string {
	ids := append([]string(nil), cfg.Architectures...)
	ids = append(ids, cfg.ModelType, cfg.LLMConfig.ModelType)
	return ids
}

func parserNameForConfig(modelDir string, cfg sourceConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := parserNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}

	return "", nil
}

func parserNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case isQwen35Family(s):
		return "qwen3.5", nil
	case strings.Contains(s, "qwen3"):
		return "qwen3", nil
	// Nemotron-H publishes NemotronHForCausalLM for text and
	// NemotronH_Nano_Omni_Reasoning_V3 for omni; model_type is nemotron_h,
	// nemotron_h_moe, or the omni name. The two stems cover all of them.
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}

func rendererNameForConfig(modelDir string, cfg sourceConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := rendererNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}

	return "", nil
}

func rendererNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case isQwen35Family(s):
		return qwen35RendererNameFromTemplate(chatTemplate), nil
	case strings.Contains(s, "qwen3"):
		return "qwen3-coder", nil
	// Nemotron-H publishes NemotronHForCausalLM for text and
	// NemotronH_Nano_Omni_Reasoning_V3 for omni; model_type is nemotron_h,
	// nemotron_h_moe, or the omni name. The two stems cover all of them.
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}
