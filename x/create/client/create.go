// Package client provides local and server-backed model creation for
// safetensors-based models.
package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/quant"
)

var errSafetensorsAdapters = errors.New("safetensors imports do not support adapters")

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template   string
	System     string
	Licenses   []string
	Adapters   []string
	Draft      string
	Parser     string
	Renderer   string
	Requires   string
	Parameters map[string]any
	Messages   []api.Message
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
			mfConfig.Licenses = append(mfConfig.Licenses, cmd.Args)
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
			mfConfig.Requires = strings.TrimPrefix(requires, "v")
		case "adapter":
			mfConfig.Adapters = append(mfConfig.Adapters, cmd.Args)
		case "message":
			role, content, _ := strings.Cut(cmd.Args, ": ")
			mfConfig.Messages = append(mfConfig.Messages, api.Message{Role: role, Content: content})
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
	Force         bool             // create even when MLX validation fails
	Modelfile     *ModelfileConfig // template/system/license/parser/renderer/parameters from Modelfile
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects safetensors source imports and existing safetensors base models.
func CreateModel(ctx context.Context, opts CreateOptions, p *progress.Progress) error {
	if opts.Modelfile != nil && len(opts.Modelfile.Adapters) > 0 {
		return errSafetensorsAdapters
	}

	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	isBaseModelWithDraft := hasDraft && !isSafetensors && create.IsSafetensorsLLMModel(opts.ModelDir)
	if err := validateSafetensorsQuantization(opts); err != nil {
		return err
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
	if isSafetensors {
		if err := validateDistinctSafetensorsSources(opts.ModelDir, opts.Modelfile); err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	spinnerKey := "create"
	mlxOpts := create.MLXValidationOptions{
		Force: opts.Force,
		Warning: func(message string) {
			fmt.Fprintf(os.Stderr, "warning: %s\n", message)
		},
	}
	// Set up progress spinner
	statusMsg := "importing safetensors model"
	spinner := progress.NewSpinner(statusMsg)
	if p != nil {
		p.Add(spinnerKey, spinner)
	}

	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		if p != nil {
			p.Add(spinnerKey, spinner)
		}
	}
	if isBaseModelWithDraft {
		draftLayers, err := create.CreateDraftLayers(
			ctx,
			opts.Modelfile.Draft,
			"draft.",
			"draft/",
			opts.DraftQuantize,
			mlxOpts,
			create.ManifestBlobStore{},
			progressFn,
		)
		if err != nil {
			spinner.Stop()
			return err
		}

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
	pipelineOpts := create.PipelineOptions{
		Quantize:   opts.Quantize,
		Validation: mlxOpts,
	}
	if opts.Modelfile != nil {
		pipelineOpts.Parser = opts.Modelfile.Parser
		pipelineOpts.Renderer = opts.Modelfile.Renderer
		pipelineOpts.Requires = opts.Modelfile.Requires
		pipelineOpts.DraftDir = opts.Modelfile.Draft
		pipelineOpts.DraftQuantize = opts.DraftQuantize
	}
	err := create.Create(
		ctx,
		opts.ModelName, opts.ModelDir, pipelineOpts,
		create.ManifestBlobStore{},
		newManifestWriter(opts),
		progressFn,
	)

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created safetensors model '%s'\n", opts.ModelName)
	return nil
}

func validateSafetensorsQuantization(opts CreateOptions) error {
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	if opts.DraftQuantize != "" && !hasDraft {
		return fmt.Errorf("--draft-quantize requires a DRAFT model")
	}
	if opts.Quantize != "" && quant.Canonical(opts.Quantize) == "" {
		return fmt.Errorf("unsupported --quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.Quantize)
	}
	if opts.DraftQuantize != "" && quant.Canonical(opts.DraftQuantize) == "" {
		return fmt.Errorf("unsupported --draft-quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.DraftQuantize)
	}
	return nil
}

func createModelFromBaseWithDraft(ctx context.Context, opts CreateOptions, draftLayers []create.LayerInfo, progressFn func(string)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	progressFn(fmt.Sprintf("loading base model %s", opts.ModelDir))
	baseName := model.ParseName(opts.ModelDir)
	if !baseName.IsValid() {
		return fmt.Errorf("invalid base model name: %s", opts.ModelDir)
	}
	baseManifest, err := manifest.ParseNamedManifest(baseName)
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
	if opts.Modelfile.Requires != "" {
		baseConfig.Requires = opts.Modelfile.Requires
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var configLayer *manifest.Layer
	for i := range baseManifest.Layers {
		layer := &baseManifest.Layers[i]
		if layer.MediaType == "application/vnd.ollama.image.json" && layer.Name == "config.json" {
			configLayer = layer
			break
		}
	}
	if configLayer == nil {
		return fmt.Errorf("base model %s does not contain config.json", opts.ModelDir)
	}

	layers := make([]create.LayerInfo, 0, len(baseManifest.Layers)+len(draftLayers))
	for _, layer := range baseManifest.Layers {
		if isDraftLayer(layer) {
			continue
		}
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
	writer := newManifestWriter(opts)
	return writer(
		ctx,
		opts.ModelName,
		create.ManifestInfo{
			ModelConfig: *baseConfig,
			ConfigLayer: create.LayerInfo{
				Digest:    configLayer.Digest,
				Size:      configLayer.Size,
				MediaType: configLayer.MediaType,
				Name:      configLayer.Name,
			},
			Layers: layers,
		},
	)
}

func isDraftLayer(layer manifest.Layer) bool {
	return layer.MediaType == manifest.MediaTypeImageDraft ||
		strings.HasPrefix(layer.Name, "draft.") ||
		strings.HasPrefix(layer.Name, "draft/")
}

func readConfigV2(m *manifest.Manifest) (*model.ConfigV2, error) {
	f, err := m.Config.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open base config: %w", err)
	}
	defer f.Close()

	var cfg model.ConfigV2
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse base config: %w", err)
	}
	return &cfg, nil
}

func validateDistinctSafetensorsSources(modelDir string, modelfile *ModelfileConfig) error {
	if modelfile == nil || modelfile.Draft == "" {
		return nil
	}

	modelInfo, err := os.Stat(modelDir)
	if err != nil {
		return fmt.Errorf("stat FROM source %s: %w", modelDir, err)
	}
	draftInfo, err := os.Stat(modelfile.Draft)
	if err != nil {
		return fmt.Errorf("stat DRAFT source %s: %w", modelfile.Draft, err)
	}
	if os.SameFile(modelInfo, draftInfo) {
		return fmt.Errorf("DRAFT must not reference the same local path as FROM: %s", modelfile.Draft)
	}
	return nil
}

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions) create.ManifestWriter {
	var template, system, draftDir string
	var license any
	var parameters map[string]any
	var messages []api.Message
	if opts.Modelfile != nil {
		template = opts.Modelfile.Template
		system = opts.Modelfile.System
		if len(opts.Modelfile.Licenses) > 0 {
			license = opts.Modelfile.Licenses
		}
		draftDir = opts.Modelfile.Draft
		parameters = opts.Modelfile.Parameters
		messages = opts.Modelfile.Messages
	}
	return create.NewSafetensorsManifestWriter(create.SafetensorsManifestOptions{
		MinVersion: create.SafetensorsMinOllamaVersion,
		DraftDir:   draftDir,
		Template:   template,
		System:     system,
		License:    license,
		Parameters: parameters,
		Messages:   messages,
	})
}
