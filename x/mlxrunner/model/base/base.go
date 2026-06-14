package base

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/tokenizer"
)

// Model is the interface that model implementations must satisfy.
type Model interface {
	Forward(b *batch.Batch, cache []cache.Cache) *mlx.Array
	Unembed(x *mlx.Array) *mlx.Array
	NumLayers() int
	Tokenizer() *tokenizer.Tokenizer
	MaxContextLength() int

	// LoadWeights receives all tensors loaded from the manifest and assigns
	// them to model fields. Model-specific logic (MLA absorption, expert
	// stacking, quantized layer creation) happens here.
	LoadWeights(tensors map[string]*mlx.Array) error
}

// DraftModel is an auxiliary model stored alongside a target model.
type DraftModel interface {
	LoadWeights(tensors map[string]*mlx.Array) error
}

// MTPDefaults holds model-provided draft-token defaults for speculative
// decoding. Environment settings in the runner may override these values.
type MTPDefaults struct {
	InitialDraftTokens int
	MaxDraftTokens     int
	Enabled            bool
}

// MTPDefaultsProvider lets a model provide MTP policy defaults from its own
// config without teaching the runner model-specific shape heuristics.
type MTPDefaultsProvider interface {
	MTPDraftDefaults(sample bool) MTPDefaults
}

// MTPDraftModel is a draft model capable of Gemma-style multi-token
// prediction from target token embeddings, target hidden states, and target KV.
type MTPDraftModel interface {
	Draft(inputEmbeds *mlx.Array, position int32, caches []cache.Cache) (logits, hidden *mlx.Array)
}

// MTPEmbeddingModel exposes the target token embedding path used by MTP drafts.
type MTPEmbeddingModel interface {
	TokenEmbeddings(inputIDs *mlx.Array) *mlx.Array
}

// DiffuseConfig holds the resolved runtime settings for one block-diffusion
// generation. The runner obtains it from the model via ResolveDiffuse and passes
// it back to Diffuse.
type DiffuseConfig struct {
	Canvas              int // canvas (block) width in tokens
	Steps               int // max denoising steps per canvas
	PredictTokens       int // requested generated tokens, capped by the runner
	MaxCanvases         int // number of autoregressive canvases to generate
	SelfCondK           int // self-conditioning top-k gather width
	StabilityThreshold  int // consecutive-stable steps before early stop (0 = always stable)
	TMin                float32
	TMax                float32
	EntropyBound        float32
	ConfidenceThreshold float32
	Seed                int64
}

// DiffusionModel is implemented by block-diffusion models (e.g. diffusion-gemma).
// The runner routes such models to its diffusion pipeline instead of the
// autoregressive one.
type DiffusionModel interface {
	Model
	// ResolveDiffuse derives the runtime config from the model's parsed defaults,
	// the requested predict length (which sets the number of canvases), an
	// optional step override (<= 0 means use the model default), and a seed.
	ResolveDiffuse(nPredict, steps int, seed int64) DiffuseConfig
	// Diffuse runs the denoising loop over prompt, invoking emit for each
	// generated token id in order. emit returning a non-nil error stops generation.
	Diffuse(ctx context.Context, prompt []int32, cfg DiffuseConfig, emit func(token int32) error) error
}

var (
	mu            sync.Mutex
	registry      = make(map[string]func(root *model.Root) (Model, error))
	draftRegistry = make(map[string]func(root *model.Root, target Model) (DraftModel, error))
)

// Register registers a model constructor by architecture name.
// Called from init() in model packages. Panics on duplicate registration.
func Register(arch string, fn func(root *model.Root) (Model, error)) {
	mu.Lock()
	defer mu.Unlock()

	if _, exists := registry[arch]; exists {
		panic(fmt.Sprintf("model architecture %q already registered", arch))
	}
	registry[arch] = fn
}

// RegisterDraft registers a draft model constructor by architecture name.
func RegisterDraft(arch string, fn func(root *model.Root, target Model) (DraftModel, error)) {
	mu.Lock()
	defer mu.Unlock()

	if _, exists := draftRegistry[arch]; exists {
		panic(fmt.Sprintf("draft model architecture %q already registered", arch))
	}
	draftRegistry[arch] = fn
}

// New reads config.json from the manifest, detects the architecture, looks up
// the registered constructor, and calls it to create the model (with config
// parsed and struct created, but weights not yet loaded).
func New(root *model.Root) (Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	var archConfig struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config.json: %w", err)
	}

	if len(archConfig.Architectures) == 0 {
		return nil, fmt.Errorf("no architectures found in config.json")
	}

	arch := archConfig.Architectures[0]
	slog.Info("Model architecture", "arch", arch)

	mu.Lock()
	fn, ok := registry[arch]
	mu.Unlock()

	if !ok {
		return nil, fmt.Errorf("unsupported architecture: %s", arch)
	}

	return fn(root)
}

// NewDraft constructs the draft model described by the manifest config, if any.
func NewDraft(root *model.Root, target Model) (DraftModel, error) {
	if root == nil || root.Draft == nil {
		return nil, nil
	}

	configPath := root.Draft.Config
	if configPath == "" {
		configPath = "draft/config.json"
	}
	configData, err := root.Manifest.ReadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", configPath, err)
	}

	var archConfig struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", configPath, err)
	}

	arch := root.Draft.Architecture
	if arch == "" && len(archConfig.Architectures) > 0 {
		arch = archConfig.Architectures[0]
	}
	if arch == "" {
		arch = archConfig.ModelType
	}
	if arch == "" {
		return nil, fmt.Errorf("no draft architecture found in %s", configPath)
	}
	slog.Info("Draft model architecture", "arch", arch)

	mu.Lock()
	fn, ok := draftRegistry[arch]
	mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("unsupported draft architecture: %s", arch)
	}

	return fn(root, target)
}

// Weights returns a function that loads model weights, then pins all
// arrays reachable from the model struct and sweeps everything else.
func Weights(m Model) func(map[string]*mlx.Array) error {
	return func(tensors map[string]*mlx.Array) error {
		if err := m.LoadWeights(tensors); err != nil {
			return err
		}

		collected := mlx.Collect(m)
		for _, arr := range collected {
			mlx.Pin(arr)
		}
		mlx.Sweep()
		mlx.Eval(collected...)

		return nil
	}
}
