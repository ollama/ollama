package base

import (
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

// DraftModel is an auxiliary model alongside a target that proposes speculative
// tokens.
type DraftModel interface {
	// Draft fuses b.Hidden (the target hidden state) into its own forward and
	// returns the head's hidden plus the projected hidden seeding the next step.
	Draft(b *batch.Batch, caches []cache.Cache) (hidden, projected *mlx.Array)

	// Unembed projects a hidden state to vocabulary logits.
	Unembed(x *mlx.Array) *mlx.Array

	// DraftCaches selects the draft model's own KV caches from the full
	// per-request slice — any subset, or nil when the draft keeps no KV.
	DraftCaches(caches []cache.Cache) []cache.Cache

	// LoadWeights assigns manifest tensors to the draft head's fields.
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

// SelfDraft is implemented by models whose draft head ships inline with the
// target weights; it returns the head, or nil when the checkpoint shipped none.
type SelfDraft interface {
	SelfDraft() DraftModel
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
