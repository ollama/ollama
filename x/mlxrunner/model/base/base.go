//go:build mlx

package base

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/tokenizer"
)

// Model is the interface that model implementations must satisfy.
type Model interface {
	Forward(inputs *mlx.Array, cache []cache.Cache) *mlx.Array
	Unembed(x *mlx.Array) *mlx.Array
	NumLayers() int
	Tokenizer() *tokenizer.Tokenizer

	// LoadWeights receives all tensors loaded from the manifest and assigns
	// them to model fields. Model-specific logic (MLA absorption, expert
	// stacking, quantized layer creation) happens here.
	LoadWeights(tensors map[string]*mlx.Array) error
}

var (
	mu       sync.Mutex
	registry = make(map[string]func(root *model.Root) (Model, error))
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
