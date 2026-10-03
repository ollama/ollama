package model

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
)

// Extractor runs a full encoder pass and returns scored spans in the input.
// It has no vocabulary projection or autoregressive KV cache.
type Extractor interface {
	LoadWeights(map[string]*mlx.Array) error
	MaxContextLength() int
	Extract(context.Context, api.ExtractRequest) (*api.ExtractResponse, error)
}

var extractors = make(map[string]func(*Root) (Extractor, error))

func RegisterExtractor(arch string, fn func(*Root) (Extractor, error)) {
	mu.Lock()
	defer mu.Unlock()
	if _, exists := extractors[arch]; exists {
		panic(fmt.Sprintf("extractor architecture %q already registered", arch))
	}
	extractors[arch] = fn
}

// NewExtractor returns nil for a model registered on the generation path.
func NewExtractor(root *Root) (Extractor, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if len(cfg.Architectures) == 0 {
		return nil, nil
	}
	mu.Lock()
	fn := extractors[cfg.Architectures[0]]
	mu.Unlock()
	if fn == nil {
		return nil, nil
	}
	return fn(root)
}
