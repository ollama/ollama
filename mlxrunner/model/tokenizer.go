package model

import (
	"errors"
	"fmt"
	"path"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// ErrNoTokenizer reports a checkpoint without a text tokenizer, or one whose
// architecture is not registered.
var ErrNoTokenizer = errors.New("no tokenizer for architecture")

// TokenizerOption describes where a checkpoint's tokenizer differs from the
// standard layout: tokenizer.json with optional tokenizer_config.json and
// generation_config.json beside config.json.
type TokenizerOption func(*tokenizerSpec)

type tokenizerSpec struct {
	dir              string
	generationConfig []byte
	none             bool
}

// WithTokenizerDir reads tokenizer.json and tokenizer_config.json from dir.
func WithTokenizerDir(dir string) TokenizerOption {
	return func(s *tokenizerSpec) { s.dir = dir }
}

// WithGenerationConfig replaces generation_config.json, for checkpoints that
// do not ship the special token IDs the model generates with.
func WithGenerationConfig(data []byte) TokenizerOption {
	return func(s *tokenizerSpec) { s.generationConfig = data }
}

// WithoutTokenizer marks an architecture with no text tokenizer.
func WithoutTokenizer() TokenizerOption {
	return func(s *tokenizerSpec) { s.none = true }
}

// Tokenizer builds the checkpoint's tokenizer as its architecture registered
// it. Model constructors and standalone tokenization share this path.
func (r *Root) Tokenizer() (*tokenizer.Tokenizer, error) {
	return LoadTokenizer(r.Manifest)
}

// LoadTokenizer builds the tokenizer for the manifest's architecture without
// reading weights. The error wraps [ErrNoTokenizer] when there is none.
func LoadTokenizer(m *manifest.Manifest) (*tokenizer.Tokenizer, error) {
	configData, err := m.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	arch, err := architecture(configData)
	if err != nil {
		return nil, err
	}

	mu.Lock()
	spec, ok := tokenizerSpecs[arch]
	mu.Unlock()
	if !ok || spec.none {
		return nil, fmt.Errorf("%w %s", ErrNoTokenizer, arch)
	}

	tokData, err := m.ReadConfig(path.Join(spec.dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData, GenerationConfigJSON: spec.generationConfig}
	if tokConfig.GenerationConfigJSON == nil {
		if tokConfig.GenerationConfigJSON, err = readOptionalConfig(m, "generation_config.json"); err != nil {
			return nil, err
		}
	}
	if tokConfig.TokenizerConfigJSON, err = readOptionalConfig(m, path.Join(spec.dir, "tokenizer_config.json")); err != nil {
		return nil, err
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	return tok, nil
}

// readOptionalConfig returns nil for a file the manifest does not list, and
// fails if a listed file cannot be read.
func readOptionalConfig(m *manifest.Manifest, name string) ([]byte, error) {
	if _, ok := m.ConfigLayer(name); !ok {
		return nil, nil
	}
	data, err := m.ReadConfig(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	return data, nil
}
