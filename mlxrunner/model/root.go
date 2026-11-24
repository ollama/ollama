package model

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	modeltypes "github.com/ollama/ollama/types/model"
)

// Root wraps a model's manifest with pre-scanned quantization metadata.
type Root struct {
	Manifest *manifest.Manifest
	Draft    *modeltypes.Draft

	// Backwards-compatible model-level quant metadata (first tensor blob).
	quantType string
	groupSize int

	// Per-tensor quantization metadata.
	tensorQuant map[string]*TensorQuantInfo
}

// Open loads a manifest for the given model name and scans tensor blobs for
// quantization metadata.
func Open(modelName string) (*Root, error) {
	m, err := manifest.ParseNamedManifest(modeltypes.ParseName(modelName))
	if err != nil {
		return nil, err
	}

	root := &Root{
		Manifest:    m,
		tensorQuant: make(map[string]*TensorQuantInfo),
	}
	root.Draft = readDraftConfig(m)

	for _, layer := range m.TensorLayers() {
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		infos, blobQuantType, blobGroupSize, err := readBlobTensorQuantInfo(blobPath)
		if err != nil {
			continue
		}

		for name, info := range infos {
			root.tensorQuant[name] = info
		}

		if root.quantType == "" && blobQuantType != "" {
			root.quantType = strings.ToUpper(blobQuantType)
			root.groupSize = blobGroupSize
			if root.groupSize == 0 {
				root.groupSize = defaultGroupSize(root.quantType)
			}
		}
	}

	return root, nil
}

func readDraftConfig(m *manifest.Manifest) *modeltypes.Draft {
	if m == nil || m.Config.Digest == "" {
		return nil
	}

	configPath, err := manifest.BlobsPath(m.Config.Digest)
	if err != nil {
		return nil
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var cfg modeltypes.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil
	}
	if cfg.Draft != nil {
		return cfg.Draft
	}

	if _, ok := m.ConfigLayer("draft/config.json"); ok {
		return &modeltypes.Draft{
			ModelFormat:  "safetensors",
			TensorPrefix: "draft.",
			Config:       "draft/config.json",
		}
	}
	return nil
}

// QuantType returns the quantization type detected from the first tensor blob metadata.
func (r *Root) QuantType() string { return r.quantType }

// GroupSize returns the quantization group size detected from the first tensor blob metadata.
func (r *Root) GroupSize() int { return r.groupSize }

// TensorQuant returns per-tensor quantization metadata if available.
func (r *Root) TensorQuant(name string) *TensorQuantInfo {
	if r == nil {
		return nil
	}
	return r.tensorQuant[name]
}

// AllTensorQuant returns a copy of the per-tensor quantization metadata.
func (r *Root) AllTensorQuant() map[string]*TensorQuantInfo {
	out := make(map[string]*TensorQuantInfo, len(r.tensorQuant))
	for k, v := range r.tensorQuant {
		if v == nil {
			continue
		}
		copy := *v
		out[k] = &copy
	}
	return out
}

// Quantization returns the model-level quantization defaults that linear
// layers fall back to, and a copy of the per-tensor metadata.
func (r *Root) Quantization() (groupSize, bits int, mode string, tensorQuant map[string]*TensorQuantInfo) {
	groupSize, bits, mode = QuantizationParams(r.QuantType())
	if gs := r.GroupSize(); gs > 0 {
		groupSize = gs
	}
	return groupSize, bits, mode, r.AllTensorQuant()
}

// LoadTokenizer loads tokenizer.json with configJSON (the model's config.json)
// and, when present, generation_config.json and tokenizer_config.json.
func (r *Root) LoadTokenizer(configJSON []byte) (*tokenizer.Tokenizer, error) {
	data, err := r.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	config := &tokenizer.TokenizerConfig{ConfigJSON: configJSON}
	if data, err := r.Manifest.ReadConfig("generation_config.json"); err == nil {
		config.GenerationConfigJSON = data
	}
	if data, err := r.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		config.TokenizerConfigJSON = data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(data, config)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	return tok, nil
}
