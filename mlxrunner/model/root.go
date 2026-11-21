package model

import (
	"encoding/json"
	"os"
	"strings"

	"github.com/ollama/ollama/manifest"
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
