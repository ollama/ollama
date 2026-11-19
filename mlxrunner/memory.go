package mlxrunner

import (
	"encoding/json"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model/hy4_flash"
)

// modelGPUBytes estimates the model's device-resident weight storage.
func modelGPUBytes(m *manifest.Manifest) (uint64, error) {
	layers := m.TensorLayers()
	excluded := map[string]bool{}
	var config struct {
		Architectures []string `json:"architectures"`
	}
	if data, err := m.ReadConfig("config.json"); err == nil && json.Unmarshal(data, &config) == nil &&
		len(config.Architectures) > 0 && config.Architectures[0] == hy4_flash.Architecture {
		for _, name := range hy4_flash.CheckpointSkippedTensorNames(data, layers) {
			excluded[name] = true
		}
	}
	var total uint64
	for _, layer := range layers {
		if !excluded[layer.Name] || !blobFullyExcluded(layer, excluded) {
			total += uint64(layer.Size)
		}
	}
	return total, nil
}

// A manifest layer names a blob, while the runner skips tensors by the
// names in its header. A mixed blob must still count toward the estimate.
func blobFullyExcluded(layer manifest.Layer, excluded map[string]bool) bool {
	path, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return false
	}
	blob, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return false
	}
	defer blob.Close()
	for _, name := range blob.ListTensors() {
		if !excluded[name] {
			return false
		}
	}
	return blob.TensorCount() > 0
}
