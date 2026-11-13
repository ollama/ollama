package mlxrunner

import (
	"encoding/json"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/deepseek_v41"
	"github.com/ollama/ollama/mlxrunner/model/qwen4_exp"
)

// modelGPUBytes excludes tensor layers the model will read from host storage.
// The selection uses each model's loader rules so a GPU-resident fallback is
// still counted in the admission estimate.
func modelGPUBytes(m *manifest.Manifest) (uint64, error) {
	var total uint64
	for _, layer := range m.TensorLayers() {
		total += uint64(layer.Size)
	}
	if _, ok := m.ConfigLayer("config.json"); !ok {
		return total, nil
	}
	var config struct {
		Architectures []string `json:"architectures"`
	}
	data, err := m.ReadConfig("config.json")
	if err == nil {
		_ = json.Unmarshal(data, &config)
	}
	// Unknown or unreadable configs use the conservative all-GPU estimate.
	if len(config.Architectures) != 1 {
		return total, nil
	}
	var selectHostTensors func(*model.Root) ([]string, error)
	switch config.Architectures[0] {
	case "DeepseekV41ForCausalLM":
		selectHostTensors = deepseek_v41.HostBackedTensorNames
	case "Qwen4ExpForConditionalGeneration":
		selectHostTensors = qwen4_exp.HostBackedTensorNames
	default:
		return total, nil
	}
	root, err := model.OpenManifest(m)
	if err != nil {
		return 0, err
	}
	names, err := selectHostTensors(root)
	if err != nil {
		return 0, err
	}
	hostBacked := make(map[string]bool, len(names))
	for _, name := range names {
		hostBacked[name] = true
	}
	var resident uint64
	for _, layer := range m.TensorLayers() {
		if !hostBacked[layer.Name] {
			resident += uint64(layer.Size)
		}
	}
	return resident, nil
}
