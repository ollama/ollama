package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
)

// Clef's config and shard index describe only its backbone. RENDERER clef adds
// the joint head and normalizes the output architecture without changing source files.
func prepareClefInventory(inv Inventory) (Inventory, error) {
	if inv.Config.Architecture() != "Qwen3_5ForConditionalGeneration" {
		return Inventory{}, fmt.Errorf("unsupported Clef backbone %q", inv.Config.Architecture())
	}
	var backbone struct {
		TextConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(inv.RawConfig, &backbone); err != nil {
		return Inventory{}, err
	}
	var cfg struct {
		HiddenSize    int `json:"hidden_size"`
		Width         int `json:"width"`
		RoutingLayers int `json:"routing_layers"`
		Layers        int `json:"layers"`
		Heads         int `json:"heads"`
		Feedforward   int `json:"feedforward"`
	}
	data, err := os.ReadFile(filepath.Join(inv.Dir, "joint_head_config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read Clef joint head config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Inventory{}, fmt.Errorf("parse Clef joint head config: %w", err)
	}
	if cfg.HiddenSize <= 0 || cfg.HiddenSize != backbone.TextConfig.HiddenSize || cfg.Width <= 0 || cfg.Heads <= 0 || cfg.Width%cfg.Heads != 0 || cfg.Feedforward <= 0 || cfg.Layers < 1 || cfg.Layers > 16 || cfg.RoutingLayers < 1 || cfg.RoutingLayers > 16 {
		return Inventory{}, fmt.Errorf("invalid Clef joint head configuration for backbone")
	}
	head, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, "joint_head.safetensors"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read Clef joint head: %w", err)
	}
	defer head.Close()
	for name, shape := range map[string][]int32{
		"memory_projection.weight": {int32(cfg.Width), int32(cfg.HiddenSize)},
		"type_embedding.weight":    {3, int32(cfg.Width)},
	} {
		tensor, err := head.GetTensor(name)
		if err != nil || !slices.Equal(tensor.Shape, shape) {
			return Inventory{}, fmt.Errorf("Clef joint head requires %s with shape %v", name, shape)
		}
	}
	for _, name := range head.ListTensors() {
		if tensor, ok := inv.Tensors[name]; ok && tensor.File != "joint_head.safetensors" {
			return Inventory{}, fmt.Errorf("Clef head duplicates backbone tensor %q", name)
		}
		tensor, err := head.GetTensor(name)
		if err != nil {
			return Inventory{}, fmt.Errorf("read Clef head tensor %s: %w", name, err)
		}
		inv.Tensors[name] = SourceTensor{Name: name, Dtype: tensor.Dtype, Shape: tensor.Shape, File: "joint_head.safetensors"}
	}
	var output map[string]json.RawMessage
	if err := json.Unmarshal(inv.RawConfig, &output); err != nil {
		return Inventory{}, err
	}
	output["architectures"] = json.RawMessage(`["ClefForDecision"]`)
	inv.RawConfig, err = json.Marshal(output)
	if err != nil {
		return Inventory{}, err
	}
	inv.Config.Architectures = []string{"ClefForDecision"}
	return inv, nil
}

// The publisher's head is not in the shard index. Transfer it without using
// its presence to select a model architecture.
func clefWeightFiles(dir string, files []string) ([]string, error) {
	info, err := os.Stat(filepath.Join(dir, "joint_head.safetensors"))
	if os.IsNotExist(err) {
		return files, nil
	}
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("Clef joint_head.safetensors must be a regular file")
	}
	if !slices.Contains(files, "joint_head.safetensors") {
		files = append(files, "joint_head.safetensors")
		slices.Sort(files)
	}
	return files, nil
}
