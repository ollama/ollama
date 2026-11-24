package create

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
)

const d3FormatID = "d3-code-readout-v1"

func qwenDecisionFormat(dir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(dir, "decision_config.json"))
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	var cfg struct {
		FormatID string `json:"format_id"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", fmt.Errorf("parse decision_config.json: %w", err)
	}
	if cfg.FormatID != "" && cfg.FormatID != d3FormatID {
		return "", fmt.Errorf("unsupported Qwen decision format %q", cfg.FormatID)
	}
	return cfg.FormatID, nil
}

func prepareD3Inventory(inv Inventory) (Inventory, error) {
	var backbone struct {
		TextConfig struct {
			HiddenSize int32 `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(inv.RawConfig, &backbone); err != nil {
		return Inventory{}, fmt.Errorf("parse d3 backbone config: %w", err)
	}
	data, err := os.ReadFile(filepath.Join(inv.Dir, "decision_config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read d3 decision config: %w", err)
	}
	var decision struct {
		FormatVersion int      `json:"format_version"`
		FormatID      string   `json:"format_id"`
		Prompt        string   `json:"prompt"`
		Codes         []string `json:"codes"`
		TokenIDs      []int32  `json:"token_ids"`
		AttentionMode string   `json:"attention_mode"`
		Pooling       string   `json:"pooling"`
		ReadoutDType  string   `json:"readout_dtype"`
	}
	if err := json.Unmarshal(data, &decision); err != nil {
		return Inventory{}, fmt.Errorf("parse d3 decision config: %w", err)
	}
	if backbone.TextConfig.HiddenSize <= 0 || decision.FormatVersion != 1 || decision.FormatID != d3FormatID || decision.Prompt != "d3" || len(decision.Codes) != 255 || len(decision.TokenIDs) != 255 || decision.AttentionMode != "noncausal_full_attention" || decision.Pooling != "last" || decision.ReadoutDType != "float32" {
		return Inventory{}, fmt.Errorf("unsupported d3 decision config (format %q, attention %q)", decision.FormatID, decision.AttentionMode)
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, "readout.safetensors"))
	if err != nil {
		return Inventory{}, fmt.Errorf("open d3 readout: %w", err)
	}
	defer reader.Close()
	if names := reader.ListTensors(); !slices.Equal(names, []string{"weight"}) {
		return Inventory{}, fmt.Errorf("d3 readout must contain only weight")
	}
	weight, err := reader.GetTensor("weight")
	if err != nil || weight.Dtype != "F32" || !slices.Equal(weight.Shape, []int32{255, backbone.TextConfig.HiddenSize}) {
		return Inventory{}, fmt.Errorf("d3 requires an FP32 readout of shape [255,%d]", backbone.TextConfig.HiddenSize)
	}
	if inv.Has("weight") {
		return Inventory{}, fmt.Errorf("d3 readout weight collides with a backbone tensor")
	}
	inv.Tensors["weight"] = SourceTensor{Name: "weight", Dtype: weight.Dtype, Shape: weight.Shape, File: "readout.safetensors"}
	return inv, nil
}
