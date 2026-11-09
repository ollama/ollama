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

// Qwen3_5Model names the publisher's backbone, not a decision-model class.
// Either decision sidecar identifies a checkpoint that needs full validation;
// preparePplxInventory checks that both files are present and compatible.
func hasPplxSidecar(dir string) (bool, error) {
	for _, name := range []string{"decision_config.json", "readout.safetensors"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return true, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return false, fmt.Errorf("check Perplexity sidecar %s: %w", name, err)
		}
	}
	return false, nil
}

// The decision readout is a separate safetensors file outside the backbone's
// shard index. Inventory must include it explicitly before planning blobs.
func preparePplxInventory(inv Inventory) (Inventory, error) {
	var backbone struct {
		TextConfig struct {
			HiddenSize int32 `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(inv.RawConfig, &backbone); err != nil {
		return Inventory{}, fmt.Errorf("parse Perplexity backbone config: %w", err)
	}
	var decision struct {
		FormatVersion int      `json:"format_version"`
		Codes         []string `json:"codes"`
		TokenIDs      []int32  `json:"token_ids"`
		AttentionMode string   `json:"attention_mode"`
		Pooling       string   `json:"pooling"`
	}
	data, err := os.ReadFile(filepath.Join(inv.Dir, "decision_config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read Perplexity decision config: %w", err)
	}
	if err := json.Unmarshal(data, &decision); err != nil {
		return Inventory{}, fmt.Errorf("parse Perplexity decision config: %w", err)
	}
	if backbone.TextConfig.HiddenSize <= 0 || decision.FormatVersion != 2 || len(decision.Codes) != 255 || len(decision.TokenIDs) != 255 {
		return Inventory{}, fmt.Errorf("invalid Perplexity decision readout configuration")
	}
	var joint struct {
		Version       int    `json:"version"`
		Template      string `json:"template"`
		Marker        string `json:"marker"`
		ReadoutOffset int    `json:"readout_offset"`
	}
	data, err = os.ReadFile(filepath.Join(inv.Dir, "joint_config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read Perplexity joint config: %w", err)
	}
	if err := json.Unmarshal(data, &joint); err != nil {
		return Inventory{}, fmt.Errorf("parse Perplexity joint config: %w", err)
	}
	if joint.Version != 3 || joint.Template != "joint-question-inline-readout-v1" || joint.Marker != "<|fim_suffix|>" || joint.ReadoutOffset != 0 {
		return Inventory{}, fmt.Errorf("unsupported Perplexity joint protocol")
	}
	if decision.AttentionMode != "noncausal_full_attention" || decision.Pooling != "last" {
		return Inventory{}, fmt.Errorf("unsupported Perplexity decision attention mode %q or pooling %q", decision.AttentionMode, decision.Pooling)
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, "readout.safetensors"))
	if err != nil {
		return Inventory{}, fmt.Errorf("open Perplexity readout: %w", err)
	}
	defer reader.Close()
	if names := reader.ListTensors(); !slices.Equal(names, []string{"weight"}) {
		return Inventory{}, fmt.Errorf("Perplexity readout must contain only weight")
	}
	weight, err := reader.GetTensor("weight")
	if err != nil || weight.Dtype != "BF16" || !slices.Equal(weight.Shape, []int32{255, backbone.TextConfig.HiddenSize}) {
		return Inventory{}, fmt.Errorf("Perplexity requires a BF16 readout of shape [255,%d]", backbone.TextConfig.HiddenSize)
	}
	if inv.Has("weight") {
		return Inventory{}, fmt.Errorf("Perplexity readout weight collides with a backbone tensor")
	}
	inv.Tensors["weight"] = SourceTensor{Name: "weight", Dtype: weight.Dtype, Shape: weight.Shape, File: "readout.safetensors"}
	return inv, nil
}

type pplxImportTransform struct{}

func newPplxImportTransform(json.RawMessage) (quantizePolicy, error) {
	return pplxImportTransform{}, nil
}

func (pplxImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Keep the decision readout at its publisher dtype (BF16 for Perplexity,
	// FP32 for d3). Apply the backbone's Qwen3.5 policy to everything else.
	if name == "weight" {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}
