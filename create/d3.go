package create

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
)

const (
	d3FormatID          = "d3-code-readout-v1"
	autoTrustFormatID   = "jev-code-readout-v1"
	readoutTensorName   = "weight"
	lmHeadReadoutTensor = "lm_head.weight"
)

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
	if cfg.FormatID != "" && cfg.FormatID != d3FormatID && cfg.FormatID != autoTrustFormatID {
		return "", fmt.Errorf("unsupported Qwen decision format %q", cfg.FormatID)
	}
	return cfg.FormatID, nil
}

type codeReadoutConfig struct {
	FormatVersion int      `json:"format_version"`
	FormatID      string   `json:"format_id"`
	Prompt        string   `json:"prompt"`
	Codes         []string `json:"codes"`
	TokenIDs      []int32  `json:"token_ids"`
	Temperature   float32  `json:"temperature"`
	AttentionMode string   `json:"attention_mode"`
	Pooling       string   `json:"pooling"`
	ReadoutDType  string   `json:"readout_dtype"`
}

func readCodeReadoutConfig(inv Inventory, prompt string, hidden, vocab int32) (codeReadoutConfig, error) {
	data, err := os.ReadFile(filepath.Join(inv.Dir, "decision_config.json"))
	if err != nil {
		return codeReadoutConfig{}, fmt.Errorf("read decision config: %w", err)
	}
	var cfg codeReadoutConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return codeReadoutConfig{}, fmt.Errorf("parse decision config: %w", err)
	}
	if hidden <= 0 || cfg.FormatVersion != 1 || cfg.FormatID == "" || cfg.Prompt != prompt || len(cfg.Codes) != 255 || len(cfg.TokenIDs) != 255 || cfg.AttentionMode != "noncausal_full_attention" || cfg.Pooling != "last" || cfg.ReadoutDType != "float32" {
		return codeReadoutConfig{}, fmt.Errorf("unsupported decision config (format %q, prompt %q, attention %q)", cfg.FormatID, cfg.Prompt, cfg.AttentionMode)
	}
	if cfg.Temperature <= 0 || math.IsNaN(float64(cfg.Temperature)) || math.IsInf(float64(cfg.Temperature), 0) {
		return codeReadoutConfig{}, fmt.Errorf("invalid decision temperature %g", cfg.Temperature)
	}
	seen := make(map[int32]bool, len(cfg.TokenIDs))
	for _, id := range cfg.TokenIDs {
		if id < 0 || (vocab > 0 && id >= vocab) || seen[id] {
			return codeReadoutConfig{}, fmt.Errorf("answer code token id %d is out of range or duplicated", id)
		}
		seen[id] = true
	}
	return cfg, nil
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
	if _, err := readCodeReadoutConfig(inv, "d3", backbone.TextConfig.HiddenSize, 0); err != nil {
		return Inventory{}, err
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, "readout.safetensors"))
	if err != nil {
		return Inventory{}, fmt.Errorf("open d3 readout: %w", err)
	}
	defer reader.Close()
	if names := reader.ListTensors(); !slices.Equal(names, []string{readoutTensorName}) {
		return Inventory{}, fmt.Errorf("d3 readout must contain only %s", readoutTensorName)
	}
	weight, err := reader.GetTensor(readoutTensorName)
	if err != nil || weight.Dtype != "F32" || !slices.Equal(weight.Shape, []int32{255, backbone.TextConfig.HiddenSize}) {
		return Inventory{}, fmt.Errorf("d3 requires an FP32 readout of shape [255,%d]", backbone.TextConfig.HiddenSize)
	}
	if inv.Has(readoutTensorName) {
		return Inventory{}, fmt.Errorf("d3 readout %s collides with a backbone tensor", readoutTensorName)
	}
	inv.Tensors[readoutTensorName] = SourceTensor{Name: readoutTensorName, Dtype: weight.Dtype, Shape: weight.Shape, File: "readout.safetensors"}
	return inv, nil
}

func prepareAutoTrustCodeReadoutInventory(inv Inventory) (Inventory, error) {
	if arch := inv.Config.Architecture(); arch != "Qwen3_5ForConditionalGeneration" && arch != "Qwen3_5Model" {
		return Inventory{}, fmt.Errorf("unsupported code-readout backbone %q", arch)
	}
	var backbone struct {
		TextConfig struct {
			HiddenSize int32 `json:"hidden_size"`
			VocabSize  int32 `json:"vocab_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(inv.RawConfig, &backbone); err != nil {
		return Inventory{}, fmt.Errorf("parse code-readout backbone config: %w", err)
	}
	cfg, err := readCodeReadoutConfig(inv, "jev", backbone.TextConfig.HiddenSize, backbone.TextConfig.VocabSize)
	if err != nil {
		return Inventory{}, err
	}
	if err := validateFoldedReadout(inv, cfg.TokenIDs, backbone.TextConfig.HiddenSize, backbone.TextConfig.VocabSize); err != nil {
		return Inventory{}, err
	}

	var output map[string]json.RawMessage
	if err := json.Unmarshal(inv.RawConfig, &output); err != nil {
		return Inventory{}, err
	}
	// Route the publisher's conditional-generation checkpoint through the decision loader.
	output["architectures"] = json.RawMessage(`["Qwen3_5Model"]`)
	inv.RawConfig, err = json.Marshal(output)
	if err != nil {
		return Inventory{}, err
	}
	inv.Config.Architectures = []string{"Qwen3_5Model"}
	return inv, nil
}

func validateFoldedReadout(inv Inventory, tokenIDs []int32, hiddenSize, vocabSize int32) error {
	if inv.Has(readoutTensorName) {
		return fmt.Errorf("folded readout cannot also ship %s", readoutTensorName)
	}
	weight, ok := inv.Tensors[lmHeadReadoutTensor]
	if !ok {
		return fmt.Errorf("checkpoint does not ship %s", lmHeadReadoutTensor)
	}
	if weight.Dtype != "BF16" || !slices.Equal(weight.Shape, []int32{vocabSize, hiddenSize}) {
		return fmt.Errorf("%s must be BF16 [%d,%d], got %s %v", lmHeadReadoutTensor, vocabSize, hiddenSize, weight.Dtype, weight.Shape)
	}
	for _, id := range tokenIDs {
		if id >= weight.Shape[0] {
			return fmt.Errorf("answer code token id %d is outside the %d-row readout", id, weight.Shape[0])
		}
	}
	return nil
}

// Keep answer-code rows at source precision during import quantization.
type foldedReadoutImportTransform struct{}

func (foldedReadoutImportTransform) quantizationType(name string, shape []int32, requested string) string {
	if name == lmHeadReadoutTensor {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}
