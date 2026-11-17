package create

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
)

const pplxLocateConfigFile = "locate-span/locate_config.json"

var pplxLocateWeights = [...]struct {
	file   string
	prefix string
}{
	{"locate-span/adapter.safetensors", "locate.adapter."},
	{"locate-span/head.safetensors", "locate.head."},
}

type pplxJointConfig struct {
	Version       int    `json:"version"`
	Template      string `json:"template"`
	Marker        string `json:"marker"`
	ReadoutOffset int    `json:"readout_offset"`
	Separator     string `json:"separator"`
	Usage         string `json:"usage"`
	MaxQuestions  *int   `json:"max_questions"`
	MaxLength     *int   `json:"max_length"`
}

func (joint pplxJointConfig) chunked() bool {
	return joint.Version == 4 && joint.Template == "joint-chunked-readout-v1" &&
		joint.Separator == "<|fim_middle|>" && joint.Usage == "nonpadding_joint_sequence_including_template_and_image_positions" &&
		joint.MaxQuestions == nil && joint.MaxLength == nil
}

func readPplxJointConfig(dir string) (pplxJointConfig, error) {
	var joint pplxJointConfig
	data, err := os.ReadFile(filepath.Join(dir, "joint_config.json"))
	if err != nil {
		return joint, fmt.Errorf("read Perplexity joint config: %w", err)
	}
	if err := json.Unmarshal(data, &joint); err != nil {
		return joint, fmt.Errorf("parse Perplexity joint config: %w", err)
	}
	return joint, nil
}

// Only the chunked PPLX protocol uses the Locate sidecars. A stray directory
// in an older checkpoint does not change which files enter the model.
func pplxLocateSourceFiles(dir string) (config string, weights []string, err error) {
	joint, err := readPplxJointConfig(dir)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil, nil
	}
	if err != nil || !joint.chunked() {
		return "", nil, err
	}
	var files []string
	for _, weight := range pplxLocateWeights {
		files = append(files, weight.file)
	}
	for _, name := range append([]string{pplxLocateConfigFile}, files...) {
		info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name)))
		if err != nil {
			return "", nil, fmt.Errorf("check Perplexity Locate %s: %w", name, err)
		}
		if !info.Mode().IsRegular() {
			return "", nil, fmt.Errorf("Perplexity Locate %s must be a regular file", name)
		}
	}
	return pplxLocateConfigFile, files, nil
}

func qwenDecisionWeightFiles(dir string, files []string) ([]string, error) {
	files, err := appendWeightSidecar(dir, files, "readout.safetensors", "Qwen decision")
	if err != nil {
		return nil, err
	}
	format, err := qwenDecisionFormat(dir)
	if err != nil || format == d3FormatID {
		return files, err
	}
	_, weights, err := pplxLocateSourceFiles(dir)
	if err != nil {
		return nil, err
	}
	if len(weights) != 0 {
		files = append(files, weights...)
		slices.Sort(files)
	}
	return files, nil
}

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
	joint, err := readPplxJointConfig(inv.Dir)
	if err != nil {
		return Inventory{}, err
	}
	legacy := joint.Version == 3 && joint.Template == "joint-question-inline-readout-v1"
	chunked := joint.chunked()
	if (!legacy && !chunked) || joint.Marker != "<|fim_suffix|>" || joint.ReadoutOffset != 0 {
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
	if chunked {
		if err := addPplxLocateInventory(&inv, backbone.TextConfig.HiddenSize); err != nil {
			return Inventory{}, err
		}
	}
	return inv, nil
}

func addPplxLocateInventory(inv *Inventory, hidden int32) error {
	var cfg struct {
		Format   string `json:"format"`
		Marker   string `json:"marker"`
		MarkerID int32  `json:"marker_id"`
		Base     struct {
			SHA256 map[string]string `json:"sha256"`
		} `json:"base"`
		LoRA struct {
			Rank   int     `json:"rank"`
			Alpha  float64 `json:"alpha"`
			Layers int     `json:"layers"`
			Merged bool    `json:"merged"`
		} `json:"lora"`
		Head struct {
			Width     int `json:"width"`
			MaxSpan   int `json:"max_span"`
			Proposals int `json:"proposals"`
		} `json:"head"`
	}
	path := filepath.Join(inv.Dir, filepath.FromSlash(pplxLocateConfigFile))
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read Perplexity Locate configuration: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse Perplexity Locate configuration: %w", err)
	}
	if cfg.Format != "pplx-decider-locate-span-v1" || cfg.Marker != "<|locate|>" || cfg.MarkerID < 0 ||
		cfg.LoRA.Rank != 64 || cfg.LoRA.Alpha != 64 || cfg.LoRA.Layers != 0 || cfg.LoRA.Merged ||
		cfg.Head.Width != 128 || cfg.Head.MaxSpan != 128 || cfg.Head.Proposals != 128 {
		return fmt.Errorf("unsupported Perplexity Locate adapter configuration")
	}
	for _, name := range []string{"config.json", "decision_config.json", "joint_config.json", "readout.safetensors", "tokenizer.json"} {
		want := cfg.Base.SHA256[name]
		if len(want) != 64 {
			return fmt.Errorf("Perplexity Locate adapter has no base digest for %s", name)
		}
		file, err := os.Open(filepath.Join(inv.Dir, name))
		if err != nil {
			return fmt.Errorf("open Perplexity Locate base %s: %w", name, err)
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, file)
		closeErr := file.Close()
		if copyErr != nil {
			return fmt.Errorf("hash Perplexity Locate base %s: %w", name, copyErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close Perplexity Locate base %s: %w", name, closeErr)
		}
		if fmt.Sprintf("%x", hash.Sum(nil)) != want {
			return fmt.Errorf("Perplexity Locate adapter does not match base %s", name)
		}
	}
	for _, item := range pplxLocateWeights {
		reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, filepath.FromSlash(item.file)))
		if err != nil {
			return fmt.Errorf("open Perplexity Locate %s: %w", item.file, err)
		}
		names := reader.ListTensors()
		if len(names) == 0 {
			reader.Close()
			return fmt.Errorf("Perplexity Locate %s has no tensors", item.file)
		}
		for _, name := range names {
			tensor, err := reader.GetTensor(name)
			if err != nil || tensor.Dtype != "F32" {
				reader.Close()
				return fmt.Errorf("Perplexity Locate tensor %s must be FP32: %v", name, err)
			}
			key := item.prefix + name
			if inv.Has(key) {
				reader.Close()
				return fmt.Errorf("duplicate Perplexity Locate tensor %s", key)
			}
			inv.Tensors[key] = SourceTensor{Name: name, Dtype: tensor.Dtype, Shape: tensor.Shape, File: item.file}
		}
		reader.Close()
	}
	if marker := inv.Tensors["locate.adapter.language_model.embed_tokens.locate"]; !slices.Equal(marker.Shape, []int32{hidden}) {
		return fmt.Errorf("Perplexity Locate marker embedding must have %d values", hidden)
	}
	for _, name := range []string{"ts", "te", "qs", "qe", "qp"} {
		if tensor := inv.Tensors["locate.head."+name+".weight"]; !slices.Equal(tensor.Shape, []int32{128, hidden}) {
			return fmt.Errorf("Perplexity Locate %s projection has wrong shape", name)
		}
	}
	return nil
}

type pplxImportTransform struct{}

func newPplxImportTransform(json.RawMessage) (quantizePolicy, error) {
	return pplxImportTransform{}, nil
}

func (pplxImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Keep the decision readout at its publisher dtype (BF16 for Perplexity,
	// FP32 for d3). Apply the backbone's Qwen3.5 policy to everything else.
	if name == "weight" || strings.HasPrefix(name, "locate.") {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}
