package create

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
)

// SourceTensor describes one tensor found in a source model: its on-disk type
// and shape and which safetensors file holds it. It carries no weight data —
// only what the header and shard index reveal.
type SourceTensor struct {
	Name  string
	Dtype string
	Shape []int32
	File  string // safetensors file basename, relative to the model directory
}

// Inventory is the immutable result of reading a source model: every tensor
// indexed by name, plus the parsed config and the model directory. Reading
// source headers happens only here; the classify, plan, and write steps work
// entirely from this listing and never re-open a source header to make a
// decision. RawConfig holds the config.json bytes so architecture-specific
// factories can parse their own fields without re-opening the file.
type Inventory struct {
	Dir       string
	Config    sourceModelConfig
	RawConfig json.RawMessage
	Tensors   map[string]SourceTensor
}

// Has reports whether a tensor with the given name exists in the source.
func (inv Inventory) Has(name string) bool {
	_, ok := inv.Tensors[name]
	return ok
}

// ReadInventory reads a source model directory into an Inventory: the config,
// the shard index, and every tensor's header. It reads no weight data. If the
// shard index references a tensor that cannot be found (a missing or truncated
// shard, e.g. a partial download), it fails rather than silently producing an
// incomplete model.
func ReadInventory(dir string) (Inventory, error) {
	cfg, rawConfig, err := readSourceModelConfig(dir)
	if err != nil {
		return Inventory{}, fmt.Errorf("read config: %w", err)
	}

	index, files, err := safetensorsWeightFiles(dir)
	if err != nil {
		return Inventory{}, err
	}

	tensors := make(map[string]SourceTensor)
	for _, file := range files {
		ext, err := safetensors.OpenForExtraction(filepath.Join(dir, file))
		if err != nil {
			if len(index) > 0 {
				return Inventory{}, fmt.Errorf("source model is incomplete: open indexed shard %s: %w", file, err)
			}
			return Inventory{}, fmt.Errorf("open %s: %w", file, err)
		}
		for _, name := range ext.ListTensors() {
			if expectedFile, indexed := index[name]; len(index) > 0 {
				if !indexed {
					continue
				}
				if expectedFile != file {
					ext.Close()
					return Inventory{}, fmt.Errorf("source model is incomplete: tensor %s is indexed in %q but found in %q", name, expectedFile, file)
				}
			}
			td, err := ext.GetTensor(name)
			if err != nil {
				ext.Close()
				return Inventory{}, fmt.Errorf("read tensor %s from %s: %w", name, file, err)
			}
			if prev, ok := tensors[name]; ok {
				ext.Close()
				return Inventory{}, fmt.Errorf("duplicate tensor %s: found in both %s and %s", name, prev.File, file)
			}
			tensors[name] = SourceTensor{
				Name:  name,
				Dtype: td.Dtype,
				Shape: td.Shape,
				File:  file,
			}
		}
		ext.Close()
	}

	// Completeness: every tensor named in the shard index must actually be
	// present. A missing shard (or an index entry whose shard lacks the
	// tensor) means missing weights, which must fail loudly here rather than
	// silently importing an incomplete model.
	for name, file := range index {
		if tensor, ok := tensors[name]; !ok {
			return Inventory{}, fmt.Errorf("source model is incomplete: tensor %s (indexed in %q) was not found", name, file)
		} else if tensor.File != file {
			return Inventory{}, fmt.Errorf("source model is incomplete: tensor %s is indexed in %q but found in %q", name, file, tensor.File)
		}
	}

	if len(tensors) == 0 {
		return Inventory{}, fmt.Errorf("no model weights found in %s", dir)
	}

	return Inventory{Dir: dir, Config: cfg, RawConfig: rawConfig, Tensors: tensors}, nil
}

// SafetensorsWeightFiles returns the source shards and supported sidecars for
// transfer. Importing a sidecar still requires its model-specific importer.
func SafetensorsWeightFiles(dir string) ([]string, error) {
	_, files, err := safetensorsWeightFiles(dir)
	if err != nil {
		return nil, err
	}
	// Clef selects its separate head through RENDERER, so its sidecar is
	// included whenever present. The Perplexity readout belongs only to its
	// Qwen3_5Model checkpoint.
	files, err = appendWeightSidecar(dir, files, "joint_head.safetensors", "Clef")
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(filepath.Join(dir, "readout.safetensors")); errors.Is(err, os.ErrNotExist) {
		return files, nil
	} else if err != nil {
		return nil, err
	}
	cfg, _, err := readSourceModelConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.Architecture() != "Qwen3_5Model" {
		return files, nil
	}
	return appendWeightSidecar(dir, files, "readout.safetensors", "Perplexity")
}

func appendWeightSidecar(dir string, files []string, name, modelName string) ([]string, error) {
	info, err := os.Stat(filepath.Join(dir, name))
	if errors.Is(err, os.ErrNotExist) {
		return files, nil
	}
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s %s must be a regular file", modelName, name)
	}
	if !slices.Contains(files, name) {
		files = append(files, name)
		slices.Sort(files)
	}
	return files, nil
}

// SafetensorsConfigFiles lists source metadata for remote transfer and local
// JSON import. Auxiliary checkpoint subfolders are excluded.
func SafetensorsConfigFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	var hasLaya bool
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		hasLaya = hasLaya || name == "rl_agent_config.json"
		if filepath.Ext(name) == ".json" || name == "chat_template.jinja" {
			names = append(names, name)
		}
	}
	if hasLaya {
		for _, name := range []string{"encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"} {
			info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name)))
			if err != nil {
				return nil, err
			}
			if !info.Mode().IsRegular() {
				return nil, fmt.Errorf("Laya metadata %s is not a regular file", name)
			}
			names = append(names, name)
		}
	}
	hasSidecar, err := hasPplxSidecar(dir)
	if err != nil {
		return nil, err
	}
	if !hasSidecar {
		return names, nil
	}
	cfg, _, err := readSourceModelConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.Architecture() != "Qwen3_5Model" {
		return names, nil
	}
	// This checkpoint has extra JSON that does not configure inference. Keep
	// only the files the importer or runner consumes.
	modelMetadata := map[string]bool{
		"config.json":                  true,
		"decision_config.json":         true,
		"joint_config.json":            true,
		"generation_config.json":       true,
		"model.safetensors.index.json": true,
		"processor_config.json":        true,
		"tokenizer.json":               true,
		"tokenizer_config.json":        true,
		"special_tokens_map.json":      true,
		"added_tokens.json":            true,
		"hf_quant_config.json":         true,
		"quantization_config.json":     true,
	}
	return slices.DeleteFunc(names, func(name string) bool {
		return filepath.Ext(name) == ".json" && !modelMetadata[name]
	}), nil
}

func safetensorsWeightFiles(dir string) (map[string]string, []string, error) {
	index, err := readSourceTensorFiles(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("read tensor index: %w", err)
	}
	files, err := inventoryShardFiles(dir, index)
	if err != nil {
		return nil, nil, err
	}
	return index, files, nil
}

func inventoryShardFiles(dir string, index map[string]string) ([]string, error) {
	if len(index) > 0 {
		files := make(map[string]struct{})
		for _, file := range index {
			if file == "" || filepath.Base(file) != file || !strings.HasSuffix(file, ".safetensors") {
				return nil, fmt.Errorf("tensor index contains invalid shard path %q", file)
			}
			files[file] = struct{}{}
		}
		return sortedKeys(files), nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	// Without an index, only the standard HF weights - a monolithic
	// model.safetensors or sharded model-*.safetensors set - are imported.
	// Other safetensors in the same repo, notably Mistral consolidated files,
	// are skipped so they cannot shadow or pollute the model tensors.
	var monolithic bool
	var files []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") || !strings.HasPrefix(entry.Name(), "model") {
			continue
		}
		if entry.Name() == "model.safetensors" {
			monolithic = true
		}
		files = append(files, entry.Name())
	}
	if monolithic && len(files) > 1 {
		return nil, fmt.Errorf("found both model.safetensors and sharded model-*.safetensors weights in %s: ambiguous source", dir)
	}
	return files, nil
}
