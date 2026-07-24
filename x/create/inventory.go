package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
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

	index, err := readSourceTensorFiles(dir)
	if err != nil {
		return Inventory{}, fmt.Errorf("read tensor index: %w", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return Inventory{}, err
	}

	// Only the standard HF weights - a monolithic model.safetensors or the
	// sharded model-*.safetensors set - are imported. Other safetensors in the
	// same repo - notably Mistral's consolidated-*.safetensors - use a layout
	// we don't support, and are skipped so they can't shadow or pollute the
	// model tensors.
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
		return Inventory{}, fmt.Errorf("found both model.safetensors and sharded model-*.safetensors weights in %s: ambiguous source", dir)
	}

	tensors := make(map[string]SourceTensor)
	for _, file := range files {
		ext, err := safetensors.OpenForExtraction(filepath.Join(dir, file))
		if err != nil {
			return Inventory{}, fmt.Errorf("open %s: %w", file, err)
		}
		for _, name := range ext.ListTensors() {
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
		if _, ok := tensors[name]; !ok {
			return Inventory{}, fmt.Errorf("source model is incomplete: tensor %s (indexed in %q) was not found", name, file)
		}
	}

	if len(tensors) == 0 {
		return Inventory{}, fmt.Errorf("no model.safetensors or model-*.safetensors weights found in %s", dir)
	}

	return Inventory{Dir: dir, Config: cfg, RawConfig: rawConfig, Tensors: tensors}, nil
}
