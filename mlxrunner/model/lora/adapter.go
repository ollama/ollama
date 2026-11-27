package lora

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
)

// Projection describes the exported PEFT factors, before any native packing.
// An expert dimension of one means the factor is shared across routed experts.
type Projection struct {
	Input, Output  int
	Experts        int
	OutputLimit    int  // unpadded vocabulary, when the native head trims padding
	ExcludeFromAll bool // PEFT all-linear normally excludes the language head
}

type Layout map[string]Projection

var layouts = map[string]func(json.RawMessage) (Layout, error){}

// RegisterLayout is called by architecture packages at initialization.
func RegisterLayout(arch string, layout func(json.RawMessage) (Layout, error)) {
	if layouts[arch] != nil {
		panic("duplicate LoRA layout for " + arch)
	}
	layouts[arch] = layout
}

func ModelLayout(data json.RawMessage) (Layout, error) {
	var cfg struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if len(cfg.Architectures) == 0 {
		return nil, fmt.Errorf("missing base architecture")
	}
	layout := layouts[cfg.Architectures[0]]
	if layout == nil {
		return nil, fmt.Errorf("MLX LoRA is unsupported for %s", cfg.Architectures[0])
	}
	return layout(data)
}

type config struct {
	Rank    int             `json:"r"`
	Alpha   *float32        `json:"lora_alpha"`
	Type    string          `json:"peft_type"`
	Targets json.RawMessage `json:"target_modules"`
}

func parseConfig(data []byte) (config, []string, error) {
	var c config
	if err := json.Unmarshal(data, &c); err != nil {
		return c, nil, err
	}
	if c.Type != "LORA" || c.Rank <= 0 || c.Alpha == nil {
		return c, nil, fmt.Errorf("expected PEFT LORA with positive r and numeric lora_alpha")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return c, nil, err
	}
	// Training metadata does not change the exported B A product. Reject
	// other nondefault options rather than silently applying a different delta.
	for key, raw := range fields {
		switch key {
		case "r", "lora_alpha", "peft_type", "target_modules", "task_type",
			"auto_mapping", "base_model_name_or_path", "revision", "peft_version",
			"inference_mode", "init_lora_weights", "lora_dropout", "megatron_core", "qalora_group_size":
			continue
		case "bias":
			if string(raw) == `"none"` {
				continue
			}
		}
		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			return c, nil, err
		}
		switch v := value.(type) {
		case nil:
			continue
		case bool:
			if !v {
				continue
			}
		case []any:
			if len(v) == 0 {
				continue
			}
		case map[string]any:
			if len(v) == 0 {
				continue
			}
		}
		return c, nil, fmt.Errorf("unsupported PEFT option %s=%s", key, raw)
	}
	if string(c.Targets) == `"all-linear"` {
		return c, nil, nil
	}
	var targets []string
	if err := json.Unmarshal(c.Targets, &targets); err != nil || len(targets) == 0 {
		return c, nil, fmt.Errorf("target_modules must be all-linear or a nonempty list of module names")
	}
	return c, targets, nil
}

type pair struct {
	dtype      string
	a, b       string
	projection Projection
}

// Checkpoint holds a validated mapping of exported tensor names onto native projections.
type Checkpoint struct {
	path  string
	pairs map[string]pair
	scale float32
	Bytes int64 // Factor storage before vocabulary trimming.
}

// Inspect reads just the bounded config and safetensors header. Create and
// inference use the same validation, including for externally supplied manifests.
func Inspect(ref manifest.AdapterRef, layout Layout) (*Checkpoint, error) {
	configPath, err := manifest.BlobsPath(ref.Config)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(configPath)
	if err != nil {
		return nil, err
	}
	if info.Size() > 1<<20 {
		return nil, fmt.Errorf("adapter_config.json exceeds 1 MiB")
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	cfg, targets, err := parseConfig(data)
	if err != nil {
		return nil, fmt.Errorf("adapter_config.json: %w", err)
	}
	path, err := manifest.BlobsPath(ref.Weights)
	if err != nil {
		return nil, err
	}
	f, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return inspectTensors(f, path, layout, cfg, targets)
}

func (layout Layout) targets(targets []string) (Layout, error) {
	expected := make(map[string]Projection)
	matched := make(map[string]bool)
	for name, p := range layout {
		if targets == nil && !p.ExcludeFromAll {
			expected[name] = p
		}
		for _, target := range targets {
			if target != "" && (name == target || strings.HasSuffix(name, "."+target)) {
				expected[name] = p
				matched[target] = true
			}
		}
	}
	for _, target := range targets {
		if !matched[target] {
			return nil, fmt.Errorf("unsupported target module %q", target)
		}
	}
	if len(expected) == 0 {
		return nil, fmt.Errorf("adapter selects no projections")
	}
	return expected, nil
}

func inspectTensors(f *safetensors.TensorExtractor, path string, layout Layout, c config, targets []string) (*Checkpoint, error) {
	expected, err := layout.targets(targets)
	if err != nil {
		return nil, err
	}
	pairs := make(map[string]pair)
	var size int64

	for _, key := range f.ListTensors() {
		name := strings.TrimPrefix(key, "base_model.model.")
		module, isA := strings.CutSuffix(name, ".lora_A.weight")
		if !isA {
			var ok bool
			module, ok = strings.CutSuffix(name, ".lora_B.weight")
			if !ok {
				return nil, fmt.Errorf("unexpected adapter tensor %q", key)
			}
		}
		p, ok := expected[module]
		if !ok {
			return nil, fmt.Errorf("adapter tensor %q has no declared supported target", key)
		}
		td, err := f.GetTensor(key)
		if err != nil {
			return nil, err
		}
		if !slices.Contains([]string{"F16", "BF16", "F32"}, td.Dtype) {
			return nil, fmt.Errorf("unsupported dtype %s for %s", td.Dtype, key)
		}
		entry := pairs[module]
		if entry.dtype != "" && entry.dtype != td.Dtype {
			return nil, fmt.Errorf("mixed factor dtypes for %s", module)
		}
		entry.dtype = td.Dtype
		size += td.Size
		entry.projection = p
		rows, cols := p.Output, c.Rank
		if isA {
			rows, cols = c.Rank, p.Input
			if entry.a != "" {
				return nil, fmt.Errorf("duplicate A factor for %s", module)
			}
			entry.a = key
		} else {
			if entry.b != "" {
				return nil, fmt.Errorf("duplicate B factor for %s", module)
			}
			entry.b = key
		}
		shape := []int{rows, cols}
		if p.Experts > 0 && len(td.Shape) == 3 {
			count := int(td.Shape[0])
			if count != 1 && count != p.Experts {
				return nil, fmt.Errorf("%s has %d expert factors, expected 1 or %d", key, count, p.Experts)
			}
			shape = append([]int{count}, shape...)
		}
		if !slices.EqualFunc(shape, td.Shape, func(a int, b int32) bool { return a == int(b) }) {
			return nil, fmt.Errorf("%s has shape %v, expected %v", key, td.Shape, shape)
		}
		pairs[module] = entry
	}
	for module := range expected {
		entry := pairs[module]
		if entry.a == "" || entry.b == "" {
			return nil, fmt.Errorf("missing LoRA factor for %s", module)
		}
	}
	return &Checkpoint{Bytes: size, path: path, pairs: pairs, scale: *c.Alpha / float32(c.Rank)}, nil
}
