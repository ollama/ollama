// Package d3 serves code-readout decision models.
package d3

import (
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type config struct {
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

type Model struct {
	*qwen3_5.Model
	Readout       *nn.Linear
	config        config
	foldedReadout *foldedReadout
}

type foldedReadout struct {
	path  string
	rows  []int32
	width int
}

// Keep d3's smaller image cap; AutoTrust uses the Qwen3.5 default.
const d3VisionMaxPixels = 1_638_400

const (
	d3FormatID        = "d3-code-readout-v1"
	autoTrustFormatID = "jev-code-readout-v1"

	lmHeadReadoutTensor = "lm_head.weight"
)

var codeReadoutFormats = []struct {
	id, prompt string
}{
	{d3FormatID, "d3"},
	{autoTrustFormatID, "jev"},
}

func codeReadoutFormat(id string) (prompt string, ok bool) {
	for _, format := range codeReadoutFormats {
		if format.id == id {
			return format.prompt, true
		}
	}
	return "", false
}

func NewModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read decision config: %w", err)
	}
	prompt, ok := codeReadoutFormat(cfg.FormatID)
	if !ok || cfg.FormatVersion != 1 || cfg.Prompt != prompt || cfg.AttentionMode != "noncausal_full_attention" || cfg.Pooling != "last" || cfg.ReadoutDType != "float32" || len(cfg.Codes) != 255 || len(cfg.TokenIDs) != 255 {
		return nil, fmt.Errorf("unsupported decision config (format %q, prompt %q, attention %q)", cfg.FormatID, cfg.Prompt, cfg.AttentionMode)
	}
	if cfg.Temperature <= 0 || math.IsNaN(float64(cfg.Temperature)) || math.IsInf(float64(cfg.Temperature), 0) {
		return nil, fmt.Errorf("invalid d3 temperature %g", cfg.Temperature)
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	m.Config.NoncausalFullAttention = true
	m.Config.PreciseGatedDeltaDecay = true
	if cfg.FormatID == d3FormatID {
		m.VisionMaxPixels = d3VisionMaxPixels
	}
	if m.Config.ModelType != "qwen3_5_text" {
		return nil, fmt.Errorf("unsupported d3 backbone %q", m.Config.ModelType)
	}
	seen := make(map[int32]bool, len(cfg.TokenIDs))
	for i, code := range cfg.Codes {
		id := cfg.TokenIDs[i]
		if id < 0 || id >= m.Config.VocabSize || seen[id] || !slices.Equal(m.Tokenizer().Encode(code, false), []int32{id}) {
			return nil, fmt.Errorf("invalid d3 answer code %q at row %d", code, i)
		}
		seen[id] = true
	}
	if cfg.FormatID == autoTrustFormatID {
		folded, err := openFoldedReadout(root, cfg.TokenIDs, m.Config.HiddenSize)
		if err != nil {
			return nil, err
		}
		m.foldedReadout = folded
	}
	return m, nil
}

func openFoldedReadout(root *model.Root, tokenIDs []int32, hidden int32) (*foldedReadout, error) {
	var layer *manifest.Layer
	for _, candidate := range root.Manifest.TensorLayers() {
		if candidate.Name == lmHeadReadoutTensor {
			layer = &candidate
			break
		}
	}
	if layer == nil {
		return nil, fmt.Errorf("checkpoint does not ship %s", lmHeadReadoutTensor)
	}
	path, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return nil, fmt.Errorf("resolve %s: %w", lmHeadReadoutTensor, err)
	}
	reader, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", lmHeadReadoutTensor, err)
	}
	defer reader.Close()
	weight, err := reader.GetTensor(lmHeadReadoutTensor)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", lmHeadReadoutTensor, err)
	}
	if weight.Dtype != "BF16" || len(weight.Shape) != 2 || weight.Shape[0] <= 0 || weight.Shape[1] != hidden {
		return nil, fmt.Errorf("%s must be BF16 [vocab,%d], got %s %v", lmHeadReadoutTensor, hidden, weight.Dtype, weight.Shape)
	}
	for _, id := range tokenIDs {
		if id < 0 || id >= weight.Shape[0] {
			return nil, fmt.Errorf("answer code token id %d is outside the %d-row lm_head", id, weight.Shape[0])
		}
	}
	return &foldedReadout{path: path, rows: slices.Clone(tokenIDs), width: int(hidden)}, nil
}

// SkippedTensorNames avoids loading the full lm_head; load reads only answer rows.
func (m *Model) SkippedTensorNames() []string {
	if m.foldedReadout == nil {
		return nil
	}
	return []string{lmHeadReadoutTensor}
}

// BF16 to FP32 widening preserves the publisher's row values exactly.
func (f *foldedReadout) load() (*mlx.Array, error) {
	file, err := os.Open(f.path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", f.path, err)
	}
	defer file.Close()
	reader, err := safetensors.OpenForExtraction(f.path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	weight, err := reader.GetTensor(lmHeadReadoutTensor)
	if err != nil {
		return nil, err
	}
	rowBytes := weight.Size / int64(weight.Shape[0])
	if rowBytes*int64(weight.Shape[0]) != weight.Size || rowBytes != int64(f.width)*2 {
		return nil, fmt.Errorf("%s row layout is %d bytes, want %d", lmHeadReadoutTensor, rowBytes, f.width*2)
	}
	values := make([]float32, 0, len(f.rows)*f.width)
	row := make([]byte, rowBytes)
	for _, index := range f.rows {
		if _, err := file.ReadAt(row, weight.Offset()+int64(index)*rowBytes); err != nil {
			return nil, fmt.Errorf("read %s row %d: %w", lmHeadReadoutTensor, index, err)
		}
		for i := 0; i+1 < len(row); i += 2 {
			bits := uint32(row[i]) | uint32(row[i+1])<<8
			values = append(values, math.Float32frombits(bits<<16))
		}
	}
	return mlx.FromValues(values, len(f.rows), f.width), nil
}

// Full-prompt attention cannot reuse a causal prefix cache.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(b *batch.Batch, _ []cache.Layer) (*mlx.Array, *mlx.Array) {
	return m.Model.Forward(b, nil)
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if m.foldedReadout != nil && tensors["weight"] != nil {
		return fmt.Errorf("folded readout cannot also load weight")
	}
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	w := tensors["weight"]
	if m.foldedReadout != nil {
		var err error
		if w, err = m.foldedReadout.load(); err != nil {
			return err
		}
	}
	if w == nil || w.DType() != mlx.DTypeFloat32 || !slices.Equal(w.Dims(), []int{255, int(m.Config.HiddenSize)}) {
		return fmt.Errorf("d3 requires an FP32 readout weight of shape [255,%d]", m.Config.HiddenSize)
	}
	m.Readout = &nn.Linear{Weight: w}
	return nil
}
