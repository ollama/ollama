// Package pplxdecider implements Perplexity's decision checkpoint on its
// Qwen3_5Model backbone.
package pplxdecider

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type config struct {
	FormatVersion int      `json:"format_version"`
	Codes         []string `json:"codes"`
	TokenIDs      []int32  `json:"token_ids"`
	Temperature   float32  `json:"temperature"`
	MaxPixels     int32    `json:"max_pixels"`
	AttentionMode string   `json:"attention_mode"`
	Pooling       string   `json:"pooling"`
}

type jointConfig struct {
	Version       int    `json:"version"`
	Template      string `json:"template"`
	Marker        string `json:"marker"`
	ReadoutOffset int    `json:"readout_offset"`
}

type Model struct {
	*qwen3_5.Model
	Readout  *nn.Linear
	config   config
	markerID int32
}

func init() { model.Register("Qwen3_5Model", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read Perplexity decision config: %w", err)
	}
	var joint jointConfig
	if err := root.Manifest.ReadConfigJSON("joint_config.json", &joint); err != nil {
		return nil, fmt.Errorf("read Perplexity joint config: %w", err)
	}
	if joint.Version != 3 || joint.Template != "joint-question-inline-readout-v1" || joint.Marker != "<|fim_suffix|>" || joint.ReadoutOffset != 0 {
		return nil, fmt.Errorf("unsupported Perplexity joint protocol")
	}
	if cfg.FormatVersion != 2 || len(cfg.Codes) != 255 || len(cfg.TokenIDs) != len(cfg.Codes) {
		return nil, fmt.Errorf("invalid Perplexity decision vocabulary or format")
	}
	if cfg.Temperature <= 0 || math.IsNaN(float64(cfg.Temperature)) || math.IsInf(float64(cfg.Temperature), 0) {
		return nil, fmt.Errorf("invalid Perplexity temperature %g", cfg.Temperature)
	}
	if cfg.Pooling != "last" {
		return nil, fmt.Errorf("unsupported Perplexity pooling %q", cfg.Pooling)
	}
	if cfg.AttentionMode != "noncausal_full_attention" {
		return nil, fmt.Errorf("unsupported Perplexity attention mode %q", cfg.AttentionMode)
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	m.Config.NoncausalFullAttention = true
	// Match the publisher's FP32 recurrent decay over long decision rows.
	m.Config.PreciseGatedDeltaDecay = true
	if cfg.MaxPixels <= 0 {
		return nil, fmt.Errorf("invalid Perplexity image limit %d", cfg.MaxPixels)
	}
	m.VisionMaxPixels = cfg.MaxPixels
	if m.Config.ModelType != "qwen3_5_text" {
		return nil, fmt.Errorf("unsupported Perplexity backbone %q", m.Config.ModelType)
	}
	seen := make(map[int32]bool, len(cfg.TokenIDs))
	for i, code := range cfg.Codes {
		id := cfg.TokenIDs[i]
		if id < 0 || id >= m.Config.VocabSize || seen[id] || !slices.Equal(m.Tokenizer().Encode(code, false), []int32{id}) {
			return nil, fmt.Errorf("invalid Perplexity answer code %q at row %d", code, i)
		}
		seen[id] = true
	}
	markerIDs := m.Tokenizer().Encode(joint.Marker, false)
	if len(markerIDs) != 1 {
		return nil, fmt.Errorf("Perplexity readout marker must be one token")
	}
	var ok bool
	if m.markerID, ok = m.Tokenizer().GetSpecialToken(joint.Marker); !ok || m.markerID != markerIDs[0] {
		return nil, fmt.Errorf("Perplexity readout marker is not a special token")
	}
	return m, nil
}

// A full-attention layer reads later tokens in the decision row. A prefix
// snapshot cannot represent the row after its question or state changes.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	return m.Model.Forward(b, nil)
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	w := tensors["weight"] // The publisher keeps its readout in readout.safetensors.
	if w == nil || w.DType() != mlx.DTypeBFloat16 || !slices.Equal(w.Dims(), []int{255, int(m.Config.HiddenSize)}) {
		return fmt.Errorf("Perplexity requires a BF16 readout weight of shape [255,%d]", m.Config.HiddenSize)
	}
	m.Readout = &nn.Linear{Weight: w}
	return nil
}
