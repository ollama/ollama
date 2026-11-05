// Package d3 implements the original Decision 3 code-readout checkpoints.
package d3

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
	Readout *nn.Linear
	config  config
}

// The publisher's image processor caps each image at 1,638,400 pixels.
const visionMaxPixels = 1_638_400

func NewModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read d3 decision config: %w", err)
	}
	if cfg.FormatVersion != 1 || cfg.FormatID != "d3-code-readout-v1" || cfg.Prompt != "d3" || cfg.AttentionMode != "noncausal_full_attention" || cfg.Pooling != "last" || cfg.ReadoutDType != "float32" || len(cfg.Codes) != 255 || len(cfg.TokenIDs) != 255 {
		return nil, fmt.Errorf("unsupported d3 decision config (format %q, attention %q)", cfg.FormatID, cfg.AttentionMode)
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
	m.VisionMaxPixels = visionMaxPixels
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
	return m, nil
}

// Full-attention layers read both directions within each complete question
// row. Reusing a causal prefix snapshot after changing state, question, or
// option order would yield incorrect hidden states.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	return m.Model.Forward(b, nil)
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	w := tensors["weight"]
	if w == nil || w.DType() != mlx.DTypeFloat32 || !slices.Equal(w.Dims(), []int{255, int(m.Config.HiddenSize)}) {
		return fmt.Errorf("d3 requires an FP32 readout weight of shape [255,%d]", m.Config.HiddenSize)
	}
	m.Readout = &nn.Linear{Weight: w}
	return nil
}
