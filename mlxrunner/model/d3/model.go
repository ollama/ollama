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

const prefixBidir = "prefix_bidir"

// The publisher's image processor caps each image at 1,638,400 pixels.
const visionMaxPixels = 1_638_400

func NewModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read d3 decision config: %w", err)
	}
	if cfg.FormatVersion != 1 || cfg.FormatID != "d3-code-readout-v1" || cfg.Prompt != "d3" || (cfg.AttentionMode != "noncausal_full_attention" && cfg.AttentionMode != prefixBidir) || cfg.Pooling != "last" || cfg.ReadoutDType != "float32" || len(cfg.Codes) != 255 || len(cfg.TokenIDs) != 255 {
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

// The runner schedules whole rows. D3.1 owns the temporary caches for the
// state and questions of each row.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(b *batch.Batch, _ []cache.Layer) (*mlx.Array, *mlx.Array) {
	if m.config.AttentionMode != prefixBidir {
		return m.Model.Forward(b, nil)
	}
	rows := make([]*mlx.Array, len(b.SeqQueryLens))
	for i, n := range b.SeqQueryLens {
		row := b.Row(i)
		layout := row.Layout[0].(*prefixLayout)
		h := m.forwardPrefixRow(row, layout)
		if padding := b.InputIDs.Dim(1) - int(n); padding > 0 {
			h = h.Concatenate(1, mlx.Zeros(h.DType(), 1, padding, h.Dim(2)))
		}
		rows[i] = h
	}
	h := mlx.Concatenate(rows, 0)
	return h, h
}

type prefixQuestion struct {
	end     int
	layout  any
	options int
}

type prefixLayout struct {
	done        <-chan struct{}
	stateEnd    int
	stateLayout any
	questions   []prefixQuestion
}

func (m *Model) forwardPrefixRow(row *batch.Batch, layout *prefixLayout) *mlx.Array {
	caches := m.Model.NewCaches()
	layers := cache.Layers(caches)
	snapshots := make([]cache.Snapshot, len(caches))
	defer func() {
		for i, c := range caches {
			if snapshots[i] != nil {
				snapshots[i].Close()
			}
			c.Free()
		}
	}()

	run := func(start, end, offset int, ropeLayout any, media []batch.MediaItem) *mlx.Array {
		return mlx.ScopedEval(func() []*mlx.Array {
			b := &batch.Batch{
				InputIDs:     row.InputIDs.Slice(mlx.Slice(), mlx.Slice(start, end)),
				SeqOffsets:   []int32{int32(offset)},
				SeqQueryLens: []int32{int32(end - start)},
				Layout:       []any{ropeLayout},
				Media:        media,
			}
			h, _ := m.Model.Forward(b, layers)
			return []*mlx.Array{h}
		})[0]
	}

	// The state sees its entire block. Each question starts from the same
	// state, with its original position offset and no preceding question.
	stateEnd := layout.stateEnd
	state := run(0, stateEnd, 0, layout.stateLayout, row.Media)
	if len(layout.questions) > 1 {
		for i, c := range caches {
			// Attention can rewind by offset. Recurrent state needs a snapshot.
			if _, ok := c.(cache.Recurrent); ok {
				snapshots[i] = c.Snapshot(0)
			}
		}
	}

	parts := []*mlx.Array{state}
	start := stateEnd
	for i, question := range layout.questions {
		select {
		case <-layout.done:
			// The scorer discards a canceled response. Preserve its shape
			// for the runner's whole-row batching path.
			parts = append(parts, mlx.Zeros(state.DType(), 1, int(row.SeqQueryLens[0])-start, state.Dim(2)))
			return mlx.Concatenate(parts, 1)
		default:
		}
		if i > 0 {
			for j, c := range caches {
				if !c.Restore(snapshots[j], stateEnd) {
					panic("D3.1: could not restore request-local state")
				}
			}
		}
		parts = append(parts, run(start, question.end, stateEnd, question.layout, nil))
		start = question.end
	}
	return mlx.Concatenate(parts, 1)
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
