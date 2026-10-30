// Package autotrustdecision serves AutoTrust decision slot-readout decision checkpoints.
package autotrustdecision

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
)

type decisionHead struct {
	Bias          []float32 `json:"bias"`
	VerbalizerIDs []int32   `json:"verbalizer_ids"`
	Slots         struct {
		Ranges      map[string][2]int `json:"ranges"`
		Verbalizers []string          `json:"verbalizers"`
		Template    string            `json:"template_version"`
	} `json:"slots"`
}

type calibration struct {
	PerKind map[string]float32 `json:"per_kind"`
}

type adapterConfig struct {
	Rank  int     `json:"r"`
	Alpha float32 `json:"lora_alpha"`
}

type Model struct {
	*qwen3_5.Model
	head        decisionHead
	temperature map[string]float32
	adapter     adapterConfig
	choiceIDs   []int32
	choiceCodes []string
}

func init() { model.Register("AutoTrustDecision", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	var head decisionHead
	if err := root.Manifest.ReadConfigJSON("decision_head.json", &head); err != nil {
		return nil, fmt.Errorf("read AutoTrust decision decision head: %w", err)
	}
	if head.Slots.Template != "bare-v1" || len(head.Bias) != 24 || len(head.VerbalizerIDs) != 24 ||
		head.Slots.Ranges["noul"] != [2]int{0, 2} {
		return nil, fmt.Errorf("unsupported AutoTrust decision slot head")
	}
	if head.Slots.Ranges["score"] != [2]int{2, 8} || head.Slots.Ranges["choice"] != [2]int{8, 24} {
		return nil, fmt.Errorf("unsupported AutoTrust decision slot ranges")
	}
	var cal calibration
	if err := root.Manifest.ReadConfigJSON("calibration.json", &cal); err != nil {
		return nil, fmt.Errorf("read AutoTrust decision calibration: %w", err)
	}
	for _, kind := range []string{"noul", "score", "choice"} {
		t := cal.PerKind[kind]
		if t <= 0 || math.IsNaN(float64(t)) || math.IsInf(float64(t), 0) {
			return nil, fmt.Errorf("invalid AutoTrust decision %s temperature %g", kind, t)
		}
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), head: head, temperature: cal.PerKind}
	if m.Config.ModelType != "qwen3_5" && m.Config.ModelType != "qwen3_5_text" {
		return nil, fmt.Errorf("unsupported AutoTrust decision backbone %q", m.Config.ModelType)
	}
	for _, id := range head.VerbalizerIDs {
		if id < 0 || id >= m.Config.VocabSize {
			return nil, fmt.Errorf("AutoTrust decision verbalizer token %d is outside the vocabulary", id)
		}
	}
	// A missing adapter config denotes a checkpoint with the adapter fused into
	// the backbone. A present but unreadable config must fail model loading.
	if _, ok := root.Manifest.ConfigLayer("adapter_config.json"); ok {
		if err := root.Manifest.ReadConfigJSON("adapter_config.json", &m.adapter); err != nil {
			return nil, fmt.Errorf("read AutoTrust decision adapter config: %w", err)
		}
		if m.adapter.Rank <= 0 || m.adapter.Alpha <= 0 {
			return nil, fmt.Errorf("invalid AutoTrust decision adapter rank %d or alpha %g", m.adapter.Rank, m.adapter.Alpha)
		}
	}
	m.choiceCodes, m.choiceIDs = choiceLabels(m.Tokenizer())
	if len(m.choiceIDs) < 255 || !slices.Equal(m.choiceIDs[:16], head.VerbalizerIDs[8:24]) {
		return nil, fmt.Errorf("AutoTrust decision tokenizer choice labels do not match the trained head")
	}
	return m, nil
}

// The publisher's LoRA applies to the System 1 backbone and lm_head only.
// Merge into the loaded BF16 tensors before Qwen packs its linear-attention
// projections. An already fused checkpoint has no adapter tensors.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	const prefix = "base_model.model."
	const aSuffix = ".lora_A.weight"
	pairs := 0
	for name, a := range tensors {
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, aSuffix) {
			continue
		}
		if m.adapter.Rank <= 0 || m.adapter.Alpha <= 0 {
			return fmt.Errorf("AutoTrust decision adapter tensors require adapter_config.json")
		}
		stem := strings.TrimSuffix(strings.TrimPrefix(name, prefix), aSuffix)
		baseName := stem + ".weight"
		bName := strings.TrimSuffix(name, aSuffix) + ".lora_B.weight"
		w, b := tensors[baseName], tensors[bName]
		if w == nil || b == nil || w.NumDims() != 2 || a.NumDims() != 2 || b.NumDims() != 2 ||
			a.Dim(0) != m.adapter.Rank || b.Dim(1) != m.adapter.Rank ||
			a.Dim(1) != w.Dim(1) || b.Dim(0) != w.Dim(0) {
			return fmt.Errorf("AutoTrust decision adapter pair %s does not match base projection", stem)
		}
		// Each merge finishes before the next pair so its FP32 intermediates
		// do not accumulate across the checkpoint's adapter projections.
		updated := mlx.ScopedEval(func() []*mlx.Array {
			return []*mlx.Array{mlx.Add(w.AsType(mlx.DTypeFloat32),
				mlx.MulScalar(mlx.Matmul(b.AsType(mlx.DTypeFloat32), a.AsType(mlx.DTypeFloat32)), m.adapter.Alpha/float32(m.adapter.Rank))).AsType(w.DType())}
		})[0]
		tensors[baseName] = updated
		delete(tensors, name)
		delete(tensors, bName)
		pairs++
	}
	for name := range tensors {
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, ".lora_B.weight") {
			return fmt.Errorf("AutoTrust decision adapter %s has no matching A tensor", name)
		}
	}
	if m.adapter.Rank > 0 && pairs == 0 {
		return fmt.Errorf("AutoTrust decision adapter configuration has no weights")
	}
	return m.Model.LoadWeights(tensors)
}
