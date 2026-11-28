package inkling2

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lora"
)

func init() { lora.RegisterLayout("Inkling2LoopForConditionalGeneration", adapterLayout) }

func adapterLayout(data json.RawMessage) (lora.Layout, error) {
	var cfg struct {
		Text textConfig `json:"text_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	c := cfg.Text
	if err := c.validate(); err != nil {
		return nil, err
	}
	if _, err := c.executionPlan(); err != nil {
		return nil, err
	}
	layout := lora.Layout{"language_model.lm_head": {Input: c.HiddenSize, Output: c.VocabSize, OutputLimit: c.UnpaddedVocab}}
	for i := range c.NumLayers {
		heads, kvHeads, dim := c.Heads, c.KVHeads, c.HeadDim
		if slices.Contains(c.LocalLayers, i) {
			heads, kvHeads, dim = c.LocalHeads, c.LocalKVHeads, c.LocalHeadDim
		}
		prefix := fmt.Sprintf("language_model.layers.%d.", i)
		for name, width := range map[string]int{"wq_du": heads * dim, "wr_du": heads * c.RelativeDim, "gate_attn": heads * dim, "wk_dv": kvHeads * dim, "wv_dv": kvHeads * dim} {
			layout[prefix+"attn."+name] = lora.Projection{Input: c.HiddenSize, Output: width}
		}
		layout[prefix+"attn.wo_ud"] = lora.Projection{Input: heads * dim, Output: c.HiddenSize}
		up := lora.Projection{Input: c.HiddenSize, Output: c.ExpertWidth, Experts: c.Experts}
		layout[prefix+"mlp.experts.w1"] = up
		layout[prefix+"mlp.experts.w3"] = up
		layout[prefix+"mlp.experts.w2"] = lora.Projection{Input: c.ExpertWidth, Output: c.HiddenSize, Experts: c.Experts}
	}
	return layout, nil
}

type attentionLoRA struct{ query, relative, gate, key, value *lora.Factors }
type expertLoRA struct{ gate, up, down *lora.Factors }

func (m *targetModel) WithAdapter(factors map[string]*lora.Factors) model.Model {
	view := *m
	view.Layers = slices.Clone(m.Layers)
	view.draftTaps = nil
	view.headLoRA = factors["language_model.lm_head"]
	for i := range view.Layers {
		prefix := fmt.Sprintf("language_model.layers.%d.", i)
		attn := prefix + "attn."
		a := &view.Layers[i].Attention
		proj := attentionLoRA{
			query: factors[attn+"wq_du"], relative: factors[attn+"wr_du"], gate: factors[attn+"gate_attn"],
			key: factors[attn+"wk_dv"], value: factors[attn+"wv_dv"],
		}
		if proj != (attentionLoRA{}) {
			a.lora = &proj
		}
		a.outputLoRA = factors[attn+"wo_ud"]
		exp := prefix + "mlp.experts."
		ffn := expertLoRA{gate: factors[exp+"w1"], up: factors[exp+"w3"], down: factors[exp+"w2"]}
		if ffn != (expertLoRA{}) {
			view.Layers[i].FeedForward.lora = &ffn
		}
	}
	return &view
}

func (a selfAttention) project(qInput, kInput, vInput *mlx.Array) (q, k, v *mlx.Array) {
	q, k, v = a.projectBase(qInput, kInput, vInput)
	l := a.lora
	if l == nil {
		return
	}
	w, r, kv := a.heads*a.headDim, a.heads*a.PositionProjection.Dim(0), a.kvHeads*a.headDim
	if l.query != nil || l.relative != nil || l.gate != nil {
		q = mlx.Concatenate([]*mlx.Array{l.query.Add(columns(q, 0, w), qInput), l.relative.Add(columns(q, w, w+r), qInput),
			l.gate.Add(columns(q, w+r, 2*w+r), qInput), columns(q, 2*w+r, 2*w+r+a.heads)}, -1)
	}
	applyKV := func(base, input *mlx.Array, f *lora.Factors) *mlx.Array {
		if f == nil {
			return base
		}
		prefix := f.Add(columns(base, 0, kv), input)
		if tail := columns(base, kv, base.Dim(-1)); tail != nil {
			return mlx.Concatenate([]*mlx.Array{prefix, tail}, -1)
		}
		return prefix
	}
	return q, applyKV(k, kInput, l.key), applyKV(v, vInput, l.value)
}

func (a selfAttention) projectOutput(input *mlx.Array) *mlx.Array {
	return a.outputLoRA.Add(projectTarget(a.Output, input), input)
}

// The exported bank can share A or B across experts. Apply each expert's
// factors before the routing-weighted outputs are summed.
func addExpert(base, input, indices *mlx.Array, f *lora.Factors) *mlx.Array {
	if f == nil {
		return base
	}
	multiply := func(x, w *mlx.Array) *mlx.Array {
		if w.NumDims() == 2 {
			return x.Matmul(w)
		}
		return mlx.GatherMM(x, w, nil, indices, false)
	}
	delta := multiply(multiply(input.AsType(f.AT.DType()), f.AT), f.BT)
	return base.Add(mlx.MulScalar(delta, f.Scale))
}

func (e routedExperts) combineLoRA(flat, indices, scores *mlx.Array) *mlx.Array {
	x := flat.ExpandDims(1).ExpandDims(1)
	gate, up := splitGateUp(e.GateUp.project(x, indices).AsType(mlx.DTypeFloat32))
	gate = addExpert(gate, x, indices, e.lora.gate)
	up = addExpert(up, x, indices, e.lora.up)
	weighted := weightedSwiGLU(gate, up, scores, e.gateCap, e.upCap, flat.DType())
	down := addExpert(e.Down.project(weighted, indices).AsType(mlx.DTypeFloat32), weighted, indices, e.lora.down)
	return mlx.Sum(down.Squeeze(2), 1, false).AsType(flat.DType())
}
