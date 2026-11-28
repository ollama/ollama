package qwen3

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lora"
)

func init() { lora.RegisterLayout("Qwen3ForCausalLM", adapterLayout) }

func adapterLayout(data json.RawMessage) (lora.Layout, error) {
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	if c.HiddenSize <= 0 || c.NumAttentionHeads <= 0 || c.NumHiddenLayers <= 0 || c.NumHiddenLayers > 1024 || c.IntermediateSize <= 0 || c.VocabSize <= 0 {
		return nil, fmt.Errorf("invalid Qwen3 projection dimensions")
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.NumKeyValueHeads <= 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	h, q, kv, inner := int(c.HiddenSize), int(c.HeadDim)*int(c.NumAttentionHeads), int(c.HeadDim)*int(c.NumKeyValueHeads), int(c.IntermediateSize)
	layout := lora.Layout{"lm_head": {Input: h, Output: int(c.VocabSize), ExcludeFromAll: true}}
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		layout[p+"self_attn.q_proj"] = lora.Projection{Input: h, Output: q}
		layout[p+"self_attn.k_proj"] = lora.Projection{Input: h, Output: kv}
		layout[p+"self_attn.v_proj"] = lora.Projection{Input: h, Output: kv}
		layout[p+"self_attn.o_proj"] = lora.Projection{Input: q, Output: h}
		layout[p+"mlp.gate_proj"] = lora.Projection{Input: h, Output: inner}
		layout[p+"mlp.up_proj"] = lora.Projection{Input: h, Output: inner}
		layout[p+"mlp.down_proj"] = lora.Projection{Input: inner, Output: h}
	}
	return layout, nil
}

func (m *Model) WithAdapter(f map[string]*lora.Factors) model.Model {
	view := *m
	view.LMHead = lora.AdaptLinear(m.LMHead, f["lm_head"])
	view.Layers = slices.Clone(m.Layers)
	for i, base := range m.Layers {
		layer, attn, mlp := *base, *base.Attention, *base.MLP
		p := fmt.Sprintf("model.layers.%d.", i)
		attn.QProj = lora.AdaptLinear(attn.QProj, f[p+"self_attn.q_proj"])
		attn.KProj = lora.AdaptLinear(attn.KProj, f[p+"self_attn.k_proj"])
		attn.VProj = lora.AdaptLinear(attn.VProj, f[p+"self_attn.v_proj"])
		attn.OProj = lora.AdaptLinear(attn.OProj, f[p+"self_attn.o_proj"])
		mlp.GateProj = lora.AdaptLinear(mlp.GateProj, f[p+"mlp.gate_proj"])
		mlp.UpProj = lora.AdaptLinear(mlp.UpProj, f[p+"mlp.up_proj"])
		mlp.DownProj = lora.AdaptLinear(mlp.DownProj, f[p+"mlp.down_proj"])
		layer.Attention, layer.MLP = &attn, &mlp
		view.Layers[i] = &layer
	}
	return &view
}
