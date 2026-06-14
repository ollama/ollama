package diffusiongemma

import (
	"context"
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	xmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// tinyArr builds a small deterministic weight of the given shape.
func tinyArr(shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.1)) * 0.05
	}
	return mlx.FromValues(data, shape...)
}

// tiny model dims (MoE disabled to keep the synthetic weight set small; the MoE
// path is inherited unchanged from gemma4).
const (
	tHidden  = 8
	tLayers  = 2
	tHeads   = 2
	tKVHeads = 1
	tHeadDim = 4
	tVocab   = 16
	tInterm  = 8
	tCanvas  = 4
)

// newTinyModel builds and loads a tiny DiffusionGemma model with random weights.
func newTinyModel(t *testing.T) *Model {
	t.Helper()
	cfgJSON := []byte(`{
		"architectures": ["DiffusionGemmaForBlockDiffusion"],
		"canvas_length": 4,
		"text_config": {
			"hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 8,
			"num_attention_heads": 2, "num_key_value_heads": 1,
			"head_dim": 4, "global_head_dim": 4, "vocab_size": 16,
			"rms_norm_eps": 1e-6, "max_position_embeddings": 64,
			"sliding_window": 0, "enable_moe_block": false,
			"layer_types": ["full_attention","full_attention"],
			"rope_parameters": {"full_attention": {"rope_theta": 10000.0}}
		}
	}`)
	cfg, err := parseTextConfig(cfgJSON)
	if err != nil {
		t.Fatalf("parseTextConfig: %v", err)
	}
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = xmodel.QuantizationParams("")

	tensors := map[string]*mlx.Array{
		"embed_tokens.weight":       tinyArr(tVocab, tHidden),
		"norm.weight":               tinyArr(tHidden),
		"self_cond_pre_norm.weight": tinyArr(tHidden),
		"self_cond_gate.weight":     tinyArr(tInterm, tHidden),
		"self_cond_up.weight":       tinyArr(tInterm, tHidden),
		"self_cond_down.weight":     tinyArr(tHidden, tInterm),
	}
	for i := range tLayers {
		p := "layers." + itoa(i)
		tensors[p+".input_layernorm.weight"] = tinyArr(tHidden)
		tensors[p+".post_attention_layernorm.weight"] = tinyArr(tHidden)
		tensors[p+".pre_feedforward_layernorm.weight"] = tinyArr(tHidden)
		tensors[p+".post_feedforward_layernorm.weight"] = tinyArr(tHidden)
		tensors[p+".self_attn.q_proj.weight"] = tinyArr(tHeads*tHeadDim, tHidden)
		tensors[p+".self_attn.k_proj.weight"] = tinyArr(tKVHeads*tHeadDim, tHidden)
		tensors[p+".self_attn.v_proj.weight"] = tinyArr(tKVHeads*tHeadDim, tHidden)
		tensors[p+".self_attn.o_proj.weight"] = tinyArr(tHidden, tHeads*tHeadDim)
		tensors[p+".self_attn.q_norm.weight"] = tinyArr(tHeadDim)
		tensors[p+".self_attn.k_norm.weight"] = tinyArr(tHeadDim)
		tensors[p+".mlp.gate_proj.weight"] = tinyArr(tInterm, tHidden)
		tensors[p+".mlp.up_proj.weight"] = tinyArr(tInterm, tHidden)
		tensors[p+".mlp.down_proj.weight"] = tinyArr(tHidden, tInterm)
	}

	m := &Model{Layers: make([]*DecoderLayer, tLayers), TextConfig: &cfg}
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	if m.SelfCond == nil {
		t.Fatal("self-conditioning MLP did not load")
	}
	// Mirror base.Weights: pin the model weights so the Diffuse loop's Sweep
	// doesn't free them (production loads via base.Weights, which does this).
	collected := mlx.Collect(m)
	for _, arr := range collected {
		mlx.Pin(arr)
	}
	mlx.Eval(collected...)
	return m
}

// TestForwardMetal validates LoadWeights, both forward phases, the canvas mask,
// the self-conditioning MLP, and Unembed end to end (shapes + finiteness).
func TestForwardMetal(t *testing.T) {
	skipIfNoMLX(t)
	m := newTinyModel(t)
	caches := m.NewCaches()

	m.prefillPrompt([]int32{1, 2, 3}, caches)
	nPast := 3

	const k = 2
	s1 := m.decodeCanvasSample([]int32{0, 0, 0, 0}, int32(nPast), nil, caches, 0.8, k, mlx.RandomKey(1))
	if len(s1.argmax) != tCanvas || len(s1.entropy) != tCanvas || len(s1.scIDs) != tCanvas*k {
		t.Fatalf("sample shapes: argmax=%d entropy=%d scIDs=%d", len(s1.argmax), len(s1.entropy), len(s1.scIDs))
	}
	assertFinite(t, "entropy-no-selfcond", s1.entropy)
	assertFinite(t, "scProbs-no-selfcond", s1.scProbs)
	for _, a := range s1.argmax {
		if a < 0 || int(a) >= tVocab {
			t.Fatalf("argmax out of vocab: %d", a)
		}
	}

	sc := &SelfCond{K: k, IDs: make([]int32, tCanvas*k), Probs: make([]float32, tCanvas*k)}
	for j := range tCanvas {
		sc.IDs[j*k], sc.IDs[j*k+1] = int32(j%tVocab), int32((j+1)%tVocab)
		sc.Probs[j*k], sc.Probs[j*k+1] = 0.7, 0.3
	}
	s2 := m.decodeCanvasSample([]int32{0, 1, 2, 3}, int32(nPast), sc, caches, 0.8, k, mlx.RandomKey(2))
	assertFinite(t, "entropy-selfcond", s2.entropy)
	t.Logf("OK: prefill + 2 decoder passes (with self-cond) ran on Metal")
}

// TestDiffuseMetal validates the full denoising loop on Metal — in particular the
// per-step cache checkpoint/rollback and multi-canvas commit (nil tokenizer => no
// EOS, so it runs all canvases).
func TestDiffuseMetal(t *testing.T) {
	skipIfNoMLX(t)
	m := newTinyModel(t)
	cfg := base.DiffuseConfig{
		Canvas: tCanvas, Steps: 3, MaxCanvases: 2, SelfCondK: 2,
		StabilityThreshold: 1, TMin: 0.4, TMax: 0.8,
		EntropyBound: 0.1, ConfidenceThreshold: 0.005, Seed: 7,
	}
	var got []int32
	if err := m.Diffuse(context.Background(), []int32{1, 2, 3}, cfg, func(tok int32) error {
		got = append(got, tok)
		return nil
	}); err != nil {
		t.Fatalf("Diffuse: %v", err)
	}
	if len(got) != cfg.MaxCanvases*cfg.Canvas {
		t.Fatalf("emitted %d tokens, want %d: %v", len(got), cfg.MaxCanvases*cfg.Canvas, got)
	}
	for _, tk := range got {
		if tk < 0 || tk >= tVocab {
			t.Errorf("token out of vocab: %d", tk)
		}
	}
	t.Logf("OK: Diffuse loop (checkpoint/rollback + commit) ran on Metal, emitted %v", got)
}

func assertFinite(t *testing.T, label string, xs []float32) {
	t.Helper()
	for i, v := range xs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: non-finite at %d: %v", label, i, v)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}
