package diffusiongemma

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestCanvasMask(t *testing.T) {
	// Encoder/causal pass: a plain causal mask.
	if got := canvasMask(false, 0, 0); !got.IsCausal() {
		t.Errorf("encoder mask: IsCausal() = false, want true")
	}
	// decoderPhase with a zero-width canvas degrades to causal (no relaxation).
	if got := canvasMask(true, 10, 0); !got.IsCausal() {
		t.Errorf("decoder mask, canvasLen=0: IsCausal() = false, want true (degrades to causal)")
	}
	// Decoder pass over a real canvas: relaxed (bidirectional within the canvas),
	// so not pure-causal, but still a restrictive (non-zero) mask vs the prefix.
	dec := canvasMask(true, 34, 256)
	if dec.IsCausal() {
		t.Errorf("decoder canvas mask: IsCausal() = true, want false (relaxed/bidirectional)")
	}
	if dec.IsZero() {
		t.Errorf("decoder canvas mask: IsZero() = true, want false (prefix stays causal)")
	}
}

func TestParseSuppressTokens(t *testing.T) {
	got := parseSuppressTokens([]byte(`{"suppress_tokens":[258883,258882]}`))
	want := []int32{258883, 258882}
	if !slices.Equal(got, want) {
		t.Fatalf("parseSuppressTokens() = %v, want %v", got, want)
	}

	if got := parseSuppressTokens([]byte(`{"eos_token_id":[1,106,50]}`)); got != nil {
		t.Fatalf("parseSuppressTokens() without suppress_tokens = %v, want nil", got)
	}
}

func TestParseDiffusionParams(t *testing.T) {
	config := []byte(`{
		"architectures": ["DiffusionGemmaForBlockDiffusion"],
		"model_type": "diffusion_gemma",
		"canvas_length": 256,
		"text_config": {"hidden_size": 2816, "num_hidden_layers": 30}
	}`)
	genConfig := []byte(`{
		"max_denoising_steps": 48,
		"t_min": 0.4,
		"t_max": 0.8,
		"stability_threshold": 2,
		"confidence_threshold": 0.005,
		"sampler_config": {"entropy_bound": 0.1}
	}`)

	p := parseDiffusionParams(config, genConfig)

	if p.CanvasLength != 256 {
		t.Errorf("CanvasLength = %d, want 256", p.CanvasLength)
	}
	if p.MaxDenoisingSteps != 48 {
		t.Errorf("MaxDenoisingSteps = %d, want 48", p.MaxDenoisingSteps)
	}
	if p.StabilityThreshold != 2 {
		t.Errorf("StabilityThreshold = %d, want 2", p.StabilityThreshold)
	}
	for _, tc := range []struct {
		name string
		got  float32
		want float32
	}{
		{"TMin", p.TMin, 0.4},
		{"TMax", p.TMax, 0.8},
		{"EntropyBound", p.EntropyBound, 0.1},
		{"ConfidenceThreshold", p.ConfidenceThreshold, 0.005},
	} {
		if math.Abs(float64(tc.got-tc.want)) > 1e-6 {
			t.Errorf("%s = %v, want %v", tc.name, tc.got, tc.want)
		}
	}
}

func TestParseDiffusionParamsMissingGenConfig(t *testing.T) {
	// canvas_length still parses from root config; gen-config fields stay zero
	// (the runner treats zero as "use built-in default").
	p := parseDiffusionParams([]byte(`{"canvas_length": 128}`), nil)
	if p.CanvasLength != 128 {
		t.Errorf("CanvasLength = %d, want 128", p.CanvasLength)
	}
	if p.MaxDenoisingSteps != 0 || p.EntropyBound != 0 {
		t.Errorf("expected zero gen-config defaults, got steps=%d entropy=%v", p.MaxDenoisingSteps, p.EntropyBound)
	}
}

func TestParseTextConfigInfersAttentionKEqV(t *testing.T) {
	cfg, err := parseTextConfig([]byte(`{
		"architectures": ["DiffusionGemmaForBlockDiffusion"],
		"text_config": {
			"num_global_key_value_heads": 2,
			"layer_types": ["sliding_attention", "full_attention"]
		}
	}`))
	if err != nil {
		t.Fatalf("parseTextConfig: %v", err)
	}
	if !cfg.AttentionKEqV {
		t.Fatal("AttentionKEqV = false, want true when global KV heads are configured")
	}
}

func TestNormalizeNativeDiffusionGemmaTensors(t *testing.T) {
	embed := mlx.New("embed")
	embedScales := mlx.New("embed_scales")
	embedBiases := mlx.New("embed_biases")
	norm := mlx.New("norm")
	qProj := mlx.New("q_proj")
	selfCondGate := mlx.New("self_cond_gate")
	selfCondScales := mlx.New("self_cond_scales")
	selfCondBiases := mlx.New("self_cond_biases")

	tensors := map[string]*mlx.Array{
		"model.decoder.embed_tokens.weight":                embed,
		"model.decoder.embed_tokens.scales":                embedScales,
		"model.decoder.embed_tokens.biases":                embedBiases,
		"model.decoder.norm.weight":                        norm,
		"model.decoder.layers.0.self_attn.q_proj.weight":   qProj,
		"model.decoder.self_conditioning.gate_proj.weight": selfCondGate,
		"model.decoder.self_conditioning.gate_proj.scales": selfCondScales,
		"model.decoder.self_conditioning.gate_proj.biases": selfCondBiases,
	}

	normalizeNativeDiffusionGemmaTensors(tensors)

	cases := map[string]*mlx.Array{
		"model.embed_tokens.weight":              embed,
		"model.embed_tokens.weight_scale":        embedScales,
		"model.embed_tokens.weight_qbias":        embedBiases,
		"model.norm.weight":                      norm,
		"model.layers.0.self_attn.q_proj.weight": qProj,
		"self_cond_gate.weight":                  selfCondGate,
		"self_cond_gate.weight_scale":            selfCondScales,
		"self_cond_gate.weight_qbias":            selfCondBiases,
	}

	for name, want := range cases {
		if got := tensors[name]; got != want {
			t.Fatalf("tensors[%q] = %p, want %p", name, got, want)
		}
	}

	if got := resolveWeightPrefix(tensors); got != "model." {
		t.Fatalf("resolveWeightPrefix() = %q, want %q", got, "model.")
	}
}
