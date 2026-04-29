package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type lagunaTestTensor struct {
	tensorBase
}

func newLagunaTestTensor(name string, shape ...uint64) Tensor {
	return &lagunaTestTensor{tensorBase: tensorBase{name: name, shape: shape}}
}

func (t *lagunaTestTensor) WriteTo(io.Writer) (int64, error) {
	return 0, nil
}

func (t *lagunaTestTensor) Clone() Tensor {
	return &lagunaTestTensor{tensorBase: tensorBase{
		name:  t.name,
		shape: append([]uint64(nil), t.shape...),
	}}
}

func TestLagunaReplacements(t *testing.T) {
	p := lagunaModel{}
	r := strings.NewReplacer(p.Replacements()...)

	tests := []struct {
		name string
		in   string
		want string
	}{
		{"embed", "model.embed_tokens.weight", "token_embd.weight"},
		{"final_norm", "model.norm.weight", "output_norm.weight"},
		{"lm_head", "lm_head.weight", "output.weight"},
		{"block prefix", "model.layers.7.input_layernorm.weight", "blk.7.attn_norm.weight"},
		{"q", "model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"},
		{"k", "model.layers.3.self_attn.k_proj.weight", "blk.3.attn_k.weight"},
		{"v", "model.layers.3.self_attn.v_proj.weight", "blk.3.attn_v.weight"},
		{"o", "model.layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight"},
		{"g", "model.layers.3.self_attn.g_proj.weight", "blk.3.attn_g.weight"},
		{"q_norm", "model.layers.3.self_attn.q_norm.weight", "blk.3.attn_q_norm.weight"},
		{"k_norm", "model.layers.3.self_attn.k_norm.weight", "blk.3.attn_k_norm.weight"},
		{"post_attn_norm", "model.layers.3.post_attention_layernorm.weight", "blk.3.ffn_norm.weight"},
		{"dense gate", "model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"},
		{"dense up", "model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"},
		{"dense down", "model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
		{"shexp gate", "model.layers.5.mlp.shared_expert.gate_proj.weight", "blk.5.ffn_gate_shexp.weight"},
		{"shexp up", "model.layers.5.mlp.shared_expert.up_proj.weight", "blk.5.ffn_up_shexp.weight"},
		{"shexp down", "model.layers.5.mlp.shared_expert.down_proj.weight", "blk.5.ffn_down_shexp.weight"},
		{"router", "model.layers.5.mlp.gate.weight", "blk.5.ffn_gate_inp.weight"},
		{"score bias", "model.layers.5.mlp.experts.e_score_correction_bias", "blk.5.exp_probs_b.bias"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := r.Replace(tc.in); got != tc.want {
				t.Errorf("Replace(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestLagunaValidateRejectsUnsupportedVariants(t *testing.T) {
	base := validLagunaTestModel()
	tests := []struct {
		name string
		edit func(*lagunaModel)
		want string
	}{
		{
			name: "per-element gating",
			edit: func(m *lagunaModel) {
				m.Gating = "per-element"
			},
			want: "unsupported attention gating",
		},
		{
			name: "attention sinks",
			edit: func(m *lagunaModel) {
				m.SwaAttentionSinkEnabled = true
			},
			want: "swa_attention_sink_enabled=true",
		},
		{
			name: "qk norm disabled",
			edit: func(m *lagunaModel) {
				m.QKNormType = "none"
			},
			want: "unsupported qk_norm_type",
		},
		{
			name: "softmax moe",
			edit: func(m *lagunaModel) {
				m.MoERouterUseSigmoid = false
			},
			want: "moe_router_use_sigmoid=false",
		},
		{
			name: "router weight on input",
			edit: func(m *lagunaModel) {
				m.MoEApplyRouterWeightOnInput = true
			},
			want: "moe_apply_router_weight_on_input=true",
		},
		{
			name: "unknown layer type",
			edit: func(m *lagunaModel) {
				m.LayerTypes[1] = "local_attention"
			},
			want: "unsupported layer_types[1]",
		},
		{
			name: "nonstandard dense layout",
			edit: func(m *lagunaModel) {
				m.MLPOnlyLayers = []uint32{0, 3}
			},
			want: "unsupported mlp_only_layers",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := base
			m.LayerTypes = append([]string(nil), base.LayerTypes...)
			m.NumAttentionHeadsPerLayer = append([]uint32(nil), base.NumAttentionHeadsPerLayer...)
			m.MLPOnlyLayers = append([]uint32(nil), base.MLPOnlyLayers...)
			tc.edit(&m)
			err := m.validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("validate() error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestLagunaGAConfigNormalizesBoolGatingAndNestedRope(t *testing.T) {
	var m lagunaModel
	if err := json.Unmarshal([]byte(`{
		"architectures": ["LagunaForCausalLM"],
		"num_hidden_layers": 1,
		"hidden_size": 8,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"gating": true,
		"num_experts": 2,
		"num_experts_per_tok": 1,
		"moe_intermediate_size": 4,
		"shared_expert_intermediate_size": 4,
		"decoder_sparse_step": 1,
		"mlp_layer_types": ["dense"],
		"rope_parameters": {
			"full_attention": {
				"rope_theta": 500000,
				"rope_type": "yarn",
				"factor": 32,
				"original_max_position_embeddings": 4096,
				"beta_fast": 64,
				"beta_slow": 1,
				"attention_factor": 1,
				"partial_rotary_factor": 0.5
			},
			"sliding_attention": {
				"rope_theta": 10000,
				"rope_type": "default",
				"partial_rotary_factor": 1
			}
		}
	}`), &m); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if err := m.validate(); err != nil {
		t.Fatalf("validate() error = %v", err)
	}
	if m.Gating != "true" {
		t.Fatalf("Gating = %q, want raw true marker", m.Gating)
	}
	if !m.Gating.perHead() {
		t.Fatal("expected bool gating to normalize as per-head support")
	}
	if m.QKNormType != "rmsnorm" {
		t.Fatalf("QKNormType = %q, want rmsnorm default", m.QKNormType)
	}
	if !m.MoERouterUseSigmoid {
		t.Fatal("MoERouterUseSigmoid should default true")
	}
	if !m.NormTopKProb {
		t.Fatal("NormTopKProb should default true")
	}
	if diff := cmp.Diff(m.MLPOnlyLayers, []uint32{0}); diff != "" {
		t.Fatalf("MLPOnlyLayers mismatch (-got +want):\n%s", diff)
	}
	if m.RopeParameters.RopeTheta != 500000 || m.RopeParameters.PartialRotaryFactor != 0.5 {
		t.Fatalf("full rope = %#v, want theta=500000 partial=0.5", m.RopeParameters)
	}
	if m.SwaRopeParameters.RopeTheta != 10000 || m.SwaRopeParameters.PartialRotaryFactor != 1 {
		t.Fatalf("swa rope = %#v, want theta=10000 partial=1", m.SwaRopeParameters)
	}
}

func validLagunaTestModel() lagunaModel {
	return lagunaModel{
		ModelParameters: ModelParameters{
			VocabSize: 32,
		},
		NumHiddenLayers:              2,
		HiddenSize:                   8,
		IntermediateSize:             16,
		NumAttentionHeads:            2,
		NumKeyValueHeads:             1,
		HeadDim:                      4,
		RMSNormEPS:                   1e-6,
		MaxPositionEmbeddings:        4096,
		SlidingWindow:                512,
		Gating:                       "per-head",
		QKNormType:                   "rmsnorm",
		LayerTypes:                   []string{"global_attention", "sliding_attention"},
		NumAttentionHeadsPerLayer:    []uint32{2, 2},
		NumExperts:                   2,
		NumExpertsPerTok:             1,
		MoEIntermediateSize:          4,
		SharedExpertIntermediateSize: 4,
		NormTopKProb:                 true,
		MoeRoutedScalingFactor:       2.5,
		MoERouterUseSigmoid:          true,
		DecoderSparseStep:            1,
		MLPOnlyLayers:                []uint32{0},
	}
}

func validLagunaTestTensors(m lagunaModel) []Tensor {
	ts := []Tensor{
		newLagunaTestTensor("token_embd.weight", uint64(m.VocabSize), uint64(m.HiddenSize)),
		newLagunaTestTensor("output_norm.weight", uint64(m.HiddenSize)),
	}

	for layer := range m.NumHiddenLayers {
		prefix := fmt.Sprintf("blk.%d", layer)
		heads := uint64(m.numHeadsForLayer(layer))
		attnWidth := heads * uint64(m.HeadDim)
		kvWidth := uint64(m.NumKeyValueHeads * m.HeadDim)
		ts = append(ts,
			newLagunaTestTensor(prefix+".attn_norm.weight", uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".ffn_norm.weight", uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".attn_q.weight", attnWidth, uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".attn_k.weight", kvWidth, uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".attn_v.weight", kvWidth, uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".attn_output.weight", uint64(m.HiddenSize), attnWidth),
			newLagunaTestTensor(prefix+".attn_g.weight", heads, uint64(m.HiddenSize)),
			newLagunaTestTensor(prefix+".attn_q_norm.weight", uint64(m.HeadDim)),
			newLagunaTestTensor(prefix+".attn_k_norm.weight", uint64(m.HeadDim)),
		)

		if m.layerUsesMoE(layer) {
			ts = append(ts,
				newLagunaTestTensor(prefix+".ffn_gate_inp.weight", uint64(m.NumExperts), uint64(m.HiddenSize)),
				newLagunaTestTensor(prefix+".exp_probs_b.bias", uint64(m.NumExperts)),
				newLagunaTestTensor(prefix+".ffn_gate_shexp.weight", uint64(m.SharedExpertIntermediateSize), uint64(m.HiddenSize)),
				newLagunaTestTensor(prefix+".ffn_up_shexp.weight", uint64(m.SharedExpertIntermediateSize), uint64(m.HiddenSize)),
				newLagunaTestTensor(prefix+".ffn_down_shexp.weight", uint64(m.HiddenSize), uint64(m.SharedExpertIntermediateSize)),
			)
			for expert := range m.NumExperts {
				ts = append(ts,
					newLagunaTestTensor(fmt.Sprintf("%s.mlp.experts.%d.gate_proj.weight", prefix, expert), uint64(m.MoEIntermediateSize), uint64(m.HiddenSize)),
					newLagunaTestTensor(fmt.Sprintf("%s.mlp.experts.%d.up_proj.weight", prefix, expert), uint64(m.MoEIntermediateSize), uint64(m.HiddenSize)),
					newLagunaTestTensor(fmt.Sprintf("%s.mlp.experts.%d.down_proj.weight", prefix, expert), uint64(m.HiddenSize), uint64(m.MoEIntermediateSize)),
				)
			}
		} else {
			ts = append(ts,
				newLagunaTestTensor(prefix+".ffn_gate.weight", uint64(m.IntermediateSize), uint64(m.HiddenSize)),
				newLagunaTestTensor(prefix+".ffn_up.weight", uint64(m.IntermediateSize), uint64(m.HiddenSize)),
				newLagunaTestTensor(prefix+".ffn_down.weight", uint64(m.HiddenSize), uint64(m.IntermediateSize)),
			)
		}
	}

	return ts
}

func TestLagunaTensorsMergeRoutedExperts(t *testing.T) {
	m := validLagunaTestModel()
	out := m.Tensors(validLagunaTestTensors(m))

	tensors := make(map[string]*ggml.Tensor, len(out))
	for _, t := range out {
		tensors[t.Name] = t
	}

	tests := map[string][]uint64{
		"blk.1.ffn_gate_exps.weight": {uint64(m.NumExperts), uint64(m.MoEIntermediateSize), uint64(m.HiddenSize)},
		"blk.1.ffn_up_exps.weight":   {uint64(m.NumExperts), uint64(m.MoEIntermediateSize), uint64(m.HiddenSize)},
		"blk.1.ffn_down_exps.weight": {uint64(m.NumExperts), uint64(m.HiddenSize), uint64(m.MoEIntermediateSize)},
	}
	for name, wantShape := range tests {
		tensor, ok := tensors[name]
		if !ok {
			t.Fatalf("missing merged tensor %q", name)
		}
		if diff := cmp.Diff(wantShape, tensor.Shape); diff != "" {
			t.Fatalf("%s shape mismatch (-want +got):\n%s", name, diff)
		}
	}

	for expert := range m.NumExperts {
		name := fmt.Sprintf("blk.1.mlp.experts.%d.gate_proj.weight", expert)
		if _, ok := tensors[name]; ok {
			t.Fatalf("unexpected unmerged expert tensor %q", name)
		}
	}
}

func TestLagunaKVShape(t *testing.T) {
	m := lagunaModel{
		NumHiddenLayers:              4,
		HiddenSize:                   128,
		IntermediateSize:             256,
		NumAttentionHeads:            8,
		NumKeyValueHeads:             4,
		HeadDim:                      16,
		RMSNormEPS:                   1e-6,
		MaxPositionEmbeddings:        4096,
		SlidingWindow:                512,
		PartialRotaryFactor:          0.5,
		Gating:                       "per-head",
		QKNormType:                   "rmsnorm",
		LayerTypes:                   []string{"full_attention", "sliding_attention", "sliding_attention", "sliding_attention"},
		NumAttentionHeadsPerLayer:    []uint32{8, 16, 16, 16},
		NumExperts:                   32,
		NumExpertsPerTok:             4,
		MoEIntermediateSize:          64,
		SharedExpertIntermediateSize: 64,
		NormTopKProb:                 true,
		MoeRoutedScalingFactor:       2.5,
		MoERouterUseSigmoid:          true,
		DecoderSparseStep:            1,
		MLPOnlyLayers:                []uint32{0},
	}
	m.RopeParameters.RopeTheta = 500000
	m.RopeParameters.RopeType = "yarn"
	m.RopeParameters.Factor = 32
	m.RopeParameters.OriginalMaxPositionEmbeddings = 4096
	m.RopeParameters.BetaFast = 64
	m.RopeParameters.BetaSlow = 1
	m.SwaRopeParameters.RopeTheta = 10000
	m.SwaRopeParameters.RopeType = "linear"
	m.SwaRopeParameters.Factor = 1
	m.SwaRopeParameters.PartialRotaryFactor = 1

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}, Template: "{% include 'chat_template.jinja' %}"})
	required := []string{
		"general.architecture",
		"tokenizer.ggml.pre",
		"laguna.block_count",
		"laguna.context_length",
		"laguna.embedding_length",
		"laguna.feed_forward_length",
		"laguna.attention.head_count",
		"laguna.attention.head_count_kv",
		"laguna.attention.key_length",
		"laguna.attention.value_length",
		"laguna.attention.layer_norm_rms_epsilon",
		"laguna.attention.sliding_window",
		"laguna.attention.layer_types",
		"laguna.attention.sliding_window_pattern",
		"laguna.attention.gating_type",
		"laguna.attention.qk_norm",
		"laguna.expert_count",
		"laguna.expert_used_count",
		"laguna.expert_feed_forward_length",
		"laguna.expert_shared_feed_forward_length",
		"laguna.expert_shared_count",
		"laguna.expert_weights_norm",
		"laguna.expert_weights_scale",
		"laguna.expert_gating_func",
		"laguna.leading_dense_block_count",
		"laguna.dense_layers",
		"laguna.rope.freq_base",
		"laguna.rope.scaling.type",
		"laguna.rope.scaling.factor",
		"laguna.rope.partial_rotary_factor",
		"laguna.rope.swa.freq_base",
		"laguna.rope.swa.scaling.type",
		"laguna.rope.dimension_count",
		"laguna.rope.swa.dimension_count",
	}
	for _, k := range required {
		if _, ok := kv[k]; !ok {
			t.Errorf("missing required KV: %s", k)
		}
	}

	if got := kv["general.architecture"]; got != "laguna" {
		t.Errorf("architecture = %v, want laguna", got)
	}
	if got := kv["tokenizer.ggml.add_bos_token"]; got != false {
		t.Errorf("tokenizer.ggml.add_bos_token = %v, want false", got)
	}
	if _, ok := kv["tokenizer.chat_template"]; ok {
		t.Fatal("tokenizer.chat_template should be omitted for Laguna")
	}
	if got := kv["laguna.expert_gating_func"]; got != lagunaGatingFuncSigmoid {
		t.Errorf("expert_gating_func = %v, want sigmoid(%d)", got, lagunaGatingFuncSigmoid)
	}
	if got := kv["laguna.leading_dense_block_count"]; got != uint32(1) {
		t.Errorf("leading_dense_block_count = %v, want 1", got)
	}
	if got := kv["laguna.rope.dimension_count"]; got != uint32(8) {
		t.Errorf("rope.dimension_count = %v, want 8", got)
	}
	if got := kv["laguna.rope.swa.dimension_count"]; got != uint32(16) {
		t.Errorf("rope.swa.dimension_count = %v, want 16", got)
	}
	if got, ok := kv["laguna.attention.layer_types"].([]uint32); !ok || len(got) != 4 || got[0] != 0 || got[1] != 1 || got[2] != 1 || got[3] != 1 {
		t.Fatalf("layer_types = %#v, want [0 1 1 1]", kv["laguna.attention.layer_types"])
	}
	if got, ok := kv["laguna.attention.sliding_window_pattern"].([]bool); !ok || len(got) != 4 || got[0] || !got[1] || !got[2] || !got[3] {
		t.Fatalf("sliding_window_pattern = %#v, want [false true true true]", kv["laguna.attention.sliding_window_pattern"])
	}
}

func TestLagunaKVYarnAttentionFactorFallback(t *testing.T) {
	m := validLagunaTestModel()
	m.RopeParameters.RopeType = "yarn"
	m.RopeParameters.Factor = 32

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	got, ok := kv["laguna.rope.scaling.attn_factor"].(float32)
	if !ok {
		t.Fatalf("attn_factor type = %T, want float32", kv["laguna.rope.scaling.attn_factor"])
	}

	want := float32(0.1*math.Log(32) + 1)
	if diff := math.Abs(float64(got - want)); diff > 1e-6 {
		t.Fatalf("attn_factor = %v, want %v", got, want)
	}
}
