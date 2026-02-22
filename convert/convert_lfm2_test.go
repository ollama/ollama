package convert

import (
	"io"
	"slices"
	"strings"
	"testing"
)

type lfm2StubTensor struct {
	tensorBase
}

func newLFM2StubTensor(name string, shape []uint64) *lfm2StubTensor {
	return &lfm2StubTensor{
		tensorBase: tensorBase{
			name:  name,
			shape: shape,
		},
	}
}

func (t *lfm2StubTensor) WriteTo(io.Writer) (int64, error) {
	return 0, nil
}

func (t *lfm2StubTensor) Clone() Tensor {
	return &lfm2StubTensor{
		tensorBase: tensorBase{
			name:  t.name,
			shape: slices.Clone(t.shape),
		},
	}
}

func TestLFM2MoEKV(t *testing.T) {
	var p lfm2Model
	p.ModelParameters.ModelType = "lfm2_moe"
	p.VocabSize = 65536
	p.HiddenSize = 2048
	p.NumHiddenLayers = 4
	p.MaxPositionEmbeddings = 128000
	p.IntermediateSize = 11776
	p.NumAttentionHeads = 32
	p.NumKeyValueHeads = 8
	p.LayerTypes = []string{"conv", "full_attention", "conv", "full_attention"}
	p.NormEps = 1e-5
	p.ConvLCache = 3
	p.MoEIntermediateSize = 1536
	p.NumExperts = 64
	p.NumExpertsPerToken = 4
	p.NumDenseLayers = 2
	p.RopeParameters.RopeTheta = 1_000_000

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["general.architecture"], "lfm2moe"; got != want {
		t.Fatalf("general.architecture = %v, want %v", got, want)
	}

	if got, want := kv["expert_count"], uint32(64); got != want {
		t.Fatalf("expert_count = %v, want %v", got, want)
	}

	if got, want := kv["expert_used_count"], uint32(4); got != want {
		t.Fatalf("expert_used_count = %v, want %v", got, want)
	}

	if got, want := kv["expert_feed_forward_length"], uint32(1536); got != want {
		t.Fatalf("expert_feed_forward_length = %v, want %v", got, want)
	}

	if got, want := kv["leading_dense_block_count"], uint32(2); got != want {
		t.Fatalf("leading_dense_block_count = %v, want %v", got, want)
	}

	if got, want := kv["expert_gating_func"], uint32(2); got != want {
		t.Fatalf("expert_gating_func = %v, want %v", got, want)
	}

	gotHeadCounts, ok := kv["attention.head_count_kv"].([]uint32)
	if !ok {
		t.Fatalf("attention.head_count_kv has unexpected type %T", kv["attention.head_count_kv"])
	}

	wantHeadCounts := []uint32{0, 8, 0, 8}
	if !slices.Equal(gotHeadCounts, wantHeadCounts) {
		t.Fatalf("attention.head_count_kv = %v, want %v", gotHeadCounts, wantHeadCounts)
	}

	if got, want := kv["rope.freq_base"], float32(1_000_000); got != want {
		t.Fatalf("rope.freq_base = %v, want %v", got, want)
	}
}

func TestLFM2DenseKV(t *testing.T) {
	p := lfm2Model{
		ModelParameters:       ModelParameters{ModelType: "lfm2", VocabSize: 32000},
		HiddenSize:            1024,
		NumHiddenLayers:       2,
		MaxPositionEmbeddings: 32768,
		IntermediateSize:      4096,
		NumAttentionHeads:     16,
		NumKeyValueHeads:      4,
		LayerTypes:            []string{"conv", "full_attention"},
		NormEps:               1e-5,
		ConvLCache:            3,
		RopeTheta:             10000,
	}

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["general.architecture"], "lfm2"; got != want {
		t.Fatalf("general.architecture = %v, want %v", got, want)
	}

	if _, ok := kv["expert_count"]; ok {
		t.Fatalf("expert_count should not be set for dense lfm2")
	}
}

func TestLFM2MoETensors(t *testing.T) {
	p := lfm2Model{
		ModelParameters: ModelParameters{ModelType: "lfm2_moe"},
		NumHiddenLayers: 4,
		NumDenseLayers:  2,
	}

	in := []Tensor{
		newLFM2StubTensor("blk.2.feed_forward.experts.0.w1.weight", []uint64{1536, 2048}),
		newLFM2StubTensor("blk.2.feed_forward.experts.1.w1.weight", []uint64{1536, 2048}),
		newLFM2StubTensor("blk.2.feed_forward.experts.0.w2.weight", []uint64{2048, 1536}),
		newLFM2StubTensor("blk.2.feed_forward.experts.1.w2.weight", []uint64{2048, 1536}),
		newLFM2StubTensor("blk.2.feed_forward.experts.0.w3.weight", []uint64{1536, 2048}),
		newLFM2StubTensor("blk.2.feed_forward.experts.1.w3.weight", []uint64{1536, 2048}),
		newLFM2StubTensor("blk.0.shortconv.conv.weight", []uint64{2048, 1, 3}),
	}

	out := p.Tensors(in)

	byName := make(map[string][]uint64, len(out))
	for _, tns := range out {
		byName[tns.Name] = tns.Shape
	}

	if got, ok := byName["blk.2.ffn_gate_exps.weight"]; !ok {
		t.Fatalf("missing merged tensor blk.2.ffn_gate_exps.weight")
	} else if !slices.Equal(got, []uint64{2, 1536, 2048}) {
		t.Fatalf("blk.2.ffn_gate_exps.weight shape = %v, want [2 1536 2048]", got)
	}

	if got, ok := byName["blk.2.ffn_down_exps.weight"]; !ok {
		t.Fatalf("missing merged tensor blk.2.ffn_down_exps.weight")
	} else if !slices.Equal(got, []uint64{2, 2048, 1536}) {
		t.Fatalf("blk.2.ffn_down_exps.weight shape = %v, want [2 2048 1536]", got)
	}

	if got, ok := byName["blk.2.ffn_up_exps.weight"]; !ok {
		t.Fatalf("missing merged tensor blk.2.ffn_up_exps.weight")
	} else if !slices.Equal(got, []uint64{2, 1536, 2048}) {
		t.Fatalf("blk.2.ffn_up_exps.weight shape = %v, want [2 1536 2048]", got)
	}

	if got, ok := byName["blk.0.shortconv.conv.weight"]; !ok {
		t.Fatalf("missing shortconv tensor")
	} else if !slices.Equal(got, []uint64{2048, 3}) {
		t.Fatalf("blk.0.shortconv.conv.weight shape = %v, want [2048 3]", got)
	}

	if _, ok := byName["blk.2.feed_forward.experts.0.w1.weight"]; ok {
		t.Fatalf("unmerged expert tensor should not be present")
	}
}

func TestLFM2MoEReplacements(t *testing.T) {
	p := lfm2Model{}
	replacer := strings.NewReplacer(p.Replacements()...)

	if got, want := replacer.Replace("model.layers.2.feed_forward.expert_bias"), "blk.2.exp_probs_b.bias"; got != want {
		t.Fatalf("expert bias replacement = %q, want %q", got, want)
	}

	if got, want := replacer.Replace("model.layers.2.feed_forward.gate.weight"), "blk.2.ffn_gate_inp.weight"; got != want {
		t.Fatalf("gate replacement = %q, want %q", got, want)
	}
}

func TestLFM2KVContextLengthEdgeCaseFallbackOverride(t *testing.T) {
	p := lfm2Model{
		ModelParameters:       ModelParameters{ModelType: "lfm2_moe", VocabSize: 65536},
		HiddenSize:            2048,
		NumHiddenLayers:       40,
		MaxPositionEmbeddings: 128000,
		IntermediateSize:      11776,
		NumAttentionHeads:     32,
		NumKeyValueHeads:      8,
		LayerTypes:            make([]string, 40),
		NormEps:               1e-5,
		ConvLCache:            3,
		MoEIntermediateSize:   1536,
		NumExperts:            64,
		NumExpertsPerToken:    4,
		NumDenseLayers:        2,
	}
	for i := 0; i < len(p.LayerTypes); i++ {
		p.LayerTypes[i] = "conv"
	}
	p.LayerTypes[2] = "full_attention"

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["context_length"], uint32(32768); got != want {
		t.Fatalf("context_length = %v, want %v", got, want)
	}
}

func TestLFM2KVContextLengthNoOverride(t *testing.T) {
	p := lfm2Model{
		ModelParameters:       ModelParameters{ModelType: "lfm2_moe", VocabSize: 65536},
		HiddenSize:            2048,
		NumHiddenLayers:       39, // mismatch: should not trigger edge case
		MaxPositionEmbeddings: 128000,
		IntermediateSize:      11776,
		NumAttentionHeads:     32,
		NumKeyValueHeads:      8,
		LayerTypes:            []string{"conv", "full_attention"},
		NormEps:               1e-5,
		ConvLCache:            3,
		MoEIntermediateSize:   1536,
		NumExperts:            64,
		NumExpertsPerToken:    4,
		NumDenseLayers:        2,
	}

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["context_length"], uint32(128000); got != want {
		t.Fatalf("context_length = %v, want %v", got, want)
	}
}

func TestLFM2KVFeedForwardLengthAutoAdjust(t *testing.T) {
	p := lfm2Model{
		ModelParameters:        ModelParameters{ModelType: "lfm2", VocabSize: 65536},
		HiddenSize:             2048,
		NumHiddenLayers:        16,
		MaxPositionEmbeddings:  128000,
		IntermediateSize:       12288, // should be ignored when block_ff_dim is set
		BlockFFDim:             12288,
		BlockAutoAdjustFFDim:   true,
		BlockMultipleOf:        256,
		BlockFFNDimMultiplier:  1.0,
		NumAttentionHeads:      32,
		NumKeyValueHeads:       8,
		LayerTypes:             []string{"conv", "full_attention"},
		NormEps:                1e-5,
		ConvLCache:             3,
	}

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["feed_forward_length"], uint32(8192); got != want {
		t.Fatalf("feed_forward_length = %v, want %v", got, want)
	}
}
