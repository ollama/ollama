package convert

import (
	"bytes"
	"encoding/binary"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func boolPtr(v bool) *bool {
	return &v
}

func readTensorData(t *testing.T, tensor *ggml.Tensor) []float32 {
	t.Helper()

	var b bytes.Buffer
	if _, err := tensor.WriteTo(&b); err != nil {
		t.Fatal(err)
	}

	numel := 1
	for _, d := range tensor.Shape {
		numel *= int(d)
	}

	values := make([]float32, numel)
	if err := binary.Read(&b, binary.LittleEndian, &values); err != nil {
		t.Fatal(err)
	}

	return values
}

func TestQwen3NextLegacyModelTypeDisablesReorder(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_next",
		},
	}

	if m.shouldReorderVHeads() {
		t.Fatalf("legacy qwen3_next model_type should not reorder v-head layout")
	}
}

func TestQwen3NextLegacyArchitectureDisablesReorder(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			Architectures: []string{"Qwen3NextForCausalLM"},
		},
	}

	if m.shouldReorderVHeads() {
		t.Fatalf("legacy Qwen3Next architecture should not reorder v-head layout")
	}
}

func TestQwen3NextKVLegacyConfig(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_next",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			MaxPositionEmbeddings: 8192,
			HiddenSize:            512,
			NumHiddenLayers:       4,
			IntermediateSize:      2048,
			NumAttentionHeads:     8,
			NumKeyValueHeads:      2,
			HeadDim:               64,
			RopeTheta:             1_000_000,
			RMSNormEPS:            1e-6,

			NumExperts:             8,
			NumExpertsPerToken:     2,
			NormTopkProb:           boolPtr(true),
			MoEIntermediateSize:    256,
			SharedExpertIntermSize: 512,

			FullAttentionInterval: 2,

			LinearConvKernelDim: 4,
			LinearKeyHeadDim:    64,
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  64,

			PartialRotaryFactor: 0.25,
		},
	}

	if err := m.parseMore(os.DirFS(t.TempDir())); err != nil {
		t.Fatal(err)
	}

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := kv["general.architecture"], "qwen35moe"; got != want {
		t.Fatalf("unexpected architecture: got %v want %v", got, want)
	}
	if got, want := kv["tokenizer.ggml.pre"], "qwen35"; got != want {
		t.Fatalf("unexpected tokenizer pre: got %v want %v", got, want)
	}

	headCountKV, ok := kv["attention.head_count_kv"].([]uint32)
	if !ok {
		t.Fatalf("attention.head_count_kv has unexpected type: %T", kv["attention.head_count_kv"])
	}
	if got, want := headCountKV, []uint32{0, 2, 0, 2}; !slices.Equal(got, want) {
		t.Fatalf("unexpected attention.head_count_kv: got %v want %v", got, want)
	}

	if _, ok := kv["ssm.v_head_reordered"]; ok {
		t.Fatalf("legacy qwen3next should not enable ssm.v_head_reordered")
	}
	if got, want := kv["norm_top_k_prob"], true; got != want {
		t.Fatalf("unexpected norm_top_k_prob: got %v want %v", got, want)
	}
}

func TestQwen35MoeOmitsNormTopKProbWhenUnset(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			MaxPositionEmbeddings: 4096,
			HiddenSize:            512,
			NumHiddenLayers:       4,
			IntermediateSize:      2048,
			NumAttentionHeads:     8,
			NumKeyValueHeads:      2,
			HeadDim:               64,
			RopeTheta:             1_000_000,
			RMSNormEPS:            1e-6,
			NumExperts:            8,
			NumExpertsPerToken:    2,
			FullAttentionInterval: 2,
			LinearConvKernelDim:   4,
			LinearKeyHeadDim:      64,
			LinearNumKeyHeads:     2,
			LinearNumValueHeads:   4,
			LinearValueHeadDim:    64,
			PartialRotaryFactor:   0.25,
		},
	}

	if err := m.parseMore(os.DirFS(t.TempDir())); err != nil {
		t.Fatal(err)
	}

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if _, ok := kv["norm_top_k_prob"]; ok {
		t.Fatalf("expected norm_top_k_prob to be omitted when not set in config")
	}
}

func TestQwen35KVFromTextConfig(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		TextConfig: &qwen3NextTextConfig{
			MaxPositionEmbeddings: 16384,
			HiddenSize:            1024,
			NumHiddenLayers:       4,
			IntermediateSize:      4096,
			NumAttentionHeads:     8,
			NumKeyValueHeads:      4,
			HeadDim:               128,
			RMSNormEPS:            1e-6,

			LayerTypes: []string{
				"linear_attention",
				"full_attention",
				"linear_attention",
				"full_attention",
			},

			LinearConvKernelDim: 4,
			LinearKeyHeadDim:    128,
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  128,

			RopeParameters: qwen3NextRopeParams{
				MRopeInterleaved:    true,
				MropeSection:        []int32{11, 11, 10},
				RopeType:            "default",
				RopeTheta:           10_000_000,
				PartialRotaryFactor: 0.25,
			},
		},
		VisionModel: qwen3NextVisionConfig{
			Depth:                  2,
			HiddenSize:             128,
			NumHeads:               4,
			InChannels:             3,
			PatchSize:              16,
			SpatialMergeSize:       2,
			RMSNormEps:             1e-6,
			RopeTheta:              10_000,
			TemporalPatchSize:      2,
			DeepstackVisualIndexes: []int32{1},
		},
		ImageTokenID:       1001,
		VisionStartTokenID: 1002,
		VisionEndTokenID:   1003,
	}
	m.VisionModel.Size.ShortestEdge = 224
	m.VisionModel.Size.LongestEdge = 4096
	m.VisionModel.ImageMean = []float32{0.5, 0.5, 0.5}
	m.VisionModel.ImageStd = []float32{0.2, 0.2, 0.2}

	if err := m.parseMore(os.DirFS(t.TempDir())); err != nil {
		t.Fatal(err)
	}

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := kv["general.architecture"], "qwen35"; got != want {
		t.Fatalf("unexpected architecture: got %v want %v", got, want)
	}

	headCountKV, ok := kv["attention.head_count_kv"].([]uint32)
	if !ok {
		t.Fatalf("attention.head_count_kv has unexpected type: %T", kv["attention.head_count_kv"])
	}
	if got, want := headCountKV, []uint32{0, 4, 0, 4}; !slices.Equal(got, want) {
		t.Fatalf("unexpected attention.head_count_kv: got %v want %v", got, want)
	}

	if got, ok := kv["ssm.v_head_reordered"].(bool); !ok || !got {
		t.Fatalf("expected ssm.v_head_reordered=true, got %v (%T)", kv["ssm.v_head_reordered"], kv["ssm.v_head_reordered"])
	}

	mrope, ok := kv["mrope_sections"].([]int32)
	if !ok {
		t.Fatalf("mrope_sections has unexpected type: %T", kv["mrope_sections"])
	}
	if got, want := mrope, []int32{11, 11, 10}; !slices.Equal(got, want) {
		t.Fatalf("unexpected mrope_sections: got %v want %v", got, want)
	}
	ropeSections, ok := kv["rope.dimension_sections"].([]int32)
	if !ok {
		t.Fatalf("rope.dimension_sections has unexpected type: %T", kv["rope.dimension_sections"])
	}
	if got, want := ropeSections, []int32{11, 11, 10}; !slices.Equal(got, want) {
		t.Fatalf("unexpected rope.dimension_sections: got %v want %v", got, want)
	}

	if got, ok := kv["rope.mrope_interleaved"].(bool); !ok || !got {
		t.Fatalf("expected rope.mrope_interleaved=true, got %v (%T)", kv["rope.mrope_interleaved"], kv["rope.mrope_interleaved"])
	}

	if got, want := kv["vision.block_count"], uint32(2); got != want {
		t.Fatalf("unexpected vision.block_count: got %v want %v", got, want)
	}
}

func TestQwen3NextReplacements(t *testing.T) {
	r := strings.NewReplacer((&qwen3NextModel{}).Replacements()...)

	if got, want := r.Replace("model.language_model.layers.1.linear_attn.in_proj_qkv.weight"), "blk.1.attn_qkv.weight"; got != want {
		t.Fatalf("unexpected language-model replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("model.visual.blocks.0.attn.qkv.weight"), "v.blk.0.attn_qkv.weight"; got != want {
		t.Fatalf("unexpected vision replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("model.layers.1.linear_attn.in_proj_qkvz.weight"), "blk.1.ssm_in.weight"; got != want {
		t.Fatalf("unexpected legacy replacement: got %q want %q", got, want)
	}
}

func TestQwen35ReordersVHeads(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.attn_gate.weight",
			shape: []uint64{4, 2},
			data:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{0, 1, 4, 5, 2, 3, 6, 7}; !slices.Equal(got, want) {
		t.Fatalf("unexpected data: got %v want %v", got, want)
	}
}

func TestQwen35ReordersAttnQKVOutputDim(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearKeyHeadDim:    1,
			LinearValueHeadDim:  1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.attn_qkv.weight",
			shape: []uint64{8, 2}, // [out_features, in_features] (HF layout)
			data: []float32{
				0, 1, // q0
				2, 3, // q1
				4, 5, // k0
				6, 7, // k1
				10, 11, // v(k0,v0)
				12, 13, // v(k0,v1)
				20, 21, // v(k1,v0)
				22, 23, // v(k1,v1)
			},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		10, 11, 20, 21, 12, 13, 22, 23,
	}; !slices.Equal(got, want) {
		t.Fatalf("unexpected qkv data: got %v want %v", got, want)
	}
}

func TestQwen35ReordersSsmOutInputDim(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.ssm_out.weight",
			shape: []uint64{2, 4},
			data:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{0, 2, 1, 3, 4, 6, 5, 7}; !slices.Equal(got, want) {
		t.Fatalf("unexpected ssm_out data: got %v want %v", got, want)
	}
}

func TestQwen35ReordersSsmBetaRows(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.ssm_beta.weight",
			shape: []uint64{4, 2},
			data:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{0, 1, 4, 5, 2, 3, 6, 7}; !slices.Equal(got, want) {
		t.Fatalf("unexpected ssm_beta data: got %v want %v", got, want)
	}
}

func TestQwen35ReordersConv1DChannelDim(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearKeyHeadDim:    1,
			LinearValueHeadDim:  1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.ssm_conv1d.weight",
			shape: []uint64{8, 2}, // [channels, kernel] after squeeze
			data: []float32{
				0, 1, // q0
				2, 3, // q1
				4, 5, // k0
				6, 7, // k1
				10, 11, // v(k0,v0)
				12, 13, // v(k0,v1)
				20, 21, // v(k1,v0)
				22, 23, // v(k1,v1)
			},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		10, 11, 20, 21, 12, 13, 22, 23,
	}; !slices.Equal(got, want) {
		t.Fatalf("unexpected conv1d data: got %v want %v", got, want)
	}
}

func TestLegacyQwen3NextDoesNotReorderVHeads(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_next",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			LinearNumKeyHeads:   2,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.attn_gate.weight",
			shape: []uint64{4, 1},
			data:  []float32{0, 1, 2, 3},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{0, 1, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("unexpected data for legacy qwen3next: got %v want %v", got, want)
	}
}

func TestQwen35MoePackedExperts(t *testing.T) {
	m := &qwen3NextModel{
		qwen3NextTextConfig: qwen3NextTextConfig{
			NumHiddenLayers: 1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.mlp.experts.gate_up_proj",
			shape: []uint64{2, 4, 3},
			data: []float32{
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11,
				12, 13, 14,
				15, 16, 17,
				18, 19, 20,
				21, 22, 23,
			},
		},
		&fakeTensor{
			name:  "blk.0.mlp.experts.down_proj",
			shape: []uint64{2, 5, 3},
			data:  make([]float32, 2*5*3),
		},
	})

	get := func(name string) *ggml.Tensor {
		for _, tensor := range out {
			if tensor.Name == name {
				return tensor
			}
		}
		return nil
	}

	gate := get("blk.0.ffn_gate_exps.weight")
	if gate == nil {
		t.Fatalf("missing tensor %q", "blk.0.ffn_gate_exps.weight")
	}
	if got, want := gate.Shape, []uint64{2, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("unexpected gate shape: got %v want %v", got, want)
	}
	if got, want := readTensorData(t, gate), []float32{
		0, 1, 2, 3, 4, 5,
		12, 13, 14, 15, 16, 17,
	}; !slices.Equal(got, want) {
		t.Fatalf("unexpected gate values: got %v want %v", got, want)
	}

	up := get("blk.0.ffn_up_exps.weight")
	if up == nil {
		t.Fatalf("missing tensor %q", "blk.0.ffn_up_exps.weight")
	}
	if got, want := up.Shape, []uint64{2, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("unexpected up shape: got %v want %v", got, want)
	}
	if got, want := readTensorData(t, up), []float32{
		6, 7, 8, 9, 10, 11,
		18, 19, 20, 21, 22, 23,
	}; !slices.Equal(got, want) {
		t.Fatalf("unexpected up values: got %v want %v", got, want)
	}

	down := get("blk.0.ffn_down_exps.weight")
	if down == nil {
		t.Fatalf("missing tensor %q", "blk.0.ffn_down_exps.weight")
	}
	if got, want := down.Shape, []uint64{2, 5, 3}; !slices.Equal(got, want) {
		t.Fatalf("unexpected down shape: got %v want %v", got, want)
	}
}

func TestQwen35SharedExpertGateKeepsMatrixShape(t *testing.T) {
	m := &qwen3NextModel{}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "blk.0.ffn_gate_inp_shexp.weight",
			shape: []uint64{1, 4},
			data:  []float32{0, 1, 2, 3},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := out[0].Shape, []uint64{1, 4}; !slices.Equal(got, want) {
		t.Fatalf("unexpected shared gate shape: got %v want %v", got, want)
	}
}
