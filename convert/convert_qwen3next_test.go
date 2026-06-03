package convert

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/d4l3k/go-bfloat16"
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

	if got, want := kv["attention.head_count_kv"], uint32(2); got != want {
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
			IntermediateSize:       512,
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

	if got, want := kv["attention.head_count_kv"], uint32(4); got != want {
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
	if got, want := ropeSections, []int32{11, 11, 10, 0}; !slices.Equal(got, want) {
		t.Fatalf("unexpected rope.dimension_sections: got %v want %v", got, want)
	}

	if got, ok := kv["rope.mrope_interleaved"].(bool); !ok || !got {
		t.Fatalf("expected rope.mrope_interleaved=true, got %v (%T)", kv["rope.mrope_interleaved"], kv["rope.mrope_interleaved"])
	}

	if got, want := kv["vision.block_count"], uint32(2); got != want {
		t.Fatalf("unexpected vision.block_count: got %v want %v", got, want)
	}
	if got, want := kv["vision.feed_forward_length"], uint32(512); got != want {
		t.Fatalf("unexpected vision.feed_forward_length: got %v want %v", got, want)
	}
}

func TestQwen35MTPTensors(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		qwen3NextTextConfig: qwen3NextTextConfig{
			NumHiddenLayers:       32,
			NumNextNPredictLayers: 1,
		},
	}

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := kv["block_count"], uint32(33); got != want {
		t.Fatalf("unexpected block_count: got %v want %v", got, want)
	}
	if got, want := kv["nextn_predict_layers"], uint32(1); got != want {
		t.Fatalf("unexpected nextn_predict_layers: got %v want %v", got, want)
	}

	tensors := m.Tensors([]Tensor{
		&fakeTensor{name: "mtp.fc.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
		&fakeTensor{name: "mtp.pre_fc_norm_embedding.weight", shape: []uint64{2}, data: []float32{0, 1}},
		&fakeTensor{name: "mtp.pre_fc_norm_hidden.weight", shape: []uint64{2}, data: []float32{0, 1}},
		&fakeTensor{name: "mtp.norm.weight", shape: []uint64{2}, data: []float32{0, 1}},
		&fakeTensor{name: "mtp.layers.0.attn_q.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
		&fakeTensor{name: "mtp.layers.0.ffn_down.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
	})

	byName := map[string]*ggml.Tensor{}
	for _, tensor := range tensors {
		byName[tensor.Name] = tensor
	}

	for _, name := range []string{
		"blk.32.nextn.eh_proj.weight",
		"blk.32.nextn.enorm.weight",
		"blk.32.nextn.hnorm.weight",
		"blk.32.nextn.shared_head_norm.weight",
		"blk.32.attn_q.weight",
		"blk.32.ffn_down.weight",
	} {
		if _, ok := byName[name]; !ok {
			t.Fatalf("missing MTP tensor %q", name)
		}
	}

	for _, name := range []string{
		"blk.32.nextn.enorm.weight",
		"blk.32.nextn.hnorm.weight",
		"blk.32.nextn.shared_head_norm.weight",
	} {
		if got, want := readTensorData(t, byName[name]), []float32{1, 2}; !slices.Equal(got, want) {
			t.Fatalf("unexpected shifted norm values for %s: got %v want %v", name, got, want)
		}
	}
}

func TestQwen35NativeSplitKV(t *testing.T) {
	m := &qwen3NextModel{
		ModelParameters: ModelParameters{
			ModelType: "qwen3_5",
		},
		TextConfig: &qwen3NextTextConfig{
			MaxPositionEmbeddings: 16384,
			HiddenSize:            2560,
			NumHiddenLayers:       4,
			IntermediateSize:      9216,
			NumAttentionHeads:     16,
			NumKeyValueHeads:      4,
			HeadDim:               256,
			RMSNormEPS:            1e-6,
			FullAttentionInterval: 2,
			LinearConvKernelDim:   4,
			LinearKeyHeadDim:      128,
			LinearNumKeyHeads:     16,
			LinearNumValueHeads:   32,
			LinearValueHeadDim:    128,
			RopeParameters: qwen3NextRopeParams{
				MRopeInterleaved:    true,
				MropeSection:        []int32{11, 11, 10},
				RopeTheta:           10_000_000,
				PartialRotaryFactor: 0.25,
			},
		},
		VisionModel: qwen3NextVisionConfig{
			Depth:                 24,
			HiddenSize:            1024,
			IntermediateSize:      4096,
			NumHeads:              16,
			NumPositionEmbeddings: 2304,
			InChannels:            3,
			OutHiddenSize:         2560,
			PatchSize:             16,
			SpatialMergeSize:      2,
		},
		ImageTokenID:       248056,
		VisionStartTokenID: 248053,
		VisionEndTokenID:   248054,
	}
	m.VisionModel.ImageMean = []float32{0.5, 0.5, 0.5}
	m.VisionModel.ImageStd = []float32{0.5, 0.5, 0.5}

	if err := m.parseMore(os.DirFS(t.TempDir())); err != nil {
		t.Fatal(err)
	}

	textKV := m.TextKV(&Tokenizer{Vocabulary: &Vocabulary{}})
	for _, key := range []string{
		"vision.block_count",
		"image_token_id",
		"vision_start_token_id",
		"vision_end_token_id",
		"mrope_sections",
		"rope.mrope_section",
		"rope.mrope_interleaved",
		"ssm.v_head_reordered",
	} {
		if _, ok := textKV[key]; ok {
			t.Fatalf("TextKV retained %q", key)
		}
	}
	if got, want := textKV["rope.dimension_sections"], []int32{11, 11, 10, 0}; !slices.Equal(got.([]int32), want) {
		t.Fatalf("unexpected rope.dimension_sections: got %v want %v", got, want)
	}

	projectorKV := m.ProjectorKV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := projectorKV["general.architecture"], "clip"; got != want {
		t.Fatalf("unexpected projector architecture: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.projector_type"], "qwen3vl_merger"; got != want {
		t.Fatalf("unexpected projector type: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.vision.feed_forward_length"], uint32(4096); got != want {
		t.Fatalf("unexpected projector feed_forward_length: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.vision.image_size"], uint32(768); got != want {
		t.Fatalf("unexpected projector image_size: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.vision.projection_dim"], uint32(2560); got != want {
		t.Fatalf("unexpected projector projection_dim: got %v want %v", got, want)
	}
}

func TestQwen35ProjectorTensors(t *testing.T) {
	m := &qwen3NextModel{
		VisionModel: qwen3NextVisionConfig{Depth: 1},
	}

	patch := &fakeTensor{
		name:  "v.patch_embed.weight",
		shape: []uint64{2, 2, 2, 1, 2},
		data:  []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	}
	tensors := m.ProjectorTensors([]Tensor{
		patch,
		&fakeTensor{name: "v.pos_embed.weight", shape: []uint64{4, 2}, data: []float32{0, 1, 2, 3, 4, 5, 6, 7}},
		&fakeTensor{name: "v.blk.0.attn_qkv.weight", shape: []uint64{6, 2}, data: make([]float32, 12), sourceDType: "BF16", kind: tensorKindFP16},
		&fakeTensor{name: "v.blk.0.mlp.linear_fc1.weight", shape: []uint64{8, 2}, data: make([]float32, 16), sourceDType: "BF16", kind: tensorKindFP16},
		&fakeTensor{name: "token_embd.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
		&fakeTensor{name: "mtp.fc.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
	})

	byName := map[string]*ggml.Tensor{}
	for _, tensor := range tensors {
		byName[tensor.Name] = tensor
	}

	if _, ok := byName["token_embd.weight"]; ok {
		t.Fatalf("projector tensors included text tensor")
	}
	if _, ok := byName["mtp.fc.weight"]; ok {
		t.Fatalf("projector tensors included MTP tensor")
	}
	if got := byName["v.position_embd.weight"]; got == nil || got.Kind != tensorKindFP32 {
		t.Fatalf("position embedding was not promoted to F32: %#v", got)
	}
	if got := byName["v.blk.0.attn_qkv.weight"]; got == nil {
		t.Fatalf("attn_qkv tensor missing")
	} else if got.Kind != tensorKindBF16 {
		t.Fatalf("attn_qkv tensor was not preserved as BF16: %#v", got)
	}
	if got := byName["v.blk.0.ffn_up.weight"]; got == nil {
		t.Fatalf("ffn_up tensor missing")
	} else if got.Kind != tensorKindBF16 {
		t.Fatalf("ffn_up tensor was not preserved as BF16: %#v", got)
	}

	first := byName["v.patch_embd.weight"]
	if first == nil {
		t.Fatalf("first patch embedding slice missing")
	}
	if got, want := first.Shape, []uint64{2, 2, 1, 2}; !slices.Equal(got, want) {
		t.Fatalf("unexpected first patch shape: got %v want %v", got, want)
	}
	if got, want := readTensorData(t, first), []float32{0, 1, 4, 5, 8, 9, 12, 13}; !slices.Equal(got, want) {
		t.Fatalf("unexpected first patch data: got %v want %v", got, want)
	}

	second := byName["v.patch_embd.weight.1"]
	if second == nil {
		t.Fatalf("second patch embedding slice missing")
	}
	if got, want := readTensorData(t, second), []float32{2, 3, 6, 7, 10, 11, 14, 15}; !slices.Equal(got, want) {
		t.Fatalf("unexpected second patch data: got %v want %v", got, want)
	}
}

func TestQwen35BF16ProjectorWriterPreservesSource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tensor.bin")
	values := []float32{1, -2, 3.5, 4.25}
	raw := bfloat16.EncodeFloat32(values)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	st := safetensor{
		fs:     os.DirFS(dir),
		path:   "tensor.bin",
		dtype:  "BF16",
		offset: 0,
		size:   int64(len(raw)),
		tensorBase: &tensorBase{
			name:  "v.blk.0.attn_qkv.weight",
			shape: []uint64{2, 2},
		},
	}
	tensor := &ggml.Tensor{
		Name:     "v.blk.0.attn_qkv.weight",
		Kind:     tensorKindBF16,
		Shape:    []uint64{2, 2},
		WriterTo: tensorBF16Writer{tensor: st},
	}

	var got bytes.Buffer
	if n, err := tensor.WriteTo(&got); err != nil {
		t.Fatal(err)
	} else if n != int64(len(raw)) {
		t.Fatalf("unexpected byte count: got %d want %d", n, len(raw))
	}
	if !bytes.Equal(got.Bytes(), raw) {
		t.Fatalf("BF16 writer changed source bytes: got %x want %x", got.Bytes(), raw)
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
	if got, want := r.Replace("model.layers.1.linear_attn.dt_bias"), "blk.1.ssm_dt.bias"; got != want {
		t.Fatalf("unexpected dt bias replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("model.layers.1.linear_attn.dt_proj.weight"), "blk.1.ssm_dt.weight"; got != want {
		t.Fatalf("unexpected dt projection replacement: got %q want %q", got, want)
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

func TestQwen35ReordersSsmDtBias(t *testing.T) {
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
			name:  "blk.0.ssm_dt.bias",
			shape: []uint64{4},
			data:  []float32{0, 1, 2, 3},
		},
	})
	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}

	if got, want := readTensorData(t, out[0]), []float32{0, 2, 1, 3}; !slices.Equal(got, want) {
		t.Fatalf("unexpected ssm_dt.bias data: got %v want %v", got, want)
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

func TestQwen35MTPMoePackedExperts(t *testing.T) {
	m := &qwen3NextModel{
		qwen3NextTextConfig: qwen3NextTextConfig{
			NumHiddenLayers:       40,
			NumNextNPredictLayers: 1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "mtp.layers.0.mlp.experts.gate_up_proj",
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
			name:  "mtp.layers.0.mlp.experts.down_proj",
			shape: []uint64{2, 5, 3},
			data:  make([]float32, 2*5*3),
		},
	})

	byName := map[string]*ggml.Tensor{}
	for _, tensor := range out {
		if strings.Contains(tensor.Name, ".mlp.experts.") {
			t.Fatalf("unexpected raw expert tensor %q", tensor.Name)
		}
		byName[tensor.Name] = tensor
	}

	gate := byName["blk.40.ffn_gate_exps.weight"]
	if gate == nil {
		t.Fatalf("missing tensor %q", "blk.40.ffn_gate_exps.weight")
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

	if _, ok := byName["blk.40.ffn_up_exps.weight"]; !ok {
		t.Fatalf("missing tensor %q", "blk.40.ffn_up_exps.weight")
	}
	if _, ok := byName["blk.40.ffn_down_exps.weight"]; !ok {
		t.Fatalf("missing tensor %q", "blk.40.ffn_down_exps.weight")
	}
}

func TestQwen35MTPMoePerExpertTensors(t *testing.T) {
	m := &qwen3NextModel{
		qwen3NextTextConfig: qwen3NextTextConfig{
			NumHiddenLayers:       40,
			NumNextNPredictLayers: 1,
		},
	}

	out := m.Tensors([]Tensor{
		&fakeTensor{
			name:  "mtp.layers.0.mlp.experts.1.gate_proj.weight",
			shape: []uint64{2, 2},
			data:  []float32{10, 11, 12, 13},
		},
		&fakeTensor{
			name:  "mtp.layers.0.mlp.experts.0.gate_proj.weight",
			shape: []uint64{2, 2},
			data:  []float32{0, 1, 2, 3},
		},
	})

	if len(out) != 1 {
		t.Fatalf("unexpected output tensor count: got %d want 1", len(out))
	}
	if got, want := out[0].Name, "blk.40.ffn_gate_exps.weight"; got != want {
		t.Fatalf("unexpected tensor name: got %q want %q", got, want)
	}
	if got, want := out[0].Shape, []uint64{2, 2, 2}; !slices.Equal(got, want) {
		t.Fatalf("unexpected tensor shape: got %v want %v", got, want)
	}
	if got, want := readTensorData(t, out[0]), []float32{0, 1, 2, 3, 10, 11, 12, 13}; !slices.Equal(got, want) {
		t.Fatalf("unexpected tensor values: got %v want %v", got, want)
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
