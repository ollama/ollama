package convert

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestQwen3VLTextAndProjectorKV(t *testing.T) {
	m := &qwen3VLModel{
		qwen3Model: qwen3Model{
			HiddenSize: 2048,
		},
	}
	m.RopeScaling.Type = "mrope"
	m.RopeScaling.MropeSection = []int32{24, 20, 20}
	m.VisionModel.Depth = 24
	m.VisionModel.HiddenSize = 1024
	m.VisionModel.IntermediateSize = 4096
	m.VisionModel.OutHiddenSize = 2048
	m.VisionModel.NumHeads = 16
	m.VisionModel.InChannels = 3
	m.VisionModel.PatchSize = 16
	m.VisionModel.SpatialMergeSize = 2
	m.VisionModel.NumPositionEmbeddings = 2304
	m.VisionModel.TemporalPatchSize = 2
	m.VisionModel.RMSNormEps = 1e-6
	m.VisionModel.RopeTheta = 10000
	m.VisionModel.DeepstackVisualIndexes = []int32{5, 11, 17}
	m.VisionModel.ImageMean = []float32{0.5, 0.5, 0.5}
	m.VisionModel.ImageStd = []float32{0.5, 0.5, 0.5}

	textKV := m.TextKV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := textKV["general.architecture"], "qwen3vl"; got != want {
		t.Fatalf("unexpected text architecture: got %v want %v", got, want)
	}
	if got, want := textKV["rope.dimension_sections"], []int32{24, 20, 20, 0}; !slices.Equal(got.([]int32), want) {
		t.Fatalf("unexpected rope.dimension_sections: got %v want %v", got, want)
	}
	if got, want := textKV["n_deepstack_layers"], uint32(3); got != want {
		t.Fatalf("unexpected n_deepstack_layers: got %v want %v", got, want)
	}
	for _, key := range []string{"vision.block_count", "vision.deepstack_visual_indexes", "rope.mrope_section"} {
		if _, ok := textKV[key]; ok {
			t.Fatalf("TextKV retained %q", key)
		}
	}

	projectorKV := m.ProjectorKV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := projectorKV["general.architecture"], "clip"; got != want {
		t.Fatalf("unexpected projector architecture: got %v want %v", got, want)
	}
	if got, want := projectorKV["general.type"], "mmproj"; got != want {
		t.Fatalf("unexpected projector type: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.projector_type"], "qwen3vl_merger"; got != want {
		t.Fatalf("unexpected projector type: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.vision.feed_forward_length"], uint32(4096); got != want {
		t.Fatalf("unexpected feed_forward_length: got %v want %v", got, want)
	}
	if got, want := projectorKV["clip.vision.image_size"], uint32(768); got != want {
		t.Fatalf("unexpected image_size: got %v want %v", got, want)
	}
	mask, ok := projectorKV["clip.vision.is_deepstack_layers"].([]bool)
	if !ok {
		t.Fatalf("deepstack mask has unexpected type: %T", projectorKV["clip.vision.is_deepstack_layers"])
	}
	if len(mask) != 24 || !mask[5] || !mask[11] || !mask[17] {
		t.Fatalf("unexpected deepstack mask: %v", mask)
	}
}

func TestQwen3VLProjectorTensors(t *testing.T) {
	m := &qwen3VLModel{}
	m.VisionModel.DeepstackVisualIndexes = []int32{5, 11, 17}

	tensors := m.ProjectorTensors([]Tensor{
		&fakeTensor{
			name:  "v.patch_embd.weight",
			shape: []uint64{2, 2, 2, 1, 2},
			data:  []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		&fakeTensor{name: "v.position_embd.weight", shape: []uint64{4, 2}, data: []float32{0, 1, 2, 3, 4, 5, 6, 7}},
		&fakeTensor{name: "v.merger.linear_fc1.weight", shape: []uint64{4, 2}, data: make([]float32, 8)},
		&fakeTensor{name: "v.merger.linear_fc2.bias", shape: []uint64{4}, data: make([]float32, 4)},
		&fakeTensor{name: "v.merger.norm.weight", shape: []uint64{2}, data: make([]float32, 2)},
		&fakeTensor{name: "v.deepstack.0.linear_fc1.weight", shape: []uint64{4, 2}, data: make([]float32, 8)},
		&fakeTensor{name: "v.deepstack.1.norm.bias", shape: []uint64{2}, data: make([]float32, 2)},
		&fakeTensor{name: "v.blk.0.attn_qkv.weight", shape: []uint64{6, 2}, data: make([]float32, 12), sourceDType: "BF16", kind: tensorKindFP16},
		&fakeTensor{name: "token_embd.weight", shape: []uint64{2, 2}, data: make([]float32, 4)},
	})

	byName := map[string]uint32{}
	for _, tensor := range tensors {
		byName[tensor.Name] = tensor.Kind
	}

	if _, ok := byName["token_embd.weight"]; ok {
		t.Fatalf("projector tensors included text tensor")
	}
	if got := byName["v.position_embd.weight"]; got != tensorKindFP32 {
		t.Fatalf("position embedding was not promoted to F32: %d", got)
	}
	if got := byName["v.blk.0.attn_qkv.weight"]; got != tensorKindBF16 {
		t.Fatalf("BF16 projector tensor was not preserved: %d", got)
	}
	for _, name := range []string{
		"mm.0.weight",
		"mm.2.bias",
		"v.post_ln.weight",
		"v.deepstack.5.fc1.weight",
		"v.deepstack.11.norm.bias",
	} {
		if _, ok := byName[name]; !ok {
			t.Fatalf("missing projector tensor %q", name)
		}
	}

	firstTensor := tensorsByName(tensors)["v.patch_embd.weight"]
	if firstTensor == nil {
		t.Fatalf("first patch embedding slice missing")
	}
	if got, want := firstTensor.Shape, []uint64{2, 2, 1, 2}; !slices.Equal(got, want) {
		t.Fatalf("unexpected first patch shape: got %v want %v", got, want)
	}
	if got, want := readTensorData(t, firstTensor), []float32{0, 1, 4, 5, 8, 9, 12, 13}; !slices.Equal(got, want) {
		t.Fatalf("unexpected first patch data: got %v want %v", got, want)
	}

	secondTensor := tensorsByName(tensors)["v.patch_embd.weight.1"]
	if secondTensor == nil {
		t.Fatalf("second patch embedding slice missing")
	}
	if got, want := readTensorData(t, secondTensor), []float32{2, 3, 6, 7, 10, 11, 14, 15}; !slices.Equal(got, want) {
		t.Fatalf("unexpected second patch data: got %v want %v", got, want)
	}
}

func tensorsByName(tensors []*ggml.Tensor) map[string]*ggml.Tensor {
	byName := map[string]*ggml.Tensor{}
	for _, tensor := range tensors {
		byName[tensor.Name] = tensor
	}
	return byName
}
