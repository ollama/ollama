package convert

import (
	"slices"
	"strings"
	"testing"
)

func TestLFM2VLTextModelKVUsesTextConfig(t *testing.T) {
	p := lfm2VLTextModel{
		TextConfig: lfm2Model{
			ModelParameters:       ModelParameters{ModelType: "lfm2", VocabSize: 65536},
			HiddenSize:            2048,
			NumHiddenLayers:       16,
			MaxPositionEmbeddings: 128000,
			IntermediateSize:      12288,
			BlockFFDim:            12288,
			BlockAutoAdjustFFDim:  true,
			BlockMultipleOf:       256,
			BlockFFNDimMultiplier: 1.0,
			NumAttentionHeads:     32,
			NumKeyValueHeads:      8,
			LayerTypes:            []string{"conv", "full_attention"},
			NormEps:               1e-5,
			ConvLCache:            3,
		},
		DownsampleFactor: 2,
		VisionConfig: struct {
			HiddenSize        uint32  `json:"hidden_size"`
			IntermediateSize  uint32  `json:"intermediate_size"`
			NumAttentionHeads uint32  `json:"num_attention_heads"`
			NumHiddenLayers   uint32  `json:"num_hidden_layers"`
			NumChannels       uint32  `json:"num_channels"`
			PatchSize         uint32  `json:"patch_size"`
			LayerNormEpsilon  float32 `json:"layer_norm_eps"`
		}{
			HiddenSize:        1152,
			IntermediateSize:  4304,
			NumAttentionHeads: 16,
			NumHiddenLayers:   27,
			NumChannels:       3,
			PatchSize:         16,
			LayerNormEpsilon:  1e-6,
		},
	}
	p.Processor.ImageProcessor.TileSize = 512
	p.Processor.ImageProcessor.ImageMean = []float32{0.5, 0.5, 0.5}
	p.Processor.ImageProcessor.ImageStd = []float32{0.5, 0.5, 0.5}

	kv := p.KV(&Tokenizer{
		Vocabulary: &Vocabulary{
			Model:  "gpt2",
			Tokens: []string{"<|pad|>", "<image>", "<|image_start|>", "<|image_end|>", "<|img_thumbnail|>"},
		},
	})

	if got, want := kv["general.architecture"], "lfm2"; got != want {
		t.Fatalf("general.architecture = %v, want %v", got, want)
	}

	if got, want := kv["feed_forward_length"], uint32(8192); got != want {
		t.Fatalf("feed_forward_length = %v, want %v", got, want)
	}

	if got, want := kv["vision.block_count"], uint32(27); got != want {
		t.Fatalf("vision.block_count = %v, want %v", got, want)
	}

	if got, want := kv["vision.image_size"], uint32(256); got != want {
		t.Fatalf("vision.image_size = %v, want %v", got, want)
	}

	if got, want := kv["vision.image_token_id"], uint32(396); got != want {
		t.Fatalf("vision.image_token_id = %v, want %v", got, want)
	}

	if got, want := kv["vision.image_start_token_id"], uint32(2); got != want {
		t.Fatalf("vision.image_start_token_id = %v, want %v", got, want)
	}

	if got, want := kv["vision.do_image_splitting"], true; got != want {
		t.Fatalf("vision.do_image_splitting = %v, want %v", got, want)
	}
	if got, want := kv["vision.min_tiles"], uint32(2); got != want {
		t.Fatalf("vision.min_tiles = %v, want %v", got, want)
	}
	if got, want := kv["vision.max_tiles"], uint32(10); got != want {
		t.Fatalf("vision.max_tiles = %v, want %v", got, want)
	}
	if got, want := kv["vision.tile_size"], uint32(512); got != want {
		t.Fatalf("vision.tile_size = %v, want %v", got, want)
	}
	if got, want := kv["vision.use_thumbnail"], true; got != want {
		t.Fatalf("vision.use_thumbnail = %v, want %v", got, want)
	}
	if got, want := kv["vision.use_image_special_tokens"], true; got != want {
		t.Fatalf("vision.use_image_special_tokens = %v, want %v", got, want)
	}
}

func TestLFM2VLTextModelTensorsIncludeVision(t *testing.T) {
	p := lfm2VLTextModel{}
	p.VisionConfig.PatchSize = 16
	p.VisionConfig.NumChannels = 3
	input := []Tensor{
		newLFM2StubTensor("model.embed_tokens.weight", []uint64{65536, 2048}),
		newLFM2StubTensor("model.layers.0.ffn_norm.weight", []uint64{2048}),
		newLFM2StubTensor("v.patch_embd.weight", []uint64{1152, 768}),
		newLFM2StubTensor("v.blk.0.attn_q.weight", []uint64{1152, 1152}),
		newLFM2StubTensor("mm.1.weight", []uint64{2048, 4608}),
	}

	out := p.Tensors(input)
	if len(out) == 0 {
		t.Fatal("expected non-empty tensor list")
	}

	foundPatch := false
	foundVision := false
	for _, tns := range out {
		if tns.Name == "v.patch_embd.weight" {
			foundPatch = true
			if !slices.Equal(tns.Shape, []uint64{1152, 3, 16, 16}) {
				t.Fatalf("v.patch_embd.weight shape = %v, want [1152 3 16 16]", tns.Shape)
			}
		}
		if strings.HasPrefix(tns.Name, "v.") || strings.HasPrefix(tns.Name, "mm.") {
			foundVision = true
		}
	}

	if !foundPatch {
		t.Fatal("expected v.patch_embd.weight in output tensors")
	}
	if !foundVision {
		t.Fatal("expected at least one vision/projector tensor in output")
	}
}

func TestLFM2VLTextModelReplacements(t *testing.T) {
	p := lfm2VLTextModel{}
	r := strings.NewReplacer(p.Replacements()...)

	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "language_model_embed_tokens",
			in:   "model.language_model.embed_tokens.weight",
			want: "token_embd.weight",
		},
		{
			name: "language_model_layers",
			in:   "model.language_model.layers.2.self_attn.q_proj.weight",
			want: "blk.2.attn_q.weight",
		},
		{
			name: "nested_language_model_prefix",
			in:   "model.language_model.model.embedding_norm.weight",
			want: "token_embd_norm.weight",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := r.Replace(tt.in); got != tt.want {
				t.Fatalf("replacement(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestLFM2VLProjectorKV(t *testing.T) {
	p := lfm2VLProjectorModel{
		DownsampleFactor:   2,
		ProjectorHiddenDim: 2048,
	}
	p.VisionModel.NumHiddenLayers = 27
	p.VisionModel.HiddenSize = 1152
	p.VisionModel.IntermediateSize = 4304
	p.VisionModel.NumAttentionHeads = 16
	p.VisionModel.PatchSize = 16
	p.VisionModel.LayerNormEpsilon = 1e-6
	p.Processor.ImageProcessor.TileSize = 512
	p.Processor.ImageProcessor.ImageMean = []float32{0.5, 0.5, 0.5}
	p.Processor.ImageProcessor.ImageStd = []float32{0.5, 0.5, 0.5}

	kv := p.KV(nil)

	if got, want := kv["general.architecture"], "clip"; got != want {
		t.Fatalf("general.architecture = %v, want %v", got, want)
	}
	if got, want := kv["clip.projector_type"], "lfm2"; got != want {
		t.Fatalf("clip.projector_type = %v, want %v", got, want)
	}
	if got, want := kv["clip.vision.image_size"], uint32(256); got != want {
		t.Fatalf("clip.vision.image_size = %v, want %v", got, want)
	}
}

func TestLFM2VLProjectorTensorsPatchReshape(t *testing.T) {
	p := lfm2VLProjectorModel{}
	p.VisionModel.NumChannels = 3
	p.VisionModel.PatchSize = 16

	input := []Tensor{
		newLFM2StubTensor("v.patch_embd.weight", []uint64{1152, 768}),
		newLFM2StubTensor("mm.1.weight", []uint64{2048, 4608}),
		newLFM2StubTensor("model.embed_tokens.weight", []uint64{65536, 2048}),
	}

	out := p.Tensors(input)
	if len(out) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(out))
	}

	var patchShape []uint64
	for _, tns := range out {
		if tns.Name == "v.patch_embd.weight" {
			patchShape = tns.Shape
			break
		}
	}

	if !slices.Equal(patchShape, []uint64{1152, 3, 16, 16}) {
		t.Fatalf("v.patch_embd.weight shape = %v, want [1152 3 16 16]", patchShape)
	}
}

func TestRepackPatchEmbeddingWeight(t *testing.T) {
	data := []float32{
		0, 1, // y=0,x=0
		2, 3, // y=0,x=1
		4, 5, // y=1,x=0
		6, 7, // y=1,x=1
	}

	got, err := repackPatchEmbeddingWeight(data, []uint64{1, 8}, 2, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []float32{0, 2, 4, 6, 1, 3, 5, 7}
	if !slices.Equal(got, want) {
		t.Fatalf("repacked data = %v, want %v", got, want)
	}
}
