package gemma4

import (
	"bytes"
	"image"
	"image/png"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestParseMultimodalConfig(t *testing.T) {
	visionJSON := []byte(`{
		"image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882,
		"vision_soft_tokens_per_image": 280,
		"vision_config": {"model_type": "gemma4_vision", "hidden_size": 1152,
			"num_hidden_layers": 27, "num_attention_heads": 16, "head_dim": 72,
			"patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280,
			"standardize": true, "rope_parameters": {"rope_theta": 100.0}},
		"text_config": {"hidden_size": 2816, "use_bidirectional_attention": "vision"}}`)

	mm, err := parseMultimodalConfig(visionJSON)
	if err != nil {
		t.Fatal(err)
	}
	if mm.VisionConfig == nil {
		t.Fatal("vision config not retained")
	}
	if mm.ImageTokenID != 258880 || mm.BOITokenID != 255999 || mm.EOITokenID != 258882 {
		t.Fatalf("token ids = %d/%d/%d", mm.ImageTokenID, mm.BOITokenID, mm.EOITokenID)
	}
	if mm.VisionConfig.RopeTheta != 100 || !mm.VisionConfig.Standardize {
		t.Fatalf("vision config = %+v", mm.VisionConfig)
	}

	cfg, err := parseTextConfig(visionJSON)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UseBidirectionalAttention != "vision" || !cfg.BidirectionalVisionAttention {
		t.Fatalf("use_bidirectional_attention = %q", cfg.UseBidirectionalAttention)
	}

	// The 12B unified family uses the encoder-free embedder.
	unified, err := parseMultimodalConfig([]byte(`{
		"image_token_id": 258880,
		"vision_config": {"model_type": "gemma4_unified_vision", "mm_embed_dim": 3840,
			"model_patch_size": 48, "num_soft_tokens": 280, "patch_size": 16,
			"pooling_kernel_size": 3, "rms_norm_eps": 1e-6}}`))
	if err != nil {
		t.Fatal(err)
	}
	uv := unified.VisionConfig
	if uv == nil || !uv.unified() {
		t.Fatal("unified vision config not retained")
	}
	if uv.NumSoftTokens != 280 {
		t.Fatalf("unified config = %+v", uv)
	}

	// An unknown vision architecture loads text-only.
	unknown, err := parseMultimodalConfig([]byte(`{
		"vision_config": {"model_type": "gemma5_vision"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if unknown.VisionConfig != nil {
		t.Fatal("unknown vision config should be ignored")
	}

	textOnly, err := parseMultimodalConfig([]byte(`{"text_config": {"hidden_size": 640}}`))
	if err != nil {
		t.Fatal(err)
	}
	if textOnly.VisionConfig != nil {
		t.Fatal("text-only checkpoint grew a vision config")
	}

	if err := validateVisionSoftTokenBudget(280); err != nil {
		t.Fatal(err)
	}
	if err := validateVisionSoftTokenBudget(300); err == nil {
		t.Fatal("budget 300 accepted")
	}
}

func visionTestModel() *Model {
	return &Model{
		TextConfig: &TextConfig{BidirectionalVisionAttention: true},
		MM:         multimodalConfig{ImageTokenID: 258880, BOITokenID: 255999, EOITokenID: 258882},
	}
}

func mediaItemAt(pos int, softTokens int32) batch.MediaItem {
	return batch.MediaItem{
		Pos:    pos,
		Opaque: preparedImage{geom: ImageGeometry{NumSoftTokens: softTokens}},
	}
}

func TestBuildMasks(t *testing.T) {
	m := visionTestModel()

	// Soft run [3, 7): boi at 2, eoi at 7.
	item := mediaItemAt(2, 4)
	chunk := func(off, qLen int32) *batch.Batch {
		return &batch.Batch{
			SeqOffsets:   []int32{off},
			SeqQueryLens: []int32{qLen},
			Media:        []batch.MediaItem{item},
		}
	}

	sliding, full := m.buildMasks(&batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{8}})
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("text-only batch left the causal fast path")
	}

	sliding, full = m.buildMasks(chunk(0, 8))
	if sliding.IsCausal() {
		t.Fatal("intersecting run did not relax the sliding mask")
	}
	if !full.IsCausal() {
		t.Fatal("full-attention mask relaxed under \"vision\" semantics")
	}

	// A decode step past the run keeps both masks on the fast path.
	sliding, full = m.buildMasks(chunk(20, 1))
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("non-intersecting item left the causal fast path")
	}

	m.BidirectionalVisionAttention = false
	sliding, full = m.buildMasks(chunk(0, 8))
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("causal-only config relaxed a mask")
	}
}

func TestScatterMedia(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := visionTestModel()
		m.HiddenSize = 4

		features := mlx.FromValues([]float32{
			1, 1, 1, 1,
			2, 2, 2, 2,
			3, 3, 3, 3,
			4, 4, 4, 4,
		}, 4, 4)
		item := mediaItemAt(1, 4) // soft run [2, 6)
		item.Features = features

		// Split evaluation: first chunk covers [0, 4) — soft rows 2, 3 — and the
		// resumed chunk [4, 8) covers rows 4, 5 with feature rows sliced by
		// overlap.
		rowVal := func(h *mlx.Array, row int) float32 {
			vals := h.Slice(mlx.Slice(0, 1), mlx.Slice(row, row+1), mlx.Slice()).AsType(mlx.DTypeFloat32)
			mlx.Eval(vals)
			return vals.Floats()[0]
		}

		first := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 4, 4), &batch.Batch{
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{4},
			Media:        []batch.MediaItem{item},
		})
		for row, want := range []float32{0, 0, 1, 2} {
			if got := rowVal(first, row); got != want {
				t.Fatalf("first chunk row %d = %v, want %v", row, got, want)
			}
		}

		resumed := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 4, 4), &batch.Batch{
			SeqOffsets:   []int32{4},
			SeqQueryLens: []int32{4},
			Media:        []batch.MediaItem{item},
		})
		for row, want := range []float32{3, 4, 0, 0} {
			if got := rowVal(resumed, row); got != want {
				t.Fatalf("resumed chunk row %d = %v, want %v", row, got, want)
			}
		}

		// Featureless items (decode) scatter nothing.
		item.Features = nil
		decode := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 4), &batch.Batch{
			SeqOffsets:   []int32{7},
			SeqQueryLens: []int32{1},
			Media:        []batch.MediaItem{item},
		})
		if got := rowVal(decode, 0); got != 0 {
			t.Fatalf("featureless scatter wrote %v", got)
		}
	})
}

func TestPrepareMediaExpansion(t *testing.T) {
	m := visionTestModel()
	m.Vision = &VisionConfig{
		ModelType:         "gemma4_vision",
		HiddenSize:        12,
		PatchSize:         16,
		PoolingKernelSize: 3,
		DefaultOutputLen:  70,
		RMSNormEps:        1e-6,
	}
	m.VisionTower = &VisionTower{}

	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 96, 96))); err != nil {
		t.Fatal(err)
	}

	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{9}},
		{Kind: "image", Data: buf.Bytes()},
	})
	if err != nil {
		t.Fatal(err)
	}

	// 96x96 at budget 70 resizes to 384x384: 24x24 patches, 64 soft tokens,
	// spliced after the one-token text run.
	wantSoft := 64
	if len(prepared.Tokens) != 1+wantSoft+2 || prepared.Tokens[0] != 9 {
		t.Fatalf("stream length = %d, want %d", len(prepared.Tokens), 1+wantSoft+2)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{1, 1 + wantSoft + 2} || item.Source != 1 {
		t.Fatalf("item range = %v source = %d", item.Range, item.Source)
	}
	exp := prepared.Tokens[item.Range[0]:item.Range[1]]
	if exp[0] != m.MM.BOITokenID || exp[len(exp)-1] != m.MM.EOITokenID {
		t.Fatal("expansion not wrapped in boi/eoi")
	}
	for _, tok := range exp[1 : len(exp)-1] {
		if tok != m.MM.ImageTokenID {
			t.Fatalf("soft token = %d", tok)
		}
	}
	if item.Dims[0] != 24*24 || item.Dims[1] != 16*16*3 {
		t.Fatalf("dims = %v", item.Dims)
	}
	if len(item.MediaData) != item.Dims[0]*item.Dims[1] {
		t.Fatalf("media data length = %d", len(item.MediaData))
	}
	p := item.Opaque.(preparedImage)
	if p.geom.NumSoftTokens != int32(wantSoft) || len(p.positions) != 2*24*24 {
		t.Fatalf("geometry = %+v, positions = %d", p.geom, len(p.positions))
	}

	if _, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte{1}}}); err == nil {
		t.Fatal("audio accepted")
	}
	if _, err := m.PrepareMedia([]base.Segment{{Kind: "image"}}); err == nil {
		t.Fatal("image with no data accepted")
	}
	m.VisionTower = nil
	if _, err := m.PrepareMedia([]base.Segment{{Kind: "image", Data: buf.Bytes()}}); err == nil {
		t.Fatal("towerless model accepted an image")
	}
}
