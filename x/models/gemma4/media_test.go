package gemma4

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func testPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	img.Set(1, 0, color.RGBA{G: 255, A: 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestParseImageTokenLimit(t *testing.T) {
	for _, tt := range []struct {
		name     string
		data     string
		fallback int
		want     int
	}{
		{name: "nested max", data: `{"image_processor":{"max_soft_tokens":560}}`, fallback: 280, want: 560},
		{name: "nested sequence", data: `{"image_processor":{"image_seq_length":140}}`, fallback: 280, want: 140},
		{name: "top level", data: `{"image_seq_length":70}`, fallback: 280, want: 70},
		{name: "invalid", data: `{`, fallback: 280, want: 280},
		{name: "default fallback", data: `{}`, fallback: 0, want: gemma4DefaultImageTokens},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseImageTokenLimit([]byte(tt.data), tt.fallback); got != tt.want {
				t.Fatalf("limit = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestPrepareMediaGemma4VisionSchemes(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  VisionConfig
	}{
		{
			name: "12b vision_embedder",
			cfg: VisionConfig{
				ModelType:         "gemma4_unified_vision",
				PatchSize:         48,
				PoolingKernelSize: 1,
				ImageSeqLength:    1,
			},
		},
		{
			name: "26b 31b vision_tower",
			cfg: VisionConfig{
				ModelType:         "gemma4_vision",
				PatchSize:         16,
				PoolingKernelSize: 3,
				ImageSeqLength:    1,
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{
				VisionConfig:      &tt.cfg,
				VisionEncoder:     &VisionEncoder{Cfg: &tt.cfg},
				MultimodalEmbed:   &MultimodalEmbedder{},
				imageStartTokenID: 10,
				imageTokenID:      11,
				imageEndTokenID:   12,
				imageTokenLimit:   1,
			}
			prepared, err := m.PrepareMedia([]base.Segment{
				{Tokens: []int32{1, 2}},
				{Kind: "image", Data: testPNG(t)},
				{Tokens: []int32{3}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if want := []int32{1, 2, 10, 11, 12, 3}; !slices.Equal(prepared.Tokens, want) {
				t.Fatalf("tokens = %v, want %v", prepared.Tokens, want)
			}
			if len(prepared.Items) != 1 {
				t.Fatalf("items = %d, want 1", len(prepared.Items))
			}
			item := prepared.Items[0]
			if item.Range != [2]int{2, 5} || item.Source != 1 {
				t.Fatalf("item range/source = %v/%d", item.Range, item.Source)
			}
			if want := []int{1, 3, 48, 48}; !slices.Equal(item.Dims, want) {
				t.Fatalf("dims = %v, want %v", item.Dims, want)
			}
			if geom := item.Opaque.(preparedImage); geom.numTokens != 1 {
				t.Fatalf("tokens = %d, want 1", geom.numTokens)
			}
			if item.Causal {
				t.Fatal("Gemma4 image expansion must stay atomic")
			}
		})
	}
}

func TestPrepareMediaRejectsUnsupportedInput(t *testing.T) {
	cfg := &VisionConfig{PatchSize: 48, PoolingKernelSize: 1, ImageSeqLength: 1}
	m := &Model{
		VisionConfig:    cfg,
		VisionEncoder:   &VisionEncoder{Cfg: cfg},
		MultimodalEmbed: &MultimodalEmbedder{},
	}
	if _, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte{1}}}); err == nil {
		t.Fatal("audio input did not fail")
	}
	text := &Model{}
	if _, err := text.PrepareMedia([]base.Segment{{Kind: "image", Data: testPNG(t)}}); err == nil {
		t.Fatal("text-only model accepted an image")
	}
}

func TestMediaAttentionMaskUsesSoftTokenRange(t *testing.T) {
	b := &batch.Batch{Media: []batch.MediaItem{{
		Seq:    0,
		Pos:    3,
		Opaque: preparedImage{numTokens: 2},
	}}}
	if mediaAttentionMask(b).IsCausal() {
		t.Fatal("image soft-token span did not relax the causal mask")
	}
}
