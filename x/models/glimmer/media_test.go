package glimmer

import (
	"bytes"
	"image"
	"image/png"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func testVisionModel() *Model {
	return &Model{
		VisionEncoder: &VisionEncoder{},
		Config: &Config{
			HasVision:              true,
			PatchTokenID:           7,
			ImageStartTokenID:      8,
			ImageEndTokenID:        9,
			VisionPatchSize:        14,
			VisionPatchTemporal:    1,
			VisionDownsampleFactor: 2,
		},
	}
}

func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewNRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestPrepareMediaSplicesExpansion(t *testing.T) {
	m := testVisionModel()
	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{1, 2}},
		{Kind: "image", Data: testPNG(t, 56, 56)},
		{Tokens: []int32{3}},
	})
	if err != nil {
		t.Fatal(err)
	}

	// 56x56 at patch stride 28 (14*2) is a 2x2 downsampled grid: 4 output
	// tokens over a 4x4 patch grid.
	wantTokens := []int32{1, 2, 8, 7, 7, 7, 7, 9, 3}
	if len(prepared.Tokens) != len(wantTokens) {
		t.Fatalf("tokens = %v, want %v", prepared.Tokens, wantTokens)
	}
	for i, tok := range wantTokens {
		if prepared.Tokens[i] != tok {
			t.Fatalf("tokens = %v, want %v", prepared.Tokens, wantTokens)
		}
	}

	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{2, 8} {
		t.Fatalf("range = %v, want [2 8]", item.Range)
	}
	if item.Source != 1 {
		t.Fatalf("source = %d, want 1", item.Source)
	}
	if !item.Causal {
		t.Fatal("expected causal expansion")
	}

	geom := item.Opaque.(preparedImage)
	if geom.gridH != 4 || geom.gridW != 4 || geom.outputTokens != 4 {
		t.Fatalf("geometry = %+v, want 4x4 grid with 4 output tokens", geom)
	}
	want := 1
	for _, d := range item.Dims {
		want *= d
	}
	if len(item.MediaData) != want {
		t.Fatalf("media data length %d does not match dims %v", len(item.MediaData), item.Dims)
	}
}

func TestPrepareMediaRejectsUnsupportedKind(t *testing.T) {
	m := testVisionModel()
	_, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte{1}}})
	if err == nil {
		t.Fatal("expected error for audio input")
	}

	text := &Model{Config: &Config{}}
	_, err = text.PrepareMedia([]base.Segment{{Kind: "image", Data: []byte{1}}})
	if err == nil {
		t.Fatal("expected error for text-only model")
	}
}
