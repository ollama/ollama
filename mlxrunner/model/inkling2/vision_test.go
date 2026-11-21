package inkling2

import (
	"bytes"
	"image"
	"image/png"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
)

func TestPrepareMediaChecksContextBeforeDecodingPixels(t *testing.T) {
	var img bytes.Buffer
	if err := png.Encode(&img, image.NewRGBA(image.Rect(0, 0, 21, 11))); err != nil {
		t.Fatal(err)
	}
	m := targetModel{
		config: textConfig{Context: 5},
		Vision: &visionEncoder{config: visionConfig{Patch: 10, Merge: 2}, imageToken: 9},
	}
	segments := []model.Segment{
		{Tokens: []int32{1}},
		{Kind: "image", Data: img.Bytes()},
		{Tokens: []int32{2}},
	}
	prepared, err := m.PrepareMedia(segments)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prepared.Tokens, []int32{1, 9, 9, 2}) || len(prepared.Items) != 1 ||
		!slices.Equal(prepared.Items[0].Dims, []int{20, 40, 3}) {
		t.Fatalf("unexpected padded image expansion: tokens %v, items %d", prepared.Tokens, len(prepared.Items))
	}
	// A PNG header is sufficient to reject the image when there is no room
	// for its two tiles. The missing pixel data would otherwise fail decoding.
	m.config.Context--
	segments[1].Data = img.Bytes()[:33]
	if _, err := m.PrepareMedia(segments); err == nil || !strings.Contains(err.Error(), "exceeds the remaining context") {
		t.Fatalf("expected context error before pixel decoding, got %v", err)
	}
}
