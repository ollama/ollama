package hy4_flash

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
)

func TestImageExpansionIncludesNewlines(t *testing.T) {
	m := &Model{
		cfg:    config{Vision: &visionConfig{Patch: 2, Merge: 2}, ImageStartTokenID: 11, ImageTokenID: 12, ImageEndTokenID: 13},
		Vision: &visionTower{},
	}
	img := &m.processor.Image
	img.Scale = 1.0 / 255
	img.Mean = []float32{0.5, 0.5, 0.5}
	img.Std = []float32{0.5, 0.5, 0.5}
	img.Size.Min, img.Size.Max = 16, 64
	testImage := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := range 8 {
		for x := range 8 {
			testImage.SetNRGBA(x, y, color.NRGBA{R: 255, A: 128})
		}
	}
	var raw bytes.Buffer
	if err := png.Encode(&raw, testImage); err != nil {
		t.Fatal(err)
	}
	got, err := m.PrepareMedia([]model.Segment{
		{Tokens: []int32{1}},
		{Kind: "image", Data: raw.Bytes()},
		{Tokens: []int32{2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{1, 11, 12, 12, 12, 12, 12, 12, 13, 2}
	if !slices.Equal(got.Tokens, want) {
		t.Fatalf("tokens %v, want %v", got.Tokens, want)
	}
	if len(got.Items) != 1 {
		t.Fatalf("expected an image, have %d items", len(got.Items))
	}
	item := got.Items[0]
	if item.Range != [2]int{1, 9} || !slices.Equal(item.Dims, []int{1, 8, 8, 3}) {
		t.Fatalf("incorrect image item range %v or dimensions %v", item.Range, item.Dims)
	}
	if pixel := item.MediaData[:3]; math.Abs(float64(pixel[0]-1)) > 1e-6 || pixel[1] != -1 || pixel[2] != -1 {
		t.Fatalf("pixel %v, want straight opaque RGB after normalization", pixel)
	}
	layout := got.Layout.(*positionLayout)
	wantPositions := [][3]int32{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {2, 2, 3}, {2, 2, 4}, {2, 3, 2}, {2, 3, 3}, {2, 3, 4}, {5, 5, 5}, {6, 6, 6}}
	if !slices.Equal(layout.positions, wantPositions) || layout.delta != -3 {
		t.Fatalf("positions %v delta %d", layout.positions, layout.delta)
	}
}
