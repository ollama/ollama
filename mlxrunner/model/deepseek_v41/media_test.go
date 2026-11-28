package deepseek_v41

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
)

func TestImageGeometryBudget(t *testing.T) {
	v := visionConfig{PatchSize: 14, DownsampleRatio: 3, MinPixels: 295936, MaxImageTokens: 1024}
	for _, shape := range [][2]int{{800, 600}, {4000, 500}, {500, 4000}, {160, 160}, {10000, 10000}} {
		t.Run(fmt.Sprintf("%dx%d", shape[0], shape[1]), func(t *testing.T) {
			w, h, g, err := imageSize(shape[0], shape[1], v)
			if err != nil {
				t.Fatal(err)
			}
			if w%v.PatchSize != 0 || h%v.PatchSize != 0 || g.tokenCount() > v.MaxImageTokens ||
				g.textRows != (g.patchRows+v.DownsampleRatio-1)/v.DownsampleRatio ||
				g.textCols != (g.patchCols+v.DownsampleRatio-1)/v.DownsampleRatio {
				t.Fatalf("image geometry = %dx%d, grid %+v; want patch alignment and <= %d tokens", w, h, g, v.MaxImageTokens)
			}
		})
	}
}

func TestPrepareImagePatchesAndExpansion(t *testing.T) {
	input := image.NewRGBA(image.Rect(0, 0, 14, 14))
	for y := range 14 {
		for x := range 14 {
			input.Set(x, y, color.RGBA{R: 255, G: 0, B: 127, A: 255})
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	m := &Model{Config: &Config{ImageTokenID: 129264, Vision: visionConfig{PatchSize: 14, DownsampleRatio: 3, MinPixels: 1, MaxImageTokens: 1024}}, VisionTower: &visionTower{}}
	prepared, err := m.PrepareMedia([]model.Segment{{Tokens: []int32{1}}, {Kind: "image", Data: encoded.Bytes()}, {Tokens: []int32{2}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 1 || prepared.Items[0].Range != [2]int{1, 5} || len(prepared.Tokens) != 6 || prepared.Tokens[5] != 2 {
		t.Fatalf("bad image expansion: %+v", prepared)
	}
	for _, id := range prepared.Tokens[1:5] {
		if id != 129264 {
			t.Fatalf("image token = %d", id)
		}
	}
	patches := prepared.Items[0].MediaData
	if len(patches) != 3*14*14 || patches[0] != 1 || patches[14*14] != -1 ||
		math.IsNaN(float64(patches[2*14*14])) || math.IsInf(float64(patches[2*14*14]), 0) ||
		math.Abs(float64(patches[2*14*14]+0.0039215686)) > 1e-6 {
		t.Fatalf("bad RGB patch data: %v %v %v", patches[0], patches[14*14], patches[2*14*14])
	}
}

func TestRejectsLargeSourceImageBeforeDecode(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewGray(image.Rect(0, 0, 7100, 7100))); err != nil {
		t.Fatal(err)
	}
	m := &Model{Config: &Config{Vision: visionConfig{PatchSize: 14, DownsampleRatio: 3, MinPixels: 1, MaxImageTokens: 1024}}}
	_, _, err := m.preprocessImage(encoded.Bytes())
	if err == nil || !strings.Contains(err.Error(), "source pixels") {
		t.Fatalf("preprocessImage() error = %v, want source pixel limit", err)
	}
}
