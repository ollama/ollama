package qwen3_5

import (
	"encoding/json"
	"image"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/videoclip"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// TestVideoGoldens checks the prompt expansion and rope positions against
// the reference processor and model for each fixture. OLLAMA_VIDEO_FIXTURES
// names a directory with fixtures/, the goldens and the publisher checkouts
// (for the tokenizer and config) below.
func TestVideoGoldens(t *testing.T) {
	dir := os.Getenv("OLLAMA_VIDEO_FIXTURES")
	if dir == "" {
		t.Skip("OLLAMA_VIDEO_FIXTURES not set")
	}
	for _, tc := range []struct{ ref, goldens string }{
		{"Qwen3.5-2B", "goldens-qwen"},
		{"Qwen3.8-27B", "goldens-qwen38"},
	} {
		t.Run(tc.ref, func(t *testing.T) { videoGoldens(t, dir, tc.ref, tc.goldens) })
	}
}

func videoGoldens(t *testing.T, dir, refName, goldenDir string) {
	ref := filepath.Join(dir, "ref", refName)
	read := func(name string) []byte {
		data, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(read("tokenizer.json"), &tokenizer.TokenizerConfig{
		ConfigJSON:          read("config.json"),
		TokenizerConfigJSON: read("tokenizer_config.json"),
	})
	if err != nil {
		t.Fatal(err)
	}
	mm, err := parseMultimodalConfig(read("config.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{MM: mm, Vision: mm.VisionConfig, VisionTower: &VisionTower{}, tok: tok}

	goldens, err := filepath.Glob(filepath.Join(dir, goldenDir, "*.qwen.json"))
	if err != nil || len(goldens) == 0 {
		t.Fatalf("no goldens: %v", err)
	}
	for _, path := range goldens {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var g struct {
				Fixture   string    `json:"fixture"`
				InputIDs  []int32   `json:"input_ids"`
				Positions [][]int32 `json:"position_ids"`
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			clip, err := os.ReadFile(filepath.Join(dir, "fixtures", g.Fixture))
			if err != nil {
				t.Fatal(err)
			}
			// The chat template puts the video between these text runs.
			start := slices.Index(g.InputIDs, mm.VisionStartTokenID)
			end := start
			for i, id := range g.InputIDs {
				if id == mm.VisionEndTokenID {
					end = i + 1
				}
			}
			prepared, err := m.PrepareMedia([]model.Segment{
				{Tokens: g.InputIDs[:start]},
				{Kind: "video", Data: clip},
				{Tokens: g.InputIDs[end:]},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.Release()
			if !slices.Equal(prepared.Tokens, g.InputIDs) {
				t.Fatalf("expansion differs from the reference: got %d tokens, want %d", len(prepared.Tokens), len(g.InputIDs))
			}
			layout := prepared.Layout.(*visionLayout)
			L := int(layout.promptLen)
			for c := range 3 {
				got := layout.positions[c*L : (c+1)*L]
				if i := firstDiff(got, g.Positions[c]); i >= 0 {
					t.Fatalf("rope channel %d differs at token %d: got %d, want %d", c, i, got[i], g.Positions[c][i])
				}
			}
			for _, item := range prepared.Items {
				want := int(item.Opaque.(preparedImage).numTokens(m.Vision.SpatialMergeSize))
				if n := item.Range[1] - item.Range[0] - 2; n != want {
					t.Fatalf("item %v covers %d pads, want %d", item.Range, n, want)
				}
			}
		})
	}
}

// TestPackPatchesMatchesImageLayout pins the shared video packing to this
// tower's image layout: a still frame in both temporal slots packs the same.
func TestPackPatchesMatchesImageLayout(t *testing.T) {
	const w, h = 64, 96
	still := image.NewNRGBA(image.Rect(0, 0, w, h))
	r := rand.New(rand.NewPCG(1, 2))
	for i := range still.Pix {
		still.Pix[i] = uint8(r.IntN(256))
	}
	for i := 3; i < len(still.Pix); i += 4 {
		still.Pix[i] = 255 // opaque, so NRGBA and RGBA bytes agree
	}
	rgba := &image.RGBA{Pix: still.Pix, Stride: still.Stride, Rect: still.Rect}
	want := patchifyImage(rgba, h/16, w/16, 16, 2, 2)
	if got := videoclip.PackPatches([]*image.NRGBA{still, still}, 16, 2); !slices.Equal(got, want) {
		t.Fatal("videoclip.PackPatches differs from patchifyImage for a still frame")
	}
}

// TestPrepareVideoReleasesOnError checks that a request failing after its
// video was prepared still removes the video's temporary copy.
func TestPrepareVideoReleasesOnError(t *testing.T) {
	dir := os.Getenv("OLLAMA_VIDEO_FIXTURES")
	if dir == "" {
		t.Skip("OLLAMA_VIDEO_FIXTURES not set")
	}
	clip, err := os.ReadFile(filepath.Join(dir, "fixtures", "silent7s.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	ref := filepath.Join(dir, "ref", "Qwen3.5-2B")
	read := func(name string) []byte {
		data, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(read("tokenizer.json"), &tokenizer.TokenizerConfig{
		ConfigJSON:          read("config.json"),
		TokenizerConfigJSON: read("tokenizer_config.json"),
	})
	if err != nil {
		t.Fatal(err)
	}
	mm, err := parseMultimodalConfig(read("config.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{MM: mm, Vision: mm.VisionConfig, VisionTower: &VisionTower{}, tok: tok}
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)

	_, err = m.PrepareMedia([]model.Segment{{Kind: "video", Data: clip}, {Kind: "video", Data: clip}, {Kind: "image", Data: []byte("not an image")}})
	if err == nil {
		t.Fatal("PrepareMedia accepted an undecodable image")
	}
	if left, _ := filepath.Glob(filepath.Join(tmp, "ollama-video-*")); len(left) != 0 {
		t.Fatalf("temporary video copies left after the error: %v", left)
	}
}

func firstDiff(a, b []int32) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}
