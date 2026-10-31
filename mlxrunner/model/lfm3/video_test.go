package lfm3

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/videoclip"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	"github.com/ollama/ollama/video"
)

// omniVideo is LFM3-Omni's shipped video_preprocessor_config.json.
var omniVideo = videoclip.Config{FPS: 2, MinFrames: 4, MaxFrames: 384, MinPixels: 4096, MaxPixels: 25165824}

// TestSoundtrackSpans checks the soundtrack runs the reference processor
// puts after each frame pair of a 10 s, 30 fps clip with 125 audio tokens.
func TestSoundtrackSpans(t *testing.T) {
	plan, err := videoclip.NewPlan(video.Meta{Frames: 300, FPS: 30, Width: 640, Height: 360}, omniVideo, 32)
	if err != nil {
		t.Fatal(err)
	}
	var runs []int
	end := 0
	for _, span := range soundtrackSpans(plan, 125, 0.08) {
		runs = append(runs, span[1]-span[0])
		if span[0] != end {
			t.Fatalf("soundtrack spans are not contiguous at %v", span)
		}
		end = span[1]
	}
	if want := []int{13, 13, 13, 13, 13, 14, 13, 13, 13, 7}; !slices.Equal(runs, want) {
		t.Errorf("got soundtrack runs %v, want %v", runs, want)
	}
	if spans := soundtrackSpans(plan, 0, 0); spans != nil {
		t.Errorf("got spans %v without a soundtrack", spans)
	}
}

// TestVideoGoldens checks PrepareMedia's expansion of each fixture against
// the reference processor's. OLLAMA_VIDEO_FIXTURES names a directory with
// fixtures/, goldens/ and ref/LFM3-Omni-2B/ (the publisher checkout at
// 101fb44, for the tokenizer and the Omni audio configs).
func TestVideoGoldens(t *testing.T) {
	dir, tok := videoFixtures(t)
	ref := filepath.Join(dir, "ref", "LFM3-Omni-2B")
	config, err := os.ReadFile(filepath.Join(ref, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	processor, err := os.ReadFile(filepath.Join(ref, "processor_config.json"))
	if err != nil {
		t.Fatal(err)
	}

	goldens, err := filepath.Glob(filepath.Join(dir, "goldens", "*.json"))
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
				Fixture   string  `json:"fixture"`
				Error     string  `json:"error"`
				WithAudio bool    `json:"with_audio"`
				InputIDs  []int32 `json:"input_ids"`
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			if g.Error != "" {
				t.Skipf("reference failed on this input: %s", g.Error)
			}
			data, err := os.ReadFile(filepath.Join(dir, "fixtures", g.Fixture))
			if err != nil {
				t.Fatal(err)
			}
			m := &VLModel{Model: &Model{tok: tok, Vision: &VisionModel{
				Config:       visionConfig{PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2},
				VideoTokenID: 125029,
				Video:        &omniVideo,
			}}}
			if g.WithAudio {
				if m.Audio, err = newAudioInput(config, processor, true); err != nil {
					t.Fatal(err)
				}
			}
			prepared, err := m.PrepareMedia([]model.Segment{{Kind: "video", Data: data}})
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.Release()

			got, want := prepared.Tokens, g.InputIDs[slices.Index(g.InputIDs, visionStartTokenID):]
			i := 0
			for i < len(got) && i < len(want) && got[i] == want[i] {
				i++
			}
			// AVFoundation trims AAC's trailing padding by the edit list; the
			// reference decoder keeps ~23 ms of it, which can add one final
			// soundtrack token before the clip's closing marker.
			paddingToken := g.WithAudio && i == len(got)-1 && i+1 < len(want) && want[i] == audioTokenID && want[i+1] == got[i]
			if i < len(got) && !paddingToken {
				t.Fatalf("expansion differs from the reference at token %d:\n got %s\nwant %s", i, render(tok, got), render(tok, want))
			}
		})
	}
}

// videoFixtures returns OLLAMA_VIDEO_FIXTURES, a directory with fixtures/,
// goldens/ and ref/LFM3-Omni-2B/, and the reference tokenizer from it.
func videoFixtures(t *testing.T) (string, *tokenizer.Tokenizer) {
	t.Helper()
	dir := os.Getenv("OLLAMA_VIDEO_FIXTURES")
	if dir == "" {
		t.Skip("OLLAMA_VIDEO_FIXTURES not set")
	}
	ref := filepath.Join(dir, "ref", "LFM3-Omni-2B")
	read := func(name string) []byte {
		data, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(read("tokenizer.json"), &tokenizer.TokenizerConfig{
		ConfigJSON:           read("config.json"),
		GenerationConfigJSON: read("generation_config.json"),
		TokenizerConfigJSON:  read("tokenizer_config.json"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return dir, tok
}

// TestPrepareVideoReleasesOnError checks that a request failing after its
// video was prepared still removes the video's temporary copy.
func TestPrepareVideoReleasesOnError(t *testing.T) {
	dir, tok := videoFixtures(t)
	clip, err := os.ReadFile(filepath.Join(dir, "fixtures", "silent7s.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	m := &VLModel{Model: &Model{tok: tok, Vision: &VisionModel{
		Config:       visionConfig{PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2},
		VideoTokenID: 125029,
		Video:        &omniVideo,
	}}}

	prepared, err := m.PrepareMedia([]model.Segment{{Kind: "video", Data: clip}})
	if err != nil {
		t.Fatal(err)
	}
	if left, _ := filepath.Glob(filepath.Join(tmp, "ollama-video-*")); len(left) != 1 {
		t.Fatalf("prepared video copies = %v, want 1", left)
	}
	prepared.Release()

	_, err = m.PrepareMedia([]model.Segment{{Kind: "video", Data: clip}, {Kind: "image", Data: []byte("not an image")}})
	if err == nil {
		t.Fatal("PrepareMedia accepted an undecodable image")
	}
	if left, _ := filepath.Glob(filepath.Join(tmp, "ollama-video-*")); len(left) != 0 {
		t.Fatalf("temporary video copies left after the error: %v", left)
	}
}

// render shows ids as text with runs of one token compressed.
func render(tok *tokenizer.Tokenizer, ids []int32) string {
	var b strings.Builder
	for i := 0; i < len(ids); {
		j := i
		for j < len(ids) && ids[j] == ids[i] {
			j++
		}
		if j-i > 2 {
			fmt.Fprintf(&b, "%s×%d", tok.Decode([]int32{ids[i]}), j-i)
		} else {
			b.WriteString(tok.Decode(ids[i:j]))
		}
		i = j
		if b.Len() > 600 {
			b.WriteString("…")
			break
		}
	}
	return b.String()
}
