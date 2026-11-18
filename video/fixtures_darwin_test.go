package video

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

// TestFixtureGoldens compares decoding of real containers against goldens
// recorded from the publisher's reference processor, and pixels against
// ffmpeg's decode of the same frames when ffmpeg is installed.
// OLLAMA_VIDEO_FIXTURES names a directory with fixtures/ and goldens/.
func TestFixtureGoldens(t *testing.T) {
	dir := os.Getenv("OLLAMA_VIDEO_FIXTURES")
	if dir == "" {
		t.Skip("OLLAMA_VIDEO_FIXTURES not set")
	}
	ffmpeg, _ := exec.LookPath("ffmpeg")
	if ffmpeg == "" {
		t.Log("ffmpeg not found: skipping the pixel comparison")
	}
	for _, name := range []string{"clip10s.mp4", "short1.3s.mp4", "silent7s.mp4", "long200s.mp4", "rot90.mov"} {
		t.Run(name, func(t *testing.T) { fixtureGolden(t, dir, ffmpeg, name) })
	}
}

func fixtureGolden(t *testing.T, dir, ffmpeg, name string) {
	stem := name[:len(name)-len(filepath.Ext(name))]
	raw, err := os.ReadFile(filepath.Join(dir, "goldens", stem+".v.json"))
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Metadata struct {
			FPS           float64 `json:"fps"`
			Total         int     `json:"total_num_frames"`
			Width, Height int
			Indices       []int `json:"frames_indices"`
		}
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	g := golden.Metadata

	path := filepath.Join(dir, "fixtures", name)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m := f.Meta()
	t.Logf("meta %+v; reference fps=%v frames=%d %dx%d", m, g.FPS, g.Total, g.Width, g.Height)
	if m.Frames != g.Total || m.Width != g.Width || m.Height != g.Height {
		t.Errorf("frames/size = %d %dx%d, want %d %dx%d", m.Frames, m.Width, m.Height, g.Total, g.Width, g.Height)
	}
	if math.Abs(m.FPS-g.FPS) > 1e-3*g.FPS {
		t.Errorf("fps = %v, want %v", m.FPS, g.FPS)
	}

	// Spooled copies have no extension.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	bare := filepath.Join(t.TempDir(), "3f2a9c")
	if err := os.WriteFile(bare, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if b, err := Open(bare); err != nil {
		t.Errorf("opening without an extension: %v", err)
	} else {
		if b.Meta() != m {
			t.Errorf("meta without an extension = %+v, want %+v", b.Meta(), m)
		}
		b.Close()
	}

	var worst float64
	err = f.Frames(context.Background(), g.Indices, func(fr Frame) error {
		if fr.Image.Rect.Dx() != g.Width || fr.Image.Rect.Dy() != g.Height {
			t.Errorf("frame %d is %v, want %dx%d", fr.Index, fr.Image.Rect, g.Width, g.Height)
		}
		if ffmpeg == "" || fr.Index%7 != 0 {
			return nil
		}
		out, err := exec.CommandContext(t.Context(), ffmpeg, "-v", "error", "-i", path, "-vf", "select=eq(n\\,"+strconv.Itoa(fr.Index)+")",
			"-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-").Output()
		if err != nil {
			return err
		}
		var sum float64
		var signed [3]float64
		for i := range len(out) / 3 {
			p := fr.Image.Pix[4*i:]
			for c := range 3 {
				d := float64(p[c]) - float64(out[3*i+c])
				sum += math.Abs(d)
				signed[c] += d
			}
		}
		n := float64(len(out) / 3)
		mad := sum / float64(len(out))
		t.Logf("frame %d: mean abs %.3f, signed RGB mean %.2f %.2f %.2f", fr.Index, mad, signed[0]/n, signed[1]/n, signed[2]/n)
		// Chroma upsampling differs between decoders at sharp edges; a
		// shifted color matrix or range would show as a bias.
		for c := range 3 {
			if math.Abs(signed[c]/n) > 2 {
				t.Errorf("frame %d channel %d is biased by %.2f vs ffmpeg", fr.Index, c, signed[c]/n)
			}
		}
		worst = max(worst, mad)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("worst mean abs diff vs ffmpeg rgb24: %.3f", worst)

	pcm, rate, err := f.PCM(600)
	if err != nil {
		t.Fatal(err)
	}
	if m.Audio != (pcm != nil) {
		t.Errorf("audio = %v but got %d samples", m.Audio, len(pcm))
	}
	if pcm != nil {
		t.Logf("pcm %d samples at %d Hz = %.3fs", len(pcm), rate, float64(len(pcm))/float64(rate))
	}
}
