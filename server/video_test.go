package server

import (
	"errors"
	"os"
	"runtime"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/video"
)

func readFixture(t *testing.T) api.ImageData {
	t.Helper()
	data, err := os.ReadFile("testdata/tiny.mp4")
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	return api.ImageData(data)
}

// Requests without a video must come back untouched, on every platform.
func TestExpandVideosPassesThroughNonVideo(t *testing.T) {
	jpeg := api.ImageData{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10}
	in := []api.ImageData{jpeg}

	out, err := expandVideos(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || string(out[0]) != string(jpeg) {
		t.Fatalf("non-video input was modified: %v", out)
	}
}

func TestExpandVideosEmpty(t *testing.T) {
	out, err := expandVideos(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("expected no media, got %d", len(out))
	}
}

// The fixture must be recognized as a video everywhere, even where it cannot be
// decoded — otherwise it would silently be treated as an image.
func TestFixtureIsDetectedAsVideo(t *testing.T) {
	if !video.IsVideo(readFixture(t)) {
		t.Fatal("fixture not detected as video")
	}
}

func TestExpandVideos(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("video decoding is only implemented on macOS")
	}

	jpeg := api.ImageData{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10}
	out, err := expandVideos([]api.ImageData{jpeg, readFixture(t)})
	if err != nil {
		t.Fatal(err)
	}

	if len(out) < 3 {
		t.Fatalf("expected the jpeg plus frames and audio, got %d", len(out))
	}
	if string(out[0]) != string(jpeg) {
		t.Fatal("existing image was not kept in place")
	}
	for i, m := range out {
		if video.IsVideo(m) {
			t.Fatalf("media %d is still a video", i)
		}
	}
	if _, ok := audioAt(out); !ok {
		t.Fatal("audio track missing")
	}
}

func audioAt(media []api.ImageData) (int, bool) {
	for i, m := range media {
		if len(m) >= 12 && string(m[:4]) == "RIFF" && string(m[8:12]) == "WAVE" {
			return i, true
		}
	}
	return 0, false
}

// Where no decoder ships with Ollama the user must get a clear reason.
func TestExpandVideosUnsupportedPlatform(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("macOS has a decoder")
	}
	if _, err := expandVideos([]api.ImageData{readFixture(t)}); !errors.Is(err, video.ErrUnsupportedPlatform) {
		t.Fatalf("expected ErrUnsupportedPlatform, got %v", err)
	}
}
