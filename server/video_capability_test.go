package server

import (
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestVideoCapability(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config model.ConfigV2
		want   bool
	}{
		{"lfm3 omni", model.ConfigV2{ModelFormat: "safetensors", Renderer: "lfm3-omni", Capabilities: []string{"completion", "vision", "audio"}}, true},
		{"lfm3 vl", model.ConfigV2{ModelFormat: "safetensors", Renderer: "lfm3", Capabilities: []string{"completion", "vision"}}, true},
		{"lfm3 text", model.ConfigV2{ModelFormat: "safetensors", Renderer: "lfm3", Capabilities: []string{"completion"}}, false},
		{"lfm3 audio", model.ConfigV2{ModelFormat: "safetensors", Renderer: "lfm3-audio", Capabilities: []string{"completion", "audio"}}, false},
		{"qwen3.5", model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion", "vision"}}, true},
		{"qwen3.8", model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.8", Capabilities: []string{"completion", "vision"}}, true},
		{"qwen3.5 text", model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion"}}, false},
		{"qwen3.5 gguf", model.ConfigV2{ModelFormat: "gguf", Renderer: "qwen3.5", Capabilities: []string{"completion", "vision"}}, false},
		{"d1 omni decision", model.ConfigV2{ModelFormat: "safetensors", Renderer: "lfm3-omni", Capabilities: []string{"decision", "vision", "audio"}}, false},
		{"other vision model", model.ConfigV2{ModelFormat: "safetensors", Renderer: "gemma4", Capabilities: []string{"completion", "vision"}}, false},
		{"recorded at create", model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion", "vision", "video"}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{Config: tc.config}
			if got := slices.Contains(m.Capabilities(), model.CapabilityVideo); got != tc.want {
				t.Errorf("video capability = %v, want %v (capabilities %v)", got, tc.want, m.Capabilities())
			}
			err := checkVideo(m, "m")
			switch {
			case !tc.want:
				if err == nil || !strings.Contains(err.Error(), "does not support video") {
					t.Errorf("checkVideo = %v, want unsupported model", err)
				}
			case runtime.GOOS == "darwin":
				if err != nil {
					t.Errorf("checkVideo = %v", err)
				}
			default: // Video decoding is macOS-only for now.
				if err == nil || !strings.Contains(err.Error(), runtime.GOOS) {
					t.Errorf("checkVideo = %v, want unsupported platform", err)
				}
			}
		})
	}
}

// A declared capability does not route video to a runner that cannot decode
// it, and is not reported.
func TestCheckVideoDeclaredOnGGUF(t *testing.T) {
	m := &Model{Config: model.ConfigV2{ModelFormat: "gguf", Renderer: "qwen3.5", Capabilities: []string{"completion", "vision", "video"}}}
	if slices.Contains(m.Capabilities(), model.CapabilityVideo) {
		t.Errorf("GGUF model reports video: %v", m.Capabilities())
	}
	if err := checkVideo(m, "m"); err == nil || !strings.Contains(err.Error(), "does not support video") {
		t.Fatalf("checkVideo = %v, want unsupported model", err)
	}
}

func TestHasVideo(t *testing.T) {
	mp4 := api.ImageData("\x00\x00\x00\x20ftypisom\x00\x00\x02\x00")
	png := api.ImageData("\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR")
	if !hasVideo([]api.ImageData{png, mp4}) {
		t.Error("an MP4 attachment was not detected")
	}
	if hasVideo([]api.ImageData{png}) || hasVideo(nil) {
		t.Error("a non-video attachment was detected as video")
	}
}
