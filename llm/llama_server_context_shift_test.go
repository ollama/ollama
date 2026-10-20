package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestServerContextShift(t *testing.T) {
	tests := []struct {
		name       string
		arch       string
		projectors []string
		enabled    bool
		want       bool
	}{
		{name: "llama keeps shift", arch: "llama", enabled: true, want: true},
		{name: "glm5-next keeps shift", arch: "glm5-next", enabled: true, want: true},
		{name: "disabled by config", arch: "llama", enabled: false, want: false},
		{name: "deepseek41 dsv4 cache", arch: "deepseek41", enabled: true, want: false},
		{name: "deepseek4 dsv4 cache", arch: "deepseek4", enabled: true, want: false},
		{name: "qwen35 im-rope", arch: "qwen35", enabled: true, want: false},
		{name: "multimodal projector", arch: "llama", projectors: []string{"mmproj.gguf"}, enabled: true, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			launch := llamaServerLaunchConfig{
				modelArch:  tt.arch,
				projectors: tt.projectors,
				opts:       api.Options{NumKeep: 4},
				config:     LlamaServerConfig{ContextShift: tt.enabled},
			}
			if got := launch.serverContextShift(); got != tt.want {
				t.Fatalf("serverContextShift() = %v, want %v", got, tt.want)
			}
			params := appendContextShiftArgs(nil, launch.opts, launch.serverContextShift())
			if got := slices.Contains(params, "--context-shift"); got != tt.want {
				t.Fatalf("params %v contain --context-shift = %v, want %v", params, got, tt.want)
			}
		})
	}
}
