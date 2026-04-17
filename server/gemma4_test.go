package server

import "testing"

func TestResolveGemma4Renderer(t *testing.T) {
	tests := []struct {
		name  string
		model *Model
		want  string
	}{
		{
			name:  "nil model falls back to legacy alias",
			model: nil,
			want:  gemma4RendererLegacy,
		},
		{
			name: "explicit small passes through",
			model: &Model{
				Config: testConfigWithRenderer(gemma4RendererSmall),
			},
			want: gemma4RendererSmall,
		},
		{
			name: "explicit large passes through",
			model: &Model{
				Config: testConfigWithRenderer(gemma4RendererLarge),
			},
			want: gemma4RendererLarge,
		},
		{
			name: "legacy e4b tag resolves small",
			model: &Model{
				Name:      "gemma4:e4b",
				ShortName: "gemma4:e4b",
				Config:    testConfigWithRenderer(gemma4RendererLegacy),
			},
			want: gemma4RendererSmall,
		},
		{
			name: "legacy 31b tag resolves large",
			model: &Model{
				Name:      "gemma4:31b-cloud",
				ShortName: "gemma4:31b-cloud",
				Config:    testConfigWithRenderer(gemma4RendererLegacy),
			},
			want: gemma4RendererLarge,
		},
		{
			name: "legacy model type resolves small",
			model: &Model{
				Config: testConfigWithRendererAndType(gemma4RendererLegacy, "4.3B"),
			},
			want: gemma4RendererSmall,
		},
		{
			name: "legacy model type resolves large",
			model: &Model{
				Config: testConfigWithRendererAndType(gemma4RendererLegacy, "25.2B"),
			},
			want: gemma4RendererLarge,
		},
		{
			name: "legacy unknown defaults small",
			model: &Model{
				Config: testConfigWithRenderer(gemma4RendererLegacy),
			},
			want: gemma4RendererSmall,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveGemma4Renderer(tt.model); got != tt.want {
				t.Fatalf("resolveGemma4Renderer() = %q, want %q", got, tt.want)
			}
		})
	}
}
