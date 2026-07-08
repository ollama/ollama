package server

import (
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestResolveQwen35Variant(t *testing.T) {
	tests := []struct {
		name  string
		model *Model
		want  string
	}{
		{
			name: "instruct from short name",
			model: &Model{
				Name:      "registry.ollama.ai/frob/qwen3.5-instruct:latest",
				ShortName: "frob/qwen3.5-instruct:latest",
				Config:    model.ConfigV2{Renderer: qwen35Legacy, Parser: qwen35Legacy},
			},
			want: qwen35Instruct,
		},
		{
			name: "instruct from base name",
			model: &Model{
				Config: model.ConfigV2{
					Renderer: qwen35Legacy,
					Parser:   qwen35Legacy,
					BaseName: "Qwen3.5-Instruct",
				},
			},
			want: qwen35Instruct,
		},
		{
			name: "thinking from short name",
			model: &Model{
				ShortName: "library/qwen3.5-thinking:latest",
				Config:    model.ConfigV2{Renderer: qwen35Legacy, Parser: qwen35Legacy},
			},
			want: qwen35Thinking,
		},
		{
			name: "plain remains legacy",
			model: &Model{
				ShortName: "library/qwen3.5:latest",
				Config:    model.ConfigV2{Renderer: qwen35Legacy, Parser: qwen35Legacy},
			},
			want: qwen35Legacy,
		},
		{
			name: "instruct from tag",
			model: &Model{
				ShortName: "library/qwen3.5:instruct",
				Config:    model.ConfigV2{Renderer: qwen35Legacy, Parser: qwen35Legacy},
			},
			want: qwen35Instruct,
		},
		{
			name: "namespace does not select instruct variant",
			model: &Model{
				ShortName: "instruct-lab/qwen3.5:latest",
				Config:    model.ConfigV2{Renderer: qwen35Legacy, Parser: qwen35Legacy},
			},
			want: qwen35Legacy,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveRendererName(tt.model); got != tt.want {
				t.Fatalf("resolveRendererName() = %q, want %q", got, tt.want)
			}
			if got := resolveParserName(tt.model); got != tt.want {
				t.Fatalf("resolveParserName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestQwen35InstructModelDoesNotAdvertiseThinking(t *testing.T) {
	m := Model{
		Name:      "registry.ollama.ai/frob/qwen3.5-instruct:latest",
		ShortName: "frob/qwen3.5-instruct:latest",
		Config:    model.ConfigV2{Parser: qwen35Legacy},
	}

	caps := m.Capabilities()
	if !slices.Contains(caps, model.CapabilityTools) {
		t.Fatalf("expected tools capability, got %#v", caps)
	}
	if slices.Contains(caps, model.CapabilityThinking) {
		t.Fatalf("did not expect thinking capability, got %#v", caps)
	}
}

func TestRenderPromptResolvesQwen35InstructRenderer(t *testing.T) {
	m := Model{
		Name:      "registry.ollama.ai/frob/qwen3.5-instruct:latest",
		ShortName: "frob/qwen3.5-instruct:latest",
		Config:    model.ConfigV2{Renderer: qwen35Legacy},
	}

	got, err := renderPrompt(&m, []api.Message{{Role: "user", Content: "Hello"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if strings.Contains(got, "<think>") || strings.Contains(got, "</think>") {
		t.Fatalf("did not expect think tags for instruct renderer, got:\n%s", got)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n") {
		t.Fatalf("expected plain assistant prefill, got:\n%s", got)
	}
}
