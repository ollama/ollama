package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestResolveThinking(t *testing.T) {
	mixed := &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}
	for _, tt := range []struct {
		name            string
		descriptor      *model.Thinking
		requested, want any
	}{
		{"omission", mixed, nil, "high"},
		{"exact", mixed, "max", "max"},
		{"unsupported low", mixed, "low", "high"},
		{"unsupported xhigh", mixed, "xhigh", "high"},
		{"future level", mixed, "future", "high"},
		{"empty level", mixed, "", "high"},
		{"case sensitive", mixed, "MAX", "high"},
		{"whitespace", mixed, " max ", "high"},
		{"off", mixed, false, false},
		{"boolean on retains behavior", mixed, true, true},
		{"unknown retains omission", nil, nil, nil},
		{"unknown retains name", nil, "xhigh", "xhigh"},
		{"invalid metadata retains request", &model.Thinking{Values: []any{false}, Default: 75}, "xhigh", "xhigh"},
		{"toggle unsupported uses on", &model.Thinking{Values: []any{false, true}, Default: true}, "low", true},
		{"off default is not lost", &model.Thinking{Values: []any{false, true}, Default: false}, "low", false},
		{"fixed thinking preserves explicit off", &model.Thinking{Values: []any{true}, Default: true}, false, false},
		{"known nonthinking default", &model.Thinking{Values: []any{false}, Default: false}, nil, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var requestedThink *api.ThinkValue
			if tt.requested != nil {
				requestedThink = &api.ThinkValue{Value: tt.requested}
			}
			think := ResolveThinking(requestedThink, tt.descriptor)
			var got any
			if think != nil {
				got = think.Value
			}
			if got != tt.want {
				t.Fatalf("got %#v, want %#v", got, tt.want)
			}
			if requestedThink != nil && requestedThink.Value != tt.requested {
				t.Fatal("mutated request")
			}
		})
	}
}

func TestThinkingRendererVariants(t *testing.T) {
	for _, name := range []string{"qwen3.5", "qwen3.8", "ornith", "qwen3-coder", "qwen3-vl-instruct", "qwen3-vl-thinking", "cogito", "deepseek3.1", "olmo3", "olmo3.1", "olmo3-think", "olmo3-32b-think", "nemotron-3-nano", "nemotron-3.5-nano", "gemma4", "gemma4-small", "gemma4-large", "functiongemma", "glm-4.7", "glm-ocr", "lfm2", "lfm2-thinking", "laguna", "poolside-v1", "cohere", "glimmer"} {
		t.Run(name, func(t *testing.T) {
			if !ThinkingForRenderer(name).Valid() {
				t.Fatal("invalid or absent descriptor")
			}
		})
	}
	if ThinkingForRenderer("unknown") != nil {
		t.Fatal("unknown renderer must not invent metadata")
	}
}
