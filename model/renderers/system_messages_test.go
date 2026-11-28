package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRenderWithRendererPreservesLaterSystemMessages(t *testing.T) {
	tests := []struct {
		name   string
		suffix string
	}{
		{name: "qwen3-coder"},
		{name: "olmo3"},
		{name: "olmo3.1"},
		{name: "olmo3-think", suffix: olmo3ThinkFunctionsSuffix},
		{name: "olmo3-32b-think", suffix: olmo3ThinkFunctionsSuffix},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []api.Message{
				{Role: "system", Content: "first instruction"},
				{Role: "user", Content: "initial question"},
				{Role: "assistant", Content: "initial answer"},
				{Role: "system", Content: "follow-up instruction"},
				{Role: "user", Content: "follow-up question"},
			}
			prompt, err := RenderWithRenderer(tt.name, messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := "<|im_start|>system\nfirst instruction" + tt.suffix + "<|im_end|>\n" +
				"<|im_start|>user\ninitial question<|im_end|>\n" +
				"<|im_start|>assistant\ninitial answer<|im_end|>\n" +
				"<|im_start|>system\nfollow-up instruction" + tt.suffix + "<|im_end|>\n" +
				"<|im_start|>user\nfollow-up question<|im_end|>\n" +
				"<|im_start|>assistant\n"
			if !strings.HasPrefix(prompt, want) {
				t.Errorf("prompt = %q, want prefix %q", prompt, want)
			}
		})
	}
}
