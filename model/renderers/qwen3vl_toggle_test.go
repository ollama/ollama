package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3VLRendererThinkToggle(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "hello"},
	}

	t.Run("thinking enabled by default when configured", func(t *testing.T) {
		r := &Qwen3VLRenderer{hasThinkingSupport: true, defaultThinking: true}
		out, err := r.Render(msgs, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out, "<think>\n") {
			t.Fatalf("expected thinking prefill in output, got: %q", out)
		}
	})

	t.Run("thinking disabled by default", func(t *testing.T) {
		r := &Qwen3VLRenderer{hasThinkingSupport: true, defaultThinking: false}
		out, err := r.Render(msgs, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(out, "<think>\n") {
			t.Fatalf("did not expect thinking prefill in output, got: %q", out)
		}
		if !strings.HasSuffix(out, "<|im_start|>assistant\n</think>\n") {
			t.Fatalf("unexpected assistant prefill: %q", out)
		}
	})

	t.Run("thinking disabled at runtime", func(t *testing.T) {
		r := &Qwen3VLRenderer{hasThinkingSupport: true, defaultThinking: true}
		out, err := r.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(out, "<think>\n") {
			t.Fatalf("did not expect thinking prefill in output, got: %q", out)
		}
		if !strings.HasSuffix(out, "<|im_start|>assistant\n</think>\n") {
			t.Fatalf("unexpected assistant prefill: %q", out)
		}
	})
}
