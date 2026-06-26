package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestOrnith9BRendererMatchesAssistantHistoryThinkBlocks(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{Role: "assistant", Content: "Hello."},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith-9b", msgs, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>

</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>
`
	if got != want {
		t.Fatalf("unexpected Ornith 9B render\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

func TestOrnith9BRendererKeepsAssistantThinkBlocksWhenThinkingDisabled(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{
			Role:     "assistant",
			Thinking: "Keep it short.",
			Content:  "Hello.",
		},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith-9b", msgs, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>
Keep it short.
</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>

</think>

`
	if got != want {
		t.Fatalf("unexpected Ornith 9B render with thinking disabled\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}
