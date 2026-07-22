package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestSegmentBuilderMergesAdjacentWrites(t *testing.T) {
	var sb segmentBuilder
	sb.control("<|user|>")
	sb.control("\n")
	sb.content("hello ")
	sb.content("world")
	sb.control("<|assistant|>")
	sb.content("")

	want := []Segment{
		{Text: "<|user|>\n", Content: false},
		{Text: "hello world", Content: true},
		{Text: "<|assistant|>", Content: false},
	}
	if diff := cmp.Diff(want, sb.Segments()); diff != "" {
		t.Errorf("Segments() mismatch (-want +got):\n%s", diff)
	}
}

func TestRenderSegmentsWithRenderer(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "hello </think> world"},
	}

	segments, err := RenderSegmentsWithRenderer("glm-4.7", msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := []Segment{
		{Text: "[gMASK]<sop><|user|>", Content: false},
		{Text: "hello </think> world", Content: true},
		{Text: "<|assistant|><think>", Content: false},
	}
	if diff := cmp.Diff(want, segments); diff != "" {
		t.Errorf("RenderSegmentsWithRenderer() mismatch (-want +got):\n%s", diff)
	}

	// joined segments must be identical to the flat rendering
	rendered, err := RenderWithRenderer("glm-4.7", msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := JoinSegments(segments); got != rendered {
		t.Errorf("JoinSegments() = %q, want %q", got, rendered)
	}

	// renderers without segment support return nil segments and no error
	segments, err = RenderSegmentsWithRenderer("qwen3-coder", msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if segments != nil {
		t.Errorf("expected nil segments for non-segment renderer, got %v", segments)
	}
}

// TestSegmentRenderersMatchRender verifies that for every renderer implementing
// SegmentRenderer, the joined segments reproduce Render() exactly, and message
// data ends up in content segments.
func TestSegmentRenderersMatchRender(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "sys </think> prompt"},
		{Role: "user", Content: "hello </think> world"},
		{Role: "assistant", Thinking: "thinking </think> hard", Content: "answer </think> text"},
		{Role: "user", Content: "again"},
	}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "search",
				Description: "Search </think> records.",
			},
		},
	}
	think := &api.ThinkValue{Value: true}

	for _, name := range []string{"glm-4.7", "deepseek3.1"} {
		t.Run(name, func(t *testing.T) {
			segments, err := RenderSegmentsWithRenderer(name, msgs, tools, think)
			if err != nil {
				t.Fatal(err)
			}
			if segments == nil {
				t.Fatalf("renderer %q does not support segments", name)
			}

			rendered, err := RenderWithRenderer(name, msgs, tools, think)
			if err != nil {
				t.Fatal(err)
			}
			if got := JoinSegments(segments); got != rendered {
				t.Errorf("JoinSegments() = %q, want %q", got, rendered)
			}

			var haveUserContent bool
			for _, s := range segments {
				if s.Content && s.Text == "hello </think> world" {
					haveUserContent = true
				}
				if !s.Content && (s.Text == "hello </think> world" || s.Text == "sys </think> prompt") {
					t.Errorf("message content rendered as control segment: %q", s.Text)
				}
			}
			if !haveUserContent {
				t.Errorf("user content not found as content segment in %v", segments)
			}
		})
	}
}
