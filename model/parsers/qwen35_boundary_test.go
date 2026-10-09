package parsers

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// A qwen-style tool call the model emits INSIDE its thinking block, with no
// </think> before it. Captured on 2026-09-24 from an omnimerge-v6 agentic run
// (reported by ollama@solidpc): the model reasoned out a one-character fix,
// emitted a well-formed `editor` call, and the client received an empty turn --
// no content and no tool call. The agentic loop read that as "the model is
// done" and ended the run, so the fix it had correctly worked out never
// applied, silently.
const qwen35CallInsideThinking = "<think>I need to fix the extra brace. The line reads `}}});` and should read `}});`." +
	"<tool_call><function=editor><parameter=path>manic_miner.html</parameter>" +
	"<parameter=new_text>}});</parameter></function></tool_call>"

// The same call after a well-formed </think>, which is the ordinary path.
const qwen35CallAfterThinking = "<think>The brace count is wrong.</think>" +
	"<tool_call><function=editor><parameter=path>manic_miner.html</parameter>" +
	"<parameter=new_text>}});</parameter></function></tool_call>"

func qwen35EditorTool() []api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("new_text", api.ToolProperty{Type: api.PropertyType{"string"}})
	return []api.Tool{{
		Function: api.ToolFunction{
			Name:       "editor",
			Parameters: api.ToolFunctionParameters{Properties: props},
		},
	}}
}

// feedQwen35 streams chunks through a fresh parser and returns everything it
// emitted, the way a caller accumulates a response.
func feedQwen35(t *testing.T, chunks []string) (calls []api.ToolCall, content, thinking string) {
	t.Helper()

	p := &Qwen35Parser{}
	p.Init(qwen35EditorTool(), nil, &api.ThinkValue{Value: true})

	var contentSb, thinkingSb strings.Builder
	for i, chunk := range chunks {
		c, th, cl, err := p.Add(chunk, i == len(chunks)-1)
		if err != nil {
			t.Fatalf("Add(%q): %v", chunk, err)
		}
		contentSb.WriteString(c)
		thinkingSb.WriteString(th)
		calls = append(calls, cl...)
	}
	return calls, contentSb.String(), thinkingSb.String()
}

func assertEditorCallRecovered(t *testing.T, label string, calls []api.ToolCall, thinking string) {
	t.Helper()

	if len(calls) != 1 {
		t.Fatalf("%s: got %d tool calls, want 1; the call was swallowed (thinking=%q)", label, len(calls), thinking)
	}
	if got := calls[0].Function.Name; got != "editor" {
		t.Errorf("%s: tool name = %q, want %q", label, got, "editor")
	}
	args := calls[0].Function.Arguments.ToMap()
	if got := args["path"]; got != "manic_miner.html" {
		t.Errorf("%s: path = %v, want manic_miner.html", label, got)
	}
	if got := args["new_text"]; got != "}});" {
		t.Errorf("%s: new_text = %v, want }});", label, got)
	}

	// The reasoning channel must never carry the call's own markup. Leaked tags
	// are how a client ends up displaying half a tool call as the model's
	// thoughts, and they are the fingerprint this bug left in the transcript:
	// an orphaned </parameter></function></tool_call> at the end of the
	// reasoning, with the opening tags gone.
	for _, tag := range []string{"<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"} {
		if strings.Contains(thinking, tag) {
			t.Errorf("%s: thinking leaked %q: %q", label, tag, thinking)
		}
	}
}

// TestAToolCallOpenedInsideThinkingSurvivesAnyChunkBoundary is the regression.
//
// A complete <tool_call> already in the buffer used to lose to a trailing
// suffix that merely COULD grow into one. The call's own body is full of `<`,
// so a chunk boundary landing on any of them -- and llama-server streams by
// token, so they land there routinely -- flushed the opening tag into the
// thinking channel along with the reasoning. The buffer kept only the `<`, the
// recovery branch could never fire again, and the rest of the call drained into
// thinking as prose.
//
// Before the fix this lost the call on 12 of the 214 two-way splits below.
func TestAToolCallOpenedInsideThinkingSurvivesAnyChunkBoundary(t *testing.T) {
	for _, raw := range []string{qwen35CallInsideThinking, qwen35CallAfterThinking} {
		for i := 1; i < len(raw); i++ {
			calls, _, thinking := feedQwen35(t, []string{raw[:i], raw[i:]})
			assertEditorCallRecovered(t, fmt.Sprintf("two-way split after byte %d", i), calls, thinking)
		}
	}
}

// TestAToolCallOpenedInsideThinkingSurvivesTokenByToken is the worst case: a
// boundary after every single byte, so every partial tag is seen.
func TestAToolCallOpenedInsideThinkingSurvivesTokenByToken(t *testing.T) {
	for _, raw := range []string{qwen35CallInsideThinking, qwen35CallAfterThinking} {
		chunks := make([]string, 0, len(raw))
		for _, r := range raw {
			chunks = append(chunks, string(r))
		}
		calls, _, thinking := feedQwen35(t, chunks)
		assertEditorCallRecovered(t, "byte by byte", calls, thinking)
	}
}

// TestTheBoundaryThatSwallowedTheCall pins the exact shape from the report: the
// chunk ends on the `<` that opens </parameter>, immediately after a complete
// opening tag. This is the minimal case, and it fails on its own if the branch
// order in eat() ever goes back.
func TestTheBoundaryThatSwallowedTheCall(t *testing.T) {
	raw := qwen35CallInsideThinking
	cut := strings.Index(raw, "</parameter>") + 1 // keep the `<`, split before `/`
	if cut <= 0 {
		t.Fatal("fixture no longer contains </parameter>")
	}

	calls, _, thinking := feedQwen35(t, []string{raw[:cut], raw[cut:]})
	assertEditorCallRecovered(t, "boundary on the `<` of </parameter>", calls, thinking)
}

// TestReasoningIsStillDeliveredWithTheCall guards the other half: recovering the
// call must not cost the reasoning that led to it.
func TestReasoningIsStillDeliveredWithTheCall(t *testing.T) {
	calls, _, thinking := feedQwen35(t, []string{qwen35CallInsideThinking})
	assertEditorCallRecovered(t, "whole response at once", calls, thinking)

	if !strings.Contains(thinking, "extra brace") {
		t.Errorf("reasoning was dropped: thinking = %q", thinking)
	}
}
