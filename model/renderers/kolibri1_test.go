package renderers

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// Frozen outputs of Kolibri-1's upstream Jinja template at
// e52eb4627d11516b0c01de49210ab5a4e4061444.
func TestKolibri1TemplateParity(t *testing.T) {
	data, err := os.ReadFile("testdata/kolibri1_render_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string
		Messages []api.Message
		Tools    []api.Tool
		Think    *api.ThinkValue
		Expected string
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := RenderWithRenderer("kolibri1", tc.Messages, tc.Tools, tc.Think)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.Expected, got); diff != "" {
				t.Errorf("template mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestKolibri1AssistantContinuation(t *testing.T) {
	got, err := RenderWithRenderer("kolibri1", []api.Message{{Role: "user", Content: "Count"}, {Role: "assistant", Content: "1, 2,"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n\n</think>\n\n1, 2,") {
		t.Fatalf("invalid continuation: %q", got)
	}
}

func TestKolibri1MinimalEffort(t *testing.T) {
	for _, effort := range []string{"minimal", "low"} {
		t.Run(effort, func(t *testing.T) {
			think := ResolveThinking(&api.ThinkValue{Value: effort}, ThinkingForRenderer("kolibri1"))
			got, err := RenderWithRenderer("kolibri1", []api.Message{{Role: "user", Content: "Hello"}}, nil, think)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got, "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed directly to the answer.") {
				t.Fatalf("%s did not render low-effort reasoning: %q", effort, got)
			}
		})
	}
}
