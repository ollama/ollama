package renderers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// These renderers use different tool-call syntax, but each must retain the
// distinction between JSON integers and decimals in a decoded API request.
func TestDecodedToolArgumentNumbersAcrossRenderers(t *testing.T) {
	const request = `{"role":"assistant","tool_calls":[{"function":{"name":"calculate","arguments":{"count":1,"ratio":1.0,"large":9007199254740993}}}]}`
	var message api.Message
	if err := json.Unmarshal([]byte(request), &message); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name     string
		renderer Renderer
		want     []string
	}{
		{"Flint", &FlintRenderer{}, []string{
			"<arg_key>count</arg_key><arg_value>1</arg_value>",
			"<arg_key>ratio</arg_key><arg_value>1.0</arg_value>",
			"<arg_key>large</arg_key><arg_value>9007199254740993</arg_value>",
		}},
		{"Laguna", &LagunaV8Renderer{}, []string{
			"<arg_key>count</arg_key><arg_value>1</arg_value>",
			"<arg_key>ratio</arg_key><arg_value>1.0</arg_value>",
			"<arg_key>large</arg_key><arg_value>9007199254740993</arg_value>",
		}},
		{"Qwen3.8", newQwen38Renderer(), []string{
			"<parameter=count>\n1\n</parameter>",
			"<parameter=ratio>\n1.0\n</parameter>",
			"<parameter=large>\n9007199254740993\n</parameter>",
		}},
		{"Glimmer", &GlimmerRenderer{}, []string{
			"<atem:parameter name=\"count\">1</atem:parameter>",
			"<atem:parameter name=\"ratio\">1.0</atem:parameter>",
			"<atem:parameter name=\"large\">9007199254740993</atem:parameter>",
		}},
		{"Gemma4", &Gemma4Renderer{}, []string{
			"count:1", "ratio:1.0", "large:9007199254740993",
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.renderer.Render([]api.Message{{Role: "user", Content: "Calculate."}, message}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Errorf("rendered tool call lacks %q: %q", want, got)
				}
			}
		})
	}
}
