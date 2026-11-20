package renderers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestNestedToolHistoryOrder(t *testing.T) {
	const history = `[{"role":"user","content":"Send it."},{"role":"assistant","tool_calls":[{"id":"call_1","function":{"name":"edit","arguments":{"box":{"z":"last-field","a":"first-field"},"rows":[{"y":"last-row","b":"first-row"}]}}}]},{"role":"tool","tool_name":"edit","tool_call_id":"call_1","content":"Done."}]`
	var messages []api.Message
	if err := json.Unmarshal([]byte(history), &messages); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name        string
		sourceOrder bool
	}{
		{"ripple", true}, {"qwen3-vl-instruct", true}, {"qwen3-vl-thinking", true}, {"cogito", true}, {"deepseek3.1", true}, {"cohere", true}, {"qwen3.5", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prompt, err := RenderWithRenderer(tc.name, messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, pair := range [][2]string{{"last-field", "first-field"}, {"last-row", "first-row"}} {
				left, right := strings.Index(prompt, pair[0]), strings.Index(prompt, pair[1])
				if left < 0 || right < 0 || (left < right) != tc.sourceOrder {
					t.Errorf("expected %q, %q in source order=%t, rendered %q", pair[0], pair[1], tc.sourceOrder, prompt)
				}
			}
		})
	}
}
