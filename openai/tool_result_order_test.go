package openai

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/model/renderers"
)

func TestFromChatRequestToolResultOrder(t *testing.T) {
	call := func(ids ...string) Message {
		m := Message{Role: "assistant"}
		for _, id := range ids {
			tc := ToolCall{ID: id, Type: "function"}
			tc.Function.Name = "get_weather"
			tc.Function.Arguments = `{"city":"` + id + `"}`
			m.ToolCalls = append(m.ToolCalls, tc)
		}
		return m
	}
	result := func(id string) Message {
		return Message{Role: "tool", ToolCallID: id, Content: "weather for " + id}
	}
	for _, tt := range []struct {
		name     string
		messages []Message
		want     []string
	}{
		{"reversed", []Message{call("a", "b"), result("b"), result("a")}, []string{"weather for a", "weather for b"}},
		{"ordered", []Message{call("a", "b"), result("a"), result("b")}, []string{"weather for a", "weather for b"}},
		{"three calls", []Message{call("a", "b", "c"), result("c"), result("a"), result("b")}, []string{"weather for a", "weather for b", "weather for c"}},
		{"missing call ID", []Message{call("", "b"), result("b"), result("")}, []string{"weather for b", "weather for "}},
		{"missing ID", []Message{call("a", "b"), result("b"), result("")}, []string{"weather for b", "weather for "}},
		{"unknown ID", []Message{call("a", "b"), result("b"), result("x")}, []string{"weather for b", "weather for x"}},
		{"duplicate result", []Message{call("a", "b", "c"), result("b"), result("a"), result("b")}, []string{"weather for b", "weather for a", "weather for b"}},
		{"duplicate call", []Message{call("a", "b", "a"), result("b"), result("a"), result("a")}, []string{"weather for b", "weather for a", "weather for a"}},
		{"incomplete", []Message{call("a", "b", "c"), result("b"), result("a")}, []string{"weather for b", "weather for a"}},
		{"extra result", []Message{call("a", "b"), result("b"), result("a"), result("c")}, []string{"weather for b", "weather for a", "weather for c"}},
		{"no assistant", []Message{result("b"), result("a")}, []string{"weather for b", "weather for a"}},
		{"interrupted", []Message{call("a", "b"), result("b"), {Role: "user", Content: "continue"}, result("a")}, []string{"weather for b", "weather for a"}},
		{"multiple turns", []Message{call("a", "b"), result("b"), result("a"), call("c", "d"), result("d"), result("c")}, []string{"weather for a", "weather for b", "weather for c", "weather for d"}},
	} {
		for _, parts := range []bool{false, true} {
			name := tt.name
			if parts {
				name += "/content parts"
			}
			t.Run(name, func(t *testing.T) {
				messages := append([]Message(nil), tt.messages...)
				if parts {
					for i := range messages {
						if messages[i].Role == "tool" {
							messages[i].Content = []any{map[string]any{"type": "text", "text": messages[i].Content}}
						}
					}
				}
				before, _ := json.Marshal(messages)
				got, err := FromChatRequest(ChatCompletionRequest{Messages: messages})
				if err != nil {
					t.Fatal(err)
				}
				var contents []string
				for _, m := range got.Messages {
					if m.Role == "tool" {
						contents = append(contents, m.Content)
					}
				}
				if diff := cmp.Diff(tt.want, contents); diff != "" {
					t.Errorf("results (-want +got):\n%s", diff)
				}
				after, _ := json.Marshal(messages)
				if string(before) != string(after) {
					t.Error("mutated caller's messages")
				}
			})
		}
	}
}

// Exercise the real renderers with the issue's two same-name weather calls.
// Reordering results must preserve the prompt, while exchanging their IDs must
// change it: otherwise the model never receives the corrected association.
func TestToolResultOrderRenderedPrompt(t *testing.T) {
	const payload = `{"model":"test","messages":[{"role":"user","content":"Return temperatures for Tokyo and Toronto."},{"role":"assistant","tool_calls":[{"id":"call_tokyo","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}},{"id":"call_toronto","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Toronto\"}"}}]},{"role":"tool","tool_call_id":"call_tokyo","content":"{\"temperature\":5,\"condition\":\"snow\"}"},{"role":"tool","tool_call_id":"call_toronto","content":"{\"temperature\":22,\"condition\":\"sunny\"}"}]}`
	for _, renderer := range []string{"gemma4", "qwen3-coder"} {
		t.Run(renderer, func(t *testing.T) {
			var req ChatCompletionRequest
			if err := json.Unmarshal([]byte(payload), &req); err != nil {
				t.Fatal(err)
			}
			render := func() string {
				t.Helper()
				converted, err := FromChatRequest(req)
				if err != nil {
					t.Fatal(err)
				}
				prompt, err := renderers.RenderWithRenderer(renderer, converted.Messages, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				return prompt
			}
			baseline := render()
			req.Messages[2], req.Messages[3] = req.Messages[3], req.Messages[2]
			if diff := cmp.Diff(baseline, render()); diff != "" {
				t.Errorf("reordered prompt (-baseline +reordered):\n%s", diff)
			}
			req.Messages[2].ToolCallID, req.Messages[3].ToolCallID = req.Messages[3].ToolCallID, req.Messages[2].ToolCallID
			if baseline == render() {
				t.Error("exchanging result IDs did not change the prompt")
			}
		})
	}
}
