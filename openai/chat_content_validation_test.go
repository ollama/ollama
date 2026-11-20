package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFromChatRequestRejectsInvalidContentWithToolCalls(t *testing.T) {
	for _, content := range []string{`42`, `true`, `{"text":"do not discard"}`} {
		for _, calls := range []string{`[]`, `[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]`} {
			t.Run(content+"/"+calls, func(t *testing.T) {
				var req ChatCompletionRequest
				body := `{"model":"test","messages":[{"role":"assistant","content":` + content + `,"tool_calls":` + calls + `}]}`
				if err := json.Unmarshal([]byte(body), &req); err != nil {
					t.Fatal(err)
				}
				if _, err := FromChatRequest(req); err == nil || !strings.Contains(err.Error(), "invalid message content type") {
					t.Fatalf("expected invalid content error, got %v", err)
				}
			})
		}
	}
}

func TestFromChatRequestAllowsAbsentContentWithToolCalls(t *testing.T) {
	for _, content := range []string{``, `"content":null,`} {
		var req ChatCompletionRequest
		body := `{"model":"test","messages":[{"role":"assistant",` + content + `"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}`
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		chat, err := FromChatRequest(req)
		if err != nil {
			t.Fatal(err)
		}
		if len(chat.Messages) != 1 || chat.Messages[0].Content != "" || len(chat.Messages[0].ToolCalls) != 1 || chat.Messages[0].ToolCalls[0].ID != "call_1" {
			t.Fatalf("tool call changed: %+v", chat.Messages)
		}
	}
}
