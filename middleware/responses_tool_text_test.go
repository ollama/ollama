package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestResponsesMiddlewareToolText(t *testing.T) {
	call := func(id string) api.ToolCall {
		return api.ToolCall{ID: id, Function: api.ToolCallFunction{
			Name: "lookup", Arguments: testArgs(map[string]any{"id": id}),
		}}
	}
	mixed := api.Message{Content: "Checking.", ToolCalls: []api.ToolCall{call("a"), call("b")}}
	for _, tt := range []struct {
		name   string
		stream bool
		chunks []api.Message
		types  []string
		texts  []string
	}{
		{"non-streaming", false, []api.Message{mixed}, []string{"message", "function_call", "function_call"}, []string{"Checking."}},
		{"same chunk", true, []api.Message{mixed}, []string{"message", "function_call", "function_call"}, []string{"Checking."}},
		{"interleaved", true, []api.Message{
			{Thinking: "Considering."},
			{Content: "Checking. "},
			{Content: "Please wait.", ToolCalls: []api.ToolCall{call("a")}},
			{Content: "Next.", ToolCalls: []api.ToolCall{call("b"), call("c")}},
			{Content: "Done."},
		}, []string{"reasoning", "message", "function_call", "message", "function_call", "function_call", "message"}, []string{"Checking. Please wait.", "Next.", "Done."}},
		{"text after tool", true, []api.Message{{ToolCalls: []api.ToolCall{call("a")}}, {Content: "Done."}}, []string{"function_call", "message"}, []string{"Done."}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.Use(ResponsesMiddleware())
			router.POST("/v1/responses", func(c *gin.Context) {
				for i, message := range tt.chunks {
					c.JSON(http.StatusOK, api.ChatResponse{Message: message, Done: i == len(tt.chunks)-1})
				}
			})
			body := fmt.Sprintf(`{"model":"test","input":"look up records","stream":%t}`, tt.stream)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			if resp.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", resp.Code, resp.Body.String())
			}

			var final struct {
				Output []map[string]any `json:"output"`
			}
			var added []string
			done := map[string]map[string]any{}
			textEvents := map[string][]string{}
			deltas := map[string]string{}
			if !tt.stream {
				if err := json.Unmarshal(resp.Body.Bytes(), &final); err != nil {
					t.Fatal(err)
				}
			} else {
				sequence := 0
				completed := false
				for _, frame := range strings.Split(resp.Body.String(), "\n\n") {
					for _, line := range strings.Split(frame, "\n") {
						if !strings.HasPrefix(line, "data: ") {
							continue
						}
						var event struct {
							Type           string          `json:"type"`
							ContentIndex   int             `json:"content_index"`
							Delta          string          `json:"delta"`
							Text           string          `json:"text"`
							Part           map[string]any  `json:"part"`
							SequenceNumber int             `json:"sequence_number"`
							OutputIndex    int             `json:"output_index"`
							ItemID         string          `json:"item_id"`
							Item           map[string]any  `json:"item"`
							Response       json.RawMessage `json:"response"`
						}
						if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
							t.Fatal(err)
						}
						if completed {
							t.Error("event after response.completed")
						}
						if event.SequenceNumber != sequence {
							t.Errorf("sequence = %d, want %d", event.SequenceNumber, sequence)
						}
						sequence++
						id := event.ItemID
						if event.Item != nil {
							id = event.Item["id"].(string)
						}
						if event.Type == "response.output_item.added" {
							if event.OutputIndex != len(added) {
								t.Errorf("new item index = %d, want %d", event.OutputIndex, len(added))
							}
							added = append(added, id)
						}
						if id != "" {
							if event.OutputIndex < 0 || event.OutputIndex >= len(added) || added[event.OutputIndex] != id {
								t.Errorf("event %s index %d does not identify item %s", event.Type, event.OutputIndex, id)
							}
							if done[id] != nil {
								t.Errorf("event %s after item %s finished", event.Type, id)
							}
						}
						if strings.HasPrefix(event.Type, "response.output_text.") || strings.HasPrefix(event.Type, "response.content_part.") {
							if event.ContentIndex != 0 {
								t.Errorf("content index = %d, want 0", event.ContentIndex)
							}
							if event.Type != "response.output_text.delta" || len(textEvents[id]) == 0 || textEvents[id][len(textEvents[id])-1] != event.Type {
								textEvents[id] = append(textEvents[id], event.Type)
							}
						}
						if event.Type == "response.output_text.delta" {
							deltas[id] += event.Delta
						}
						if event.Type == "response.output_text.done" && event.Text != deltas[id] {
							t.Errorf("completed text = %q, deltas = %q", event.Text, deltas[id])
						}
						if event.Type == "response.content_part.done" && event.Part["text"] != deltas[id] {
							t.Errorf("completed part = %v, deltas = %q", event.Part, deltas[id])
						}
						if event.Type == "response.output_item.done" {
							done[id] = event.Item
						}
						if event.Type == "response.completed" {
							completed = true
							if err := json.Unmarshal(event.Response, &final); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
			}

			if tt.stream && (len(added) != len(final.Output) || len(done) != len(added)) {
				t.Errorf("item counts: added %d, done %d, final %d", len(added), len(done), len(final.Output))
			}

			var types, texts []string
			for i, item := range final.Output {
				types = append(types, item["type"].(string))
				id := item["id"].(string)
				if tt.stream && (i >= len(added) || added[i] != id || !reflect.DeepEqual(done[id], item)) {
					t.Errorf("final output[%d] = %#v does not match streamed item", i, item)
				}
				if item["type"] == "message" {
					texts = append(texts, item["content"].([]any)[0].(map[string]any)["text"].(string))
					if tt.stream && deltas[id] != texts[len(texts)-1] {
						t.Errorf("message text = %q, deltas = %q", texts[len(texts)-1], deltas[id])
					}
					wantEvents := []string{"response.content_part.added", "response.output_text.delta", "response.output_text.done", "response.content_part.done"}
					if tt.stream && !reflect.DeepEqual(textEvents[id], wantEvents) {
						t.Errorf("text events = %v, want %v", textEvents[id], wantEvents)
					}
				}
			}
			if !reflect.DeepEqual(types, tt.types) || !reflect.DeepEqual(texts, tt.texts) {
				t.Errorf("output types = %v, texts = %v; want %v, %v", types, texts, tt.types, tt.texts)
			}
		})
	}
}
