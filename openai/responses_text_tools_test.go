package openai

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestResponsesStreamTextThenTools(t *testing.T) {
	for _, text := range []string{"", "Checking the weather."} {
		for _, toolCount := range []int{1, 2} {
			t.Run(fmt.Sprintf("text=%t/tools=%d", text != "", toolCount), func(t *testing.T) {
				c := NewResponsesStreamConverter("resp_order", "msg_order", "test-model", ResponsesRequest{})
				var events []ResponsesStreamEvent
				if text != "" {
					events = append(events, c.Process(api.ChatResponse{Message: api.Message{Content: text}})...)
				}
				var calls []api.ToolCall
				for i := 0; i < toolCount; i++ {
					calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{
						Name: "weather", Arguments: testArgs(map[string]any{"city": "Paris"}),
					}})
				}
				events = append(events, c.Process(api.ChatResponse{Message: api.Message{ToolCalls: calls}})...)
				events = append(events, c.Process(api.ChatResponse{Done: true})...)
				added, done := map[int]any{}, map[int]any{}
				var output []any
				textDone, partDone := 0, 0
				for _, event := range events {
					data := event.Data.(map[string]any)
					switch event.Event {
					case "response.output_item.added":
						index := data["output_index"].(int)
						if text != "" && data["item"].(map[string]any)["type"] == "function_call" && done[0] == nil {
							t.Error("function call opened before message was closed")
						}
						if _, exists := added[index]; exists {
							t.Errorf("duplicate added index %d", index)
						}
						added[index] = data["item"].(map[string]any)["id"]
					case "response.output_item.done":
						if _, exists := done[data["output_index"].(int)]; exists {
							t.Error("item closed twice")
						}
						done[data["output_index"].(int)] = data["item"].(map[string]any)["id"]
					case "response.output_text.done":
						textDone++
					case "response.content_part.done":
						partDone++
					case "response.completed":
						output = data["response"].(map[string]any)["output"].([]any)
					}
				}
				wantCount, wantTextDone := toolCount, 0
				if text != "" {
					wantCount++
					wantTextDone = 1
				}
				if len(output) != wantCount || len(added) != wantCount || len(done) != wantCount {
					t.Errorf("output/added/done counts = %d/%d/%d, want %d", len(output), len(added), len(done), wantCount)
				}
				if textDone != wantTextDone || partDone != wantTextDone {
					t.Errorf("text/part done counts = %d/%d, want %d", textDone, partDone, wantTextDone)
				}
				for index, raw := range output {
					id := raw.(map[string]any)["id"]
					if id != added[index] || id != done[index] {
						t.Errorf("output[%d] ID %v, added %v, done %v", index, id, added[index], done[index])
					}
				}
			})
		}
	}
}

func TestResponsesStreamEmptyToolsKeepMessageOpen(t *testing.T) {
	c := NewResponsesStreamConverter("resp_empty", "msg_empty", "test-model", ResponsesRequest{})
	c.Process(api.ChatResponse{Message: api.Message{Content: "text"}})
	if events := c.EmitFunctionCallItems(nil); len(events) != 0 {
		t.Fatalf("empty tool list emitted events: %#v", events)
	}
	if !c.contentStarted || c.outputIndex != 0 {
		t.Fatal("empty tool list changed the open message")
	}
}

func TestResponsesStreamReasoningThenTools(t *testing.T) {
	for _, direct := range []bool{false, true} {
		for _, toolCount := range []int{1, 2} {
			t.Run(fmt.Sprintf("direct=%t/tools=%d", direct, toolCount), func(t *testing.T) {
				c := NewResponsesStreamConverter("resp_reasoning", "msg_reasoning", "test-model", ResponsesRequest{})
				events := c.Process(api.ChatResponse{Message: api.Message{Thinking: "Checking the weather."}})
				var calls []api.ToolCall
				for i := 0; i < toolCount; i++ {
					calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{
						Name: "weather", Arguments: testArgs(map[string]any{"city": "Paris"}),
					}})
				}
				if direct {
					events = append(events, c.EmitFunctionCallItems(calls)...)
				} else {
					events = append(events, c.Process(api.ChatResponse{Message: api.Message{ToolCalls: calls}})...)
				}
				events = append(events, c.Process(api.ChatResponse{Done: true})...)
				added, done := map[int]any{}, map[int]any{}
				var output []any
				summaryDone := 0
				for _, event := range events {
					data := event.Data.(map[string]any)
					switch event.Event {
					case "response.output_item.added":
						index := data["output_index"].(int)
						item := data["item"].(map[string]any)
						if item["type"] == "function_call" && done[0] == nil {
							t.Error("function call opened before reasoning was closed")
						}
						if _, exists := added[index]; exists {
							t.Errorf("duplicate added index %d", index)
						}
						added[index] = item["id"]
					case "response.output_item.done":
						index := data["output_index"].(int)
						if _, exists := done[index]; exists {
							t.Errorf("item %d closed twice", index)
						}
						done[index] = data["item"].(map[string]any)["id"]
					case "response.reasoning_summary_text.done":
						summaryDone++
						if data["output_index"] != 0 || data["text"] != "Checking the weather." {
							t.Errorf("unexpected reasoning completion: %#v", data)
						}
					case "response.completed":
						output = data["response"].(map[string]any)["output"].([]any)
					}
				}
				want := toolCount + 1
				if len(output) != want || len(added) != want || len(done) != want || summaryDone != 1 {
					t.Fatalf("output/added/done/summary counts = %d/%d/%d/%d, want %d/%d/%d/1", len(output), len(added), len(done), summaryDone, want, want, want)
				}
				for index, raw := range output {
					item := raw.(map[string]any)
					if item["id"] != added[index] || item["id"] != done[index] {
						t.Errorf("output[%d] ID %v, added %v, done %v", index, item["id"], added[index], done[index])
					}
					if index == 0 && item["type"] != "reasoning" {
						t.Error("reasoning is not first in final output")
					}
				}
			})
		}
	}

}

func TestResponsesStreamEmptyToolsKeepReasoningOpen(t *testing.T) {
	c := NewResponsesStreamConverter("resp_empty_reasoning", "msg_empty_reasoning", "test-model", ResponsesRequest{})
	c.Process(api.ChatResponse{Message: api.Message{Thinking: "thinking"}})
	if events := c.EmitFunctionCallItems(nil); len(events) != 0 {
		t.Fatalf("empty tool list emitted events: %#v", events)
	}
	if !c.reasoningStarted || c.outputIndex != 0 || c.accumulatedThinking != "thinking" {
		t.Fatal("empty tool list changed the open reasoning")
	}
}
