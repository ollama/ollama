package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestResponsesMiddlewareToolSearchInput(t *testing.T) {
	var captured *api.ChatRequest
	router := gin.New()
	router.Use(ResponsesMiddleware(), captureRequestMiddleware(&captured))
	router.POST("/v1/responses", func(c *gin.Context) { c.Status(http.StatusOK) })

	body := `{
		"model":"test",
		"tools":[{
			"type":"tool_search",
			"execution":"client",
			"description":"Find tools",
			"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
		}],
		"input":[
			{"type":"tool_search_call","id":"ts_1","call_id":"call_search","execution":"client","status":"completed","arguments":{"query":"orders"}},
			{"type":"tool_search_output","id":"tso_1","call_id":"call_search","execution":"client","status":"completed","tools":[
				{"type":"namespace","name":"orders","tools":[
					{"type":"function","name":"lookup_order","description":"Look up an order","parameters":{"type":"object"}}
				]}
			]}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.Code, resp.Body.String())
	}
	if captured == nil {
		t.Fatal("request was not converted")
	}
	if len(captured.Tools) != 1 || captured.Tools[0].Function.Name != "tool_search" {
		t.Fatalf("native tools = %#v", captured.Tools)
	}
	if len(captured.Messages) != 2 {
		t.Fatalf("messages = %#v", captured.Messages)
	}
	if got := captured.Messages[0].ToolCalls[0].Function.Name; got != "tool_search" {
		t.Fatalf("search call name = %q", got)
	}
	if got := captured.Messages[1]; got.Role != "tool" || got.ToolName != "tool_search" || got.Content != `[{"description":"Look up an order","name":"orders.lookup_order","parameters":{"type":"object"},"type":"function"}]` {
		t.Fatalf("search output = %#v", got)
	}
}

func TestResponsesMiddlewareStreamsToolSearchCall(t *testing.T) {
	router := gin.New()
	router.Use(ResponsesMiddleware())
	router.POST("/v1/responses", func(c *gin.Context) {
		c.JSON(http.StatusOK, api.ChatResponse{
			Message: api.Message{ToolCalls: []api.ToolCall{{
				ID: "call_search",
				Function: api.ToolCallFunction{
					Name:      "tool_search",
					Arguments: testArgs(map[string]any{"query": "orders", "limit": 5}),
				},
			}}},
			Done: true,
		})
	})

	body := `{
		"model":"test",
		"stream":true,
		"input":"look up an order",
		"tools":[{
			"type":"tool_search",
			"execution":"client",
			"description":"Find tools",
			"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
		}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.Code, resp.Body.String())
	}
	var sawDone, sawCompleted bool
	for _, frame := range strings.Split(resp.Body.String(), "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
				break
			}
		}
		if data == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatal(err)
		}
		switch event["type"] {
		case "response.output_item.done":
			item := event["item"].(map[string]any)
			if item["type"] != "tool_search_call" || item["call_id"] != "call_search" || item["execution"] != "client" {
				t.Fatalf("item = %#v", item)
			}
			arguments := item["arguments"].(map[string]any)
			if arguments["query"] != "orders" || arguments["limit"] != float64(5) {
				t.Fatalf("arguments = %#v", arguments)
			}
			sawDone = true
		case "response.completed":
			response := event["response"].(map[string]any)
			output := response["output"].([]any)
			if len(output) != 1 || output[0].(map[string]any)["type"] != "tool_search_call" {
				t.Fatalf("completed output = %#v", output)
			}
			sawCompleted = true
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			t.Fatalf("unexpected function-call event: %s", data)
		}
	}
	if !sawDone || !sawCompleted {
		t.Fatalf("saw output_item.done=%v response.completed=%v; body=%s", sawDone, sawCompleted, resp.Body.String())
	}
}
