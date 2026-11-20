package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

const (
	ccBashTool        = `{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}`
	ccToolSearchTool  = `{"name":"ToolSearch","description":"Load deferred tools","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}`
	ccPlaceholderTool = `{"name":"DeferredToolPlaceholder","description":"Reserved placeholder that keeps deferred tool loading active; never call this tool.","input_schema":{"type":"object","properties":{}},"defer_loading":true}`
	ccMonitorTool     = `{"name":"Monitor","description":"Watch a process; prints <lines> & exits","input_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout_ms":{"type":"number"}},"required":["timeout_ms"]},"defer_loading":true}`

	ccMonitorDefinition = `[{"type":"function","name":"Monitor","description":"Watch a process; prints <lines> & exits","parameters":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout_ms":{"type":"number"}},"required":["timeout_ms"]}}]`
)

func ccRequest(t *testing.T, tools, messages string) MessagesRequest {
	t.Helper()
	var req MessagesRequest
	body := `{"model":"glm-5.3:cloud","max_tokens":1024,"stream":true,"tools":[` + tools + `],"messages":[` + messages + `]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	return req
}

func ccPreLoad(t *testing.T) MessagesRequest {
	return ccRequest(t, ccBashTool+","+ccToolSearchTool+","+ccPlaceholderTool,
		`{"role":"user","content":"watch the build"}`)
}

func ccPostLoad(t *testing.T) MessagesRequest {
	return ccRequest(t, ccBashTool+","+ccMonitorTool+","+ccToolSearchTool+","+ccPlaceholderTool,
		`{"role":"user","content":"watch the build"},
		 {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"ToolSearch","input":{"query":"select:Monitor"}}]},
		 {"role":"user","content":[
		   {"type":"tool_result","tool_use_id":"call_1","content":[{"type":"tool_reference","tool_name":"Monitor"}]},
		   {"type":"text","text":"Tool loaded."}]}`)
}

func toolNames(tools api.Tools) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

func toolMessage(t *testing.T, messages []api.Message) api.Message {
	t.Helper()
	for _, m := range messages {
		if m.Role == "tool" {
			return m
		}
	}
	t.Fatalf("no tool message in %+v", messages)
	return api.Message{}
}

func TestFromMessagesRequest_DeferredPlaceholderNotDeclared(t *testing.T) {
	result, err := FromMessagesRequest(ccPreLoad(t))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"Bash", "ToolSearch"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_ToolReferenceExpandsDefinition(t *testing.T) {
	result, err := FromMessagesRequest(ccPostLoad(t))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d: %+v", len(result.Messages), result.Messages)
	}
	tool := result.Messages[2]
	if tool.Role != "tool" || tool.ToolCallID != "call_1" {
		t.Fatalf("expected the ToolSearch result as message 2, got %+v", tool)
	}
	if tool.Content != ccMonitorDefinition {
		t.Errorf("tool result content mismatch\nwant %s\ngot  %s", ccMonitorDefinition, tool.Content)
	}
	if got := result.Messages[3]; got.Role != "user" || got.Content != "Tool loaded." {
		t.Errorf("expected the sibling text as a user message, got %+v", got)
	}
}

func TestFromMessagesRequest_ReferencedDeferredToolDeclaredByDefault(t *testing.T) {
	result, err := FromMessagesRequest(ccPostLoad(t))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"Bash", "Monitor", "ToolSearch"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_ToolReferenceUnknownTool(t *testing.T) {
	req := ccRequest(t, ccToolSearchTool,
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"tool_reference","tool_name":"Nope"}]}]}`)
	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := toolMessage(t, result.Messages).Content; got != "" {
		t.Errorf("unexpected content: %s", got)
	}
	if diff := cmp.Diff([]string{"ToolSearch"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_DeferredToolCalledInHistoryStaysDeclared(t *testing.T) {
	req := ccRequest(t, ccBashTool+","+ccMonitorTool+","+ccToolSearchTool+","+ccPlaceholderTool,
		`{"role":"user","content":"watch the build"},
		 {"role":"assistant","content":[{"type":"tool_use","id":"call_2","name":"Monitor","input":{"timeout_ms":1000}}]},
		 {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_2","content":"done"}]}`)
	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"Bash", "Monitor", "ToolSearch"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_ToolReferenceMixedWithText(t *testing.T) {
	req := ccRequest(t, ccToolSearchTool+","+ccMonitorTool,
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[
		   {"type":"text","text":"Found 1 tool."},{"type":"tool_reference","tool_name":"Monitor"}]}]}`)
	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tool := toolMessage(t, result.Messages)
	if want := "Found 1 tool.\n\n" + ccMonitorDefinition; tool.Content != want {
		t.Errorf("content mismatch\nwant %q\ngot  %q", want, tool.Content)
	}
	if diff := cmp.Diff([]string{"ToolSearch", "Monitor"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_ServerToolSearchKeepsDeferredTools(t *testing.T) {
	req := ccRequest(t, `{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"},`+ccMonitorTool,
		`{"role":"user","content":"hi"}`)
	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"tool_search_tool_regex", "Monitor"}, toolNames(result.Tools)); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

func TestFromMessagesRequest_InvalidDeferredToolSchema(t *testing.T) {
	req := ccPostLoad(t)
	for i := range req.Tools {
		if req.Tools[i].Name == "Monitor" {
			req.Tools[i].InputSchema = json.RawMessage(`{"type":`)
		}
	}
	if _, err := FromMessagesRequest(req); err == nil {
		t.Fatal("expected an error for an invalid input_schema on a referenced tool")
	}
}
