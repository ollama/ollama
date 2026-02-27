package anthropic

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/ollama/ollama/api"
)

// Trace truncation limits.
const (
	TraceMaxStringRunes = 240
	TraceMaxSliceItems  = 8
	TraceMaxMapEntries  = 16
	TraceMaxDepth       = 4
)

// TraceTruncateString shortens s to TraceMaxStringRunes, appending a count of
// omitted characters when truncated.
func TraceTruncateString(s string) string {
	if len(s) == 0 {
		return s
	}
	runes := []rune(s)
	if len(runes) <= TraceMaxStringRunes {
		return s
	}
	return fmt.Sprintf("%s...(+%d chars)", string(runes[:TraceMaxStringRunes]), len(runes)-TraceMaxStringRunes)
}

// TraceJSON round-trips v through JSON and returns a compacted representation.
func TraceJSON(v any) any {
	if v == nil {
		return nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return map[string]any{"marshal_error": err.Error(), "type": fmt.Sprintf("%T", v)}
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return TraceTruncateString(string(data))
	}
	return TraceCompactValue(out, 0)
}

// TraceCompactValue recursively truncates strings, slices, and maps for trace
// output. depth tracks recursion to enforce TraceMaxDepth.
func TraceCompactValue(v any, depth int) any {
	if v == nil {
		return nil
	}
	if depth >= TraceMaxDepth {
		switch t := v.(type) {
		case string:
			return TraceTruncateString(t)
		case []any:
			return fmt.Sprintf("<array len=%d>", len(t))
		case map[string]any:
			return fmt.Sprintf("<object keys=%d>", len(t))
		default:
			return fmt.Sprintf("<%T>", v)
		}
	}
	switch t := v.(type) {
	case string:
		return TraceTruncateString(t)
	case []any:
		limit := min(len(t), TraceMaxSliceItems)
		out := make([]any, 0, limit+1)
		for i := range limit {
			out = append(out, TraceCompactValue(t[i], depth+1))
		}
		if len(t) > limit {
			out = append(out, fmt.Sprintf("... +%d more items", len(t)-limit))
		}
		return out
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		limit := min(len(keys), TraceMaxMapEntries)
		out := make(map[string]any, limit+1)
		for i := range limit {
			out[keys[i]] = TraceCompactValue(t[keys[i]], depth+1)
		}
		if len(keys) > limit {
			out["__truncated_keys"] = len(keys) - limit
		}
		return out
	default:
		return t
	}
}

// ---------------------------------------------------------------------------
// Anthropic request/response tracing
// ---------------------------------------------------------------------------

// TraceMessagesRequest returns a compact trace representation of a MessagesRequest.
func TraceMessagesRequest(r MessagesRequest) map[string]any {
	return map[string]any{
		"model":          r.Model,
		"max_tokens":     r.MaxTokens,
		"messages":       traceMessageParams(r.Messages),
		"system":         traceAnthropicContent(r.System),
		"stream":         r.Stream,
		"tools":          traceTools(r.Tools),
		"tool_choice":    TraceJSON(r.ToolChoice),
		"thinking":       TraceJSON(r.Thinking),
		"stop_sequences": r.StopSequences,
		"temperature":    ptrVal(r.Temperature),
		"top_p":          ptrVal(r.TopP),
		"top_k":          ptrVal(r.TopK),
	}
}

// TraceMessagesResponse returns a compact trace representation of a MessagesResponse.
func TraceMessagesResponse(r MessagesResponse) map[string]any {
	return map[string]any{
		"id":          r.ID,
		"model":       r.Model,
		"content":     TraceJSON(r.Content),
		"stop_reason": r.StopReason,
		"usage":       r.Usage,
	}
}

func traceMessageParams(msgs []MessageParam) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, map[string]any{
			"role":    m.Role,
			"content": traceAnthropicContent(m.Content),
		})
	}
	return out
}

func traceAnthropicContent(content any) any {
	switch c := content.(type) {
	case nil:
		return nil
	case string:
		return TraceTruncateString(c)
	case []any:
		blocks := make([]any, 0, len(c))
		for _, block := range c {
			blockMap, ok := block.(map[string]any)
			if !ok {
				blocks = append(blocks, TraceCompactValue(block, 0))
				continue
			}
			blocks = append(blocks, traceAnthropicBlock(blockMap))
		}
		return blocks
	default:
		return TraceJSON(c)
	}
}

func traceAnthropicBlock(block map[string]any) map[string]any {
	blockType, _ := block["type"].(string)
	out := map[string]any{"type": blockType}
	switch blockType {
	case "text":
		if text, ok := block["text"].(string); ok {
			out["text"] = TraceTruncateString(text)
		} else {
			out["text"] = TraceCompactValue(block["text"], 0)
		}
	case "thinking":
		if thinking, ok := block["thinking"].(string); ok {
			out["thinking"] = TraceTruncateString(thinking)
		} else {
			out["thinking"] = TraceCompactValue(block["thinking"], 0)
		}
	case "tool_use", "server_tool_use":
		out["id"] = block["id"]
		out["name"] = block["name"]
		out["input"] = TraceCompactValue(block["input"], 0)
	case "tool_result", "web_search_tool_result":
		out["tool_use_id"] = block["tool_use_id"]
		out["content"] = TraceCompactValue(block["content"], 0)
	case "image":
		if source, ok := block["source"].(map[string]any); ok {
			out["source"] = map[string]any{
				"type":       source["type"],
				"media_type": source["media_type"],
				"url":        source["url"],
				"data_len":   len(fmt.Sprint(source["data"])),
			}
		}
	default:
		out["block"] = TraceCompactValue(block, 0)
	}
	return out
}

func traceTools(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, TraceTool(t))
	}
	return out
}

// TraceTool returns a compact trace representation of an Anthropic Tool.
func TraceTool(t Tool) map[string]any {
	return map[string]any{
		"type":         t.Type,
		"name":         t.Name,
		"description":  TraceTruncateString(t.Description),
		"input_schema": TraceJSON(t.InputSchema),
		"max_uses":     t.MaxUses,
	}
}

// ContentBlockTypes returns the type strings from content (when it's []any blocks).
func ContentBlockTypes(content any) []string {
	blocks, ok := content.([]any)
	if !ok {
		return nil
	}
	types := make([]string, 0, len(blocks))
	for _, block := range blocks {
		blockMap, ok := block.(map[string]any)
		if !ok {
			types = append(types, fmt.Sprintf("%T", block))
			continue
		}
		t, _ := blockMap["type"].(string)
		types = append(types, t)
	}
	return types
}

func ptrVal[T any](v *T) any {
	if v == nil {
		return nil
	}
	return *v
}

// ---------------------------------------------------------------------------
// Ollama api.* tracing (shared between anthropic and middleware packages)
// ---------------------------------------------------------------------------

// TraceChatRequest returns a compact trace representation of an Ollama ChatRequest.
func TraceChatRequest(req *api.ChatRequest) map[string]any {
	if req == nil {
		return nil
	}
	stream := false
	if req.Stream != nil {
		stream = *req.Stream
	}
	return map[string]any{
		"model":    req.Model,
		"messages": TraceAPIMessages(req.Messages),
		"tools":    TraceAPITools(req.Tools),
		"stream":   stream,
		"options":  req.Options,
		"think":    TraceJSON(req.Think),
	}
}

// TraceChatResponse returns a compact trace representation of an Ollama ChatResponse.
func TraceChatResponse(resp api.ChatResponse) map[string]any {
	return map[string]any{
		"model":       resp.Model,
		"done":        resp.Done,
		"done_reason": resp.DoneReason,
		"message":     TraceAPIMessage(resp.Message),
		"metrics":     TraceJSON(resp.Metrics),
	}
}

// TraceAPIMessages returns compact trace representations for a slice of api.Message.
func TraceAPIMessages(msgs []api.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, TraceAPIMessage(m))
	}
	return out
}

// TraceAPIMessage returns a compact trace representation of a single api.Message.
func TraceAPIMessage(m api.Message) map[string]any {
	return map[string]any{
		"role":         m.Role,
		"content":      TraceTruncateString(m.Content),
		"thinking":     TraceTruncateString(m.Thinking),
		"images":       traceImageSizes(m.Images),
		"tool_calls":   traceToolCalls(m.ToolCalls),
		"tool_name":    m.ToolName,
		"tool_call_id": m.ToolCallID,
	}
}

func traceImageSizes(images []api.ImageData) []int {
	if len(images) == 0 {
		return nil
	}
	sizes := make([]int, 0, len(images))
	for _, img := range images {
		sizes = append(sizes, len(img))
	}
	return sizes
}

// TraceAPITools returns compact trace representations for a slice of api.Tool.
func TraceAPITools(tools api.Tools) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, TraceAPITool(t))
	}
	return out
}

// TraceAPITool returns a compact trace representation of a single api.Tool.
func TraceAPITool(t api.Tool) map[string]any {
	return map[string]any{
		"type":        t.Type,
		"name":        t.Function.Name,
		"description": TraceTruncateString(t.Function.Description),
		"parameters":  TraceJSON(t.Function.Parameters),
	}
}

// TraceToolCall returns a compact trace representation of an api.ToolCall.
func TraceToolCall(tc api.ToolCall) map[string]any {
	return map[string]any{
		"id":   tc.ID,
		"name": tc.Function.Name,
		"args": TraceJSON(tc.Function.Arguments),
	}
}

func traceToolCalls(tcs []api.ToolCall) []map[string]any {
	if len(tcs) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(tcs))
	for _, tc := range tcs {
		out = append(out, TraceToolCall(tc))
	}
	return out
}
