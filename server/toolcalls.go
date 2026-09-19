package server

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// completeToolCalls drops tool calls the model never finished writing.
//
// A response that ends on the token limit can stop in the middle of a tool
// call's arguments. The fields that did arrive parse cleanly, so the call
// reaches the caller looking complete while a required argument is simply
// absent — a coding agent then rejects it with a schema error that names a
// field the model was still on its way to producing, and the reason (the
// response was cut short) is nowhere in sight.
//
// done_reason already says "length", but a tool call is the part of a response
// callers act on, and acting on half a tool call is worse than not seeing one.
// So an unfinished call is dropped rather than handed over, and only when the
// generation actually stopped at the limit: a call missing a required argument
// in a response that ended normally is the model's own error to report, not
// something to hide.
func completeToolCalls(toolCalls []api.ToolCall, tools []api.Tool, truncated bool) []api.ToolCall {
	if !truncated || len(toolCalls) == 0 {
		return toolCalls
	}

	required := make(map[string][]string, len(tools))
	for _, tool := range tools {
		required[tool.Function.Name] = tool.Function.Parameters.Required
	}

	kept := toolCalls[:0]
	for _, call := range toolCalls {
		if missing := missingArguments(call, required[call.Function.Name]); missing != "" {
			slog.Warn("dropping a tool call the model did not finish writing",
				"tool", call.Function.Name, "missing", missing, "done_reason", "length")
			continue
		}
		slog.Log(context.TODO(), logutil.LevelTrace, "keeping a complete tool call from a truncated response", "tool", call.Function.Name)
		kept = append(kept, call)
	}
	return kept
}

// missingArguments returns the first required argument the call does not carry,
// or the empty string when it carries all of them.
func missingArguments(call api.ToolCall, required []string) string {
	for _, name := range required {
		if _, ok := call.Function.Arguments.Get(name); !ok {
			return name
		}
	}
	return ""
}
