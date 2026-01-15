package openai

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

const rawArgumentsKey = "__raw_arguments"

func parseToolCallArguments(argStr string) (api.ToolCallFunctionArguments, bool) {
	dec := json.NewDecoder(strings.NewReader(argStr))
	dec.UseNumber()

	var v any
	if err := dec.Decode(&v); err != nil {
		// Try when the argument itself is a JSON-encoded string.
		var s string
		if err := json.Unmarshal([]byte(argStr), &s); err != nil {
			return nil, false
		}
		dec = json.NewDecoder(strings.NewReader(s))
		dec.UseNumber()
		if err := dec.Decode(&v); err != nil {
			return nil, false
		}
	}

	switch val := v.(type) {
	case map[string]any:
		return api.ToolCallFunctionArguments(val), true
	case nil:
		return api.ToolCallFunctionArguments{}, true
	default:
		return nil, false
	}
}
