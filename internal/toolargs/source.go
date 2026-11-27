// Package toolargs preserves JSON number spelling when rendering tool calls.
package toolargs

import (
	"bytes"
	"encoding/json"
	"iter"
)

// State holds the original JSON values for arguments decoded by the API.
// The API embeds it privately so its sourceJSON method is available to RenderValues
// without adding a method to ToolCallFunctionArguments' public Go API.
type State struct {
	raw map[string]json.RawMessage
}

func Decode(data []byte) (State, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return State{}, err
	}
	if len(raw) == 0 {
		return State{}, nil
	}
	return State{raw: raw}, nil
}

func Invalidate(state *State, key string) {
	delete(state.raw, key)
}

func (state State) sourceJSON(key string) (json.RawMessage, bool) {
	raw, ok := state.raw[key]
	return raw, ok
}

type arguments interface {
	All() iter.Seq2[string, any]
	sourceJSON(string) (json.RawMessage, bool)
}

// The API's private embedding supplies this half of the renderer contract.
var _ interface {
	sourceJSON(string) (json.RawMessage, bool)
} = State{}

// RenderValues returns renderer values with the original spelling of JSON numbers.
// Programmatically supplied numbers keep their Go types.
func RenderValues(args arguments) iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for key, value := range args.All() {
			if raw, ok := args.sourceJSON(key); ok {
				switch value.(type) {
				case float64:
					value = json.Number(raw)
				case map[string]any, []any:
					dec := json.NewDecoder(bytes.NewReader(raw))
					dec.UseNumber()
					var original any
					if dec.Decode(&original) == nil {
						// Normalize both original and newly supplied numbers to
						// json.Number, as the JSON input path does.
						if encoded, err := json.Marshal(restoreNumbers(value, original)); err == nil {
							dec = json.NewDecoder(bytes.NewReader(encoded))
							dec.UseNumber()
							var restored any
							if dec.Decode(&restored) == nil {
								value = restored
							}
						}
					}
				}
			}
			if !yield(key, value) {
				return
			}
		}
	}
}

// A caller can mutate a map or slice returned by Get. Restore source numbers
// only where the decoded value has not changed; leave new values as supplied.
func restoreNumbers(current, original any) any {
	switch original := original.(type) {
	case json.Number:
		if number, ok := current.(float64); ok {
			if decoded, err := original.Float64(); err == nil && number == decoded {
				return original
			}
		}
	case map[string]any:
		if values, ok := current.(map[string]any); ok {
			result := make(map[string]any, len(values))
			for key, value := range values {
				result[key] = restoreNumbers(value, original[key])
			}
			return result
		}
	case []any:
		if values, ok := current.([]any); ok {
			result := make([]any, len(values))
			for i, value := range values {
				if i < len(original) {
					result[i] = restoreNumbers(value, original[i])
				} else {
					result[i] = value
				}
			}
			return result
		}
	}
	return current
}
