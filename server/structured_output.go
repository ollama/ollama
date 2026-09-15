package server

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

// mlxThinkingTags returns the delimiters used by an MLX model's reasoning
// output. Renderer-backed models do not necessarily carry a Go template, so
// the parser name is also used for the Qwen family that uses <think> tags.
func mlxThinkingTags(m *Model, openingTag, closingTag string) (string, string) {
	if openingTag != "" && closingTag != "" {
		return openingTag, closingTag
	}
	if m == nil {
		return "", ""
	}

	switch m.Config.Parser {
	case "qwen3", "qwen3-thinking", "qwen3.5", "ornith", "qwen3-vl-thinking":
		return "<think>", "</think>"
	default:
		return "", ""
	}
}

// mlxStructuredOutputFormat builds the single-pass grammar used by MLX when
// thinking and structured output are requested together. The prompt normally
// ends with the opening thinking tag, so the generated output starts with
// unconstrained thinking text, then the closing tag, then the schema-constrained
// response. When the prompt did not prefill the opening tag, include the whole
// tag in the grammar instead.
func mlxStructuredOutputFormat(format json.RawMessage, prompt, openingTag, closingTag string) json.RawMessage {
	if len(format) == 0 || string(format) == "null" || string(format) == `""` || openingTag == "" || closingTag == "" {
		return nil
	}

	// A caller that already supplied a structural tag should not get wrapped a
	// second time. The public API normally supplies a JSON Schema, but keeping
	// this path idempotent makes the MLX boundary safe for future callers.
	var envelope struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(format, &envelope) == nil && envelope.Type == "structural_tag" {
		return append(json.RawMessage(nil), format...)
	}

	schema := format
	if string(schema) == `"json"` {
		schema = json.RawMessage(`{"type":"object"}`)
	}

	jsonSchema := map[string]any{
		"type":        "json_schema",
		"json_schema": schema,
	}
	var elements []any
	if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
		elements = []any{
			map[string]any{
				"type":     "any_text",
				"excludes": []string{closingTag},
			},
			map[string]any{
				"type":  "const_string",
				"value": closingTag,
			},
			jsonSchema,
		}
	} else {
		elements = []any{
			map[string]any{
				"type":  "tag",
				"begin": openingTag,
				"content": map[string]any{
					"type": "any_text",
				},
				"end": closingTag,
			},
			jsonSchema,
		}
	}

	grammar := map[string]any{
		"type": "structural_tag",
		"format": map[string]any{
			"type":     "sequence",
			"elements": elements,
		},
	}
	encoded, err := json.Marshal(grammar)
	if err != nil {
		// All values above are JSON-native, so this is defensive. Returning nil
		// makes the caller retain the existing two-pass behavior.
		return nil
	}
	return encoded
}

// mlxSinglePassFormat returns the structural grammar when MLX can handle the
// complete thinking-to-content transition itself. A nil result means the
// caller should keep the existing restart-based path.
func mlxSinglePassFormat(m *Model, format json.RawMessage, prompt, openingTag, closingTag string, think *api.ThinkValue) json.RawMessage {
	if m == nil || !m.IsMLX() || len(format) == 0 || think == nil || !think.Bool() {
		return nil
	}
	openingTag, closingTag = mlxThinkingTags(m, openingTag, closingTag)
	return mlxStructuredOutputFormat(format, prompt, openingTag, closingTag)
}
