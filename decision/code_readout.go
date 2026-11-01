package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/model/renderers"
)

// codeReadoutField compiles a question for a positional code readout. Choice
// order assigns answer codes, and optional labels use the readout format's
// truthiness defaults. Encoders apply their own additional limits afterward.
func codeReadoutField(name string, q Question) (compiledField, []string, error) {
	field := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return field, nil, fmt.Errorf("field name must not be empty")
	}
	field.Description = "Choose the best matching option."
	if raw := bytes.TrimSpace(q.Instructions); !codeReadoutFalsy(raw) {
		text, err := codeReadoutDescribe(raw)
		if err != nil {
			return field, nil, fmt.Errorf("instructions: %w", err)
		}
		field.Description = text
	}
	var descriptions []string
	add := func(value, description any, prompt string) {
		field.Choices = append(field.Choices, Choice{Value: value, Description: description})
		descriptions = append(descriptions, prompt)
	}
	switch q.Type {
	case "choice":
		criteria := orderedmap.New[string, json.RawMessage]()
		if err := json.Unmarshal(q.Criteria, criteria); err != nil {
			return field, nil, fmt.Errorf("choice criteria must be an object")
		}
		for key, raw := range criteria.All() {
			if strings.TrimSpace(key) == "" {
				return field, nil, fmt.Errorf("choice key must not be empty")
			}
			var value any
			if err := json.Unmarshal(raw, &value); err != nil {
				return field, nil, err
			}
			prompt := key
			if value != nil {
				text, err := codeReadoutDescribe(raw)
				if err != nil {
					return field, nil, err
				}
				prompt += ": " + text
			}
			add(key, value, prompt)
		}
	case "noul":
		criteria := orderedmap.New[string, json.RawMessage]()
		if !codeReadoutFalsy(q.Criteria) {
			if err := json.Unmarshal(q.Criteria, criteria); err != nil {
				return field, nil, fmt.Errorf("noul criteria must be an object")
			}
		}
		for _, item := range []struct {
			key, fallback string
			value         bool
		}{{"false", "No / false", false}, {"true", "Yes / true", true}} {
			description := item.fallback
			if raw, ok := criteria.Get(item.key); ok && !codeReadoutFalsy(raw) {
				var err error
				description, err = codeReadoutDescribe(raw)
				if err != nil {
					return field, nil, err
				}
			}
			add(item.value, description, description)
		}
		for key := range criteria.All() {
			if key != "false" && key != "true" {
				return field, nil, fmt.Errorf("unknown noul criterion %q", key)
			}
		}
	case "score":
		var criteria []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return field, nil, fmt.Errorf("score criteria must be an array")
		}
		for i, raw := range criteria {
			text, err := codeReadoutDescribe(raw)
			if err != nil {
				return field, nil, err
			}
			var value any
			if err := json.Unmarshal(raw, &value); err != nil {
				return field, nil, err
			}
			add(strconv.Itoa(i), value, text)
		}
	default:
		return field, nil, fmt.Errorf("type must be choice, noul, or score")
	}
	if q.Type == "score" && len(field.Choices) < 2 {
		return field, nil, fmt.Errorf("score criteria must contain at least two levels")
	}
	if len(field.Choices) < 1 || len(field.Choices) > 255 || (q.Type == "noul" && len(field.Choices) != 2) {
		return field, nil, fmt.Errorf("criteria must contain 1–255 options")
	}
	return field, descriptions, nil
}

func codeReadoutDescribe(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || !json.Valid(raw) {
		return "", fmt.Errorf("invalid JSON value")
	}
	if raw[0] == '"' {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return "", err
		}
		return value, nil
	}
	return renderers.PythonJSON(raw, 0)
}

// The publisher formats use value truthiness for optional instructions and
// yes/no descriptions. JSON has no truthiness rule, so map its value types here.
func codeReadoutFalsy(raw json.RawMessage) bool {
	if len(bytes.TrimSpace(raw)) == 0 {
		return true
	}
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return false
	}
	switch value := value.(type) {
	case nil:
		return true
	case bool:
		return !value
	case float64:
		return value == 0
	case string:
		return value == ""
	case []any:
		return len(value) == 0
	case map[string]any:
		return len(value) == 0
	}
	return false
}
