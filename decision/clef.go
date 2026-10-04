package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/llm"
)

// encodeClef preserves the reference encoder's JSON and segment boundaries,
// retaining its choice order for the shared answer decoder.
func encodeClef(req Request, c *Compiled) error {
	state, err := clefContent(req.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	input := llm.ScoreRequest{Images: req.Images, ImagePosition: 1}
	add := func(text string) [2]int {
		n := len(input.Segments)
		input.Segments = append(input.Segments, text)
		return [2]int{n, n + 1}
	}
	add("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")
	add(state)
	add("\n\nSCHEMA FIELDS:\n")
	for name, q := range req.Questions.All() {
		f, err := clefField(name, q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		add(fmt.Sprintf("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", len(c.fields)+1, name, q.Type))
		field := llm.ScoreField{Question: add(f.Description)}
		switch q.Type {
		case "choice":
			field.Type = 1
		case "score":
			field.Type = 2
		}
		add("\nALLOWED OPTIONS:\n")
		for j, option := range f.Choices {
			add(fmt.Sprintf("OPTION %d: ", j+1))
			semantics := map[string]any{"option_id": fmt.Sprint(option.Value)}
			if option.Description != nil {
				semantics["description"] = option.Description
			}
			raw, err := clefJSON(semantics)
			if err != nil {
				return err
			}
			field.Options = append(field.Options, add(raw))
			add("\n")
		}
		add("END FIELD\n")
		input.Fields = append(input.Fields, field)
		c.fields = append(c.fields, f)
	}
	add("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
	c.Request = input
	return nil
}

func clefField(name string, q Question) (compiledField, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, fmt.Errorf("field name must not be empty")
	}
	f.Description = name
	if raw := bytes.TrimSpace(q.Instructions); len(raw) > 0 && string(raw) != "null" && string(raw) != `""` {
		var err error
		f.Description, err = clefContent(raw)
		if err != nil {
			return f, fmt.Errorf("instructions: %w", err)
		}
	}
	var criteria any
	if len(q.Criteria) > 0 {
		var err error
		criteria, err = clefValue(q.Criteria)
		if err != nil {
			return f, fmt.Errorf("criteria: %w", err)
		}
	}
	add := func(value, description any) {
		f.Choices = append(f.Choices, Choice{Value: value, Description: description})
	}
	switch q.Type {
	case "noul":
		descriptions := map[string]any{"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
		if criteria != nil {
			values, ok := criteria.(map[string]any)
			if !ok {
				return f, fmt.Errorf("noul criteria must be an object")
			}
			for key, value := range values {
				if key != "true" && key != "false" {
					return f, fmt.Errorf("unknown noul criterion %q", key)
				}
				descriptions[key] = value
			}
		}
		add(true, descriptions["true"])
		add(false, descriptions["false"])
	case "choice":
		values, ok := criteria.(map[string]any)
		if !ok {
			return f, fmt.Errorf("choice criteria must be an object")
		}
		for _, key := range slices.Sorted(maps.Keys(values)) {
			if strings.TrimSpace(key) == "" {
				return f, fmt.Errorf("choice keys must not be empty")
			}
			add(key, values[key])
		}
	case "score":
		values, ok := criteria.([]any)
		if !ok {
			return f, fmt.Errorf("score criteria must be an array")
		}
		for i, value := range values {
			add(strconv.Itoa(i), value)
		}
	default:
		return f, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(f.Choices) < 2 || len(f.Choices) > 26 {
		return f, fmt.Errorf("criteria must contain 2–26 candidates")
	}
	return f, nil
}

func clefValue(raw json.RawMessage) (any, error) {
	if !json.Valid(raw) {
		return nil, fmt.Errorf("must be a JSON value")
	}
	var value any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	err := dec.Decode(&value)
	return value, err
}

func clefContent(raw json.RawMessage) (string, error) {
	value, err := clefValue(raw)
	if err != nil {
		return "", err
	}
	if s, ok := value.(string); ok {
		return s, nil
	}
	return clefJSON(value)
}

// clefJSON matches Python's json.dumps(ensure_ascii=False, sort_keys=True,
// separators=(",", ":")), including its integer/float distinction.
// encoding/json sorts map keys, but its number and Unicode escaping rules
// differ from the reference and would change the model's input tokens.
func clefJSON(value any) (string, error) {
	switch v := value.(type) {
	case nil:
		return "null", nil
	case bool:
		return strconv.FormatBool(v), nil
	case string:
		var b strings.Builder
		b.WriteByte('"')
		for _, r := range v {
			switch r {
			case '"', '\\':
				b.WriteByte('\\')
				b.WriteRune(r)
			case '\b':
				b.WriteString(`\b`)
			case '\f':
				b.WriteString(`\f`)
			case '\n':
				b.WriteString(`\n`)
			case '\r':
				b.WriteString(`\r`)
			case '\t':
				b.WriteString(`\t`)
			default:
				if r < 0x20 {
					fmt.Fprintf(&b, `\u%04x`, r)
				} else {
					b.WriteRune(r)
				}
			}
		}
		b.WriteByte('"')
		return b.String(), nil
	case json.Number:
		if !strings.ContainsAny(string(v), ".eE") {
			if v == "-0" {
				return "0", nil
			}
			return string(v), nil
		}
		n, err := v.Float64()
		if math.IsInf(n, 0) {
			if n < 0 {
				return "-Infinity", nil
			}
			return "Infinity", nil
		}
		if err != nil {
			return "", err
		}
		if a := math.Abs(n); a >= 1e16 || (a != 0 && a < 1e-4) {
			return strconv.FormatFloat(n, 'e', -1, 64), nil
		}
		s := strconv.FormatFloat(n, 'f', -1, 64)
		if !strings.ContainsRune(s, '.') {
			s += ".0"
		}
		return s, nil
	case []any:
		parts := make([]string, len(v))
		for i, child := range v {
			text, err := clefJSON(child)
			if err != nil {
				return "", err
			}
			parts[i] = text
		}
		return "[" + strings.Join(parts, ",") + "]", nil
	case map[string]any:
		parts := make([]string, 0, len(v))
		for _, key := range slices.Sorted(maps.Keys(v)) {
			name, _ := clefJSON(key)
			text, err := clefJSON(v[key])
			if err != nil {
				return "", err
			}
			parts = append(parts, name+":"+text)
		}
		return "{" + strings.Join(parts, ",") + "}", nil
	default:
		return "", fmt.Errorf("unsupported JSON value %T", value)
	}
}
