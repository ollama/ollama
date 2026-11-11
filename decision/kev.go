package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
)

func validateKevLimits(req Request) error {
	if req.Questions == nil || req.Questions.Len() < 1 || req.Questions.Len() > 64 {
		return fmt.Errorf("questions must contain 1–64 fields")
	}
	if len(req.Images) != 0 || len(req.Videos) != 0 {
		return fmt.Errorf("Kev accepts text state only")
	}
	if len(req.State) == 0 || !json.Valid(req.State) {
		return fmt.Errorf("state must be a JSON value")
	}
	return nil
}

func encodeKev(req Request, c *Compiled) error {
	state, err := kevContent(req.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	c.Request.State = state
	for name, question := range req.Questions.All() {
		field, err := kevField(name, question)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		c.fields = append(c.fields, field)
		c.Request.Rows = append(c.Request.Rows, llm.ScoreRow{Question: &llm.ScoreQuestion{
			Type: question.Type, Instructions: field.Description, Options: field.options,
		}})
	}
	return nil
}

func kevField(name string, question Question) (compiledField, error) {
	field := compiledField{Field: Field{Name: name}, typ: question.Type}
	if strings.TrimSpace(name) == "" {
		return field, fmt.Errorf("field name must not be empty")
	}
	if len(question.Instructions) != 0 {
		var err error
		field.Description, err = kevContent(question.Instructions)
		if err != nil {
			return field, fmt.Errorf("instructions: %w", err)
		}
	}
	add := func(value any, description, option string) {
		field.Choices = append(field.Choices, Choice{Value: value, Description: description})
		field.options = append(field.options, option)
	}
	switch question.Type {
	case "noul":
		criteria := orderedmap.New[string, json.RawMessage]()
		if len(question.Criteria) > 0 && string(question.Criteria) != "null" {
			if err := json.Unmarshal(question.Criteria, criteria); err != nil {
				return field, fmt.Errorf("noul criteria must be an object")
			}
		}
		for _, item := range []struct {
			key   string
			label string
			value bool
		}{{"false", "no", false}, {"true", "yes", true}} {
			description := ""
			if raw, ok := criteria.Get(item.key); ok {
				var err error
				description, err = kevContent(raw)
				if err != nil {
					return field, err
				}
			}
			option := item.label
			if description != "" {
				option += ": " + description
			}
			add(item.value, description, option)
		}
		for key := range criteria.All() {
			if key != "false" && key != "true" {
				return field, fmt.Errorf("unknown noul criterion %q", key)
			}
		}
	case "choice":
		criteria := orderedmap.New[string, json.RawMessage]()
		if err := json.Unmarshal(question.Criteria, criteria); err != nil {
			return field, fmt.Errorf("choice criteria must be an object")
		}
		for key, raw := range criteria.All() {
			if strings.TrimSpace(key) == "" {
				return field, fmt.Errorf("choice keys must not be empty")
			}
			description, err := kevContent(raw)
			if err != nil {
				return field, err
			}
			option := key
			if description != "" {
				option += ": " + description
			}
			add(key, description, option)
		}
	case "score":
		var criteria []json.RawMessage
		if err := json.Unmarshal(question.Criteria, &criteria); err != nil {
			return field, fmt.Errorf("score criteria must be an array")
		}
		for index, raw := range criteria {
			text, err := kevContent(raw)
			if err != nil {
				return field, err
			}
			add(strconv.Itoa(index), text, text)
		}
	default:
		return field, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(field.options) < 1 || len(field.options) > 255 || (question.Type == "noul" && len(field.options) != 2) {
		return field, fmt.Errorf("criteria must contain 1–255 options")
	}
	return field, nil
}

// kevContent renders JSON values as the labels and indented lines seen during
// Kev training. A streaming decoder preserves object key order.
func kevContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	if !json.Valid(raw) {
		return "", fmt.Errorf("invalid JSON value")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	value, _, err := kevValue(dec, 0)
	if err != nil {
		return "", err
	}
	if _, err = dec.Token(); err != io.EOF {
		return "", fmt.Errorf("extra JSON input")
	}
	return value, nil
}

func kevValue(dec *json.Decoder, depth int) (string, bool, error) {
	if depth > 64 {
		return "", false, fmt.Errorf("content nesting exceeds 64 levels")
	}
	token, err := dec.Token()
	if err != nil {
		return "", false, err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return kevScalar(token), false, nil
	}
	pad := strings.Repeat("  ", depth)
	var lines []string
	switch delim {
	case '{':
		for dec.More() {
			key, err := dec.Token()
			if err != nil {
				return "", false, err
			}
			value, nested, err := kevValue(dec, depth+1)
			if err != nil {
				return "", false, err
			}
			separator := ": "
			if nested {
				separator = ":\n"
			}
			lines = append(lines, pad+key.(string)+separator+value)
		}
	case '[':
		for dec.More() {
			value, _, err := kevValue(dec, depth+1)
			if err != nil {
				return "", false, err
			}
			lines = append(lines, pad+"- "+strings.TrimLeftFunc(value, unicode.IsSpace))
		}
	default:
		return "", false, fmt.Errorf("unexpected JSON delimiter %q", delim)
	}
	if _, err := dec.Token(); err != nil {
		return "", false, err
	}
	return strings.Join(lines, "\n"), true, nil
}

func kevScalar(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case bool:
		if v {
			return "True"
		}
		return "False"
	case string:
		return v
	case json.Number:
		if !strings.ContainsAny(string(v), ".eE") {
			return string(v)
		}
		f, err := strconv.ParseFloat(string(v), 64)
		if err != nil {
			return string(v)
		}
		if f == 0 || math.Abs(f) >= 1e-4 && math.Abs(f) < 1e16 {
			out := strconv.FormatFloat(f, 'f', -1, 64)
			if !strings.Contains(out, ".") {
				out += ".0"
			}
			return out
		}
		return strconv.FormatFloat(f, 'e', -1, 64)
	default:
		return fmt.Sprint(v)
	}
}

func roundKev(value float64) float64 { return math.RoundToEven(value*1e4) / 1e4 }

func kevConfidence(probabilities []float64, kind string) float64 {
	if len(probabilities) == 1 {
		return 1
	}
	mode := 0
	for i := 1; i < len(probabilities); i++ {
		if probabilities[i] > probabilities[mode] {
			mode = i
		}
	}
	if kind == "choice" {
		uniform := 1 / float64(len(probabilities))
		return (probabilities[mode] - uniform) / (1 - uniform)
	}
	center := float64(len(probabilities)-1) / 2
	var spread, deviation float64
	for i, p := range probabilities {
		spread += p * math.Abs(float64(i-mode))
		deviation += math.Abs(float64(i) - center)
	}
	return max(0, 1-spread/(deviation/float64(len(probabilities))))
}
