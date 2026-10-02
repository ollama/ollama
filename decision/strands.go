package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
)

// Strands Decider's pointer head reads each question independently. Its prompt
// and option boundaries follow strands_decider/prompting.py (Apache-2.0).
func encodeStrands(req Request, c *Compiled) error {
	state, err := strandsContent(req.State)
	if err != nil || strings.TrimSpace(state) == "" {
		return fmt.Errorf("state must be a nonempty string, object, or array")
	}
	for name, q := range req.Questions.All() {
		f, err := strandsField(name, q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		header, typ := "Decide whether the statement is true of the state.", 0
		if q.Type == "choice" {
			header, typ = "Select exactly one option.", 1
		} else if q.Type == "score" {
			header, typ = "Rate the state against the ordered levels below (lowest first).", 2
		}
		var prompt strings.Builder
		fmt.Fprintf(&prompt, "<question type=\"%s\">\n%s\n%s\n<options>\n", q.Type, header, f.Description)
		row := llm.ScorePointerRow{Prefix: "<state>\n" + state + "\n</state>\n", Type: typ}
		for i, option := range f.Choices {
			start := prompt.Len()
			fmt.Fprintf(&prompt, "%d. %v", i+1, option.Value)
			description := strings.Join(strings.Fields(option.Description.(string)), " ")
			if description != "" {
				prompt.WriteString(" — " + description)
			}
			row.Options = append(row.Options, [2]int{start, prompt.Len()})
			prompt.WriteByte('\n')
		}
		prompt.WriteString("</options>\n</question>\n<answer>")
		row.Prompt = prompt.String()
		c.Request.PointerRows = append(c.Request.PointerRows, row)
		c.fields = append(c.fields, f)
	}
	return nil
}

func strandsField(name string, q Question) (compiledField, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, fmt.Errorf("field name must not be empty")
	}
	var err error
	f.Description, err = strandsContent(q.Instructions)
	if err != nil || strings.TrimSpace(f.Description) == "" {
		return f, fmt.Errorf("instructions must be a nonempty string, object, or array")
	}
	add := func(value any, description string) {
		f.Choices = append(f.Choices, Choice{Value: value, Description: strings.TrimSpace(description)})
	}
	limit := 255
	switch q.Type {
	case "noul":
		criteria := map[string]string{
			"false": "the statement does not hold for this state",
			"true":  "the statement holds for this state",
		}
		if raw := bytes.TrimSpace(q.Criteria); len(raw) > 0 && string(raw) != "null" {
			var values map[string]*string
			if err := json.Unmarshal(raw, &values); err != nil {
				return f, fmt.Errorf("noul criteria must be an object of true/false descriptions")
			}
			for k, v := range values {
				if (k != "true" && k != "false") || v == nil {
					return f, fmt.Errorf("noul criteria must contain only true/false string descriptions")
				}
				criteria[k] = *v
			}
		}
		add(false, criteria["false"])
		add(true, criteria["true"])
	case "choice":
		criteria := orderedmap.New[string, *string]()
		if err := json.Unmarshal(q.Criteria, criteria); err != nil {
			return f, fmt.Errorf("choice criteria must map option keys to string descriptions")
		}
		for k, v := range criteria.All() {
			if strings.TrimSpace(k) == "" {
				return f, fmt.Errorf("choice keys must not be empty")
			}
			if v == nil {
				return f, fmt.Errorf("choice descriptions must be strings")
			}
			add(k, *v)
		}
	case "score":
		limit = 10
		var criteria []*string
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return f, fmt.Errorf("score criteria must be an array of descriptions")
		}
		for i, v := range criteria {
			if v == nil {
				return f, fmt.Errorf("score descriptions must be strings")
			}
			add(strconv.Itoa(i), *v)
		}
	default:
		return f, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(f.Choices) < 2 || len(f.Choices) > limit {
		return f, fmt.Errorf("criteria must contain 2–%d candidates", limit)
	}
	return f, nil
}

// Match json.dumps(indent=2, ensure_ascii=False, sort_keys=False), preserving
// object order and Python's number formatting as well as literal Unicode.
func strandsContent(raw json.RawMessage) (string, error) {
	if _, err := content(raw); err != nil {
		return "", err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var render func(int) (string, error)
	render = func(depth int) (string, error) {
		token, err := dec.Token()
		if err != nil {
			return "", err
		}
		if delim, ok := token.(json.Delim); ok {
			var parts []string
			for dec.More() {
				prefix := strings.Repeat("  ", depth+1)
				if delim == '{' {
					key, err := dec.Token()
					if err != nil {
						return "", err
					}
					quoted, _ := clefJSON(key)
					prefix += quoted + ": "
				}
				value, err := render(depth + 1)
				if err != nil {
					return "", err
				}
				parts = append(parts, prefix+value)
			}
			end, err := dec.Token()
			if err != nil {
				return "", err
			}
			if len(parts) == 0 {
				return fmt.Sprint(delim.String(), end), nil
			}
			return delim.String() + "\n" + strings.Join(parts, ",\n") + "\n" + strings.Repeat("  ", depth) + fmt.Sprint(end), nil
		}
		if depth == 0 {
			return strings.TrimSpace(token.(string)), nil
		}
		return clefJSON(token)
	}
	return render(0)
}

func strandsConfidence(p []float64, typ string, smoothing float64) float64 {
	n := float64(len(p))
	if typ == "choice" {
		return max(0, min(1, (n*slices.Max(p)-1)/(n-1)))
	}
	var mean, variance float64
	for i, v := range p {
		mean += float64(i) * v
	}
	for i, v := range p {
		variance += v * math.Pow(float64(i)-mean, 2)
	}
	sigma, ceiling, floor := math.Sqrt(variance), (n-1)/2, math.Sqrt(max(0, smoothing))
	if ceiling <= floor {
		if sigma <= floor {
			return 1
		}
		return 0
	}
	return max(0, min(1, (ceiling-sigma)/(ceiling-floor)))
}
