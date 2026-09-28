// Package decision compiles typed decision requests into candidate-scoring
// prompts and converts the resulting logits into /v1/systemone responses.
package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
)

type compiledField struct {
	Field
	typ string
}

type Compiled struct {
	Request  llm.ScoreRequest
	fields   []compiledField
	messages [][]api.Message
}

func Compile(req Request) (*Compiled, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if req.Questions.Len() < 1 || req.Questions.Len() > 64 {
		return nil, fmt.Errorf("questions must contain 1–64 fields")
	}
	context, err := content(req.State)
	if err != nil {
		return nil, fmt.Errorf("state: %w", err)
	}
	if strings.TrimSpace(context) == "" {
		return nil, fmt.Errorf("state must not be empty")
	}
	c := &Compiled{}
	for name, q := range req.Questions.All() {
		f, err := compileField(name, q)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", name, err)
		}
		c.fields = append(c.fields, f)
	}
	data, err := json.Marshal(struct {
		Context string          `json:"context"`
		Schema  []compiledField `json:"schema"`
	}{context, c.fields})
	if err != nil {
		return nil, err
	}
	for _, f := range c.fields {
		name, err := json.Marshal(f.Name)
		if err != nil {
			return nil, err
		}
		c.messages = append(c.messages, []api.Message{
			{Role: "user", Content: string(data) + "\n\nRequested field: " + string(name)},
		})
		var row llm.ScoreRow
		for _, choice := range f.Choices {
			row.Candidates = append(row.Candidates, choice.Code)
		}
		c.Request.Rows = append(c.Request.Rows, row)
	}
	return c, nil
}

// Render prepares scoring prompts using the model's system prompt and chat template.
// The caller must set Request.MaxTokens to the loaded context size before scoring.
func (c *Compiled) Render(render func([]api.Message) (string, error)) error {
	for i, messages := range c.messages {
		prompt, err := render(messages)
		if err != nil {
			return err
		}
		c.Request.Rows[i].Prompt = prompt
	}
	return nil
}

func content(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return "", fmt.Errorf("must be a string, object, or array")
	}
	switch raw[0] {
	case '"':
		var s string
		err := json.Unmarshal(raw, &s)
		return s, err
	case '{', '[':
		var b bytes.Buffer
		err := json.Compact(&b, raw)
		return b.String(), err
	default:
		return "", fmt.Errorf("must be a string, object, or array")
	}
}

func compileField(name string, q Question) (compiledField, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, fmt.Errorf("field name must not be empty")
	}
	description, err := content(q.Instructions)
	if err != nil || strings.TrimSpace(description) == "" {
		return f, fmt.Errorf("instructions must be a nonempty string, object, or array")
	}
	f.Description = description
	add := func(value any, description string) {
		f.Choices = append(f.Choices, Choice{string(rune('A' + len(f.Choices))), value, description})
	}
	switch q.Type {
	case "noul":
		criteria := orderedmap.New[string, *string]()
		if len(q.Criteria) > 0 {
			if err := json.Unmarshal(q.Criteria, criteria); err != nil || string(q.Criteria) == "null" {
				return f, fmt.Errorf("noul criteria must be an object of true/false descriptions")
			}
		}
		no, yes := "No", "Yes"
		for key, value := range criteria.All() {
			if value == nil {
				return f, fmt.Errorf("noul descriptions must be strings")
			}
			switch key {
			case "false":
				no = *value
			case "true":
				yes = *value
			default:
				return f, fmt.Errorf("unknown noul criterion %q", key)
			}
		}
		add(false, no)
		add(true, yes)
	case "choice":
		criteria := orderedmap.New[string, *string]()
		if err := json.Unmarshal(q.Criteria, criteria); err != nil {
			return f, fmt.Errorf("choice criteria must map option keys to descriptions or null")
		}
		for key, value := range criteria.All() {
			if strings.TrimSpace(key) == "" {
				return f, fmt.Errorf("choice keys must not be empty")
			}
			description := key
			if value != nil {
				description = *value
			}
			add(key, description)
		}
	case "score":
		var criteria []*string
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return f, fmt.Errorf("score criteria must be an array of descriptions")
		}
		for i, description := range criteria {
			if description == nil {
				return f, fmt.Errorf("score descriptions must be strings")
			}
			add(strconv.Itoa(i), *description)
		}
	default:
		return f, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(f.Choices) < 2 || len(f.Choices) > 26 {
		return f, fmt.Errorf("criteria must contain 2–26 candidates")
	}
	return f, nil
}

func (c *Compiled) Answer(model string, result llm.ScoreResponse) (Response, error) {
	response := Response{Model: model, Answers: &Answers{}, Usage: Usage{InputTokens: result.InputTokens, OutputTokens: result.OutputTokens}}
	if len(result.Logits) != len(c.fields) {
		return response, fmt.Errorf("scorer returned %d rows for %d questions", len(result.Logits), len(c.fields))
	}
	for i, f := range c.fields {
		logits := result.Logits[i]
		if len(logits) != len(f.Choices) {
			return response, fmt.Errorf("scorer returned the wrong number of candidates for %q", f.Name)
		}
		peak := slices.Max(logits)
		p := make([]float64, len(logits))
		var sum float64
		for j, logit := range logits {
			if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
				return response, fmt.Errorf("scorer returned a non-finite logit for %q", f.Name)
			}
			p[j] = math.Exp(float64(logit) - float64(peak))
			sum += p[j]
		}
		var entropy, score float64
		for j := range p {
			p[j] /= sum
			score += float64(j) * p[j]
			if p[j] > 0 {
				entropy -= p[j] * math.Log(p[j])
			}
		}
		if f.typ == "noul" {
			response.Answers.Set(f.Name, NoulAnswer{Type: f.typ, Noul: p[1]})
			continue
		}
		probabilities := &Probabilities{}
		legend := &Legend{}
		for j, choice := range f.Choices {
			key := choice.Value.(string)
			probabilities.Set(key, p[j])
			legend.Set(key, choice.Description)
		}
		confidence := max(0, min(1, 1-entropy/math.Log(float64(len(p)))))
		if f.typ == "choice" {
			winner := slices.Index(p, slices.Max(p))
			response.Answers.Set(f.Name, ChoiceAnswer{f.typ, f.Choices[winner].Value.(string), probabilities, confidence})
		} else {
			response.Answers.Set(f.Name, ScoreAnswer{f.typ, score, legend, probabilities, confidence})
		}
	}
	return response, nil
}
