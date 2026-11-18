package decision

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
)

// Hindsight's decision models answer choices of 2–255 options and scores of
// 2–10 levels, with optional instructions and descriptions.
const (
	hindsightMaxChoices = 255
	hindsightMinScore   = 2
	hindsightMaxScore   = 10
)

// encodeHindsight checks the request and keeps it whole for the model, which
// renders every question into one sequence and returns one logit per option
// in the order of each field's Choices.
func encodeHindsight(req Request, c *Compiled) error {
	if req.Questions == nil || req.Questions.Len() < 1 {
		return fmt.Errorf("questions must contain at least one field")
	}
	if len(req.Images) != 0 || len(req.Videos) != 0 {
		return fmt.Errorf("Hindsight decision models accept text state only")
	}
	if _, err := Content(req.State); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	native := &llm.NativeDecision{State: req.State}
	for name, q := range req.Questions.All() {
		field, err := hindsightField(name, q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		c.fields = append(c.fields, field)
		raw, err := json.Marshal(q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		native.Questions = append(native.Questions, llm.NativeQuestion{Name: name, Question: raw})
	}
	c.Request.Native = native
	return nil
}

// hindsightContent is a description as the model reads it: text as is, JSON
// as Python-style text, and a missing or null value as nothing.
func hindsightContent(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	return Content(raw)
}

func hindsightField(name string, q Question) (compiledField, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, fmt.Errorf("field name must not be empty")
	}
	if len(q.Instructions) != 0 && string(q.Instructions) != "null" {
		description, err := Content(q.Instructions)
		if err != nil {
			return f, fmt.Errorf("instructions: %w", err)
		}
		f.Description = description
	}
	add := func(value, description any) {
		f.Choices = append(f.Choices, Choice{Value: value, Description: description})
	}
	switch q.Type {
	case "noul":
		criteria := orderedmap.New[string, json.RawMessage]()
		if len(q.Criteria) > 0 && string(q.Criteria) != "null" {
			if err := json.Unmarshal(q.Criteria, criteria); err != nil {
				return f, fmt.Errorf("noul criteria must be an object of true/false descriptions")
			}
		}
		for _, key := range []string{"false", "true"} {
			raw, _ := criteria.Get(key)
			description, err := hindsightContent(raw)
			if err != nil {
				return f, fmt.Errorf("noul criterion %q: %w", key, err)
			}
			add(key == "true", description)
		}
	case "choice":
		criteria := orderedmap.New[string, json.RawMessage]()
		if err := json.Unmarshal(q.Criteria, criteria); err != nil {
			return f, fmt.Errorf("choice criteria must map option keys to descriptions or null")
		}
		for key, raw := range criteria.All() {
			description, err := hindsightContent(raw)
			if err != nil {
				return f, fmt.Errorf("option %q: %w", key, err)
			}
			add(key, description)
		}
		if n := len(f.Choices); n < 2 || n > hindsightMaxChoices {
			return f, fmt.Errorf("%d options; a choice takes 2–%d options", n, hindsightMaxChoices)
		}
	case "score":
		var criteria []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return f, fmt.Errorf("score criteria must be an array of levels")
		}
		if n := len(criteria); n < hindsightMinScore || n > hindsightMaxScore {
			return f, fmt.Errorf("%d levels; a score takes %d–%d levels", n, hindsightMinScore, hindsightMaxScore)
		}
		for i, raw := range criteria {
			description, err := hindsightContent(raw)
			if err != nil {
				return f, fmt.Errorf("level %d: %w", i, err)
			}
			add(strconv.Itoa(i), description)
		}
	default:
		return f, fmt.Errorf("type must be choice, noul, or score")
	}
	return f, nil
}
