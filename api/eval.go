package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// EvalRequest evaluates a state against named questions. State and instructions
// accept a JSON string, object, or array, as in the TypeSafe evaluation API.
type EvalRequest struct {
	Model     string                  `json:"model"`
	State     json.RawMessage         `json:"state"`
	Questions map[string]EvalQuestion `json:"questions"`
}

// EvalQuestion is a yes/no (noul), categorical (choice), or ordinal (score)
// question. Criteria is an optional object of true/false descriptions for noul,
// a required map of option names to string or null descriptions for choice,
// or a required ordered array of at least two descriptions for score.
type EvalQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

type EvalResponse struct {
	Model   string                `json:"model"`
	Answers map[string]EvalAnswer `json:"answers"`
	Usage   EvalUsage             `json:"usage"`
}

// EvalAnswer contains the fields for its question type. Pointers preserve valid
// zero-valued answers while omitting fields belonging to other question types.
type EvalAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

type EvalUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func (r EvalRequest) Validate() error {
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("model is required")
	}
	if !evalContent(r.State) {
		return fmt.Errorf("state must be a string, object, or array")
	}
	if len(r.Questions) == 0 {
		return fmt.Errorf("questions must contain at least one question")
	}
	// Stable order also makes validation errors reproducible.
	ids := make([]string, 0, len(r.Questions))
	for id := range r.Questions {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		if err := r.Questions[id].Validate(); err != nil {
			return fmt.Errorf("questions[%q]: %w", id, err)
		}
	}
	return nil
}

func evalContent(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return json.Valid(raw) && (raw[0] == '"' || raw[0] == '{' || raw[0] == '[')
}

func (q EvalQuestion) Validate() error {
	if !evalContent(q.Instructions) {
		return fmt.Errorf("instructions must be a string, object, or array")
	}
	criteria := bytes.TrimSpace(q.Criteria)
	switch q.Type {
	case "noul":
		if len(criteria) == 0 {
			return nil
		}
		var descriptions map[string]json.RawMessage
		if err := json.Unmarshal(criteria, &descriptions); err != nil || descriptions == nil {
			return fmt.Errorf("noul criteria must be an object of true/false descriptions")
		}
		for key, value := range descriptions {
			if (key != "true" && key != "false") || len(value) == 0 || value[0] != '"' {
				return fmt.Errorf("noul criteria must contain only true/false string descriptions")
			}
		}
	case "choice":
		var options map[string]json.RawMessage
		if err := json.Unmarshal(criteria, &options); err != nil || len(options) == 0 {
			return fmt.Errorf("choice criteria must be a nonempty map of options to string or null descriptions")
		}
		for _, value := range options {
			if len(value) == 0 || (value[0] != '"' && !bytes.Equal(value, []byte("null"))) {
				return fmt.Errorf("choice criteria descriptions must be strings or null")
			}
		}
	case "score":
		var levels []json.RawMessage
		if err := json.Unmarshal(criteria, &levels); err != nil || len(levels) < 2 {
			return fmt.Errorf("score criteria must be an array of at least two string descriptions")
		}
		for _, level := range levels {
			if len(level) == 0 || level[0] != '"' {
				return fmt.Errorf("score criteria descriptions must be strings")
			}
		}
	default:
		return fmt.Errorf("type must be noul, choice, or score")
	}
	return nil
}
