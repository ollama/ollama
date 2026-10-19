package d1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/internal/orderedmap"
)

// option is one choice criterion: its key and description (nil when the
// request gives null or the empty string).
type option struct {
	name string
	desc *string
}

// question is one System One question.
type question struct {
	name    string
	typ     string // noul, choice or score
	said    string
	noulYes *string
	noulNo  *string
	hasNoul bool
	options []option
	levels  []string
}

// rawQuestion is the question object as the request wrote it.
type rawQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

func present(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && !bytes.Equal(raw, []byte("null"))
}

// parseQuestion reads a question the decision layer validated as for every
// System One model into the form the prompt needs.
func parseQuestion(name string, raw json.RawMessage) (question, error) {
	q := question{name: name}
	var r rawQuestion
	if err := json.Unmarshal(raw, &r); err != nil {
		return q, fmt.Errorf("must be an object: %w", err)
	}
	q.typ = r.Type
	said, err := decision.Content(r.Instructions)
	if err != nil || strings.TrimSpace(said) == "" {
		return q, fmt.Errorf("instructions must be a nonempty string, object, or array")
	}
	q.said = said
	switch q.typ {
	case "noul":
		if present(r.Criteria) {
			criteria := orderedmap.New[string, *string]()
			if err := json.Unmarshal(r.Criteria, criteria); err != nil {
				return q, fmt.Errorf("noul criteria must map true and false to descriptions")
			}
			for key, value := range criteria.All() {
				switch key {
				case "true":
					q.noulYes = value
				case "false":
					q.noulNo = value
				default:
					return q, fmt.Errorf("unknown noul criterion %q", key)
				}
			}
			q.hasNoul = criteria.Len() > 0
		}
	case "choice":
		criteria := orderedmap.New[string, *string]()
		if err := json.Unmarshal(r.Criteria, criteria); err != nil || !present(r.Criteria) {
			return q, fmt.Errorf("choice criteria must map option keys to descriptions or null")
		}
		for key, value := range criteria.All() {
			if value != nil && *value == "" {
				value = nil
			}
			q.options = append(q.options, option{key, value})
		}
		if len(q.options) < 2 || len(q.options) > maxOptions {
			return q, fmt.Errorf("criteria must contain 2–%d candidates", maxOptions)
		}
	case "score":
		if err := json.Unmarshal(r.Criteria, &q.levels); err != nil || len(q.levels) < 2 || len(q.levels) > 10 {
			return q, fmt.Errorf("score criteria must be an array of 2–10 levels")
		}
	default:
		return q, fmt.Errorf("type must be choice, noul, or score")
	}
	return q, nil
}

// maxOptions is System One's choice candidate limit.
const maxOptions = 26
