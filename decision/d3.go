package decision

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
)

// encodeD3 retains the publisher's question and option order. The Qwen
// backbone scores a complete prompt independently for each question.
func encodeD3(req Request, c *Compiled) error {
	return encodeIndependentQwen(req, c, qwenDecisionLimits{minChoices: 1, minScore: 2, maxScore: 10})
}

func encodeAutoTrustDecision(req Request, c *Compiled) error {
	return encodeIndependentQwen(req, c, qwenDecisionLimits{minChoices: 2, minScore: 6, maxScore: 6})
}

type qwenDecisionLimits struct {
	minChoices int
	minScore   int
	maxScore   int
}

func encodeIndependentQwen(req Request, c *Compiled, limits qwenDecisionLimits) error {
	if n := req.Questions.Len(); n < 1 || n > 64 {
		return fmt.Errorf("questions must contain 1–64 fields")
	}
	if len(req.Videos) > 0 {
		return fmt.Errorf("video inputs are not supported")
	}
	if err := validateImages(req.Images); err != nil {
		return err
	}
	state := bytes.TrimSpace(req.State)
	var stateText string
	if len(state) == 0 || bytes.Equal(state, []byte("null")) || bytes.Equal(state, []byte(`""`)) {
		stateText = "(empty)"
	} else {
		var err error
		stateText, err = pplxDescribe(state)
		if err != nil {
			return fmt.Errorf("state: %w", err)
		}
	}
	c.Request = llm.ScoreRequest{State: stateText, Images: req.Images}
	for name, question := range req.Questions.All() {
		field, descriptions, err := pplxField(name, question)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		if question.Type == "score" {
			if len(field.Choices) < limits.minScore || len(field.Choices) > limits.maxScore {
				if limits.minScore == limits.maxScore {
					return fmt.Errorf("question %q: score criteria must contain exactly %d levels", name, limits.minScore)
				}
				return fmt.Errorf("question %q: score criteria must contain %d–%d levels", name, limits.minScore, limits.maxScore)
			}
			seen := make(map[string]bool, len(descriptions))
			for _, description := range descriptions {
				if description == "" || seen[description] {
					return fmt.Errorf("question %q: score levels must be distinct and nonempty", name)
				}
				seen[description] = true
			}
			for i := range field.Choices {
				field.Choices[i].Value = strconv.Itoa(i)
				if text, ok := field.Choices[i].Description.(string); ok {
					field.Choices[i].Description = text
				} else {
					canonical, err := renderers.PythonJSONCompact(field.Choices[i].Description)
					if err != nil {
						return fmt.Errorf("question %q level %d: %w", name, i, err)
					}
					field.Choices[i].Description = canonical
				}
			}
		}
		if question.Type == "choice" && len(field.Choices) < limits.minChoices {
			return fmt.Errorf("question %q: choice criteria must contain at least %d options", name, limits.minChoices)
		}
		c.fields = append(c.fields, field)
		c.Request.Rows = append(c.Request.Rows, llm.ScoreRow{Question: &llm.ScoreQuestion{
			Type: question.Type, Instructions: field.Description, Options: descriptions,
		}})
	}
	return nil
}
