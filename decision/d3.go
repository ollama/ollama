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
			if len(field.Choices) > 10 {
				return fmt.Errorf("question %q: score criteria must contain 2–10 levels", name)
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
		c.fields = append(c.fields, field)
		c.Request.Rows = append(c.Request.Rows, llm.ScoreRow{Question: &llm.ScoreQuestion{
			Type: question.Type, Instructions: field.Description, Options: descriptions,
		}})
	}
	return nil
}
