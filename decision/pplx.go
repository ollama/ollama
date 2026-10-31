package decision

import (
	"fmt"

	"github.com/ollama/ollama/llm"
)

// encodePPLX preserves question order and JSON object order. The publisher's
// readout assigns answer codes by position, so sorting choices changes scores.
func encodePPLX(req Request, c *Compiled) error {
	state, err := codeReadoutDescribe(req.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	c.Request = llm.ScoreRequest{Images: req.Images}
	for name, question := range req.Questions.All() {
		field, descriptions, err := codeReadoutField(name, question)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		c.fields = append(c.fields, field)
		c.Request.Rows = append(c.Request.Rows, llm.ScoreRow{Question: &llm.ScoreQuestion{
			Type: question.Type, Instructions: field.Description, Options: descriptions,
		}})
	}
	c.Request.State = state
	return nil
}
