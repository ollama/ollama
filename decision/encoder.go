package decision

import (
	"encoding/json"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// An encoder prepares model input from a validated request and its compiled
// fields. It may reorder choices; the shared answer decoder uses that order.
type encoder func(Request, *Compiled) error

func encoderFor(name string) encoder {
	switch name {
	case "":
		return encodeCandidates
	case "clef":
		return encodeClef
	default:
		return nil
	}
}

// encodeCandidates builds the candidate-scoring input shared by Nimble and Tev.
func encodeCandidates(req Request, c *Compiled) error {
	context, err := content(req.State)
	if err != nil {
		return err
	}
	data, err := json.Marshal(struct {
		Context string          `json:"context"`
		Schema  []compiledField `json:"schema"`
	}{context, c.fields})
	if err != nil {
		return err
	}
	for _, f := range c.fields {
		name, err := json.Marshal(f.Name)
		if err != nil {
			return err
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
	return nil
}
