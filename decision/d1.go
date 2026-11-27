package decision

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/llm"
)

// encodeD1 validates the request as for every decision model, then passes its
// questions to the d1 model, which renders them and reads its own answers.
func encodeD1(req Request, c *Compiled) error {
	if err := ValidateLimits(req, "d1"); err != nil {
		return err
	}
	if err := validateImages(req.Images); err != nil {
		return err
	}
	if _, err := Content(req.State); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	native := &llm.NativeDecision{State: req.State}
	for name, q := range req.Questions.All() {
		if _, err := compileField(name, q); err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		raw, err := json.Marshal(q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		native.Questions = append(native.Questions, llm.NativeQuestion{Name: name, Question: raw})
		c.names = append(c.names, name)
	}
	c.Request.Native = native
	c.Request.Images = req.Images
	return nil
}

func (c *Compiled) answerD1(model string, result llm.ScoreResponse) (Response, error) {
	response := Response{
		Model: model, Answers: &Answers{},
		Usage:                 Usage{InputTokens: result.InputTokens, OutputTokens: result.OutputTokens},
		PromptEvalCachedCount: result.CachedTokens,
	}
	if len(result.Answers) != len(c.names) {
		return response, fmt.Errorf("model returned %d answers for %d questions", len(result.Answers), len(c.names))
	}
	for i, name := range c.names {
		response.Answers.Set(name, result.Answers[i])
	}
	return response, nil
}
