package decision

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ollama/ollama/model/renderers"
)

// Match the HTTP body budget while bounding expansion across question rows.
const maxCompiledPromptBytes = 32 << 20

// ErrPromptTooLarge reports that compiled scoring prompts exceed their byte budget.
var ErrPromptTooLarge = errors.New("decision request is too large")

func decisionPrompts(state json.RawMessage, context string, fields []compiledField, renderer string) ([]string, error) {
	if len(fields) > 0 && len(context) > maxCompiledPromptBytes/len(fields) {
		return nil, ErrPromptTooLarge
	}
	var prompts []string
	var total int
	if renderer == "tev1" {
		for _, f := range fields {
			if len(f.Choices) > 24 {
				return nil, fmt.Errorf("question %q: Tev1 supports at most 24 candidates", f.Name)
			}
			type option struct {
				Label       string `json:"label"`
				Key         string `json:"key"`
				Description string `json:"description"`
			}
			var options []option
			for _, c := range f.Choices {
				description, ok := c.Description.(string)
				if c.Description == nil {
					// Tev1 describes every option: noul as No/Yes, bare choices by key.
					description, ok = fmt.Sprint(c.Value), true
					switch c.Value {
					case false:
						description = "No"
					case true:
						description = "Yes"
					}
				}
				if !ok || strings.TrimSpace(description) == "" {
					return nil, fmt.Errorf("question %q: Tev1 requires nonempty candidate descriptions", f.Name)
				}
				options = append(options, option{c.Code, fmt.Sprint(c.Value), description})
			}
			// Tev1 was trained with object states nested in the payload.
			prompt, err := renderers.PythonJSON(struct {
				State    json.RawMessage `json:"state"`
				Question string          `json:"question"`
				Options  []option        `json:"options"`
			}{state, f.Description, options}, 0)
			if err != nil {
				return nil, err
			}
			if len(prompt) > maxCompiledPromptBytes-total {
				return nil, ErrPromptTooLarge
			}
			total += len(prompt)
			prompts = append(prompts, prompt)
		}
		return prompts, nil
	}

	data, err := renderers.PythonJSON(struct {
		Context string          `json:"context"`
		Schema  []compiledField `json:"schema"`
	}{context, fields}, 0)
	if err != nil {
		return nil, err
	}
	// Nimble escapes chat delimiters in both the schema and the requested name.
	escape := strings.NewReplacer("<", `\u003c`, ">", `\u003e`)
	if len(data)+5*(strings.Count(data, "<")+strings.Count(data, ">")) > maxCompiledPromptBytes {
		return nil, ErrPromptTooLarge
	}
	data = escape.Replace(data)
	for _, f := range fields {
		name, err := renderers.PythonJSON(f.Name, 0)
		if err != nil {
			return nil, err
		}
		suffix := "\n\nRequested field: " + escape.Replace(name)
		if len(data)+len(suffix) > maxCompiledPromptBytes-total {
			return nil, ErrPromptTooLarge
		}
		prompt := data + suffix
		total += len(prompt)
		prompts = append(prompts, prompt)
	}
	return prompts, nil
}
