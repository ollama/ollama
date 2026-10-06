package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/model/renderers"
)

func decisionPrompts(context string, fields []compiledField, renderer string) ([]string, error) {
	var prompts []string
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
				if !ok || strings.TrimSpace(description) == "" {
					return nil, fmt.Errorf("question %q: Tev1 requires nonempty candidate descriptions", f.Name)
				}
				options = append(options, option{c.Code, fmt.Sprint(c.Value), description})
			}
			prompt, err := promptJSON(struct {
				State    string   `json:"state"`
				Question string   `json:"question"`
				Options  []option `json:"options"`
			}{context, f.Description, options})
			if err != nil {
				return nil, err
			}
			prompts = append(prompts, prompt)
		}
		return prompts, nil
	}

	data, err := promptJSON(struct {
		Context string          `json:"context"`
		Schema  []compiledField `json:"schema"`
	}{context, fields})
	if err != nil {
		return nil, err
	}
	// Nimble escapes chat delimiters in both the schema and the requested name.
	escape := strings.NewReplacer("<", `\u003c`, ">", `\u003e`)
	for _, f := range fields {
		name, err := promptJSON(f.Name)
		if err != nil {
			return nil, err
		}
		prompts = append(prompts, escape.Replace(data)+"\n\nRequested field: "+escape.Replace(name))
	}
	return prompts, nil
}

// Both publishers serialize prompts with Python's json.dumps(ensure_ascii=False).
func promptJSON(v any) (string, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	data := renderers.AddJSONSpaces(bytes.TrimSuffix(b.Bytes(), []byte{'\n'}))
	var out strings.Builder
	for i := 0; i < len(data); i++ {
		// encoding/json always escapes these two runes. Leave literal backslash
		// sequences alone by consuming each escape as a unit.
		if data[i] == '\\' && i+1 < len(data) {
			if bytes.HasPrefix(data[i:], []byte(`\u2028`)) {
				out.WriteRune('\u2028')
				i += 5
				continue
			}
			if bytes.HasPrefix(data[i:], []byte(`\u2029`)) {
				out.WriteRune('\u2029')
				i += 5
				continue
			}
			out.WriteByte(data[i])
			i++
		}
		out.WriteByte(data[i])
	}
	return out.String(), nil
}
