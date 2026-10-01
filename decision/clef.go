package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/llm"
)

// EncodeClef prepares the request validated by Compile for Clef's decision head.
// It preserves the reference encoder's segment boundaries and retains its choice
// order for the shared answer decoder.
func (c *Compiled) EncodeClef(req Request) error {
	state, err := clefContent(req.State)
	if err != nil {
		return err
	}
	input := llm.ScoreRequest{}
	add := func(text string) [2]int {
		n := len(input.Segments)
		input.Segments = append(input.Segments, text)
		return [2]int{n, n + 1}
	}
	add("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")
	add(state)
	add("\n\nSCHEMA FIELDS:\n")
	for i := range c.fields {
		f := &c.fields[i]
		q, _ := req.Questions.Get(f.Name)
		description, err := clefContent(q.Instructions)
		if err != nil {
			return err
		}
		add(fmt.Sprintf("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", i+1, f.Name, q.Type))
		field := llm.ScoreField{Question: add(description)}
		switch q.Type {
		case "noul":
			field.Type = 0
			// Clef encodes true before false; Nimble uses the reverse order.
			if f.Choices[0].Value != true {
				slices.Reverse(f.Choices)
			}
			descriptions := map[string]string{"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
			if len(q.Criteria) > 0 {
				_ = json.Unmarshal(q.Criteria, &descriptions)
			}
			f.Choices[0].Description = descriptions["true"]
			f.Choices[1].Description = descriptions["false"]
		case "choice":
			field.Type = 1
			slices.SortFunc(f.Choices, func(a, b Choice) int { return strings.Compare(a.Value.(string), b.Value.(string)) })
		case "score":
			field.Type = 2
		}
		add("\nALLOWED OPTIONS:\n")
		var original map[string]*string
		if q.Type == "choice" {
			_ = json.Unmarshal(q.Criteria, &original)
		}
		for j, option := range f.Choices {
			add(fmt.Sprintf("OPTION %d: ", j+1))
			id := fmt.Sprint(option.Value)
			semantics := map[string]any{"option_id": id, "description": option.Description}
			if q.Type == "choice" && original[id] == nil {
				delete(semantics, "description")
			}
			raw, err := clefJSON(semantics)
			if err != nil {
				return err
			}
			field.Options = append(field.Options, add(raw))
			add("\n")
		}
		add("END FIELD\n")
		input.Fields = append(input.Fields, field)
	}
	add("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
	c.Request = input
	c.messages = nil
	return nil
}

func clefJSON(value any) (string, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	err := enc.Encode(value)
	return strings.TrimSuffix(b.String(), "\n"), err
}

func clefContent(raw json.RawMessage) (string, error) {
	var value any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&value); err != nil {
		return "", err
	}
	if s, ok := value.(string); ok {
		return s, nil
	}
	return clefJSON(value)
}
