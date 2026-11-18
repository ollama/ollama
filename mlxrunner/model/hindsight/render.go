package hindsight

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
)

// render reproduces the token ids of the model's reference renderer,
// serving/hindsight_decision/typed_render.py in the hs-decide-1-flash repo.
// Framing text is encoded with control tokens; request text never is.

const (
	maxChoices = 255
	imStart    = "<|im_start|>"
	imEnd      = "<|im_end|>"
)

var noulCodes = []string{"no", "yes"}

type question struct {
	name         string
	kind         string
	instructions string
	keys         []string
	descriptions []string
}

type rendered struct {
	ids    []int32
	slots  []int
	groups [][][]int32 // per question, per option, the token ids of its code's spellings
}

// contentText is text as is, structured content as Python-style JSON, and a
// missing value as the empty string.
func contentText(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		err := json.Unmarshal(raw, &s)
		return s, err
	}
	return renderers.PythonJSON(raw, 0)
}

func parseQuestion(q llm.NativeQuestion) (question, error) {
	var raw struct {
		Type         string          `json:"type"`
		Instructions json.RawMessage `json:"instructions"`
		Criteria     json.RawMessage `json:"criteria"`
	}
	out := question{name: q.Name}
	if err := json.Unmarshal(q.Question, &raw); err != nil {
		return out, err
	}
	out.kind = raw.Type
	var err error
	if out.instructions, err = contentText(raw.Instructions); err != nil {
		return out, err
	}
	switch raw.Type {
	case "noul":
		criteria := orderedmap.New[string, json.RawMessage]()
		if len(raw.Criteria) > 0 && string(raw.Criteria) != "null" {
			if err := json.Unmarshal(raw.Criteria, criteria); err != nil {
				return out, err
			}
		}
		out.keys = []string{"false", "true"}
		for _, key := range out.keys {
			value, _ := criteria.Get(key)
			text, err := contentText(value)
			if err != nil {
				return out, err
			}
			out.descriptions = append(out.descriptions, text)
		}
	case "score":
		var levels []json.RawMessage
		if err := json.Unmarshal(raw.Criteria, &levels); err != nil {
			return out, err
		}
		for i, level := range levels {
			text, err := contentText(level)
			if err != nil {
				return out, err
			}
			out.keys = append(out.keys, strconv.Itoa(i))
			out.descriptions = append(out.descriptions, text)
		}
	case "choice":
		criteria := orderedmap.New[string, json.RawMessage]()
		if err := json.Unmarshal(raw.Criteria, criteria); err != nil {
			return out, err
		}
		for key, value := range criteria.All() {
			text, err := contentText(value)
			if err != nil {
				return out, err
			}
			out.keys = append(out.keys, key)
			out.descriptions = append(out.descriptions, text)
		}
		if len(out.keys) > maxChoices {
			return out, fmt.Errorf("%d options; at most %d", len(out.keys), maxChoices)
		}
	default:
		return out, fmt.Errorf("unknown question type %q", raw.Type)
	}
	return out, nil
}

func (m *Model) codesFor(q question) []string {
	switch q.kind {
	case "noul":
		return noulCodes
	case "score":
		codes := make([]string, len(q.keys))
		for i := range codes {
			codes[i] = strconv.Itoa(i)
		}
		return codes
	default:
		return m.codes[:len(q.keys)]
	}
}

func spellings(code string) []string {
	if code == "no" || code == "yes" {
		upper := strings.ToUpper(code[:1]) + code[1:]
		return []string{code, upper, " " + code, " " + upper}
	}
	return []string{code, " " + code}
}

func (m *Model) aliasGroups(q question) ([][]int32, error) {
	var groups [][]int32
	for _, code := range m.codesFor(q) {
		var ids []int32
		for _, form := range spellings(code) {
			id := m.singleToken(form)
			if id >= 0 && !slices.Contains(ids, id) {
				ids = append(ids, id)
			}
		}
		if len(ids) == 0 {
			return nil, fmt.Errorf("code %q is not a single token in this tokenizer", code)
		}
		groups = append(groups, ids)
	}
	return groups, nil
}

func (m *Model) render(native *llm.NativeDecision) (rendered, error) {
	var out rendered
	state, err := contentText(native.State)
	if err != nil {
		return out, fmt.Errorf("state: %w", err)
	}
	tok := m.tok
	enc := func(s string) []int32 { return tok.Encode(s, false) }
	user := func(s string) []int32 { return tok.EncodeText(s) }

	questions := make([]question, len(native.Questions))
	for i, q := range native.Questions {
		if questions[i], err = parseQuestion(q); err != nil {
			return out, fmt.Errorf("question %q: %w", q.Name, err)
		}
	}

	ids := enc(imStart + "user\n")
	ids = append(ids, enc("State:\n")...)
	ids = append(ids, user(state)...)
	ids = append(ids, enc("\n\n")...)
	ids = append(ids, enc("Questions:\n")...)
	for i, q := range questions {
		label := fmt.Sprintf("Q%d", i+1)
		ids = append(ids, user(fmt.Sprintf("[%s] (%s) ", label, q.kind))...)
		ids = append(ids, user(q.instructions)...)
		ids = append(ids, enc("\n")...)
		for j, code := range m.codesFor(q) {
			key, description := q.keys[j], q.descriptions[j]
			ids = append(ids, enc("  "+code)...)
			switch q.kind {
			case "noul":
				if description != "" {
					ids = append(ids, enc(": ")...)
					ids = append(ids, user(description)...)
				}
			case "score":
				ids = append(ids, enc(": ")...)
				ids = append(ids, user(description)...)
			default:
				ids = append(ids, enc(": ")...)
				ids = append(ids, user(key)...)
				if description != "" && description != key {
					ids = append(ids, enc(": ")...)
					ids = append(ids, user(description)...)
				}
			}
			ids = append(ids, enc("\n")...)
		}
		ids = append(ids, enc("\n")...)
	}
	ids = append(ids, enc(imEnd+"\n"+imStart+"assistant\n")...)
	for i, q := range questions {
		line := append(user(fmt.Sprintf("Q%d", i+1)), enc(":")...)
		out.slots = append(out.slots, len(ids)+len(line)-1)
		groups, err := m.aliasGroups(q)
		if err != nil {
			return out, err
		}
		out.groups = append(out.groups, groups)
		ids = append(ids, line...)
		ids = append(ids, enc("\n")...)
	}
	out.ids = ids
	return out, nil
}
