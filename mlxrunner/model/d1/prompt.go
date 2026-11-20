package d1

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

const (
	userTurn       = "<|startoftext|><|im_start|>user\n"
	assistantTurn  = "<|im_end|>\n<|im_start|>assistant\n<think></think>"
	questionHeader = "QUESTION:\n"
)

// slot is a question's answer position, as an offset from its last token,
// and one vocabulary id group per answer, max-pooled.
type slot struct {
	offset int
	groups [][]int32
}

// renderer writes d1 prompts for one tokenizer.
type renderer struct {
	tok      *tokenizer.Tokenizer
	specials []string // special-token strings, longest first
	yes, no  []int32
	digits   [][]int32 // "0".."9"
}

func newRenderer(tok *tokenizer.Tokenizer, specials []string) (*renderer, error) {
	r := &renderer{tok: tok, specials: slices.Clone(specials)}
	slices.SortStableFunc(r.specials, func(a, b string) int { return len(b) - len(a) })
	r.yes = r.singleIDs("yes", "Yes", "YES")
	r.no = r.singleIDs("no", "No", "NO")
	if len(r.yes) == 0 || len(r.no) == 0 {
		return nil, fmt.Errorf("d1 tokenizer has no single-token yes/no")
	}
	for i := range 10 {
		g := r.singleIDs(strconv.Itoa(i))
		if len(g) == 0 {
			return nil, fmt.Errorf("d1 tokenizer has no single-token digit %d", i)
		}
		r.digits = append(r.digits, g)
	}
	return r, nil
}

// single returns the id text encodes to when it is exactly one token.
func (r *renderer) single(text string) (int32, bool) {
	ids := r.tok.Encode(text, false)
	if len(ids) != 1 {
		return 0, false
	}
	return ids[0], true
}

// singleIDs returns the distinct ids of the texts that are single tokens.
func (r *renderer) singleIDs(texts ...string) []int32 {
	var ids []int32
	for _, t := range texts {
		if id, ok := r.single(t); ok && !slices.Contains(ids, id) {
			ids = append(ids, id)
		}
	}
	return ids
}

// plain breaks every special-token string in caller text with a zero-width
// space after its first character, so it tokenizes as text.
func (r *renderer) plain(text string) string {
	runes := []rune(text)
	for changed := true; changed; {
		changed = false
		s := string(runes)
		for _, special := range r.specials {
			if len(special) < 2 || !strings.Contains(s, special) {
				continue
			}
			var out []rune
			byteIndex := 0
			for _, rr := range runes {
				out = append(out, rr)
				if strings.HasPrefix(s[byteIndex:], special) {
					out = append(out, '​')
					changed = true
				}
				byteIndex += utf8.RuneLen(rr)
			}
			runes = out
			s = string(runes)
		}
	}
	return string(runes)
}

// optionCodes are the codes a choice's options are written with.
func optionCodes(names []string) []string {
	letters := true
	for _, n := range names {
		rs := []rune(strings.TrimSpace(n))
		if len(rs) != 1 || !unicode.IsLetter(rs[0]) {
			letters = false
			break
		}
	}
	codes := make([]string, len(names))
	for i, n := range names {
		switch {
		case letters:
			codes[i] = strings.TrimSpace(n)
		default:
			codes[i] = string(rune('A' + i))
		}
	}
	return codes
}

// fallbackCodes replace codes that are not single unused tokens, in order.
var fallbackCodes = func() []string {
	var pool []string
	for c := 'A'; c <= 'Z'; c++ {
		pool = append(pool, string(c))
	}
	for i := range 100 {
		pool = append(pool, fmt.Sprintf("%02d", i))
	}
	for c := 'a'; c <= 'z'; c++ {
		pool = append(pool, string(c))
	}
	return pool
}()

// aliases assigns each option a distinct single-token code and its readout group.
func (r *renderer) aliases(names []string) ([]string, [][]int32, error) {
	used := map[int32]bool{}
	take := func(code string) (int32, bool) {
		id, ok := r.single(code)
		if !ok || used[id] {
			return 0, false
		}
		used[id] = true
		return id, true
	}
	var codes []string
	var groups [][]int32
	for _, code := range optionCodes(names) {
		id, ok := take(code)
		if !ok {
			for _, alt := range fallbackCodes {
				if id, ok = take(alt); ok {
					code = alt
					break
				}
			}
			if !ok {
				return nil, nil, fmt.Errorf("no single-token code left for %d options", len(names))
			}
		}
		group := []int32{id}
		if spaced, ok := r.single(" " + code); ok && spaced != id {
			group = append(group, spaced)
		}
		codes = append(codes, code)
		groups = append(groups, group)
	}
	return codes, groups, nil
}

func describe(s *string) string {
	if s == nil {
		return "None"
	}
	return *s
}

// suffix writes a question after the state, ending in the assistant turn
// whose last token is the answer slot.
func (r *renderer) suffix(q *question) (string, slot, error) {
	var body string
	var s slot
	switch q.typ {
	case "noul":
		extra := ""
		if q.hasNoul {
			extra = "\nYes: " + describe(q.noulYes) + "\nNo: " + describe(q.noulNo)
		}
		body = q.said + extra + "\n\nReply with yes or no only."
		s.groups = [][]int32{r.yes, r.no}
	case "choice":
		names := make([]string, len(q.options))
		for i, o := range q.options {
			names[i] = o.name
		}
		codes, groups, err := r.aliases(names)
		if err != nil {
			return "", s, err
		}
		var lines []string
		for i, o := range q.options {
			text := strings.ReplaceAll(o.name, "_", " ")
			if o.desc != nil {
				text = *o.desc
			}
			lines = append(lines, codes[i]+" "+text)
		}
		body = q.said + "\n\nOptions:\n" + strings.Join(lines, "\n") + "\n\nReply with the option code only."
		s.groups = groups
	case "score":
		var lines []string
		for i, level := range q.levels {
			lines = append(lines, strconv.Itoa(i)+" "+level)
		}
		body = q.said + "\n\n" + strings.Join(lines, "\n") + "\n\nReply with a single digit 0-" + strconv.Itoa(len(q.levels)-1) + " only."
		s.groups = r.digits[:len(q.levels)]
	default:
		return "", s, fmt.Errorf("unsupported question type %q", q.typ)
	}
	return r.plain(body) + assistantTurn, s, nil
}

// stateText is the state as the prompt writes it, before escaping: a string
// verbatim, other JSON as Python's indented json.dumps.
func stateText(raw json.RawMessage) (string, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return "", fmt.Errorf("state must be a string, object, or array")
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("state: %w", err)
		}
		return s, nil
	}
	text, err := pyJSONIndent2(raw)
	if err != nil {
		return "", fmt.Errorf("state: %w", err)
	}
	return text, nil
}
