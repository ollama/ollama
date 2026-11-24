package tokenizer

import (
	"encoding/json"
	"fmt"
	"iter"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

const defaultBPEPattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

type pretokenizer struct {
	re              *regexp.Regexp
	whitespaceGroup int
	rightDigits     bool
	behavior        string
	invert          bool
	prefixSpace     bool
	byteLevel       bool
}

func loadPretokenizers(data json.RawMessage) ([]pretokenizer, error) {
	if len(data) == 0 || string(data) == "null" {
		pt, err := compileSplit(defaultBPEPattern, "Isolated", false)
		return []pretokenizer{pt}, err
	}
	var all []pretokenizer
	pending := []json.RawMessage{data}
	for len(pending) > 0 {
		data := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		var raw struct {
			Type          string            `json:"type"`
			Pretokenizers []json.RawMessage `json:"pretokenizers"`
			Pattern       struct {
				Regex  string  `json:"Regex"`
				String *string `json:"String"`
			} `json:"pattern"`
			Behavior       string `json:"behavior"`
			Invert         bool   `json:"invert"`
			UseRegex       *bool  `json:"use_regex"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("invalid pretokenizer: %w", err)
		}
		var pt pretokenizer
		var err error
		switch raw.Type {
		case "Sequence":
			// Stack children in reverse to retain their published execution order.
			for i := len(raw.Pretokenizers) - 1; i >= 0; i-- {
				pending = append(pending, raw.Pretokenizers[i])
			}
			continue
		case "Split":
			pattern := raw.Pattern.Regex
			if raw.Pattern.String != nil {
				pattern = regexp.QuoteMeta(*raw.Pattern.String)
			}
			pt, err = compileSplit(pattern, raw.Behavior, raw.Invert)
		case "ByteLevel":
			if raw.UseRegex == nil || *raw.UseRegex {
				pt, err = compileSplit(defaultBPEPattern, "Isolated", false)
			}
			pt.prefixSpace = raw.AddPrefixSpace
			pt.byteLevel = true
		default:
			return nil, fmt.Errorf("unsupported pretokenizer %q", raw.Type)
		}
		if err != nil {
			return nil, err
		}
		all = append(all, pt)
	}
	for i, pt := range all {
		// Byte mapping happens in encodeChunkInto, after the splitting pipeline.
		if pt.byteLevel {
			if i != len(all)-1 {
				return nil, fmt.Errorf("ByteLevel must be the final pretokenizer")
			}
			if pt.re == nil && !pt.prefixSpace {
				all = all[:i]
			}
		}
	}
	return all, nil
}

func compileSplit(pattern, behavior string, invert bool) (pretokenizer, error) {
	pt := pretokenizer{behavior: behavior, invert: invert, whitespaceGroup: -1}
	switch behavior {
	case "", "Isolated", "Removed", "MergedWithPrevious", "MergedWithNext", "Contiguous":
	default:
		return pt, fmt.Errorf("unsupported split behavior %q", behavior)
	}
	switch pattern {
	// RE2 cannot express these lookaheads. Decimal groups are matched from
	// the right at a Unicode word boundary; a maximal newline run already
	// satisfies the negative newline lookahead.
	case `\d{1,3}(?=(?:\d{3})*\b)`:
		pt.rightDigits = true
		pattern = `\p{Nd}+`
	case `(?:\r?\n)+(?!\r?\n)`:
		pattern = `(?:\r?\n)+`
	}
	// Only the matched trailing-whitespace alternative may backtrack. Other
	// alternatives that consume whitespace keep their original boundaries.
	const whitespaceSuffix = `\s+(?!\S)|\s+`
	prefix := strings.TrimSuffix(pattern, whitespaceSuffix)
	alternative := strings.HasSuffix(prefix, "|")
	if alternative {
		// An odd number of backslashes makes the pipe a literal.
		for i := len(prefix) - 2; i >= 0 && prefix[i] == '\\'; i-- {
			alternative = !alternative
		}
	}
	trailingWhitespace := strings.HasSuffix(pattern, whitespaceSuffix) && (prefix == "" || alternative)
	if trailingWhitespace {
		pattern = prefix + `(\s+)|\s+`
	}
	// Only noncapturing groups and the publisher contraction groups are
	// supported here. Oniguruma's other flag groups can differ from RE2:
	// full case folding matches ss to ß, and m changes dot/newline behavior.
	const contractions = `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
	groups := strings.ReplaceAll(strings.ReplaceAll(pattern, contractions, ""), "(?:", "")
	if strings.Contains(groups, "(?") {
		return pt, fmt.Errorf("unsupported pretokenizer group in %q", pattern)
	}
	// Oniguruma's whitespace class is Unicode; Go's \s is ASCII only.
	const spaces = `\p{Z}\x{85}\x{9}-\x{D}`
	var rewritten strings.Builder
	inClass := false
	for i := 0; i < len(pattern); i++ {
		switch pattern[i] {
		case '[':
			if inClass {
				return pt, fmt.Errorf("unsupported nested character class in %q", pattern)
			}
			inClass = true
		case ']':
			inClass = false
		case '&':
			if inClass && i+1 < len(pattern) && pattern[i+1] == '&' {
				return pt, fmt.Errorf("unsupported character-class intersection in %q", pattern)
			}
		case '^', '$':
			if !inClass {
				return pt, fmt.Errorf("unsupported pretokenizer anchor in %q", pattern)
			}
		case '\\':
			if i+1 < len(pattern) {
				next := pattern[i+1]
				if next == 's' || next == 'd' || next == 'w' || next == 'S' || next == 'D' || next == 'W' {
					class := spaces
					switch next {
					case 'd', 'D':
						class = `\p{Nd}`
					case 'w', 'W':
						class = `\p{L}\p{M}\p{N}\p{Pc}`
					}
					negated := next >= 'A' && next <= 'Z'
					if negated && inClass {
						return pt, fmt.Errorf("unsupported negated Unicode class in %q", pattern)
					}
					if !inClass {
						rewritten.WriteByte('[')
						if negated {
							rewritten.WriteByte('^')
						}
					}
					rewritten.WriteString(class)
					if !inClass {
						rewritten.WriteByte(']')
					}
					i++
					continue
				}
				if next == 'b' || next == 'B' || next == 'A' || next == 'z' {
					return pt, fmt.Errorf("unsupported pretokenizer boundary in %q", pattern)
				}
				rewritten.WriteByte(pattern[i])
				rewritten.WriteByte(next)
				i++
				continue
			}
		}
		rewritten.WriteByte(pattern[i])
	}
	var err error
	pt.re, err = regexp.Compile(rewritten.String())
	if err != nil {
		return pt, fmt.Errorf("unsupported pretokenizer regex %q: %w", pattern, err)
	}
	if pt.re.MatchString("") {
		return pt, fmt.Errorf("pretokenizer regex %q matches empty input", pattern)
	}
	if trailingWhitespace {
		pt.whitespaceGroup = pt.re.NumSubexp()
	}
	return pt, nil
}

func (p pretokenizer) matches(text string) iter.Seq2[int, int] {
	return func(yield func(int, int) bool) {
		for offset := 0; offset < len(text); {
			loc := p.re.FindStringSubmatchIndex(text[offset:])
			if loc == nil {
				break
			}
			start, end := offset+loc[0], offset+loc[1]
			if p.rightDigits {
				r, _ := utf8.DecodeRuneInString(text[end:])
				if end == len(text) || !(unicode.IsLetter(r) || unicode.IsMark(r) || unicode.IsNumber(r) || unicode.Is(unicode.Pc, r)) {
					count := utf8.RuneCountInString(text[start:end])
					size := count % 3
					if size == 0 {
						size = 3
					}
					cursor := start
					for i, r := range text[start:end] {
						size--
						if size == 0 {
							next := start + i + utf8.RuneLen(r)
							if !yield(cursor, next) {
								return
							}
							cursor = next
							size = 3
						}
					}
				}
				offset = end
				continue
			}
			if p.whitespaceGroup > 0 && loc[2*p.whitespaceGroup] >= 0 && end < len(text) {
				_, size := utf8.DecodeLastRuneInString(text[start:end])
				if end-start > size {
					end -= size
				}
			}
			if !yield(start, end) {
				return
			}
			offset = end
		}
	}
}

func (p pretokenizer) split(text string) []string {
	if text == "" {
		return nil
	}
	if p.prefixSpace && !strings.HasPrefix(text, " ") {
		text = " " + text
	}
	if p.re == nil {
		return []string{text}
	}
	var result []string
	pending := ""
	previousDelimiter := false
	emit := func(text string, delimiter bool) {
		switch p.behavior {
		case "Removed":
			if !delimiter {
				result = append(result, text)
			}
		case "MergedWithPrevious":
			if delimiter && !previousDelimiter && len(result) > 0 {
				result[len(result)-1] += text
			} else {
				result = append(result, text)
			}
		case "MergedWithNext":
			if delimiter {
				if pending != "" {
					result = append(result, pending)
				}
				pending = text
			} else {
				result = append(result, pending+text)
				pending = ""
			}
		case "Contiguous":
			if len(result) > 0 && delimiter == previousDelimiter {
				result[len(result)-1] += text
			} else {
				result = append(result, text)
			}
		default:
			result = append(result, text)
		}
		previousDelimiter = delimiter
	}
	offset := 0
	for start, end := range p.matches(text) {
		if start > offset {
			emit(text[offset:start], p.invert)
		}
		emit(text[start:end], !p.invert)
		offset = end
	}
	if offset < len(text) {
		emit(text[offset:], p.invert)
	}
	if pending != "" {
		result = append(result, pending)
	}

	return result
}
