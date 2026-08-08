package structured

import (
	"fmt"
	"strconv"
	"unicode/utf8"
)

// gbnfToGrammar compiles a set of named GBNF rule bodies into a Grammar
// rooted at root. Only the subset emitted by the schema converter and the
// grammarJSON constant is supported: literals, character classes, groups,
// alternation, rule references, comments, and the ? * + {n} {n,} {n,m}
// repetitions. Postfix operators bind to the whole preceding literal or
// group, matching llama.cpp's GBNF parser.
func gbnfToGrammar(rules map[string]string, root string) (*Grammar, error) {
	b := newBuilder()
	ids := make(map[string]int32, len(rules))
	for name := range rules {
		ids[name] = b.reserve()
	}
	rootID, ok := ids[root]
	if !ok {
		return nil, fmt.Errorf("gbnf: root rule %q is not defined", root)
	}
	b.g.root = rootID

	for name, body := range rules {
		p := &gbnfParser{b: b, ids: ids, src: body, rule: name}
		alts, err := p.parseAlternates()
		if err != nil {
			return nil, err
		}
		p.skipSpace()
		if p.pos < len(p.src) {
			return nil, fmt.Errorf("gbnf: rule %q: unexpected %q at offset %d", name, p.src[p.pos], p.pos)
		}
		b.define(ids[name], alts...)
	}
	return b.g, nil
}

type gbnfParser struct {
	b    *builder
	ids  map[string]int32
	src  string
	rule string
	pos  int
}

func (p *gbnfParser) errf(format string, args ...any) error {
	return fmt.Errorf("gbnf: rule %q: %s (offset %d)", p.rule, fmt.Sprintf(format, args...), p.pos)
}

func (p *gbnfParser) skipSpace() {
	for p.pos < len(p.src) {
		switch p.src[p.pos] {
		case ' ', '\t', '\r', '\n':
			p.pos++
		case '#':
			for p.pos < len(p.src) && p.src[p.pos] != '\n' {
				p.pos++
			}
		default:
			return
		}
	}
}

func (p *gbnfParser) parseAlternates() ([]seq, error) {
	first, err := p.parseSequence()
	if err != nil {
		return nil, err
	}
	alts := []seq{first}
	for {
		p.skipSpace()
		if p.pos >= len(p.src) || p.src[p.pos] != '|' {
			return alts, nil
		}
		p.pos++
		next, err := p.parseSequence()
		if err != nil {
			return nil, err
		}
		alts = append(alts, next)
	}
}

func (p *gbnfParser) parseSequence() (seq, error) {
	out := seq{}
	for {
		p.skipSpace()
		if p.pos >= len(p.src) {
			return out, nil
		}
		switch c := p.src[p.pos]; {
		case c == ')' || c == '|':
			return out, nil
		case c == '"':
			unit, err := p.parseLiteral()
			if err != nil {
				return nil, err
			}
			if out, err = p.appendWithPostfix(out, unit); err != nil {
				return nil, err
			}
		case c == '[':
			e, err := p.parseClass()
			if err != nil {
				return nil, err
			}
			var err2 error
			if out, err2 = p.appendWithPostfix(out, seq{e}); err2 != nil {
				return nil, err2
			}
		case c == '(':
			p.pos++
			alts, err := p.parseAlternates()
			if err != nil {
				return nil, err
			}
			p.skipSpace()
			if p.pos >= len(p.src) || p.src[p.pos] != ')' {
				return nil, p.errf("unterminated group")
			}
			p.pos++
			var unit seq
			if len(alts) == 1 {
				unit = alts[0]
			} else {
				unit = seq{p.b.group(alts...)}
			}
			if out, err = p.appendWithPostfix(out, unit); err != nil {
				return nil, err
			}
		case isRuleNameChar(c):
			start := p.pos
			for p.pos < len(p.src) && isRuleNameChar(p.src[p.pos]) {
				p.pos++
			}
			name := p.src[start:p.pos]
			id, ok := p.ids[name]
			if !ok {
				return nil, p.errf("reference to undefined rule %q", name)
			}
			var err error
			if out, err = p.appendWithPostfix(out, seq{ref(id)}); err != nil {
				return nil, err
			}
		default:
			return nil, p.errf("unexpected character %q", c)
		}
	}
}

// appendWithPostfix applies any ? * + {n,m} operator following the just
// parsed unit, then concatenates it onto out.
func (p *gbnfParser) appendWithPostfix(out, unit seq) (seq, error) {
	p.skipSpace()
	if p.pos < len(p.src) {
		switch p.src[p.pos] {
		case '?':
			p.pos++
			return append(out, p.b.opt(unit)), nil
		case '*':
			p.pos++
			return append(out, p.b.star(unit)), nil
		case '+':
			p.pos++
			return append(out, p.b.plus(unit)), nil
		case '{':
			p.pos++
			minCount, err := p.parseInt()
			if err != nil {
				return nil, err
			}
			maxCount := minCount
			p.skipSpace()
			if p.pos < len(p.src) && p.src[p.pos] == ',' {
				p.pos++
				p.skipSpace()
				if p.pos < len(p.src) && p.src[p.pos] == '}' {
					maxCount = -1
				} else if maxCount, err = p.parseInt(); err != nil {
					return nil, err
				}
			}
			p.skipSpace()
			if p.pos >= len(p.src) || p.src[p.pos] != '}' {
				return nil, p.errf("unterminated repetition bounds")
			}
			p.pos++
			if maxCount >= 0 && maxCount < minCount {
				return nil, p.errf("repetition max %d < min %d", maxCount, minCount)
			}
			return append(out, p.b.repeat(unit, minCount, maxCount)), nil
		}
	}
	return append(out, unit...), nil
}

func (p *gbnfParser) parseInt() (int, error) {
	p.skipSpace()
	start := p.pos
	for p.pos < len(p.src) && p.src[p.pos] >= '0' && p.src[p.pos] <= '9' {
		p.pos++
	}
	if p.pos == start {
		return 0, p.errf("expected number")
	}
	n, err := strconv.Atoi(p.src[start:p.pos])
	if err != nil {
		return 0, p.errf("bad number %q", p.src[start:p.pos])
	}
	return n, nil
}

// parseLiteral parses a "..." literal into a sequence of single-byte
// elements.
func (p *gbnfParser) parseLiteral() (seq, error) {
	p.pos++ // opening quote
	out := seq{}
	for {
		if p.pos >= len(p.src) {
			return nil, p.errf("unterminated literal")
		}
		c := p.src[p.pos]
		if c == '"' {
			p.pos++
			return out, nil
		}
		bs, err := p.parseCharBytes()
		if err != nil {
			return nil, err
		}
		for _, b := range bs {
			out = append(out, chars(b))
		}
	}
}

// parseClass parses a [...] character class into one byte element.
func (p *gbnfParser) parseClass() (elem, error) {
	p.pos++ // opening bracket
	var set byteSet
	negate := false
	if p.pos < len(p.src) && p.src[p.pos] == '^' {
		negate = true
		p.pos++
	}
	for {
		if p.pos >= len(p.src) {
			return elem{}, p.errf("unterminated character class")
		}
		if p.src[p.pos] == ']' {
			p.pos++
			break
		}
		lo, err := p.parseClassByte()
		if err != nil {
			return elem{}, err
		}
		hi := lo
		if p.pos+1 < len(p.src) && p.src[p.pos] == '-' && p.src[p.pos+1] != ']' {
			p.pos++
			if hi, err = p.parseClassByte(); err != nil {
				return elem{}, err
			}
			if hi < lo {
				return elem{}, p.errf("inverted class range")
			}
		}
		set.addRange(lo, hi)
	}
	if negate {
		set.invert()
	}
	return elem{kind: elemBytes, set: set}, nil
}

// parseClassByte parses one class entry endpoint, which must be a single
// byte (multi-byte escapes are not part of the emitted subset).
func (p *gbnfParser) parseClassByte() (byte, error) {
	bs, err := p.parseCharBytes()
	if err != nil {
		return 0, err
	}
	if len(bs) != 1 {
		return 0, p.errf("multi-byte character in class")
	}
	return bs[0], nil
}

// parseCharBytes parses one character — raw or escaped — returning its
// bytes. Raw multi-byte UTF-8 is passed through byte by byte.
func (p *gbnfParser) parseCharBytes() ([]byte, error) {
	c := p.src[p.pos]
	if c != '\\' {
		p.pos++
		return []byte{c}, nil
	}
	p.pos++
	if p.pos >= len(p.src) {
		return nil, p.errf("dangling escape")
	}
	e := p.src[p.pos]
	p.pos++
	switch e {
	case 'n':
		return []byte{'\n'}, nil
	case 'r':
		return []byte{'\r'}, nil
	case 't':
		return []byte{'\t'}, nil
	case '\\', '"', '[', ']', '-', '^', '/':
		return []byte{e}, nil
	case 'x':
		v, err := p.parseHex(2)
		if err != nil {
			return nil, err
		}
		return []byte{byte(v)}, nil
	case 'u':
		v, err := p.parseHex(4)
		if err != nil {
			return nil, err
		}
		return utf8.AppendRune(nil, rune(v)), nil
	default:
		return nil, p.errf("unsupported escape \\%c", e)
	}
}

func (p *gbnfParser) parseHex(n int) (uint32, error) {
	if p.pos+n > len(p.src) {
		return 0, p.errf("truncated hex escape")
	}
	v, err := strconv.ParseUint(p.src[p.pos:p.pos+n], 16, 32)
	if err != nil {
		return 0, p.errf("bad hex escape %q", p.src[p.pos:p.pos+n])
	}
	p.pos += n
	return uint32(v), nil
}

func isRuleNameChar(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-'
}
