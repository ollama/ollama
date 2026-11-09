package d1

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// maxRenderedState bounds indented output: indentation grows with the square
// of nesting depth, so a small deeply nested value renders to gigabytes.
var maxRenderedState = 8 << 20

// pyJSONIndent2 renders a JSON value exactly as Python
// json.dumps(json.loads(raw), ensure_ascii=False, indent=2).
func pyJSONIndent2(raw json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := pyDecode(dec)
	if err != nil {
		return "", err
	}
	if _, err := dec.Token(); err != io.EOF {
		return "", errors.New("d1: trailing data after JSON value")
	}
	var b strings.Builder
	pyWrite(&b, v, 0)
	if b.Len() > maxRenderedState {
		return "", fmt.Errorf("renders to more than %d bytes as indented JSON", maxRenderedState)
	}
	return b.String(), nil
}

// pyObject keeps keys in first-seen order; a repeated key keeps its first
// position and takes the last value, as a Python dict does.
type pyObject struct {
	keys   []string
	values map[string]any
}

func pyDecode(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := &pyObject{values: map[string]any{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("d1: object key %v", kt)
				}
				v, err := pyDecode(dec)
				if err != nil {
					return nil, err
				}
				if _, seen := obj.values[key]; !seen {
					obj.keys = append(obj.keys, key)
				}
				obj.values[key] = v
			}
			_, err := dec.Token()
			return obj, err
		case '[':
			list := []any{}
			for dec.More() {
				v, err := pyDecode(dec)
				if err != nil {
					return nil, err
				}
				list = append(list, v)
			}
			_, err := dec.Token()
			return list, err
		}
		return nil, fmt.Errorf("d1: unexpected %v", t)
	default:
		return t, nil
	}
}

func pyWrite(b *strings.Builder, v any, depth int) {
	if b.Len() > maxRenderedState {
		return
	}
	newline := func(d int) {
		if b.Len() > maxRenderedState {
			return
		}
		b.WriteByte('\n')
		b.WriteString(strings.Repeat("  ", d))
	}
	switch t := v.(type) {
	case *pyObject:
		if len(t.keys) == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteByte('{')
		for i, k := range t.keys {
			if i > 0 {
				b.WriteByte(',')
			}
			newline(depth + 1)
			b.WriteString(pyJSONString(k))
			b.WriteString(": ")
			pyWrite(b, t.values[k], depth+1)
		}
		newline(depth)
		b.WriteByte('}')
	case []any:
		if len(t) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
			}
			newline(depth + 1)
			pyWrite(b, e, depth+1)
		}
		newline(depth)
		b.WriteByte(']')
	case string:
		b.WriteString(pyJSONString(t))
	case json.Number:
		b.WriteString(pyNumber(string(t)))
	case bool:
		b.WriteString(strconv.FormatBool(t))
	case nil:
		b.WriteString("null")
	}
}

// pyNumber renders a JSON number literal as Python does after json.loads:
// integer literals stay exact integers, the rest become float reprs.
func pyNumber(lit string) string {
	if !strings.ContainsAny(lit, ".eE") {
		if strings.TrimLeft(lit, "-0") == "" {
			return "0"
		}
		return lit
	}
	// Overflow gives ±Inf and underflow 0 with ErrRange, as Python's float() does.
	f, _ := strconv.ParseFloat(lit, 64)
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}
	return pyRepr(f)
}

// pyRepr formats a finite float like Python repr: shortest round-trip digits,
// positional for exponents in [-4, 16) with at least one fractional digit,
// scientific otherwise.
func pyRepr(f float64) string {
	sci := strconv.FormatFloat(f, 'e', -1, 64)
	sign := ""
	if sci[0] == '-' {
		sign, sci = "-", sci[1:]
	}
	mant, expStr, _ := strings.Cut(sci, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)

	if exp < -4 || exp >= 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		es := "+"
		if exp < 0 {
			es, exp = "-", -exp
		}
		return fmt.Sprintf("%s%se%s%02d", sign, m, es, exp)
	}
	if exp < 0 {
		return sign + "0." + strings.Repeat("0", -exp-1) + digits
	}
	if len(digits) <= exp+1 {
		return sign + digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	}
	return sign + digits[:exp+1] + "." + digits[exp+1:]
}

// pyJSONString renders s as Python json.dumps(s, ensure_ascii=False).
func pyJSONString(s string) string {
	var b strings.Builder
	b.Grow(len(s) + 2)
	b.WriteByte('"')
	for i := range len(s) {
		c := s[i]
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if c < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, c)
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}
