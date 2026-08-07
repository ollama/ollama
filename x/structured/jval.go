package structured

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// jval is a parsed JSON value that preserves object key order and number
// source text, which encoding/json's map-based decoding discards. The
// schema converter needs both: property order determines the emitted
// grammar, and enum/const literals reproduce the schema's own number
// formatting.
type jkind uint8

const (
	jNull jkind = iota
	jBool
	jNum
	jStr
	jArr
	jObj
)

type jval struct {
	kind jkind
	b    bool
	num  string // number source text
	str  string
	arr  []*jval
	obj  []jkv
}

type jkv struct {
	k string
	v *jval
}

// parseOrdered decodes one JSON document into a jval tree.
func parseOrdered(data []byte) (*jval, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := parseValue(dec)
	if err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after JSON document")
	}
	return v, nil
}

func parseValue(dec *json.Decoder) (*jval, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case nil:
		return &jval{kind: jNull}, nil
	case bool:
		return &jval{kind: jBool, b: t}, nil
	case json.Number:
		return &jval{kind: jNum, num: t.String()}, nil
	case string:
		return &jval{kind: jStr, str: t}, nil
	case json.Delim:
		switch t {
		case '{':
			out := &jval{kind: jObj}
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := keyTok.(string)
				if !ok {
					return nil, fmt.Errorf("unexpected object key %v", keyTok)
				}
				val, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				out.obj = append(out.obj, jkv{k: key, v: val})
			}
			if _, err := dec.Token(); err != nil { // closing }
				return nil, err
			}
			return out, nil
		case '[':
			out := &jval{kind: jArr}
			for dec.More() {
				val, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				out.arr = append(out.arr, val)
			}
			if _, err := dec.Token(); err != nil { // closing ]
				return nil, err
			}
			return out, nil
		}
	}
	return nil, fmt.Errorf("unexpected token %v", tok)
}

// get returns the value for key, or nil when v is not an object or lacks
// the key. Nil-safe.
func (v *jval) get(key string) *jval {
	if v == nil || v.kind != jObj {
		return nil
	}
	for _, kv := range v.obj {
		if kv.k == key {
			return kv.v
		}
	}
	return nil
}

func (v *jval) has(key string) bool { return v.get(key) != nil }

// isEmptyObject reports whether v is {} — the "any schema" per b10091's
// object fallback.
func (v *jval) isEmptyObject() bool {
	return v != nil && v.kind == jObj && len(v.obj) == 0
}

// typeString returns the schema's "type" when it is a string, else "".
func (v *jval) typeString() string {
	if t := v.get("type"); t != nil && t.kind == jStr {
		return t.str
	}
	return ""
}

// typeIsNullOr reports whether "type" is absent or equals s — the guard
// most converter branches use.
func (v *jval) typeIsNullOr(s string) bool {
	t := v.get("type")
	return t == nil || (t.kind == jStr && t.str == s)
}

// int64Value converts a numeric schema value to int64, truncating floats
// the way nlohmann's get<int64_t> does.
func (v *jval) int64Value() (int64, error) {
	if v == nil || v.kind != jNum {
		return 0, fmt.Errorf("expected a number, got %s", v.dump())
	}
	if i, err := strconv.ParseInt(v.num, 10, 64); err == nil {
		return i, nil
	}
	f, err := strconv.ParseFloat(v.num, 64)
	if err != nil {
		return 0, err
	}
	return int64(f), nil
}

// intValue converts a numeric schema value to int (for item/length bounds).
func (v *jval) intValue() (int, error) {
	i, err := v.int64Value()
	return int(i), err
}

// copyReplaceType returns a shallow copy of the object with its "type"
// entry replaced in place, preserving key order.
func (v *jval) copyReplaceType(t string) *jval {
	out := &jval{kind: jObj, obj: make([]jkv, 0, len(v.obj)+1)}
	replaced := false
	for _, kv := range v.obj {
		if kv.k == "type" {
			out.obj = append(out.obj, jkv{k: "type", v: &jval{kind: jStr, str: t}})
			replaced = true
		} else {
			out.obj = append(out.obj, kv)
		}
	}
	if !replaced {
		out.obj = append(out.obj, jkv{k: "type", v: &jval{kind: jStr, str: t}})
	}
	return out
}

// dump renders the value as compact JSON, preserving object order and
// number source text.
func (v *jval) dump() string {
	var sb strings.Builder
	v.dumpTo(&sb)
	return sb.String()
}

func (v *jval) dumpTo(sb *strings.Builder) {
	if v == nil {
		sb.WriteString("null")
		return
	}
	switch v.kind {
	case jNull:
		sb.WriteString("null")
	case jBool:
		if v.b {
			sb.WriteString("true")
		} else {
			sb.WriteString("false")
		}
	case jNum:
		sb.WriteString(v.num)
	case jStr:
		dumpJSONString(sb, v.str)
	case jArr:
		sb.WriteByte('[')
		for i, e := range v.arr {
			if i > 0 {
				sb.WriteByte(',')
			}
			e.dumpTo(sb)
		}
		sb.WriteByte(']')
	case jObj:
		sb.WriteByte('{')
		for i, kv := range v.obj {
			if i > 0 {
				sb.WriteByte(',')
			}
			dumpJSONString(sb, kv.k)
			sb.WriteByte(':')
			kv.v.dumpTo(sb)
		}
		sb.WriteByte('}')
	}
}

// dumpJSONString writes s as a JSON string literal: '"' and '\\' and
// control characters escaped, everything else (including non-ASCII UTF-8)
// verbatim.
func dumpJSONString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '"':
			sb.WriteString(`\"`)
		case c == '\\':
			sb.WriteString(`\\`)
		case c == '\b':
			sb.WriteString(`\b`)
		case c == '\f':
			sb.WriteString(`\f`)
		case c == '\n':
			sb.WriteString(`\n`)
		case c == '\r':
			sb.WriteString(`\r`)
		case c == '\t':
			sb.WriteString(`\t`)
		case c < 0x20:
			fmt.Fprintf(sb, `\u%04x`, c)
		default:
			sb.WriteByte(c)
		}
	}
	sb.WriteByte('"')
}
