package renderers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"iter"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/toolargs"
)

// marshalWithSpaces marshals v to JSON and adds a space after each ':' and ','
// that appears outside of string values. This matches the formatting expected
// by certain model architectures.
func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return AddJSONSpaces(b), nil
}

func sortedToolCallArgumentValues(args api.ToolCallFunctionArguments) iter.Seq2[string, any] {
	type entry struct {
		name  string
		value any
	}
	var entries []entry
	for name, value := range toolargs.RenderValues(&args) {
		entries = append(entries, entry{name: name, value: value})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].name < entries[j].name })
	return func(yield func(string, any) bool) {
		for _, entry := range entries {
			if !yield(entry.name, entry.value) {
				return
			}
		}
	}
}

// AddJSONSpaces adds Python-style separator spaces to compact JSON without
// changing string values or escaping.
func AddJSONSpaces(b []byte) []byte {
	out := make([]byte, 0, len(b)+len(b)/8)
	inStr, esc := false, false
	for _, c := range b {
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
			out = append(out, c)
		case ':':
			out = append(out, ':', ' ')
		case ',':
			out = append(out, ',', ' ')
		default:
			out = append(out, c)
		}
	}
	return out
}

const maxPythonJSONDepth = 64

// PythonJSON renders v like Python's json.dumps(v, ensure_ascii=False),
// indented by indent spaces if indent > 0.
func PythonJSON(v any, indent int) (string, error) {
	sep := ", "
	if indent > 0 {
		sep = ","
	}
	return pythonJSON(v, indent, sep, ": ")
}

// PythonJSONCompact renders v like json.dumps(v, ensure_ascii=False, separators=(",", ":")).
func PythonJSONCompact(v any) (string, error) {
	return pythonJSON(v, 0, ",", ":")
}

func pythonJSON(v any, indent int, sep, colon string) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var render func(int) (string, error)
	render = func(depth int) (string, error) {
		if depth > maxPythonJSONDepth {
			return "", fmt.Errorf("JSON nesting exceeds %d levels", maxPythonJSONDepth)
		}
		token, err := dec.Token()
		if err != nil {
			return "", err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return pythonScalar(token)
		}
		var parts []string
		for dec.More() {
			var part string
			if indent > 0 {
				part = "\n" + strings.Repeat(" ", indent*(depth+1))
			}
			if delim == '{' {
				key, err := dec.Token()
				if err != nil {
					return "", err
				}
				quoted, _ := pythonScalar(key)
				part += quoted + colon
			}
			value, err := render(depth + 1)
			if err != nil {
				return "", err
			}
			parts = append(parts, part+value)
		}
		end, err := dec.Token()
		if err != nil {
			return "", err
		}
		closing := fmt.Sprint(end)
		if indent > 0 && len(parts) > 0 {
			closing = "\n" + strings.Repeat(" ", indent*depth) + closing
		}
		return delim.String() + strings.Join(parts, sep) + closing, nil
	}
	return render(0)
}

func pythonScalar(token json.Token) (string, error) {
	switch v := token.(type) {
	case nil:
		return "null", nil
	case bool:
		return strconv.FormatBool(v), nil
	case string:
		var b strings.Builder
		b.WriteByte('"')
		for _, r := range v {
			switch r {
			case '"', '\\':
				b.WriteByte('\\')
				b.WriteRune(r)
			case '\b':
				b.WriteString(`\b`)
			case '\f':
				b.WriteString(`\f`)
			case '\n':
				b.WriteString(`\n`)
			case '\r':
				b.WriteString(`\r`)
			case '\t':
				b.WriteString(`\t`)
			default:
				if r < 0x20 {
					fmt.Fprintf(&b, `\u%04x`, r)
				} else {
					b.WriteRune(r)
				}
			}
		}
		b.WriteByte('"')
		return b.String(), nil
	case json.Number:
		if !strings.ContainsAny(string(v), ".eE") {
			if v == "-0" {
				return "0", nil
			}
			return string(v), nil
		}
		n, err := v.Float64()
		if math.IsInf(n, 0) {
			if n < 0 {
				return "-Infinity", nil
			}
			return "Infinity", nil
		}
		if err != nil {
			return "", err
		}
		if a := math.Abs(n); a >= 1e16 || (a != 0 && a < 1e-4) {
			return strconv.FormatFloat(n, 'e', -1, 64), nil
		}
		s := strconv.FormatFloat(n, 'f', -1, 64)
		if !strings.ContainsRune(s, '.') {
			s += ".0"
		}
		return s, nil
	default:
		return "", fmt.Errorf("unsupported JSON value %T", token)
	}
}
