package renderers

import (
	"bytes"
	"encoding/json"
)

// marshalWithSpaces marshals v to JSON and adds a space after each ':' and ','
// that appears outside of string values. This matches the formatting expected
// by certain model architectures.
func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return addJSONSpaces(b), nil
}

// marshalWithSpacesNoHTMLEscape matches Python/Jinja JSON string escaping while
// retaining the spaced separators used by model chat templates.
func marshalWithSpacesNoHTMLEscape(v any) ([]byte, error) {
	var b bytes.Buffer
	encoder := json.NewEncoder(&b)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(v); err != nil {
		return nil, err
	}
	return addJSONSpaces(bytes.TrimSuffix(b.Bytes(), []byte{'\n'})), nil
}

func addJSONSpaces(b []byte) []byte {
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
