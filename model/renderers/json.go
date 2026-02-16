package renderers

import "encoding/json"

// marshalWithSpaces marshals v to JSON and adds a space after each ':' and ','
// that appears outside of string values. This matches the formatting expected
// by certain model architectures.
func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

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
	return out, nil
}
