package jsonutil

import (
	"bytes"
	"encoding/json"
)

// Marshal matches json.Marshal except it leaves <, >, and & unescaped.
func Marshal(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(v); err != nil {
		return nil, err
	}

	return bytes.TrimSuffix(buf.Bytes(), []byte{'\n'}), nil
}
