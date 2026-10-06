// Package mapx provides an ordered map with JSON support, wrapping
// tailscale.com/types/mapx.OrderedMap.
package mapx

import (
	"bytes"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"

	"tailscale.com/types/mapx"
)

// OrderedMap is a map that maintains the order of its keys, and that
// marshals to and from JSON preserving that order.
//
// As with a Go map, the keys must marshal to and unmarshal from JSON
// strings; in practice K is a string or a named string type.
type OrderedMap[K comparable, V any] struct {
	mapx.OrderedMap[K, V]
}

// Len returns the number of entries in m. A nil map has length zero.
func (m *OrderedMap[K, V]) Len() (n int) {
	if m != nil {
		for range m.All() {
			n++
		}
	}
	return n
}

// MarshalJSON implements json.Marshaler, encoding the map as a JSON object
// with the keys in insertion order.
func (m OrderedMap[K, V]) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	first := true
	for k, v := range m.All() {
		if !first {
			buf.WriteByte(',')
		}
		first = false
		kb, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		vb, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		buf.Write(kb)
		buf.WriteByte(':')
		buf.Write(vb)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// UnmarshalJSON implements json.Unmarshaler, decoding a JSON object into the
// map with the keys in the order they appear in data. JSON null clears the
// map, and an empty object unmarshals to an empty map, like a Go map.
func (m *OrderedMap[K, V]) UnmarshalJSON(data []byte) error {
	dec := jsontext.NewDecoder(bytes.NewReader(data))
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	if tok.Kind() == 'n' { // JSON null
		*m = OrderedMap[K, V]{}
		return nil
	}
	if tok.Kind() != '{' {
		return fmt.Errorf("expected a JSON object, got %s", tok)
	}
	*m = OrderedMap[K, V]{}
	for dec.PeekKind() != '}' {
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		// A token is valid only until the next call on the decoder,
		// so extract the object name before reading the value.
		name := tok.String()
		kb, err := json.Marshal(name)
		if err != nil {
			return err
		}
		var k K
		if err := json.Unmarshal(kb, &k); err != nil {
			return err
		}
		var v V
		if err := json.UnmarshalDecode(dec, &v); err != nil {
			return err
		}
		m.Set(k, v)
	}
	// Consume the closing '}'.
	_, err = dec.ReadToken()
	return err
}
