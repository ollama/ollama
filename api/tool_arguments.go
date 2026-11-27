package api

import (
	"bytes"
	"encoding/json"
	"maps"
	"reflect"
	"slices"

	"github.com/ollama/ollama/internal/orderedmap"
)

// Key order belongs to the particular decoded object, even if an SDK caller
// moves it within an array. The live map holds all values and mutations.
type argumentOrder map[reflect.Value][]string

func (t *ToolCallFunctionArguments) decode(data []byte) error {
	if err := json.Unmarshal(data, t.om); err != nil {
		return err
	}
	if bytes.Count(data, []byte{'{'}) < 2 {
		return nil
	}
	order := make(argumentOrder)
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := order.record(dec, t.om); err != nil {
		return err
	}
	if len(order) > 0 {
		t.order = &order
	}
	return nil
}

// Record only object keys. The ordinary decoder above owns all Go values,
// duplicate-key handling, UTF-8 replacement and numeric errors.
func (o argumentOrder) record(dec *json.Decoder, live any) error {
	token, err := dec.Token()
	if err != nil {
		return err
	}
	switch token {
	case json.Delim('{'):
		fields, _ := live.(map[string]any)
		outer, _ := live.(*orderedmap.Map[string, any])
		var keys []string
		seen := make(map[string]bool, len(fields))
		for dec.More() {
			key, err := dec.Token()
			if err != nil {
				return err
			}
			name := key.(string)
			child := fields[name]
			if outer != nil {
				child, _ = outer.Get(name)
			} else if fields != nil && !seen[name] {
				keys = append(keys, name)
				seen[name] = true
			}
			if err := o.record(dec, child); err != nil {
				return err
			}
		}
		if len(fields) > 0 {
			o[reflect.ValueOf(fields)] = keys
		}
	case json.Delim('['):
		items, _ := live.([]any)
		for i := 0; dec.More(); i++ {
			var child any
			if i < len(items) {
				child = items[i]
			}
			if err := o.record(dec, child); err != nil {
				return err
			}
		}
	default:
		return nil
	}
	_, err = dec.Token()
	return err
}

// The length distinguishes two views starting at the same slice element.
type argumentContainer struct {
	address uintptr
	length  int
}

type orderedArgumentWriter struct {
	bytes.Buffer
	order    argumentOrder
	visiting map[argumentContainer]bool
}

func (t ToolCallFunctionArguments) encode() ([]byte, error) {
	w := orderedArgumentWriter{order: *t.order, visiting: make(map[argumentContainer]bool)}
	w.WriteByte('{')
	for name, value := range t.om.All() {
		if err := w.field(name, value); err != nil {
			return nil, err
		}
	}
	w.WriteByte('}')
	return w.Bytes(), nil
}

func (w *orderedArgumentWriter) field(name string, value any) error {
	if w.Bytes()[w.Len()-1] != '{' {
		w.WriteByte(',')
	}
	key, err := json.Marshal(name)
	if err != nil {
		return err
	}
	w.Write(key)
	w.WriteByte(':')
	return w.value(value)
}

func (w *orderedArgumentWriter) value(value any) error {
	switch v := value.(type) {
	case map[string]any, []any:
		ref := reflect.ValueOf(v)
		if ref.IsNil() {
			w.WriteString("null")
			return nil
		}
		id := argumentContainer{ref.Pointer(), ref.Len()}
		if w.visiting[id] {
			return &json.UnsupportedValueError{Value: ref, Str: "encountered a cycle via " + ref.Type().String()}
		}
		w.visiting[id] = true
		defer delete(w.visiting, id)
	default:
		b, err := json.Marshal(value)
		w.Write(b)
		return err
	}
	switch v := value.(type) {
	case map[string]any:
		w.WriteByte('{')
		keys := w.order[reflect.ValueOf(v)]
		written := 0
		for _, key := range keys {
			if child, ok := v[key]; ok {
				if err := w.field(key, child); err != nil {
					return err
				}
				written++
			}
		}
		if written < len(v) {
			added := maps.Clone(v)
			for _, k := range keys {
				delete(added, k)
			}
			for _, k := range slices.Sorted(maps.Keys(added)) {
				if err := w.field(k, added[k]); err != nil {
					return err
				}
			}
		}
		w.WriteByte('}')
	case []any:
		w.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				w.WriteByte(',')
			}
			if err := w.value(item); err != nil {
				return err
			}
		}
		w.WriteByte(']')
	}
	return nil
}
