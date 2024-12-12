package jsonschema

import (
	"bytes"
	"encoding/json"
	"errors"
)

// Schema holds a JSON schema.
type Schema struct {
	// Name is the name of the property. For the parent/root property, this
	// is "root". For child properties, this is the name of the property.
	Name string `json:"-"`

	// Type is the type of the property.
	//
	// TODO: Union types (e.g. make this a []string).
	Type string

	// PrefixItems is a list of schemas for each item in a tuple. By
	// default, the tuple is "closed." unless Items is set to true or a
	// valid Schema.
	PrefixItems []*Schema

	// Items is the schema for each item in a list.
	//
	// If it is missing, or its JSON value is "null" or "false", it is nil.
	// If the JSON value is "true", it is set to the empty Schema. If the
	// JSON value is an object, it will be decoded as a Schema.
	Items *Schema

	// MinItems specifies the minimum number of items allowed in a list.
	MinItems int

	// MaxItems specifies the maximum number of items allowed in a list.
	MaxItems int

	// Properties is the schema for each property of an object.
	Properties []*Schema

	// Format is the format of the property. This is used to validate the
	// property against a specific format.
	//
	// It is the callers responsibility to validate the property against
	// the format.
	Format string

	// Minimum specifies the minimum value for numeric properties.
	Minimum float64

	// Maximum specifies the maximum value for numeric properties.
	Maximum float64

	// Enum is a list of valid values for the property.
	Enum []json.RawMessage
}

func (s *Schema) UnmarshalJSON(data []byte) error {
	type S Schema
	w := struct {
		Properties props
		Items      items
		*S
	}{
		S: (*S)(s),
	}
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Items.set {
		s.Items = &w.Items.Schema
	}
	s.Properties = w.Properties
	return nil
}

type items struct {
	Schema
	set bool
}

func (s *items) UnmarshalJSON(data []byte) error {
	switch b := data[0]; b {
	case 't':
		*s = items{set: true}
	case '{':
		type I items
		if err := json.Unmarshal(data, (*I)(s)); err != nil {
			return err
		}
		s.set = true
	case 'n', 'f':
	default:
		return errors.New("invalid Items")
	}
	return nil
}

// EffectiveType returns the effective type of the schema. If the Type field is
// not empty, it is returned; otherwise:
//
//   - If the schema has both Properties and Items, it returns an empty string.
//   - If the schema has Properties, it returns "object".
//   - If the schema has Items, it returns "array".
//   - If the schema has neither Properties nor Items, it returns "value".
//
// The returned string is never empty.
func (d *Schema) EffectiveType() string {
	if d.Type == "" {
		if len(d.Properties) > 0 {
			return "object"
		}
		if len(d.PrefixItems) > 0 || d.Items != nil {
			return "array"
		}
		return "value"
	}
	return d.Type
}

// props is an ordered list of properties. The order of the properties
// is the order in which they were defined in the schema.
type props []*Schema

var _ json.Unmarshaler = (*props)(nil)

func (v *props) UnmarshalJSON(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if data[0] != '{' {
		return errors.New("expected object")
	}

	d := json.NewDecoder(bytes.NewReader(data))

	// TODO(bmizerany): Consider DisallowUnknownFields. Currently, we, like
	// llama.cpp, ignore unknown fields, which could be lead to unexpected
	// behavior for clients of this package, since they may not be aware
	// that "additionalFields", "itemsPrefix", etc, are being ignored.
	//
	// For now, just do what llama.cpp does.

	t, err := d.Token()
	if err != nil {
		return err
	}
	if t != json.Delim('{') {
		return errors.New("expected object")
	}
	for d.More() {
		// Use the first token (map key) as the property name, then
		// decode the rest of the object fields into a Schema and
		// append.
		t, err := d.Token()
		if err != nil {
			return err
		}
		if t == json.Delim('}') {
			return nil
		}
		s := &Schema{
			Name: t.(string),
		}
		if err := d.Decode(s); err != nil {
			return err
		}
		*v = append(*v, s)
	}
	return nil
}
