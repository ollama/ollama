// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// This file is derived from the schema.go and util.go files of
// github.com/google/jsonschema-go (v0.4.3), with the map-valued keyword
// fields changed to ordered maps.

package jsonschema

import (
	"bytes"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/ollama/ollama/types/mapx"
)

func (s *Schema) UnmarshalJSON(data []byte) error {
	// A JSON boolean is a valid schema.
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		if b {
			// true is the empty schema, which validates everything.
			*s = Schema{}
		} else {
			// false is the schema that validates nothing.
			*s = *falseSchema()
		}
		return nil
	}

	var js schemaJSON
	if err := unmarshalStructWithMap(data, &js, "Extra"); err != nil {
		return err
	}

	*s = Schema{
		ID:                    js.ID,
		Schema:                js.Schema,
		Ref:                   js.Ref,
		Comment:               js.Comment,
		Defs:                  js.Defs,
		Definitions:           js.Definitions,
		Vocabulary:            js.Vocabulary,
		Anchor:                js.Anchor,
		DynamicAnchor:         js.DynamicAnchor,
		DynamicRef:            js.DynamicRef,
		Title:                 js.Title,
		Description:           js.Description,
		Default:               js.Default,
		Deprecated:            js.Deprecated,
		ReadOnly:              js.ReadOnly,
		WriteOnly:             js.WriteOnly,
		Examples:              js.Examples,
		Enum:                  js.Enum,
		MultipleOf:            js.MultipleOf,
		Minimum:               js.Minimum,
		Maximum:               js.Maximum,
		ExclusiveMinimum:      js.ExclusiveMinimum,
		ExclusiveMaximum:      js.ExclusiveMaximum,
		Pattern:               js.Pattern,
		PrefixItems:           js.PrefixItems,
		AdditionalItems:       js.AdditionalItems,
		UniqueItems:           js.UniqueItems,
		Contains:              js.Contains,
		UnevaluatedItems:      js.UnevaluatedItems,
		Required:              js.Required,
		DependentRequired:     js.DependentRequired,
		Properties:            js.Properties,
		PatternProperties:     js.PatternProperties,
		AdditionalProperties:  js.AdditionalProperties,
		PropertyNames:         js.PropertyNames,
		UnevaluatedProperties: js.UnevaluatedProperties,
		AllOf:                 js.AllOf,
		AnyOf:                 js.AnyOf,
		OneOf:                 js.OneOf,
		Not:                   js.Not,
		If:                    js.If,
		Then:                  js.Then,
		Else:                  js.Else,
		DependentSchemas:      js.DependentSchemas,
		ContentEncoding:       js.ContentEncoding,
		ContentMediaType:      js.ContentMediaType,
		ContentSchema:         js.ContentSchema,
		Format:                js.Format,
		Extra:                 js.Extra,
	}

	// Unmarshal "type" as either Type or Types.
	if len(js.Type) > 0 {
		var err error
		switch js.Type[0] {
		case '"':
			err = json.Unmarshal(js.Type, &s.Type)
		case '[':
			err = json.Unmarshal(js.Type, &s.Types)
		default:
			err = fmt.Errorf(`invalid value for "type": %q`, js.Type)
		}
		if err != nil {
			return err
		}
	}

	// Unmarshal "items" as either Items or ItemsArray.
	if len(js.Items) > 0 {
		var err error
		switch js.Items[0] {
		case '[':
			var schemas []*Schema
			err = json.Unmarshal(js.Items, &schemas)
			s.ItemsArray = schemas
		default:
			var schema Schema
			err = json.Unmarshal(js.Items, &schema)
			s.Items = &schema
		}
		if err != nil {
			return err
		}
	}

	// Unmarshal "dependencies" values as either string arrays or schemas
	// and assign them to DependencyStrings or DependencySchemas.
	if len(js.Dependencies) > 0 {
		var deps mapx.OrderedMap[string, jsontext.Value]
		if err := json.Unmarshal(js.Dependencies, &deps); err != nil {
			return err
		}
		for k, v := range deps.All() {
			if len(v) == 0 {
				continue
			}
			switch v[0] {
			case '[':
				var dstrings []string
				if err := json.Unmarshal(v, &dstrings); err != nil {
					return err
				}
				if s.DependencyStrings == nil {
					s.DependencyStrings = new(mapx.OrderedMap[string, []string])
				}
				s.DependencyStrings.Set(k, dstrings)
			default:
				var dschema Schema
				if err := json.Unmarshal(v, &dschema); err != nil {
					return err
				}
				if s.DependencySchemas == nil {
					s.DependencySchemas = new(mapx.OrderedMap[string, *Schema])
				}
				s.DependencySchemas.Set(k, &dschema)
			}
		}
	}

	// Setting Const to a pointer to null will marshal properly, but won't
	// unmarshal: the *any is set to nil, not a pointer to nil.
	if err := unmarshalAnyPtr(&s.Const, js.Const); err != nil {
		return err
	}

	set := func(dst **int, src *integer) {
		if src != nil {
			*dst = new(int(*src))
		}
	}

	set(&s.MinLength, js.MinLength)
	set(&s.MaxLength, js.MaxLength)
	set(&s.MinItems, js.MinItems)
	set(&s.MaxItems, js.MaxItems)
	set(&s.MinProperties, js.MinProperties)
	set(&s.MaxProperties, js.MaxProperties)
	set(&s.MinContains, js.MinContains)
	set(&s.MaxContains, js.MaxContains)

	return nil
}

// unmarshalStructWithMap is the inverse of [marshalStructWithMap].
// T has the same restrictions as in that function.
func unmarshalStructWithMap[T any](data []byte, v *T, mapField string) error {
	// Unmarshal into the struct, ignoring unknown fields.
	if err := json.Unmarshal(data, v); err != nil {
		return err
	}
	// Unmarshal into the map, preserving the order of the keys.
	var m mapx.OrderedMap[string, any]
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	// Delete from the map the fields of the struct.
	for n := range jsonNames(reflect.TypeFor[T]()) {
		m.Delete(n)
	}
	if m.Len() != 0 {
		reflect.ValueOf(v).Elem().FieldByName(mapField).Set(reflect.ValueOf(&m))
	}
	return nil
}

func unmarshalAnyPtr(p **any, raw jsontext.Value) error {
	if len(raw) == 0 {
		return nil
	}
	if bytes.Equal(raw, []byte("null")) {
		*p = new(any)
		return nil
	}
	return json.Unmarshal(raw, p)
}

type integer int32 // for the integer-valued fields of Schema

func (ip *integer) UnmarshalJSON(data []byte) error {
	if len(data) == 0 {
		// nothing to do
		return nil
	}
	// If there is a decimal point, src is a floating-point number.
	var i int64
	if bytes.ContainsRune(data, '.') {
		var f float64
		if err := json.Unmarshal(data, &f); err != nil {
			return errors.New("not a number")
		}
		i = int64(f)
		if float64(i) != f {
			return errors.New("not an integer value")
		}
	} else {
		if err := json.Unmarshal(data, &i); err != nil {
			return errors.New("cannot be unmarshaled into an int")
		}
	}
	// Ensure behavior is the same on both 32-bit and 64-bit systems.
	if i < math.MinInt32 || i > math.MaxInt32 {
		return errors.New("integer is out of range")
	}
	*ip = integer(i)
	return nil
}

// intToInteger converts a *int to a *integer, or nil to nil.
func intToInteger(p *int) *integer {
	if p == nil {
		return nil
	}
	return new(integer(*p))
}
