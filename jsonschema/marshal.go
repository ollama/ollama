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
	"fmt"
	"reflect"

	"github.com/ollama/ollama/types/mapx"
)

// schemaJSON is the intermediate representation that [Schema.MarshalJSON] and
// [Schema.UnmarshalJSON] marshal and unmarshal with the encoding/json/v2 package.
//
// It mirrors [Schema], except that:
//   - the union-typed keywords ("type", "items" and draft-07 "dependencies")
//     are jsontext.Values, handled explicitly;
//   - the integer-valued keywords use the [integer] type, so that JSON
//     numbers are range-checked on unmarshal.
//
// The map-valued keywords are [mapx.OrderedMap]s, which marshal to and from
// JSON with their keys in insertion order.
//
// encoding/json/v2 writes fields in declaration order, and this type keeps the
// keyword order used by github.com/google/jsonschema-go, so that the two
// packages marshal identically apart from the ordering of map keys.
//
// The jsontext.Value and bool fields use the omitzero tag option rather than
// omitempty: under encoding/json/v2, omitempty omits any field whose JSON
// encoding would be null, "", {} or [], which would also drop the keywords
// "const":null, "default":null, "properties":{} and the false-valued boolean
// keywords. omitzero omits only the zero value (a nil Value or a false bool),
// matching what encoding/json v1 did with omitempty. The map-valued keywords
// use it for the same reason: it omits a nil map but keeps an empty one.
type schemaJSON struct {
	// Keywords that require special handling.
	// Declared first so that they are written first, as in the original
	// package.
	Type         jsontext.Value                    `json:"type,omitzero"`
	Items        jsontext.Value                    `json:"items,omitzero"`
	Dependencies jsontext.Value                    `json:"dependencies,omitzero"`
	Properties   *mapx.OrderedMap[string, *Schema] `json:"properties,omitzero"`
	Const        jsontext.Value                    `json:"const,omitzero"`

	// The remaining keywords, in the order in which they are declared in
	// [Schema].
	ID                    string                             `json:"$id,omitempty"`
	Schema                string                             `json:"$schema,omitempty"`
	Ref                   string                             `json:"$ref,omitempty"`
	Comment               string                             `json:"$comment,omitempty"`
	Defs                  *mapx.OrderedMap[string, *Schema]  `json:"$defs,omitzero"`
	Definitions           *mapx.OrderedMap[string, *Schema]  `json:"definitions,omitzero"`
	Anchor                string                             `json:"$anchor,omitempty"`
	DynamicAnchor         string                             `json:"$dynamicAnchor,omitempty"`
	DynamicRef            string                             `json:"$dynamicRef,omitempty"`
	Vocabulary            *mapx.OrderedMap[string, bool]     `json:"$vocabulary,omitzero"`
	Title                 string                             `json:"title,omitempty"`
	Description           string                             `json:"description,omitempty"`
	Default               jsontext.Value                     `json:"default,omitzero"`
	Deprecated            bool                               `json:"deprecated,omitzero"`
	ReadOnly              bool                               `json:"readOnly,omitzero"`
	WriteOnly             bool                               `json:"writeOnly,omitzero"`
	Examples              []any                              `json:"examples,omitempty"`
	Enum                  []any                              `json:"enum,omitempty"`
	MultipleOf            *float64                           `json:"multipleOf,omitempty"`
	Minimum               *float64                           `json:"minimum,omitempty"`
	Maximum               *float64                           `json:"maximum,omitempty"`
	ExclusiveMinimum      *float64                           `json:"exclusiveMinimum,omitempty"`
	ExclusiveMaximum      *float64                           `json:"exclusiveMaximum,omitempty"`
	MinLength             *integer                           `json:"minLength,omitempty"`
	MaxLength             *integer                           `json:"maxLength,omitempty"`
	Pattern               string                             `json:"pattern,omitempty"`
	PrefixItems           []*Schema                          `json:"prefixItems,omitempty"`
	MinItems              *integer                           `json:"minItems,omitempty"`
	MaxItems              *integer                           `json:"maxItems,omitempty"`
	AdditionalItems       *Schema                            `json:"additionalItems,omitempty"`
	UniqueItems           bool                               `json:"uniqueItems,omitzero"`
	Contains              *Schema                            `json:"contains,omitempty"`
	MinContains           *integer                           `json:"minContains,omitempty"`
	MaxContains           *integer                           `json:"maxContains,omitempty"`
	UnevaluatedItems      *Schema                            `json:"unevaluatedItems,omitempty"`
	MinProperties         *integer                           `json:"minProperties,omitempty"`
	MaxProperties         *integer                           `json:"maxProperties,omitempty"`
	Required              []string                           `json:"required,omitempty"`
	DependentRequired     *mapx.OrderedMap[string, []string] `json:"dependentRequired,omitzero"`
	PatternProperties     *mapx.OrderedMap[string, *Schema]  `json:"patternProperties,omitzero"`
	AdditionalProperties  *Schema                            `json:"additionalProperties,omitempty"`
	PropertyNames         *Schema                            `json:"propertyNames,omitempty"`
	UnevaluatedProperties *Schema                            `json:"unevaluatedProperties,omitempty"`
	AllOf                 []*Schema                          `json:"allOf,omitempty"`
	AnyOf                 []*Schema                          `json:"anyOf,omitempty"`
	OneOf                 []*Schema                          `json:"oneOf,omitempty"`
	Not                   *Schema                            `json:"not,omitempty"`
	If                    *Schema                            `json:"if,omitempty"`
	Then                  *Schema                            `json:"then,omitempty"`
	Else                  *Schema                            `json:"else,omitempty"`
	DependentSchemas      *mapx.OrderedMap[string, *Schema]  `json:"dependentSchemas,omitzero"`
	ContentEncoding       string                             `json:"contentEncoding,omitempty"`
	ContentMediaType      string                             `json:"contentMediaType,omitempty"`
	ContentSchema         *Schema                            `json:"contentSchema,omitempty"`
	Format                string                             `json:"format,omitempty"`

	// Extra holds additional keywords beyond those above. It is merged
	// into the output by [marshalStructWithMap], and collected from the
	// input by [unmarshalStructWithMap].
	Extra *mapx.OrderedMap[string, any] `json:"-"`
}

func (s Schema) MarshalJSON() ([]byte, error) {
	// NOTE: Use a value receiver here to avoid the encoding/json bugs
	// described in golang/go#22967, golang/go#33993, and golang/go#55890.
	// With a pointer receiver, MarshalJSON is only called for Schema in
	// some cases (for example when the field value is addressable, or not
	// stored as a map value), which leads to inconsistent JSON encoding.
	// A value receiver makes Schema itself implement json.Marshaler and
	// ensures that encoding/json always calls this method.
	if err := s.basicChecks(); err != nil {
		return nil, err
	}
	// Marshal either Type or Types as "type".
	var typ jsontext.Value
	switch {
	case s.Type != "":
		t, err := json.Marshal(s.Type)
		if err != nil {
			return nil, err
		}
		typ = t
	case s.Types != nil:
		t, err := json.Marshal(s.Types)
		if err != nil {
			return nil, err
		}
		typ = t
	}

	// Marshal either Items or ItemsArray as "items".
	var items jsontext.Value
	switch {
	case s.Items != nil:
		i, err := json.Marshal(s.Items)
		if err != nil {
			return nil, err
		}
		items = i
	case s.ItemsArray != nil:
		i, err := json.Marshal(s.ItemsArray)
		if err != nil {
			return nil, err
		}
		items = i
	}

	// Merge DependencySchemas and DependencyStrings into "dependencies",
	// schemas first, in insertion order.
	var deps jsontext.Value
	if s.DependencySchemas != nil || s.DependencyStrings != nil {
		var merged mapx.OrderedMap[string, any]
		if s.DependencySchemas != nil {
			for k, v := range s.DependencySchemas.All() {
				merged.Set(k, v)
			}
		}
		if s.DependencyStrings != nil {
			for k, v := range s.DependencyStrings.All() {
				merged.Set(k, v)
			}
		}
		d, err := json.Marshal(&merged)
		if err != nil {
			return nil, err
		}
		deps = d
	}

	js := schemaJSON{
		Type:         typ,
		Items:        items,
		Dependencies: deps,

		ID:                    s.ID,
		Schema:                s.Schema,
		Ref:                   s.Ref,
		Comment:               s.Comment,
		Anchor:                s.Anchor,
		DynamicAnchor:         s.DynamicAnchor,
		DynamicRef:            s.DynamicRef,
		Title:                 s.Title,
		Description:           s.Description,
		Default:               s.Default,
		Deprecated:            s.Deprecated,
		ReadOnly:              s.ReadOnly,
		WriteOnly:             s.WriteOnly,
		Examples:              s.Examples,
		Enum:                  s.Enum,
		MultipleOf:            s.MultipleOf,
		Minimum:               s.Minimum,
		Maximum:               s.Maximum,
		ExclusiveMinimum:      s.ExclusiveMinimum,
		ExclusiveMaximum:      s.ExclusiveMaximum,
		MinLength:             intToInteger(s.MinLength),
		MaxLength:             intToInteger(s.MaxLength),
		Pattern:               s.Pattern,
		PrefixItems:           s.PrefixItems,
		MinItems:              intToInteger(s.MinItems),
		MaxItems:              intToInteger(s.MaxItems),
		AdditionalItems:       s.AdditionalItems,
		UniqueItems:           s.UniqueItems,
		Contains:              s.Contains,
		MinContains:           intToInteger(s.MinContains),
		MaxContains:           intToInteger(s.MaxContains),
		UnevaluatedItems:      s.UnevaluatedItems,
		MinProperties:         intToInteger(s.MinProperties),
		MaxProperties:         intToInteger(s.MaxProperties),
		Required:              s.Required,
		AdditionalProperties:  s.AdditionalProperties,
		PropertyNames:         s.PropertyNames,
		UnevaluatedProperties: s.UnevaluatedProperties,
		AllOf:                 s.AllOf,
		AnyOf:                 s.AnyOf,
		OneOf:                 s.OneOf,
		Not:                   s.Not,
		If:                    s.If,
		Then:                  s.Then,
		Else:                  s.Else,
		ContentEncoding:       s.ContentEncoding,
		ContentMediaType:      s.ContentMediaType,
		ContentSchema:         s.ContentSchema,
		Format:                s.Format,
		Extra:                 s.Extra,
	}

	// Marshal the map-valued keywords, even if the empty map (but not nil),
	// in insertion order.
	cv, err := constJSON(s.Const)
	if err != nil {
		return nil, err
	}
	js.Const = cv
	js.Defs = s.Defs
	js.Definitions = s.Definitions
	js.Vocabulary = s.Vocabulary
	js.DependentRequired = s.DependentRequired
	js.Properties = orderedProperties(s.Properties, s.PropertyOrder)
	js.PatternProperties = s.PatternProperties
	js.DependentSchemas = s.DependentSchemas

	bs, err := marshalStructWithMap(&js, "Extra")
	if err != nil {
		return nil, err
	}
	// Marshal {} as true and {"not":true} as false.
	// It is wasteful to do this here instead of earlier, but much easier.
	switch {
	case bytes.Equal(bs, []byte(`{}`)):
		bs = []byte("true")
	case bytes.Equal(bs, []byte(`{"not":true}`)):
		bs = []byte("false")
	}
	return bs, nil
}

// constJSON marshals a *any Const value, distinguishing a pointer to JSON null
// (which marshals as "null") from a nil pointer (which is omitted).
func constJSON(c *any) (jsontext.Value, error) {
	if c == nil {
		return nil, nil
	}
	bs, err := json.Marshal(*c)
	if err != nil {
		return nil, err
	}
	return bs, nil
}

// orderedProperties reorders the properties map: the properties listed in
// order first (if any), followed by the remaining properties in the order
// they were inserted. It returns nil if props is nil.
func orderedProperties(props *mapx.OrderedMap[string, *Schema], order []string) *mapx.OrderedMap[string, *Schema] {
	if props == nil {
		return nil
	}
	var ordered mapx.OrderedMap[string, *Schema]
	listed := make(map[string]bool, len(order))
	for _, name := range order {
		if prop, ok := props.GetOk(name); ok {
			ordered.Set(name, prop)
			listed[name] = true
		}
	}
	for name, prop := range props.All() {
		if !listed[name] {
			ordered.Set(name, prop)
		}
	}
	return &ordered
}

// marshalStructWithMap marshals its first argument to JSON, treating the field named
// mapField as an embedded map. The first argument must be a pointer to
// a struct. The underlying type of mapField must be a *mapx.OrderedMap[string, any],
// and it must have a "-" json tag, meaning it will not be marshaled.
//
// For example, given this struct:
//
//	type S struct {
//	   A int
//	   Extra *mapx.OrderedMap[string, any] `json:"-"`
//	}
//
// and this value:
//
//	s := S{A: 1, Extra: om("B", 2)}
//
// the call marshalJSONWithMap(s, "Extra") would return
//
//	{"A": 1, "B": 2}
//
// It is an error if the map contains the same key as another struct field's
// JSON name.
func marshalStructWithMap[T any](s *T, mapField string) ([]byte, error) {
	// Marshal the struct and the map separately, and concatenate the bytes.
	// This strategy is dramatically less complicated than
	// constructing a synthetic struct or map with the combined keys.
	if s == nil {
		return []byte("null"), nil
	}
	s2 := *s
	vMapField := reflect.ValueOf(&s2).Elem().FieldByName(mapField)
	mapVal := vMapField.Interface().(*mapx.OrderedMap[string, any])

	// Check for duplicates.
	if mapVal != nil {
		names := jsonNames(reflect.TypeFor[T]())
		for key := range mapVal.All() {
			if names[key] {
				return nil, fmt.Errorf("map key %q duplicates struct field", key)
			}
		}
	}

	structBytes, err := json.Marshal(s2)
	if err != nil {
		return nil, fmt.Errorf("marshalStructWithMap(%+v): %w", s, err)
	}
	if mapVal.Len() == 0 {
		return structBytes, nil
	}
	mapBytes, err := json.Marshal(mapVal)
	if err != nil {
		return nil, err
	}
	if len(structBytes) == 2 { // must be "{}"
		return mapBytes, nil
	}
	// "{X}" + "{Y}" => "{X,Y}"
	res := append(structBytes[:len(structBytes)-1], ',')
	res = append(res, mapBytes[1:]...)
	return res, nil
}
