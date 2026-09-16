// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// This file is derived from the schema.go file of
// github.com/google/jsonschema-go (v0.4.3), with the map-valued keyword
// fields changed to ordered maps.

package jsonschema

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"github.com/ollama/ollama/types/mapx"
)

// A Schema is a JSON schema object.
// It supports the keywords of both draft-07 and the 2020-12 draft
// specifications:
//   - Draft-07: https://json-schema.org/draft-07/draft-handrews-json-schema-01
//     and https://json-schema.org/draft-07/draft-handrews-json-schema-validation-01
//   - Draft 2020-12: https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-01
//     and https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-validation-01
//
// A Schema value may have non-zero values for more than one field:
// all relevant non-zero fields are used.
// There is one exception to provide more Go type-safety: the Type and Types fields
// are mutually exclusive.
//
// Unlike github.com/google/jsonschema-go, from which this package is derived,
// the map-valued fields are [mapx.OrderedMap]s rather than Go maps, so that
// marshaled JSON preserves the order in which keys were inserted.
//
// Since this struct is a Go representation of a JSON value, it inherits JSON's
// distinction between nil and empty. Nil maps and slices are considered absent,
// but empty ones are present and are marshaled. For example,
//
//	Schema{Enum: nil}
//
// is equivalent to an empty schema, while
//
//	Schema{Enum: []any{}}
//
// is a schema with an "enum" keyword whose value is the empty array.
type Schema struct {
	// core
	ID          string                            `json:"$id,omitempty"`
	Schema      string                            `json:"$schema,omitempty"`
	Ref         string                            `json:"$ref,omitempty"`
	Comment     string                            `json:"$comment,omitempty"`
	Defs        *mapx.OrderedMap[string, *Schema] `json:"$defs,omitempty"`
	Definitions *mapx.OrderedMap[string, *Schema] `json:"definitions,omitempty"`

	// split draft 7 Dependencies into DependencySchemas and DependencyStrings
	DependencySchemas *mapx.OrderedMap[string, *Schema]  `json:"-"`
	DependencyStrings *mapx.OrderedMap[string, []string] `json:"-"`

	Anchor        string                         `json:"$anchor,omitempty"`
	DynamicAnchor string                         `json:"$dynamicAnchor,omitempty"`
	DynamicRef    string                         `json:"$dynamicRef,omitempty"`
	Vocabulary    *mapx.OrderedMap[string, bool] `json:"$vocabulary,omitempty"`

	// metadata
	Title       string          `json:"title,omitempty"`
	Description string          `json:"description,omitempty"`
	Default     json.RawMessage `json:"default,omitempty"`
	Deprecated  bool            `json:"deprecated,omitempty"`
	ReadOnly    bool            `json:"readOnly,omitempty"`
	WriteOnly   bool            `json:"writeOnly,omitempty"`
	Examples    []any           `json:"examples,omitempty"`

	// validation
	// Use Type for a single type, or Types for multiple types; never both.
	Type  string   `json:"-"`
	Types []string `json:"-"`
	Enum  []any    `json:"enum,omitempty"`
	// Const is *any because a JSON null (Go nil) is a valid value.
	Const            *any     `json:"const,omitempty"`
	MultipleOf       *float64 `json:"multipleOf,omitempty"`
	Minimum          *float64 `json:"minimum,omitempty"`
	Maximum          *float64 `json:"maximum,omitempty"`
	ExclusiveMinimum *float64 `json:"exclusiveMinimum,omitempty"`
	ExclusiveMaximum *float64 `json:"exclusiveMaximum,omitempty"`
	MinLength        *int     `json:"minLength,omitempty"`
	MaxLength        *int     `json:"maxLength,omitempty"`
	Pattern          string   `json:"pattern,omitempty"`

	// arrays
	PrefixItems      []*Schema `json:"prefixItems,omitempty"`
	Items            *Schema   `json:"-"`
	ItemsArray       []*Schema `json:"-"`
	MinItems         *int      `json:"minItems,omitempty"`
	MaxItems         *int      `json:"maxItems,omitempty"`
	AdditionalItems  *Schema   `json:"additionalItems,omitempty"`
	UniqueItems      bool      `json:"uniqueItems,omitempty"`
	Contains         *Schema   `json:"contains,omitempty"`
	MinContains      *int      `json:"minContains,omitempty"` // *int, not int: default is 1, not 0
	MaxContains      *int      `json:"maxContains,omitempty"`
	UnevaluatedItems *Schema   `json:"unevaluatedItems,omitempty"`

	// objects
	MinProperties         *int                               `json:"minProperties,omitempty"`
	MaxProperties         *int                               `json:"maxProperties,omitempty"`
	Required              []string                           `json:"required,omitempty"`
	DependentRequired     *mapx.OrderedMap[string, []string] `json:"dependentRequired,omitempty"`
	Properties            *mapx.OrderedMap[string, *Schema]  `json:"properties,omitempty"`
	PatternProperties     *mapx.OrderedMap[string, *Schema]  `json:"patternProperties,omitempty"`
	AdditionalProperties  *Schema                            `json:"additionalProperties,omitempty"`
	PropertyNames         *Schema                            `json:"propertyNames,omitempty"`
	UnevaluatedProperties *Schema                            `json:"unevaluatedProperties,omitempty"`

	// logic
	AllOf []*Schema `json:"allOf,omitempty"`
	AnyOf []*Schema `json:"anyOf,omitempty"`
	OneOf []*Schema `json:"oneOf,omitempty"`
	Not   *Schema   `json:"not,omitempty"`

	// conditional
	If               *Schema                           `json:"if,omitempty"`
	Then             *Schema                           `json:"then,omitempty"`
	Else             *Schema                           `json:"else,omitempty"`
	DependentSchemas *mapx.OrderedMap[string, *Schema] `json:"dependentSchemas,omitempty"`

	// other
	// https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-validation-00#rfc.section-8
	ContentEncoding  string  `json:"contentEncoding,omitempty"`
	ContentMediaType string  `json:"contentMediaType,omitempty"`
	ContentSchema    *Schema `json:"contentSchema,omitempty"`

	// https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-validation-00#rfc.section.7
	Format string `json:"format,omitempty"`

	// Extra allows for additional keywords beyond those specified.
	Extra *mapx.OrderedMap[string, any] `json:"-"`

	// PropertyOrder records the ordering of properties for JSON rendering.
	//
	// If PropertyOrder is set, it controls the relative ordering of properties in [Schema.MarshalJSON].
	// The rendered JSON first lists any properties that appear in the PropertyOrder slice in the order
	// they appear, followed by all other properties in the order they were inserted into Properties.
	PropertyOrder []string `json:"-"`
}

// falseSchema returns a new Schema tree that fails to validate any value.
func falseSchema() *Schema {
	return &Schema{Not: &Schema{}}
}

// String returns a short description of the schema.
func (s *Schema) String() string {
	if s.ID != "" {
		return s.ID
	}
	if a := cmp.Or(s.Anchor, s.DynamicAnchor); a != "" {
		return fmt.Sprintf("anchor %s", a)
	}
	return "<anonymous schema>"
}

// CloneSchemas returns a copy of s.
// The copy is shallow except for sub-schemas, which are themelves copied with CloneSchemas.
// This allows both s and s.CloneSchemas() to appear as sub-schemas of the same parent.
// Ordered maps are cloned in insertion order.
func (s *Schema) CloneSchemas() *Schema {
	if s == nil {
		return nil
	}
	s2 := *s
	v := reflect.ValueOf(&s2)
	for _, info := range schemaFieldInfos {
		fv := v.Elem().FieldByIndex(info.sf.Index)
		switch info.sf.Type {
		case schemaType:
			sscss := fv.Interface().(*Schema)
			fv.Set(reflect.ValueOf(sscss.CloneSchemas()))

		case schemaSliceType:
			slice := fv.Interface().([]*Schema)
			slice = slices.Clone(slice)
			for i, ss := range slice {
				slice[i] = ss.CloneSchemas()
			}
			fv.Set(reflect.ValueOf(slice))

		case schemaMapType:
			m := fv.Interface().(*mapx.OrderedMap[string, *Schema])
			if m == nil {
				fv.Set(reflect.ValueOf(m))
				break
			}
			var m2 mapx.OrderedMap[string, *Schema]
			for k, ss := range m.All() {
				m2.Set(k, ss.CloneSchemas())
			}
			fv.Set(reflect.ValueOf(&m2))
		}
	}
	return &s2
}

func (s *Schema) basicChecks() error {
	if s.Type != "" && s.Types != nil {
		return errors.New("both Type and Types are set; at most one should be")
	}
	if s.Defs != nil && s.Definitions != nil {
		return errors.New("both Defs and Definitions are set; at most one should be")
	}
	if s.Items != nil && s.ItemsArray != nil {
		return errors.New("both Items and ItemsArray are set; at most one should be")
	}
	propertyOrderSeen := make(map[string]bool)
	for _, val := range s.PropertyOrder {
		if _, ok := propertyOrderSeen[val]; ok {
			// Duplicate found
			return fmt.Errorf("property order slice cannot contain duplicate entries, found duplicate %q", val)
		}
		propertyOrderSeen[val] = true
	}

	if s.DependencySchemas != nil {
		for key := range s.DependencySchemas.All() {
			// Check if the key exists in the dependency strings map
			if s.DependencyStrings != nil && s.DependencyStrings.Contains(key) {
				return fmt.Errorf("dependency key %q cannot be defined as both a schema and a string array", key)
			}
		}
	}
	return nil
}

var (
	schemaType      = reflect.TypeFor[*Schema]()
	schemaSliceType = reflect.TypeFor[[]*Schema]()
	schemaMapType   = reflect.TypeFor[*mapx.OrderedMap[string, *Schema]]()
)

type structFieldInfo struct {
	sf       reflect.StructField
	jsonName string
}

// the visible fields of Schema that contain schemas, sorted by JSON name
var schemaFieldInfos []structFieldInfo

func init() {
	t := reflect.VisibleFields(reflect.TypeFor[Schema]())
	for _, sf := range t {
		info := fieldJSONInfo(sf)
		if !info.omit {
			schemaFieldInfos = append(schemaFieldInfos, structFieldInfo{sf, info.name})
		} else {
			// The items and dependencies keywords are split into separate
			// fields to handle the union types they support in JSON, and
			// their fields have a "-" json tag. We still need them in
			// schemaFieldInfos so CloneSchemas copies the subschemas they
			// contain, so we add them here, assigning the jsonName of the
			// original keyword.
			switch sf.Name {
			case "Items", "ItemsArray":
				schemaFieldInfos = append(schemaFieldInfos, structFieldInfo{sf, "items"})
			case "DependencySchemas", "DependencyStrings":
				schemaFieldInfos = append(schemaFieldInfos, structFieldInfo{sf, "dependencies"})
			}
		}
	}
	slices.SortFunc(schemaFieldInfos, func(i1, i2 structFieldInfo) int {
		return cmp.Compare(i1.jsonName, i2.jsonName)
	})
}
