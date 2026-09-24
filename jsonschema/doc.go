// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Package jsonschema defines a [Schema] type that represents a JSON schema,
// and supports marshaling and unmarshaling it to and from JSON.
//
// It is derived from the jsonschema package of
// github.com/google/jsonschema-go (v0.4.3, MIT license), with one main
// difference: the map-valued keywords ("properties", "$defs",
// "patternProperties" and so on) are [mapx.OrderedMap]s instead of Go maps,
// so that marshaled JSON is deterministic: object keys are written in the
// order they were inserted, and unmarshaling preserves the order of keys in
// the input.
//
// The parts of the original package that this package does not include:
// validation, reference resolution, schema inference from Go types, and JSON
// Pointer navigation.
//
// For example:
//
//	s := &jsonschema.Schema{
//		Type: "object",
//		Properties: new(mapx.OrderedMap[string, *jsonschema.Schema]),
//	}
//	s.Properties.Set("name", &jsonschema.Schema{Type: "string"})
//	s.Properties.Set("age", &jsonschema.Schema{Type: "integer"})
//	data, err := json.Marshal(s)
//	// data == {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}
package jsonschema
