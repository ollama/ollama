// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// This file is derived from the util.go file of
// github.com/google/jsonschema-go (v0.4.3).

package jsonschema

import (
	"reflect"
	"strings"
	"sync"
)

var jsonNamesMap sync.Map // from reflect.Type to map[string]bool

// jsonNames returns the set of JSON object keys that t will marshal into,
// including fields from embedded structs in t.
// t must be a struct type.
func jsonNames(t reflect.Type) map[string]bool {
	// Lock not necessary: at worst we'll duplicate work.
	if val, ok := jsonNamesMap.Load(t); ok {
		return val.(map[string]bool)
	}
	m := map[string]bool{}
	for field := range t.Fields() {
		// handle embedded structs
		if field.Anonymous {
			fieldType := field.Type
			if fieldType.Kind() == reflect.Pointer {
				fieldType = fieldType.Elem()
			}
			for n := range jsonNames(fieldType) {
				m[n] = true
			}
			continue
		}
		info := fieldJSONInfo(field)
		if !info.omit {
			m[info.name] = true
		}
	}
	jsonNamesMap.Store(t, m)
	return m
}

type jsonInfo struct {
	omit     bool            // unexported or first tag element is "-"
	name     string          // Go field name or first tag element. Empty if omit is true.
	settings map[string]bool // "omitempty", "omitzero", etc.
}

// fieldJSONInfo reports information about how encoding/json
// handles the given struct field.
// If the field is unexported, jsonInfo.omit is true and no other jsonInfo field
// is populated.
// If the field is exported and has no tag, then name is the field's name and all
// other fields are false.
// Otherwise, the information is obtained from the tag.
func fieldJSONInfo(f reflect.StructField) jsonInfo {
	if !f.IsExported() {
		return jsonInfo{omit: true}
	}
	info := jsonInfo{name: f.Name}
	if tag, ok := f.Tag.Lookup("json"); ok {
		name, rest, found := strings.Cut(tag, ",")
		// "-" means omit, but "-," means the name is "-"
		if name == "-" && !found {
			return jsonInfo{omit: true}
		}
		if name != "" {
			info.name = name
		}
		if len(rest) > 0 {
			info.settings = map[string]bool{}
			for s := range strings.SplitSeq(rest, ",") {
				info.settings[s] = true
			}
		}
	}
	return info
}
