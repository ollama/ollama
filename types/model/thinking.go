package model

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Thinking describes explicit thinking controls and the default used on omission.
// Values are booleans or named strings. A nil descriptor means unknown.
type Thinking struct {
	Values  []any `json:"values"`
	Default any   `json:"default"`
}

// Valid reports whether the descriptor has distinct controls and a supported default.
func (t *Thinking) Valid() bool {
	if t == nil || len(t.Values) == 0 {
		return false
	}
	for i, value := range t.Values {
		switch v := value.(type) {
		case bool:
		case string:
			if v == "" {
				return false
			}
		default:
			return false
		}
		if slices.Contains(t.Values[:i], value) {
			return false
		}
	}
	return t.Supports(t.Default)
}

// Supports reports whether a boolean or named level is explicitly advertised.
// Other names may still be accepted by the endpoint and resolve to the default.
func (t *Thinking) Supports(value any) bool {
	if t == nil {
		return false
	}
	switch value.(type) {
	case bool, string:
		return slices.Contains(t.Values, value)
	default:
		return false
	}
}

// Clone returns an independent copy.
func (t *Thinking) Clone() *Thinking {
	if t == nil {
		return nil
	}
	return &Thinking{Values: slices.Clone(t.Values), Default: t.Default}
}

// ThinkValue represents a boolean or model-defined thinking level.
type ThinkValue struct {
	// Value can be a bool or string
	Value any
}

// IsValid checks the transport type. The model resolves supported effort names.
func (t *ThinkValue) IsValid() bool {
	if t == nil || t.Value == nil {
		return true // nil is valid (means not set)
	}

	switch t.Value.(type) {
	case bool:
		return true
	case string:
		return true
	default:
		return false
	}
}

// IsBool returns true if the value is a boolean
func (t *ThinkValue) IsBool() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(bool)
	return ok
}

// IsString returns true if the value is a string
func (t *ThinkValue) IsString() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(string)
	return ok
}

// Bool returns the value as a bool (true if enabled in any way)
func (t *ThinkValue) Bool() bool {
	if t == nil || t.Value == nil {
		return false
	}

	switch v := t.Value.(type) {
	case bool:
		return v
	case string:
		// Named levels request thinking; the renderer resolves the actual level.
		return true
	default:
		return false
	}
}

// String returns the value as a string
func (t *ThinkValue) String() string {
	if t == nil || t.Value == nil {
		return ""
	}

	switch v := t.Value.(type) {
	case string:
		return v
	case bool:
		if v {
			return "medium" // Default level when just true
		}
		return ""
	default:
		return ""
	}
}

// UnmarshalJSON implements json.Unmarshaler
func (t *ThinkValue) UnmarshalJSON(data []byte) error {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch value.(type) {
	case nil, bool, string:
		t.Value = value
		return nil
	default:
		return fmt.Errorf("think must be a boolean or string")
	}
}

// MarshalJSON implements json.Marshaler
func (t *ThinkValue) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}
