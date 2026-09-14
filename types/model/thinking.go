package model

import "slices"

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
