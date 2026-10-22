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

// ResolveRequest maps recognized effort names to the nearest advertised level,
// preferring the higher level on a tie and lexical order for same-rank aliases.
// Exact model-defined names stay unchanged.
// Off and unknown names are rejected when a known model cannot honor them.
// A boolean true enables a named-level model using its existing default level.
func (t *Thinking) ResolveRequest(requested *ThinkValue) (*ThinkValue, error) {
	if !t.Valid() || requested == nil || requested.Value == nil {
		return requested, nil
	}
	if t.Supports(requested.Value) {
		return requested, nil
	}
	if requested.Value == true {
		for _, value := range t.Values {
			if _, named := value.(string); named {
				return requested, nil
			}
		}
	}
	if level, named := requested.Value.(string); named {
		if rank := thinkingLevelRank(level); rank >= 0 {
			closest, closestRank, closestDistance := "", -1, 0
			for _, value := range t.Values {
				candidate, named := value.(string)
				if !named {
					continue
				}
				candidateRank := thinkingLevelRank(candidate)
				if candidateRank < 0 {
					continue
				}
				distance := candidateRank - rank
				if distance < 0 {
					distance = -distance
				}
				preferCandidate := candidateRank > closestRank || (candidateRank == closestRank && candidate < closest)
				if closest == "" || distance < closestDistance || (distance == closestDistance && preferCandidate) {
					closest, closestRank, closestDistance = candidate, candidateRank, distance
				}
			}
			if closest != "" {
				return &ThinkValue{Value: closest}, nil
			}
		}
	}
	values, _ := json.Marshal(t.Values)
	if requested.Value == false {
		return nil, fmt.Errorf("this model cannot disable thinking; supported values: %s", values)
	}
	return nil, fmt.Errorf("unsupported thinking value %v; supported values: %s", requested.Value, values)
}

// Only established effort names have a cross-model ordering. The legacy maximum
// aliases share a rank; exact advertised names are preserved before mapping.
func thinkingLevelRank(level string) int {
	switch level {
	case "minimal":
		return 0
	case "low":
		return 1
	case "medium":
		return 2
	case "high":
		return 3
	case "xhigh", "max", "ultra":
		return 4
	default:
		return -1
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
