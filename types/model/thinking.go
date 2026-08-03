package model

import (
	"encoding/json"
	"fmt"
	"math"
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

// ThinkValue represents a boolean, a model-defined thinking level, or a
// positive integer thinking-token budget.
type ThinkValue struct {
	// Value can be a bool, string or int
	Value any
}

// thinkLevels are the effort levels that carry a thinking budget, weakest
// first. A model can define levels of its own; those are handed to it as a
// string and carry no budget, since the table below has no share for them.
var thinkLevels = []string{"minimal", "low", "medium", "high", "max"}

// thinkBudgetFraction maps an effort level to the share of the context window
// the model is allowed to spend on thinking. Without a cap, models that
// support long reasoning traces can loop until the context is exhausted and
// never emit an answer. The steps halve rather than crowding the top of the
// range: how long a model thinks depends on the prompt, not on how much room
// it was given, so shares near the whole context stop bounding anything once
// the context is large. A level absent from this table stays valid: it reaches
// the model as a string and simply carries no budget.
var thinkBudgetFraction = map[string][2]int{
	"max":     {4, 5},
	"high":    {1, 2},
	"medium":  {1, 4},
	"low":     {1, 8},
	"minimal": {1, 16},
}

// ThinkLevels returns the effort levels that carry a thinking budget, weakest
// first.
func ThinkLevels() []string {
	return slices.Clone(thinkLevels)
}

// IsThinkLevel reports whether a string is an effort level with a budget.
func IsThinkLevel(level string) bool {
	return slices.Contains(thinkLevels, level)
}

// IsValid checks the transport type. The model resolves supported effort names.
func (t *ThinkValue) IsValid() bool {
	if t == nil || t.Value == nil {
		return true // nil is valid (means not set)
	}

	switch v := t.Value.(type) {
	case bool:
		return true
	case string:
		return true
	case int:
		return v > 0
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

// IsInt returns true if the value is an explicit thinking-token budget
func (t *ThinkValue) IsInt() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(int)
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
	case int:
		// A budget only makes sense when thinking is on
		return v > 0
	default:
		return false
	}
}

// BudgetTokens returns the number of tokens the model may spend inside a
// thinking block, or 0 when thinking is unrestricted. An explicit integer
// value is used as-is; effort levels are a share of window, the room the
// response has to work in — see api.ThinkBudgetWindow.
func (t *ThinkValue) BudgetTokens(window int) int {
	if t == nil || t.Value == nil {
		return 0
	}

	switch v := t.Value.(type) {
	case int:
		if v > 0 {
			return v
		}
	case string:
		frac, ok := thinkBudgetFraction[v]
		if !ok || window <= 0 {
			return 0
		}
		// Round down so the budget never consumes the whole window
		if budget := window * frac[0] / frac[1]; budget > 0 {
			return budget
		}
	}

	return 0
}

// Level returns the effort level to hand a model that consumes levels
// directly, or "" when there is none to send. Models that take a level as a
// string recognise low, medium and high, and gpt-oss writes whatever it is
// given straight into its system prompt, so the levels outside that range are
// reported as the nearest one they know. The budget is unaffected and keeps
// the share of the context the requested level asked for.
func (t *ThinkValue) Level() string {
	switch level := t.String(); level {
	case "max":
		return "high"
	case "minimal":
		return "low"
	default:
		return level
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
	switch v := value.(type) {
	case nil, bool, string:
		t.Value = value
		return nil
	case float64:
		// An explicit thinking-token budget
		if v != math.Trunc(v) {
			return fmt.Errorf("invalid think budget: %v (must be a whole number of tokens)", v)
		}
		if v <= 0 {
			return fmt.Errorf("invalid think budget: %v (must be greater than 0; use false to disable thinking)", v)
		}
		t.Value = int(v)
		return nil
	default:
		return fmt.Errorf("think must be a boolean, a string or a positive thinking-token budget")
	}
}

// MarshalJSON implements json.Marshaler
func (t *ThinkValue) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}
