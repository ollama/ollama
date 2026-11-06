package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
)

// ThinkBudget bounds how many tokens a model may spend inside a thinking
// block: a positive token count, or a named share of the room the response has.
//
// It is a setting of its own. A thinking level ([ThinkValue]) is the model's
// control and is handed to the model as written; it never bounds anything. A
// budget is enforced by the sampler and says nothing to the model about how to
// reason. The two share some words -- "low", "medium", "high" -- and nothing
// else: a request can carry either, both or neither.
type ThinkBudget struct {
	// Value is an int (tokens) or a string (a share name)
	Value any
}

// ErrInvalidThinkBudget is wrapped by every refusal of a think_budget value,
// so a server can answer it as the client's error rather than its own.
var ErrInvalidThinkBudget = errors.New("invalid think budget")

// thinkBudgetShares are the names a budget can be written as, smallest first.
var thinkBudgetShares = []string{"minimal", "low", "medium", "high", "max"}

// thinkBudgetFraction maps a share name to the part of the response window the
// model is allowed to spend on thinking. Without a cap, models that support
// long reasoning traces can loop until the context is exhausted and never emit
// an answer. The steps halve rather than crowding the top of the range: how
// long a model thinks depends on the prompt, not on how much room it was
// given, so shares near the whole window stop bounding anything once the
// window is large.
var thinkBudgetFraction = map[string][2]int{
	"max":     {4, 5},
	"high":    {1, 2},
	"medium":  {1, 4},
	"low":     {1, 8},
	"minimal": {1, 16},
}

// thinkBudgetAliases are alternative spellings of a share. The AI SDK calls
// the top of its scale "xhigh", which is the position "max" holds here, so a
// client built on that vocabulary sends it verbatim.
var thinkBudgetAliases = map[string]string{
	"xhigh": "max",
}

func canonicalThinkBudgetShare(share string) string {
	if canonical, ok := thinkBudgetAliases[share]; ok {
		return canonical
	}
	return share
}

// ThinkBudgetShares returns the names a budget can be written as, smallest
// first.
func ThinkBudgetShares() []string {
	return slices.Clone(thinkBudgetShares)
}

// IsThinkBudgetShare reports whether a string names a share of the window.
func IsThinkBudgetShare(share string) bool {
	return slices.Contains(thinkBudgetShares, canonicalThinkBudgetShare(share))
}

// Tokens returns the number of tokens the model may spend inside a thinking
// block, or 0 when there is no bound. A token count is used as written; a
// share is taken of window, the room the response has to work in.
func (b *ThinkBudget) Tokens(window int) int {
	if b == nil || b.Value == nil {
		return 0
	}

	switch v := b.Value.(type) {
	case int:
		if v > 0 {
			return v
		}
	case string:
		frac, ok := thinkBudgetFraction[canonicalThinkBudgetShare(v)]
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

// String returns the budget in the form it was written.
func (b *ThinkBudget) String() string {
	if b == nil || b.Value == nil {
		return ""
	}
	switch v := b.Value.(type) {
	case int:
		return strconv.Itoa(v)
	case string:
		return v
	default:
		return ""
	}
}

// UnmarshalJSON accepts a positive whole number of tokens or a share name. A
// count written as a string is read as a count, because that is how a client
// that keeps the setting in a text field sends it.
func (b *ThinkBudget) UnmarshalJSON(data []byte) error {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch v := value.(type) {
	case nil:
		b.Value = nil
		return nil
	case float64:
		count, err := thinkBudgetCount(v)
		if err != nil {
			return err
		}
		b.Value = count
		return nil
	case string:
		if number, err := strconv.ParseFloat(v, 64); err == nil {
			count, err := thinkBudgetCount(number)
			if err != nil {
				return err
			}
			b.Value = count
			return nil
		}
		if !IsThinkBudgetShare(v) {
			return fmt.Errorf("%w: %q (must be a positive number of tokens or one of %q)", ErrInvalidThinkBudget, v, thinkBudgetShares)
		}
		b.Value = v
		return nil
	default:
		return fmt.Errorf("%w: must be a positive number of tokens or one of %q", ErrInvalidThinkBudget, thinkBudgetShares)
	}
}

func thinkBudgetCount(v float64) (int, error) {
	if v != math.Trunc(v) {
		return 0, fmt.Errorf("%w: %v (must be a whole number of tokens)", ErrInvalidThinkBudget, v)
	}
	if v <= 0 {
		return 0, fmt.Errorf("%w: %v (must be greater than 0; leave it out for no bound)", ErrInvalidThinkBudget, v)
	}
	return int(v), nil
}

// MarshalJSON writes the budget in the form it was written.
func (b *ThinkBudget) MarshalJSON() ([]byte, error) {
	if b == nil || b.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(b.Value)
}
