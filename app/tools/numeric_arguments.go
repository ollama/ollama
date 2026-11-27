//go:build windows || darwin

package tools

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// integerArgument converts only integer-valued parameters at the tool boundary.
// Keep the original argument map untouched for history and other consumers.
func integerArgument(args map[string]any, name string, fallback int) (int, error) {
	value, present := args[name]
	if !present || value == nil {
		return fallback, nil
	}
	invalid := func() (int, error) { return 0, fmt.Errorf("%s must be an integer within the platform int range", name) }
	switch value := value.(type) {
	case int:
		return value, nil
	case float64:
		limit := math.Ldexp(1, strconv.IntSize-1)
		if math.IsNaN(value) || math.IsInf(value, 0) || math.Trunc(value) != value || value < -limit || value >= limit {
			return invalid()
		}
		return int(value), nil
	case json.Number:
		if !json.Valid([]byte(value)) {
			return invalid()
		}
		// A lexical integer needs no floating-point conversion.
		if n, err := strconv.ParseInt(string(value), 10, strconv.IntSize); err == nil {
			return int(n), nil
		}
		// Accept integer-valued decimal/exponent forms too, without rounding them.
		f, err := value.Float64()
		limit := math.Ldexp(1, strconv.IntSize-1)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < -limit || f > limit {
			return invalid()
		}
		if f == 0 {
			mantissa := strings.FieldsFunc(string(value), func(r rune) bool { return r == 'e' || r == 'E' })[0]
			if strings.Trim(mantissa, "-+0.") == "" {
				return 0, nil
			}
			// Reject underflow before constructing a potentially enormous denominator.
			return invalid()
		}
		exact, ok := new(big.Rat).SetString(string(value))
		if !ok || !exact.IsInt() || !exact.Num().IsInt64() {
			return invalid()
		}
		n := exact.Num().Int64()
		if int64(int(n)) != n {
			return invalid()
		}
		return int(n), nil
	default:
		return invalid()
	}
}
