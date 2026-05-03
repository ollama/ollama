package cmputil

// A Compare function for booleans as they aren't strictly comparable.
// Sorts True before False.
// Returns 0 if equal, -1 if a is true, and 1 if b is true.
func CompareBool(a, b bool) int {
	switch {
	case a == b:
		return 0
	case a:
		return -1
	default:
		return 1
	}
}
