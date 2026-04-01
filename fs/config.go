package fs

import "iter"

type Config interface {
	Architecture() string
	String(string, ...string) string
	Uint(string, ...uint32) uint32
	Float(string, ...float32) float32
	Bool(string, ...bool) bool

	Strings(string, ...[]string) []string
	Ints(string, ...[]int32) []int32
	Floats(string, ...[]float32) []float32
	Bools(string, ...[]bool) []bool

	Len() int
	Keys() iter.Seq[string]
	Value(key string) any
}
