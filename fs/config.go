package fs

type Config interface {
	Architecture() string
	String(string, ...string) string
	Uint(string, ...uint32) uint32
	Float(string, ...float32) float32
	Bool(string, ...bool) bool

	Strings(string, ...[]string) []string
	Ints(string, ...[]int32) []int32
	Floats(string, ...[]float32) []float32
}
