package gguf

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func split(name string, values map[string][]any) (matched []any, unmatched []any) {
	for key, value := range values {
		if key == name {
			matched = value
		} else {
			unmatched = append(unmatched, value...)
		}
	}
	return
}

func TestValue(t *testing.T) {
	values := map[string][]any{
		"int64":   {int(42), int8(42), int16(42), int32(42), int64(42)},
		"uint64":  {uint(42), uint8(42), uint16(42), uint32(42), uint64(42)},
		"float64": {float32(42), float64(42)},
		"string":  {"42", "hello"},
		"bool":    {true, false},
	}

	t.Run("int64", func(t *testing.T) {
		matched, unmatched := split("int64", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if i64 := kv.Int(); i64 != 42 {
				t.Errorf("expected 42, got %d", i64)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if i64 := kv.Int(); i64 != 0 {
				t.Errorf("expected 42, got %d", i64)
			}
		}
	})

	t.Run("uint64", func(t *testing.T) {
		matched, unmatched := split("uint64", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if u64 := kv.Uint(); u64 != 42 {
				t.Errorf("expected 42, got %d", u64)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if u64 := kv.Uint(); u64 != 0 {
				t.Errorf("expected 42, got %d", u64)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		matched, unmatched := split("float64", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if f64 := kv.Float(); f64 != 42 {
				t.Errorf("expected 42, got %f", f64)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if f64 := kv.Float(); f64 != 0 {
				t.Errorf("expected 42, got %f", f64)
			}
		}
	})

	t.Run("string", func(t *testing.T) {
		matched, unmatched := split("string", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if s := kv.String(); s != v {
				t.Errorf("expected 42, got %s", s)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if s := kv.String(); s != "" {
				t.Errorf("expected 42, got %s", s)
			}
		}
	})

	t.Run("bool", func(t *testing.T) {
		matched, unmatched := split("bool", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if b := kv.Bool(); b != v {
				t.Errorf("expected true, got %v", b)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if b := kv.Bool(); b != false {
				t.Errorf("expected false, got %v", b)
			}
		}
	})
}

func TestValues(t *testing.T) {
	values := map[string][]any{
		"int64s":   {[]int{42}, []int8{42}, []int16{42}, []int32{42}, []int64{42}},
		"uint64s":  {[]uint{42}, []uint8{42}, []uint16{42}, []uint32{42}, []uint64{42}},
		"float64s": {[]float32{42}, []float64{42}},
		"strings":  {[]string{"42"}, []string{"hello"}},
		"bools":    {[]bool{true}, []bool{false}},
	}

	t.Run("int64s", func(t *testing.T) {
		matched, unmatched := split("int64s", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if diff := cmp.Diff(kv.Ints(), []int64{42}); diff != "" {
				t.Errorf("diff: %s", diff)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if i64s := kv.Ints(); i64s != nil {
				t.Errorf("expected nil, got %v", i64s)
			}
		}
	})

	t.Run("uint64s", func(t *testing.T) {
		matched, unmatched := split("uint64s", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if diff := cmp.Diff(kv.Uints(), []uint64{42}); diff != "" {
				t.Errorf("diff: %s", diff)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if u64s := kv.Uints(); u64s != nil {
				t.Errorf("expected nil, got %v", u64s)
			}
		}
	})

	t.Run("float64s", func(t *testing.T) {
		matched, unmatched := split("float64s", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if diff := cmp.Diff(kv.Floats(), []float64{42}); diff != "" {
				t.Errorf("diff: %s", diff)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if f64s := kv.Floats(); f64s != nil {
				t.Errorf("expected nil, got %v", f64s)
			}
		}
	})

	t.Run("strings", func(t *testing.T) {
		matched, unmatched := split("strings", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if diff := cmp.Diff(kv.Strings(), v); diff != "" {
				t.Errorf("diff: %s", diff)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if s := kv.Strings(); s != nil {
				t.Errorf("expected nil, got %v", s)
			}
		}
	})

	t.Run("bools", func(t *testing.T) {
		matched, unmatched := split("bools", values)
		for _, v := range matched {
			kv := KeyValue{"key", Value{v}}
			if diff := cmp.Diff(kv.Bools(), v); diff != "" {
				t.Errorf("diff: %s", diff)
			}
		}

		for _, v := range unmatched {
			kv := KeyValue{"key", Value{v}}
			if b := kv.Bools(); b != nil {
				t.Errorf("expected nil, got %v", b)
			}
		}
	})
}
