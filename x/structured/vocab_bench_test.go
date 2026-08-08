package structured

import (
	"fmt"
	"math/rand"
	"testing"
)

// benchVocab approximates a real tokenizer: every single byte, plus ~256k
// multi-byte pieces of common word-like shapes.
func benchVocab() *Vocab {
	rng := rand.New(rand.NewSource(1))
	pieces := make([][]byte, 0, 260000)
	for b := 0; b < 256; b++ {
		pieces = append(pieces, []byte{byte(b)})
	}
	const letters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,:\"{}[]"
	seen := map[string]bool{}
	for len(pieces) < 260000 {
		n := 1 + rng.Intn(12)
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = letters[rng.Intn(len(letters))]
		}
		if seen[string(buf)] {
			continue
		}
		seen[string(buf)] = true
		pieces = append(pieces, buf)
	}
	eos := int32(len(pieces))
	pieces = append(pieces, nil)
	return NewVocab(pieces, []int32{eos})
}

func BenchmarkMaskColdInString(b *testing.B) {
	v := benchVocab()
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		b.Fatal(err)
	}
	m := g.NewMatcher()
	m.Advance([]byte(`{"key`)) // in-string: the widest allowed set
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.cache = make(map[uint64]*Mask) // force cold
		v.Mask(m)
	}
}

func BenchmarkMaskWarm(b *testing.B) {
	v := benchVocab()
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		b.Fatal(err)
	}
	m := g.NewMatcher()
	m.Advance([]byte(`{"key`))
	v.Mask(m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Mask(m)
	}
}

func BenchmarkMaskGenerationSequence(b *testing.B) {
	// A full pass over a representative generation, states cached across
	// tokens the way a real decode is.
	v := benchVocab()
	g, err := Compile([]byte(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}`))
	if err != nil {
		b.Fatal(err)
	}
	text := `{"name":"Ada Lovelace, mathematician and writer","age":36}`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := g.NewMatcher()
		for j := 0; j < len(text); j++ {
			v.Mask(m)
			if !m.AdvanceByte(text[j]) {
				b.Fatalf("rejected at %d", j)
			}
		}
	}
	b.ReportMetric(float64(len(text)), "tokens/op")
	_ = fmt.Sprint()
}
