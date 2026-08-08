package structured

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// walkVocab is byte-complete: every byte is a single-byte token, so a
// grammar state can never have an empty mask unless only EOS remains.
// A few multi-byte tokens exercise cross-boundary pieces.
func walkVocab() (*Vocab, [][]byte, int32) {
	pieces := make([][]byte, 0, 262)
	for b := 0; b < 256; b++ {
		pieces = append(pieces, []byte{byte(b)})
	}
	for _, s := range []string{`{"`, `":`, `",`, `"}`, "true", "null"} {
		pieces = append(pieces, []byte(s))
	}
	eos := int32(len(pieces))
	pieces = append(pieces, nil) // EOS
	return NewVocab(pieces, []int32{eos}), pieces, eos
}

// closingBias orders candidate tokens to steer long walks toward
// completion. '"' comes first: inside a string it is the closer, while
// '}' and ']' would be ordinary content and pad forever.
var closingBias = []byte{'"', '}', ']', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

func walkGrammar(t *testing.T, g *Grammar, seed int64) string {
	t.Helper()
	v, pieces, eos := walkVocab()
	rng := rand.New(rand.NewSource(seed))
	m := g.NewMatcher()
	var out []byte

	const maxSteps, softCap = 4000, 300
	for step := 0; step < maxSteps; step++ {
		mask := v.Mask(m)

		var candidates []int32
		for id := int32(0); id < int32(len(pieces)); id++ {
			if mask.Allowed(id) {
				candidates = append(candidates, id)
			}
		}
		if len(candidates) == 0 {
			t.Fatalf("empty mask after %q", out)
		}

		var pick int32 = -1
		if mask.Allowed(eos) && (step >= softCap || rng.Intn(4) == 0) {
			break
		}
		if step >= softCap {
			// Past the soft cap, prefer closing-biased bytes so deep
			// nesting unwinds.
			for _, b := range closingBias {
				if mask.Allowed(int32(b)) {
					pick = int32(b)
					break
				}
			}
		}
		if pick < 0 {
			pick = candidates[rng.Intn(len(candidates))]
			if pick == eos {
				break
			}
		}

		if !m.Advance(pieces[pick]) {
			t.Fatalf("mask allowed token %d (%q) but Advance rejected it after %q", pick, pieces[pick], out)
		}
		out = append(out, pieces[pick]...)
	}

	if !m.CanComplete() {
		t.Fatalf("walk did not reach a complete state: %q", out)
	}
	return string(out)
}

func TestWalkJSONFormatAlwaysValid(t *testing.T) {
	g := mustCompileJSON(t)
	for seed := int64(0); seed < 20; seed++ {
		out := walkGrammar(t, g, seed)
		if !json.Valid([]byte(out)) {
			t.Fatalf("seed %d: invalid JSON: %q", seed, out)
		}
		// UseNumber: the grammar (like llama-server's) permits numbers
		// that overflow float64, e.g. huge exponents.
		var v map[string]any
		dec := json.NewDecoder(strings.NewReader(out))
		dec.UseNumber()
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("seed %d: root is not an object: %v: %q", seed, err, out)
		}
	}
}

func TestWalkSchemaShapesOutput(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"},"maxItems":3}},"required":["name","age"]}`)
	for seed := int64(0); seed < 20; seed++ {
		out := walkGrammar(t, g, seed)
		var v struct {
			Name *string `json:"name"`
			Age  *int64  `json:"age"`
			Tags []any   `json:"tags"`
		}
		dec := json.NewDecoder(strings.NewReader(out))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("seed %d: %v: %q", seed, err, out)
		}
		if v.Name == nil || v.Age == nil {
			t.Fatalf("seed %d: required property missing: %q", seed, out)
		}
		if len(v.Tags) > 3 {
			t.Fatalf("seed %d: maxItems violated: %q", seed, out)
		}
	}
}

func TestWalkEnumOnlyEmitsMembers(t *testing.T) {
	g := compileSchema(t, `{"enum":["alpha","beta",42]}`)
	seen := map[string]bool{}
	for seed := int64(0); seed < 30; seed++ {
		out := walkGrammar(t, g, seed)
		var v any
		if err := json.Unmarshal([]byte(out), &v); err != nil {
			t.Fatalf("seed %d: %v: %q", seed, err, out)
		}
		switch val := v.(type) {
		case string:
			if val != "alpha" && val != "beta" {
				t.Fatalf("seed %d: enum violated: %q", seed, out)
			}
		case float64:
			if val != 42 {
				t.Fatalf("seed %d: enum violated: %q", seed, out)
			}
		default:
			t.Fatalf("seed %d: enum violated: %q", seed, out)
		}
		seen[out] = true
	}
	if len(seen) < 2 {
		t.Errorf("walks never diversified across enum members: %v", seen)
	}
}

func TestWalkRecursiveRefTerminates(t *testing.T) {
	g := compileSchema(t, `{"$ref":"#/$defs/node","$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"},"maxItems":2}},"required":["v"]}}}`)
	for seed := int64(0); seed < 10; seed++ {
		out := walkGrammar(t, g, seed)
		if !json.Valid([]byte(out)) {
			t.Fatalf("seed %d: invalid JSON: %q", seed, out)
		}
	}
}

func TestWalkIntegerBoundsHold(t *testing.T) {
	g := compileSchema(t, `{"type":"integer","minimum":-25,"maximum":170}`)
	for seed := int64(0); seed < 40; seed++ {
		out := walkGrammar(t, g, seed)
		var v json.Number
		if err := json.Unmarshal([]byte(out), &v); err != nil {
			t.Fatalf("seed %d: %v: %q", seed, err, out)
		}
		n, err := v.Int64()
		if err != nil {
			t.Fatalf("seed %d: not an integer: %q", seed, out)
		}
		if n < -25 || n > 170 {
			t.Fatalf("seed %d: bounds violated: %q", seed, out)
		}
	}
}
