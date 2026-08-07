package structured

import "testing"

// testVocab builds a small synthetic vocabulary. Token ids:
//
//	0:"{"  1:"}"  2:`"`  3:"a"  4:`":`  5:"1"  6:","  7:" "
//	8:`{"` 9:"["  10:"" (special, never allowed)  11:"ab"  12:`"}`
//	13: EOS (nil piece)
func testVocab() *Vocab {
	pieces := [][]byte{
		[]byte("{"), []byte("}"), []byte(`"`), []byte("a"),
		[]byte(`":`), []byte("1"), []byte(","), []byte(" "),
		[]byte(`{"`), []byte("["), nil, []byte("ab"), []byte(`"}`),
		nil,
	}
	return NewVocab(pieces, []int32{13})
}

func maskFor(t *testing.T, v *Vocab, input string) *Mask {
	t.Helper()
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		t.Fatal(err)
	}
	m := g.NewMatcher()
	if !m.Advance([]byte(input)) {
		t.Fatalf("prefix %q rejected", input)
	}
	return v.Mask(m)
}

func checkMask(t *testing.T, mask *Mask, allowed []int32, disallowed []int32) {
	t.Helper()
	for _, id := range allowed {
		if !mask.Allowed(id) {
			t.Errorf("token %d disallowed, want allowed", id)
		}
	}
	for _, id := range disallowed {
		if mask.Allowed(id) {
			t.Errorf("token %d allowed, want disallowed", id)
		}
	}
}

func TestMaskAtRoot(t *testing.T) {
	v := testVocab()
	mask := maskFor(t, v, "")
	// grammarJSON root is an object with no leading whitespace: only
	// tokens starting with "{" can begin.
	checkMask(t, mask,
		[]int32{0, 8},
		[]int32{1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13})
}

func TestMaskAfterOpenBrace(t *testing.T) {
	v := testVocab()
	mask := maskFor(t, v, "{")
	// ws | key string | close. `"}` and `":` are valid prefixes: '"'
	// opens a key whose first content byte is '}' or ':'.
	checkMask(t, mask,
		[]int32{1, 2, 4, 7, 12},
		[]int32{0, 3, 5, 6, 8, 9, 10, 11, 13})
}

func TestMaskInString(t *testing.T) {
	v := testVocab()
	mask := maskFor(t, v, `{"`)
	// Inside a key: any string char continues; '"' closes; `":` closes
	// and starts the separator; `{"` is { as content then a close quote.
	// `"}` is NOT valid: after the key's closing quote grammarJSON
	// requires ":" immediately.
	checkMask(t, mask,
		[]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11},
		[]int32{10, 12, 13})
}

func TestMaskAtValuePosition(t *testing.T) {
	v := testVocab()
	mask := maskFor(t, v, `{"a":`)
	// Value position: ws, object, array, string (including `":` and `"}`
	// as string-open prefixes), or number.
	checkMask(t, mask,
		[]int32{0, 2, 4, 5, 7, 8, 9, 12},
		[]int32{1, 3, 6, 10, 11, 13})
}

func TestMaskEOSOnlyWhenComplete(t *testing.T) {
	v := testVocab()
	mask := maskFor(t, v, `{"a":1}`)
	// The root object is closed: nothing may follow but EOS.
	checkMask(t, mask,
		[]int32{13},
		[]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
}

func TestMaskEOSWhileExtensible(t *testing.T) {
	v := testVocab()
	// {} is complete, but a schema-style grammar with trailing space keeps
	// ws extensible; grammarJSON's root has no trailing ws, so after {}
	// only EOS remains.
	mask := maskFor(t, v, `{}`)
	checkMask(t, mask, []int32{13}, []int32{0, 1, 2, 5, 7})

	// Mid-number: digits extend, and the object can also close later, but
	// EOS is disallowed because the root object is still open.
	mask = maskFor(t, v, `{"a":1`)
	checkMask(t, mask, []int32{1, 5, 6, 7}, []int32{13, 2, 3, 12})
}

func TestMaskMemoized(t *testing.T) {
	v := testVocab()
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		t.Fatal(err)
	}
	a, b := g.NewMatcher(), g.NewMatcher()
	a.Advance([]byte(`{"a`))
	b.Advance([]byte(`{"a`))
	ma, mb := v.Mask(a), v.Mask(b)
	if ma != mb {
		t.Error("equal states did not hit the mask cache")
	}
}

func TestMaskDuplicatePieces(t *testing.T) {
	// Real tokenizers decode distinct ids to identical bytes (e.g. a raw
	// space token and the <0x20> byte-fallback token). All duplicates
	// must be allowed, and longer pieces sharing the prefix must still
	// be walked.
	pieces := [][]byte{
		[]byte("{"), []byte("{"), []byte(`{"`), nil,
	}
	v := NewVocab(pieces, []int32{3})
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		t.Fatal(err)
	}
	mask := v.Mask(g.NewMatcher())
	checkMask(t, mask, []int32{0, 1, 2}, []int32{3})
}

func TestMaskDistinctGrammarsDoNotCollide(t *testing.T) {
	v := testVocab()
	gJSON, err := Compile([]byte(`"json"`))
	if err != nil {
		t.Fatal(err)
	}
	gArr, err := Compile([]byte(`{"type":"array","items":{"type":"integer"}}`))
	if err != nil {
		t.Fatal(err)
	}
	mJSON := v.Mask(gJSON.NewMatcher())
	mArr := v.Mask(gArr.NewMatcher())
	if mJSON.Allowed(9) {
		t.Error(`grammarJSON root allowed "["`)
	}
	if !mArr.Allowed(9) {
		t.Error(`array schema root disallowed "["`)
	}
}
