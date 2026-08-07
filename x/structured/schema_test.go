package structured

import (
	"fmt"
	"strings"
	"testing"
)

func compileSchema(t *testing.T, schema string) *Grammar {
	t.Helper()
	g, err := Compile([]byte(schema))
	if err != nil {
		t.Fatalf("Compile(%s): %v", schema, err)
	}
	return g
}

func checkAccepts(t *testing.T, g *Grammar, ok []string, bad []string) {
	t.Helper()
	for _, s := range ok {
		if !accepts(g, s) {
			m := g.NewMatcher()
			n := advanceString(m, s)
			t.Errorf("%q rejected (stops at byte %d)", s, n)
		}
	}
	for _, s := range bad {
		if accepts(g, s) {
			t.Errorf("%q accepted", s)
		}
	}
}

func TestSchemaRequiredPropertyOrder(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}`)
	checkAccepts(t, g,
		[]string{
			`{"name":"x","age":3}`,
			`{ "name" : "x" , "age" : 3 }`, // b10091 kv rules allow space around ":" and after ","
		},
		[]string{
			`{"age":3,"name":"x"}`,           // declaration order is forced
			`{"name":"x"}`,                   // missing required
			`{"name":"x","age":3,"extra":1}`, // closed object by default
			`{}`,
		})
}

func TestSchemaOptionalPropertyOrder(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}},"required":["name"]}`)
	checkAccepts(t, g,
		[]string{
			`{"name":"x"}`,
			`{"name":"x","age":3}`,
			`{"name":"x","city":"c"}`,
			`{"name":"x","age":3,"city":"c"}`,
		},
		[]string{
			`{"name":"x","city":"c","age":3}`, // optional props keep declaration order
			`{"age":3}`,
		})
}

func TestSchemaAllOptionalProperties(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}`)
	checkAccepts(t, g,
		[]string{`{}`, `{"a":1}`, `{"b":2}`, `{"a":1,"b":2}`},
		[]string{`{"b":2,"a":1}`, `{"c":3}`})
}

func TestSchemaAdditionalPropertiesTyped(t *testing.T) {
	g := compileSchema(t, `{"type":"object","additionalProperties":{"type":"integer"}}`)
	checkAccepts(t, g,
		[]string{`{}`, `{"x":1}`, `{"x":1,"y":2}`},
		[]string{`{"x":"s"}`})
}

func TestSchemaAdditionalPropertiesWithDeclared(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":true}`)
	checkAccepts(t, g,
		[]string{
			`{"name":"x"}`,
			`{"name":"x","other":1}`,
			`{"name":"x","names":[1]}`, // extension of a declared key is a valid extra key
		},
		[]string{
			`{"name":"x","name":"y"}`, // extra keys must not collide with declared ones
			// b10091 _not_strings quirk: prefixes of declared keys are
			// also unreachable as extra keys ("nam" cannot close mid-trie).
			`{"name":"x","nam":true}`,
			`{"other":1}`,
		})
}

func TestSchemaEnum(t *testing.T) {
	g := compileSchema(t, `{"enum":["red","green",7,null,true]}`)
	checkAccepts(t, g,
		[]string{`"red"`, `"green"`, `7`, `null`, `true`},
		[]string{`"blue"`, `8`, `false`, `"RED"`})
}

func TestSchemaConst(t *testing.T) {
	g := compileSchema(t, `{"const":{"kind":"point","x":1}}`)
	checkAccepts(t, g,
		[]string{`{"kind":"point","x":1}`},
		[]string{`{"kind":"point","x":2}`, `{"x":1,"kind":"point"}`})
}

func TestSchemaArrayBounds(t *testing.T) {
	g := compileSchema(t, `{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}`)
	checkAccepts(t, g,
		[]string{`[1]`, `[1,2]`, `[ 1 , 2 , 3 ]`},
		[]string{`[]`, `[1,2,3,4]`, `["a"]`})

	free := compileSchema(t, `{"type":"array"}`)
	checkAccepts(t, free, []string{`[]`, `[1,"a",{"b":[]}]`}, []string{`{"a":1}`})
}

func TestSchemaTuple(t *testing.T) {
	g := compileSchema(t, `{"prefixItems":[{"type":"integer"},{"type":"string"}]}`)
	checkAccepts(t, g,
		[]string{`[1,"a"]`, `[ 1 , "a" ]`},
		[]string{`[1]`, `[1,"a",2]`, `["a",1]`})
}

func TestSchemaAnyOf(t *testing.T) {
	g := compileSchema(t, `{"anyOf":[{"type":"integer"},{"type":"null"}]}`)
	checkAccepts(t, g, []string{`4`, `-17`, `null`}, []string{`"x"`, `4.5`, `true`})
}

func TestSchemaTypeArray(t *testing.T) {
	g := compileSchema(t, `{"type":["string","null"]}`)
	checkAccepts(t, g, []string{`"s"`, `null`}, []string{`4`, `true`})
}

func TestSchemaRefRecursion(t *testing.T) {
	g := compileSchema(t, `{"$ref":"#/$defs/node","$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}}}`)
	checkAccepts(t, g,
		[]string{
			`{"v":1}`,
			`{"v":1,"kids":[]}`,
			`{"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4}]}]}`,
		},
		[]string{`{"v":"s"}`, `{"kids":[]}`, `{"v":1,"kids":[2]}`})
}

func TestSchemaIntegerBoundsExhaustive(t *testing.T) {
	g := compileSchema(t, `{"type":"integer","minimum":3,"maximum":42}`)
	for i := -60; i <= 99; i++ {
		s := fmt.Sprintf("%d", i)
		want := i >= 3 && i <= 42
		if got := accepts(g, s); got != want {
			t.Errorf("bounds [3,42]: %q accepted=%v want %v", s, got, want)
		}
	}
	if accepts(g, "03") {
		t.Error("leading zero accepted")
	}
}

func TestSchemaIntegerBoundsVariants(t *testing.T) {
	cases := []struct {
		schema string
		ok     []string
		bad    []string
	}{
		{`{"type":"integer","minimum":10}`, []string{"10", "99", "12345"}, []string{"9", "-1", "0"}},
		{`{"type":"integer","maximum":-4}`, []string{"-4", "-100"}, []string{"-3", "0", "5"}},
		{`{"type":"integer","minimum":-12,"maximum":-4}`, []string{"-12", "-7", "-4"}, []string{"-13", "-3", "0", "4"}},
		{`{"type":"integer","minimum":-3,"maximum":7}`, []string{"-3", "0", "7"}, []string{"-4", "8"}},
		{`{"type":"integer","exclusiveMinimum":3,"exclusiveMaximum":6}`, []string{"4", "5"}, []string{"3", "6"}},
	}
	for _, c := range cases {
		g := compileSchema(t, c.schema)
		checkAccepts(t, g, c.ok, c.bad)
	}
}

func TestSchemaStringLength(t *testing.T) {
	g := compileSchema(t, `{"type":"string","minLength":2,"maxLength":4}`)
	checkAccepts(t, g,
		[]string{`"ab"`, `"abcd"`},
		[]string{`"a"`, `"abcde"`, `""`})
}

func TestSchemaStringFormats(t *testing.T) {
	date := compileSchema(t, `{"type":"string","format":"date"}`)
	checkAccepts(t, date, []string{`"2026-08-07"`}, []string{`"2026-13-07"`, `"20260807"`})

	dt := compileSchema(t, `{"type":"string","format":"date-time"}`)
	checkAccepts(t, dt,
		[]string{`"2026-08-07T12:34:56Z"`, `"2026-08-07T12:34:56.123+02:00"`},
		[]string{`"2026-08-07 12:34:56"`})

	uuid := compileSchema(t, `{"type":"string","format":"uuid"}`)
	checkAccepts(t, uuid,
		[]string{`"01234567-89ab-CDEF-0123-456789abcdef"`},
		[]string{`"01234567-89ab-CDEF-0123-456789abcde"`, `"0123456789abCDEF0123456789abcdef"`})
}

func TestSchemaAllOfMerge(t *testing.T) {
	g := compileSchema(t, `{"type":"object","allOf":[{"properties":{"a":{"type":"integer"}}},{"anyOf":[{"properties":{"b":{"type":"integer"}}}]}]}`)
	checkAccepts(t, g,
		[]string{`{"a":1}`, `{"a":1,"b":2}`},
		[]string{`{"b":2}`, `{}`})
}

func TestSchemaFreeFormObject(t *testing.T) {
	for _, schema := range []string{`{}`, `{"type":"object"}`} {
		g := compileSchema(t, schema)
		checkAccepts(t, g,
			[]string{`{}`, `{"x":[1,"a",{"y":null}]}`},
			[]string{`[1]`, `"s"`})
	}
}

func TestSchemaPrimitiveDigitCap(t *testing.T) {
	// The b10091 integral-part rule caps integers at 16 digits.
	g := compileSchema(t, `{"type":"integer"}`)
	checkAccepts(t, g,
		[]string{"0", "-5", strings.Repeat("9", 16)},
		[]string{strings.Repeat("9", 17), "007"})
}

func TestSchemaNumberIgnoresBounds(t *testing.T) {
	// b10091 only enforces minimum/maximum for integers; numbers fall back
	// to the unconstrained number primitive.
	g := compileSchema(t, `{"type":"number","minimum":10}`)
	checkAccepts(t, g, []string{"5", "10.5", "-3.25"}, []string{`"s"`})
}

func TestSchemaValueTrailingSpace(t *testing.T) {
	// Primitive rules end with the space rule, so values may carry
	// bounded trailing whitespace (at most "\n\n" plus 20 spaces/tabs).
	g := compileSchema(t, `{"type":"integer"}`)
	checkAccepts(t, g,
		[]string{"7", "7 ", "7\n  "},
		[]string{"7  ", "7\n\n\n"}) // two spaces / three newlines exceed SPACE_RULE
}

func TestSchemaUnsupportedRejected(t *testing.T) {
	cases := []struct {
		schema  string
		mention string
	}{
		{`{"type":"string","pattern":"^a+$"}`, "pattern"},
		{`{"$ref":"https://example.com/schema.json"}`, "ref"},
		{`{"type":"frobnicate"}`, ""},
		{`{"not":{"type":"string"}}`, ""},
	}
	for _, c := range cases {
		_, err := Compile([]byte(c.schema))
		if err == nil {
			t.Errorf("Compile(%s): expected error", c.schema)
			continue
		}
		if c.mention != "" && !strings.Contains(strings.ToLower(err.Error()), c.mention) {
			t.Errorf("Compile(%s): error %q does not mention %q", c.schema, err, c.mention)
		}
	}
}

func TestSchemaPropertyNameEscaping(t *testing.T) {
	g := compileSchema(t, `{"type":"object","properties":{"quo\"te":{"type":"integer"}},"required":["quo\"te"]}`)
	checkAccepts(t, g, []string{`{"quo\"te":1}`}, []string{`{"quote":1}`})
}
