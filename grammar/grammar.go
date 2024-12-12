package grammar

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"iter"
	"strconv"

	"github.com/ollama/ollama/jsonschema"
)

const jsonTerms = `
# Unicode
#
# Unicode characters can be specified directly in the grammar, for example
# hiragana ::= [ぁ-ゟ], or with escapes: 8-bit (\xXX), 16-bit (\uXXXX) or 32-bit
# (\UXXXXXXXX).
unicode ::= \x{hex}{2} | \u{hex}{4} | \U{hex}{8}

# JSON grammar from RFC 7159
null    ::= "null"
object  ::= "{" (kv ("," kv)*)? "}"
array   ::= "[" (value ("," value)*)? "]"
kv      ::= string ":" value
integer ::= "0" | [1-9] [0-9]*
number  ::= "-"? integer frac? exp?
frac    ::= "." [0-9]+
exp     ::= ("e" | "E") ("+" | "-") [0-9]+
string  ::= "\"" char* "\""
escape  ::= ["/" | "b" | "f" | "n" | "r" | "t" | unicode]
char    ::= [^"\\] | escape
space   ::= (" " | "\t" | "\n" | "\r")*
hex     ::= [0-9] | [a-f] | [A-F]
boolean ::= "true" | "false"
value   ::= object | array | string | number | boolean | "null"

# User-defined
`

func GrammarFromSchema(buf []byte, jsonSchema []byte) ([]byte, error) {
	var s *jsonschema.Schema
	if err := json.Unmarshal(jsonSchema, &s); err != nil {
		return nil, err
	}

	g := builder{
		b:   *bytes.NewBuffer(buf),
		pad: len("root"),
	}
	g.b.WriteString(jsonTerms)

	for id := range dependencies(s) {
		g.pad = max(g.pad, len(id))
	}

	ids := make(map[*jsonschema.Schema]string)
	for id, s := range dependencies(s) {
		ids[s] = id
		g.define(id)
		if err := grammarFromSchema(&g, ids, s); err != nil {
			return nil, err
		}
	}
	g.define("root")
	if err := grammarFromSchema(&g, ids, s); err != nil {
		return nil, err
	}
	g.define("") // finalize the last rule
	return g.b.Bytes(), nil
}

func grammarFromSchema(g *builder, ids map[*jsonschema.Schema]string, s *jsonschema.Schema) error {
	switch typ := s.EffectiveType(); typ {
	case "array":
		if len(s.PrefixItems) == 0 && s.Items == nil {
			g.u("array")
		} else {
			g.q("[")
			for i, s := range s.PrefixItems {
				if i > 0 {
					g.q(",")
				}
				g.u(ids[s])
			}
			if s.Items != nil {
				g.u("(")
				g.q(",")
				g.u(ids[s.Items])
				g.u(")*")
			}
			g.q("]")
		}
	case "object":
		if len(s.Properties) == 0 {
			g.u("object")
		} else {
			g.q("{")
			for i, p := range s.Properties {
				name := ids[p]
				if i > 0 {
					g.q(",")
				}
				g.q(p.Name)
				g.q(":")
				g.u(name)
			}
			g.q("}")
		}
	case "number":
		buildConstrainedNumber(g, s)
	case "string", "boolean", "value", "null", "integer":
		g.u(typ)
	default:
		return fmt.Errorf("%s: unsupported type %q", s.Name, typ)
	}
	return nil
}

// dependencies returns a sequence of all child dependencies of the schema in
// post-order.
//
// The first value is the id/pointer to the dependency, and the second value
// is the schema.
func dependencies(s *jsonschema.Schema) iter.Seq2[string, *jsonschema.Schema] {
	return func(yield func(string, *jsonschema.Schema) bool) {
		id := cmp.Or(s.Name, "root")
		for _, p := range s.Properties {
			for did, d := range dependencies(p) {
				id := fmt.Sprintf("%s_%s", id, did)
				if !yield(id, d) {
					return
				}
			}
			id := fmt.Sprintf("%s_%s", id, p.Name)
			if !yield(id, p) {
				return
			}
		}
		for i, p := range s.PrefixItems {
			id := fmt.Sprintf("tuple_%d", i)
			for did, d := range dependencies(p) {
				id := fmt.Sprintf("%s_%s", id, did)
				if !yield(id, d) {
					return
				}
			}
			if !yield(id, p) {
				return
			}
		}
		if s.Items != nil {
			id := fmt.Sprintf("tuple_%d", len(s.PrefixItems))
			for did, d := range dependencies(s.Items) {
				id := fmt.Sprintf("%s_%s", id, did)
				if !yield(id, d) {
					return
				}
			}
			if !yield(id, s.Items) {
				return
			}
		}
	}
}

type builder struct {
	b     bytes.Buffer
	pad   int
	rules int
	items int
}

// define terminates the current rule, if any, and then either starts a new
// rule or does nothing else if the name is empty.
func (b *builder) define(name string) {
	if b.rules > 0 {
		b.b.WriteString(";\n")
	}
	if name == "" {
		return
	}
	fmt.Fprintf(&b.b, "% -*s", b.pad, name)
	b.b.WriteString(" ::=")
	b.rules++
	b.items = 0
}

// alt starts a new alternative in the current rule.
func (b *builder) alt() {
	b.b.WriteString(" |")
}

// quote appends a terminal to the current rule.
func (b *builder) q(s string) {
	if b.items > 0 {
		b.b.WriteString(" ")
	}
	b.b.WriteString(" ")
	b.b.WriteString(strconv.Quote(s))
}

// u appends a non-terminal to the current rule.
func (b *builder) u(s string) {
	if b.items > 0 {
		b.b.WriteString(" ")
	}
	b.b.WriteString(" ")
	b.b.WriteString(s)
}

func buildConstrainedNumber(b *builder, s *jsonschema.Schema) {
	if s.Minimum == 0 && s.Maximum == 0 {
		b.u("TODO")
	} else {
		b.u("number")
	}
}
