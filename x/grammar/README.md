# grammar

Grammar-constrained decoding for LLM outputs using MLX.

## Performance

Performance depends on hardware, vocabulary size, grammar, and whether you
evaluate the MLX graph. See [Benchmarks](#benchmarks) for how to measure on your
setup.

### Design choices that keep masking fast

| Technique | Impact |
|-----------|--------|
| Precomputed token analysis | Terminal matches computed once at startup |
| Mask caching by grammar state signature | Reuse masks for repeated parser states |
| Partitioned tokens | Exact matches separated from DP candidates |

### Comparison Notes

- **llama.cpp**: Decodes each token to UTF-8, checks against PDA. No caching.
- **Outlines**: FSM-based. Compilation can take 40s-10min for complex schemas. Fast after compile.
- **XGrammar**: PDA with 99% context-independent tokens precomputed. State-of-the-art before this.
- **x/grammar**: Precomputed token analysis + mask caching by grammar state signature.

## Usage

```go
import (
    "github.com/ollama/ollama/x/grammar"
    "github.com/ollama/ollama/x/grammar/schema"
)

// Use built-in JSON grammar
g, _ := grammar.JSONGrammar()

// Or from JSON Schema (OpenAI-compatible)
g, _ := schema.Grammar(`{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}`)

// Or parse custom EBNF
g, _ := grammar.ParseEBNF(myGrammar, "root")

// Create engine with model vocabulary
engine, _ := grammar.NewEngine(g, vocab)
defer engine.Close()

// Generation loop
for !engine.IsComplete() {
    logits := model.Forward(tokens)
    masked := engine.ApplyMask(logits)  // Invalid tokens → -inf
    nextToken := sample(masked)
    engine.Accept(nextToken)
}
// Output conforms to the grammar when you only sample from masked tokens and call Accept
```

## EBNF Syntax

```ebnf
rule = expression .           # Rule definition (ends with .)
"literal"                      # Literal string
"a" … "z"                      # Character range (inclusive)
( a | b )                      # Grouping with alternation
[ optional ]                   # Optional (0 or 1)
{ repeated }                   # Repetition (0 or more)
```

### Example: JSON Grammar

```ebnf
json = value .

value = object | array | string | number | "true" | "false" | "null" .

object = "{" ws "}" | "{" members "}" .
members = member { "," member } .
member = ws string ws ":" element .

array = "[" ws "]" | "[" elements "]" .
elements = element { "," element } .
element = ws value ws .

string = "\"" { character } "\"" .
character = unescaped | escaped .
unescaped = " " | "!" | "#" … "[" | "]" … "~" .
escaped = "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" ) .

number = [ "-" ] integer [ fraction ] [ exponent ] .
integer = "0" | onenine { digit } .
fraction = "." digit { digit } .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
digit = "0" … "9" .
onenine = "1" … "9" .

ws = { " " | "\t" | "\n" | "\r" } .
```

### Example: Custom Schema

```ebnf
root = "{" ws name_field "," ws age_field ws "}" .

name_field = "\"name\"" ws ":" ws string .
age_field = "\"age\"" ws ":" ws number .

string = "\"" { char } "\"" .
char = " " | "!" | "#" … "~" .

number = [ "-" ] digit { digit } .
digit = "0" … "9" .

ws = { " " | "\n" } .
```

## JSON Schema Support

OpenAI-compatible JSON Schema support with automatic EBNF generation:

```go
schema := `{
    "type": "object",
    "properties": {
        "user": {"$ref": "#/$defs/User"}
    },
    "required": ["user"],
    "$defs": {
        "User": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "role": {"enum": ["admin", "user", "guest"]}
            },
            "required": ["name", "email", "role"]
        }
    }
}`

grammar, _ := schema.Grammar(schema)
```

### Supported Features

| Feature | Example |
|---------|---------|
| Basic types | `string`, `integer`, `number`, `boolean`, `null` |
| Objects | `properties`, `required` |
| Arrays | `items`, `minItems`, `maxItems` |
| Enums | `enum: ["a", "b", "c"]` |
| Constants | `const: "value"` |
| Union types | `anyOf`, `oneOf`, `type: ["string", "null"]` |
| References | `$ref: "#/$defs/Name"`, `$defs` |
| Formats | `date`, `time`, `date-time`, `email`, `uuid`, `ipv4` |

## Benchmarks

```bash
# Run all tests
go test -tags mlx ./x/grammar/...

# Run benchmarks
go test -tags mlx ./x/grammar/ -bench=.

# Compare with llama.cpp (outputs JSON)
go run -tags mlx ./x/grammar/cmd/compare -vocab-size 128000 -iterations 500

# Compare with a more complex schema
go run -tags mlx ./x/grammar/cmd/compare \
  -gbnf x/grammar/cmd/compare/complex.gbnf \
  -schema x/grammar/cmd/compare/complex.schema.json \
  -vocab-size 128000 -iterations 500
```

## References

- [XGrammar Paper](https://arxiv.org/abs/2411.15100) - Flexible and Efficient Structured Generation
- [Outlines](https://github.com/dottxt-ai/outlines) - Structured Text Generation
- [JSONSchemaBench](https://arxiv.org/abs/2501.10868) - Benchmark for Structured Outputs
