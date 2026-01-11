# Example Grammars

This directory contains example EBNF grammars for constrained decoding.

## Usage

```bash
go run -tags mlx ./x/imagegen/cmd/engine/ \
  -model /path/to/model \
  -prompt "Your prompt" \
  -grammar x/grammar/grammars/json.ebnf \
  -grammar-start value
```

## Available Grammars

| File | Start Rule | Description |
|------|------------|-------------|
| `json.ebnf` | `value` | Standard JSON (RFC 8259) |
| `expression.ebnf` | `expr` | Arithmetic expressions (+, -, *, /, parens) |
| `identifier.ebnf` | `ident` | Programming language identifiers |
| `boolean.ebnf` | `expr` | Boolean expressions (AND, OR, NOT) |
| `list.ebnf` | `list` | Comma-separated word list |
| `yesno.ebnf` | `response` | Simple yes/no responses |
| `date.ebnf` | `date` | Dates in YYYY-MM-DD format |
| `email.ebnf` | `email` | Basic email addresses |
| `phone.ebnf` | `phone` | US phone numbers |
| `hexcolor.ebnf` | `color` | CSS hex colors (#RGB or #RRGGBB) |
| `url.ebnf` | `url` | HTTP/HTTPS URLs |

## Grammar Syntax

**Note:** Comments are not supported. Grammar files must contain only EBNF productions.

The grammars use EBNF notation:

- `=` defines a production rule
- `|` is alternation (or)
- `{ }` is repetition (zero or more)
- `[ ]` is optional (zero or one)
- `" "` is a literal string
- `…` is a character range (e.g., `"a" … "z"`)
- `.` ends a production

## Writing Custom Grammars

1. Define your grammar in a `.ebnf` file
2. Choose a start rule name
3. Pass `-grammar path/to/grammar.ebnf -grammar-start rulename`

Example custom grammar for RGB colors:

```ebnf
color = "#" hexdigit hexdigit hexdigit hexdigit hexdigit hexdigit .
hexdigit = "0" … "9" | "a" … "f" | "A" … "F" .
```
