//go:build mlx

package constrained

import (
	"sync"
)

// JSONGrammar is the EBNF grammar for JSON.
// Based on https://www.json.org/json-en.html
//
// Note: This grammar operates at the character level. The engine will
// map tokens to character sequences for validation.
const JSONGrammar = `
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
unescaped = " " | "!" | "#" … "~" .
escaped = "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode ) .
unicode = "u" hex hex hex hex .
hex = "0" … "9" | "A" … "F" | "a" … "f" .

number = [ "-" ] integer [ fraction ] [ exponent ] .
integer = "0" | onenine { digit } .
fraction = "." digit { digit } .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
digit = "0" … "9" .
onenine = "1" … "9" .

ws = { " " | "\t" | "\n" | "\r" } .
`

// JSONGrammarSimplified is a token-oriented JSON grammar.
// This is simpler and more suitable for token-level constrained decoding.
// Tokens are treated as atomic units. We use lowercase names to avoid
// EBNF's lexical production rules.
//
// The grammar is factored to avoid common prefixes in alternatives,
// which simplifies PDA construction.
const JSONGrammarSimplified = `
json = value .

value = object | array | string | number | "true" | "false" | "null" .

object = "{" [ members ] "}" .
members = member { "," member } .
member = string ":" value .

array = "[" [ elements ] "]" .
elements = value { "," value } .

string = "STRING" .
number = "NUMBER" .
`

var (
	jsonPDA     *PDA
	jsonPDAOnce sync.Once
	jsonPDAErr  error
)

// GetJSONPDA returns the compiled PDA for JSON grammar.
// The PDA is compiled once and cached.
func GetJSONPDA() (*PDA, error) {
	jsonPDAOnce.Do(func() {
		jsonPDA, jsonPDAErr = CompileString(JSONGrammarSimplified, "json")
	})
	return jsonPDA, jsonPDAErr
}

// JSONRuntime wraps a PDA runtime with JSON-specific helpers.
type JSONRuntime struct {
	*Runtime
}

// NewJSONRuntime creates a new JSON validation runtime.
func NewJSONRuntime() (*JSONRuntime, error) {
	pda, err := GetJSONPDA()
	if err != nil {
		return nil, err
	}
	return &JSONRuntime{Runtime: NewRuntime(pda)}, nil
}

// TokenType represents the type of a JSON token.
type TokenType int

const (
	TokenUnknown     TokenType = iota
	TokenObjectStart           // {
	TokenObjectEnd             // }
	TokenArrayStart            // [
	TokenArrayEnd              // ]
	TokenColon                 // :
	TokenComma                 // ,
	TokenString                // "..."
	TokenNumber                // 123, -1.5e10, etc.
	TokenTrue                  // true
	TokenFalse                 // false
	TokenNull                  // null
	TokenWhitespace            // space, tab, newline
)

// ClassifyToken determines the JSON token type from a string.
func ClassifyToken(s string) TokenType {
	if len(s) == 0 {
		return TokenUnknown
	}

	// Check single-character tokens
	if len(s) == 1 {
		switch s[0] {
		case '{':
			return TokenObjectStart
		case '}':
			return TokenObjectEnd
		case '[':
			return TokenArrayStart
		case ']':
			return TokenArrayEnd
		case ':':
			return TokenColon
		case ',':
			return TokenComma
		case ' ', '\t', '\n', '\r':
			return TokenWhitespace
		}
	}

	// Check keywords
	switch s {
	case "true":
		return TokenTrue
	case "false":
		return TokenFalse
	case "null":
		return TokenNull
	}

	// Check if it's a string (starts and ends with quote)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return TokenString
	}

	// Check if it starts with quote (partial string)
	if s[0] == '"' {
		return TokenString
	}

	// Check if it could be a number
	if isNumberStart(s[0]) {
		return TokenNumber
	}

	return TokenUnknown
}

func isNumberStart(c byte) bool {
	return c == '-' || (c >= '0' && c <= '9')
}

// TokenToGrammarSymbol converts a token type to the grammar symbol it should match.
func TokenToGrammarSymbol(t TokenType) string {
	switch t {
	case TokenObjectStart:
		return "{"
	case TokenObjectEnd:
		return "}"
	case TokenArrayStart:
		return "["
	case TokenArrayEnd:
		return "]"
	case TokenColon:
		return ":"
	case TokenComma:
		return ","
	case TokenString:
		return "STRING"
	case TokenNumber:
		return "NUMBER"
	case TokenTrue:
		return "true"
	case TokenFalse:
		return "false"
	case TokenNull:
		return "null"
	default:
		return ""
	}
}

// AcceptToken validates and accepts a token string.
func (j *JSONRuntime) AcceptToken(token string) bool {
	tokenType := ClassifyToken(token)
	if tokenType == TokenUnknown || tokenType == TokenWhitespace {
		// Whitespace is always allowed (we skip it in the simplified grammar)
		return tokenType == TokenWhitespace
	}

	symbol := TokenToGrammarSymbol(tokenType)
	return j.Accept(symbol)
}

// ValidTokenTypes returns the valid token types from the current state.
func (j *JSONRuntime) ValidTokenTypes() []TokenType {
	validSymbols := j.ValidInputs()
	types := make([]TokenType, 0, len(validSymbols))

	for _, sym := range validSymbols {
		switch sym {
		case "{":
			types = append(types, TokenObjectStart)
		case "}":
			types = append(types, TokenObjectEnd)
		case "[":
			types = append(types, TokenArrayStart)
		case "]":
			types = append(types, TokenArrayEnd)
		case ":":
			types = append(types, TokenColon)
		case ",":
			types = append(types, TokenComma)
		case "STRING":
			types = append(types, TokenString)
		case "NUMBER":
			types = append(types, TokenNumber)
		case "true":
			types = append(types, TokenTrue)
		case "false":
			types = append(types, TokenFalse)
		case "null":
			types = append(types, TokenNull)
		}
	}

	return types
}
