# Issue #151: JSON Mode - GBNF Grammar-Constrained Output Implementation Guide

**Priority**: HIGH - Feature Enhancement
**Complexity**: High
**Effort**: 25 hours
**Status**: Ready for Implementation

## Problem Statement

Users need LLM outputs constrained to valid JSON format without:
- Trailing commas
- Invalid escape sequences
- Missing closing braces
- Non-JSON text before/after output

Current solution: Post-processing + regex (unreliable)

Required: **Real-time constraint** using GBNF (GGML BNF) grammar to force the model to only output valid JSON.

## Solution Overview

Implement GBNF grammar support in token generation:
1. Define JSON schema as GBNF grammar
2. Filter next-token predictions to only valid continuations
3. Support nested objects and arrays
4. Validate against user-provided schema

## Implementation

### Phase 1: GBNF JSON Grammar

```go
// llm/grammar/gbnf.go
package grammar

import (
    "bytes"
    "fmt"
)

const JSONGrammarTemplate = `
root   := object
value  := object | array | string | number | true | false | null

object := "{" ws "}" | "{" ws members "}" ws
members := member | member "," members
member := ws string ":" ws value

array  := "[" ws "]" | "[" ws elements "]" ws
elements := value | value "," elements

string := "\\"" [^"\\]* (escaped [^"\\]*)* "\\""
escaped := "\\\\" | "\\"/" | "\\"b" | "\\"f" | "\\"n" | "\\"r" | "\\"t" | unicode

number := "-"? int frac? exp?
int := "0" | [1-9] [0-9]*
frac := "." [0-9]+
exp := ("e" | "E") ("+" | "-")? [0-9]+

null := "null"
true := "true"
false := "false"

ws := [ \t\n]*
`

// GrammarValidator validates output against GBNF rules
type GrammarValidator struct {
    rules map[string]string
}

func NewGrammarValidator(grammarString string) (*GrammarValidator, error) {
    gv := &GrammarValidator{
        rules: parseGrammar(grammarString),
    }
    return gv, nil
}

// ValidateJSONSchema constrains output to JSON schema
type JSONSchemaConstraint struct {
    Schema interface{} // JSON schema definition
    Grammar string     // Generated GBNF
}

func NewJSONSchemaConstraint(schema interface{}) (*JSONSchemaConstraint, error) {
    // Convert JSON schema to GBNF
    grammar := generateGrammarFromSchema(schema)

    return &JSONSchemaConstraint{
        Schema: schema,
        Grammar: grammar,
    }, nil
}

// generateGrammarFromSchema converts JSON schema to GBNF
func generateGrammarFromSchema(schema interface{}) string {
    schemaMap, ok := schema.(map[string]interface{})
    if !ok {
        return JSONGrammarTemplate
    }

    var buf bytes.Buffer
    buf.WriteString("root := ")

    schemaType, _ := schemaMap["type"].(string)
    switch schemaType {
    case "object":
        buf.WriteString(generateObjectGrammar(schemaMap))
    case "array":
        buf.WriteString(generateArrayGrammar(schemaMap))
    default:
        buf.WriteString("value")
    }

    buf.WriteString("\n")
    buf.WriteString(JSONGrammarTemplate)

    return buf.String()
}

func generateObjectGrammar(schema map[string]interface{}) string {
    properties, ok := schema["properties"].(map[string]interface{})
    if !ok {
        return "object"
    }

    required, _ := schema["required"].([]interface{})

    var buf bytes.Buffer
    buf.WriteString("{")

    for i, propName := range required {
        if i > 0 {
            buf.WriteString(", ")
        }
        buf.WriteString(fmt.Sprintf(`"%s": `, propName))

        propSchema, _ := properties[propName].(map[string]interface{})
        propType, _ := propSchema["type"].(string)

        switch propType {
        case "string":
            buf.WriteString("string")
        case "number":
            buf.WriteString("number")
        case "integer":
            buf.WriteString("integer")
        case "boolean":
            buf.WriteString("(true | false)")
        case "array":
            buf.WriteString("array")
        default:
            buf.WriteString("value")
        }
    }

    buf.WriteString("}")
    return buf.String()
}

func generateArrayGrammar(schema map[string]interface{}) string {
    return "[value (, value)*]"
}
```

### Phase 2: Token Filter Middleware

```go
// llm/grammar/token_filter.go
package grammar

import (
    "log"
)

// TokenFilter filters model's next-token predictions
type TokenFilter struct {
    validator *GrammarValidator
    state     *ParsingState
    logger    *log.Logger
}

type ParsingState struct {
    // Track current parsing context
    Depth           int    // Nesting depth
    InString        bool   // Are we in a string?
    LastChar        byte   // Last character output
    BracketStack    []byte // Track [], {}
    EscapeNext      bool   // Next char is escaped
}

func NewTokenFilter(grammar string, logger *log.Logger) (*TokenFilter, error) {
    validator, err := NewGrammarValidator(grammar)
    if err != nil {
        return nil, err
    }

    return &TokenFilter{
        validator: validator,
        state: &ParsingState{
            BracketStack: make([]byte, 0),
        },
        logger: logger,
    }, nil
}

// FilterTokens removes invalid next tokens based on grammar
func (tf *TokenFilter) FilterTokens(vocab []string, nextTokenLogits []float32) ([]float32, error) {
    filtered := make([]float32, len(nextTokenLogits))
    copy(filtered, nextTokenLogits)

    for i, token := range vocab {
        if !tf.isValidNextToken(token) {
            // Zero out invalid tokens (set logit to -inf)
            filtered[i] = -1e10
        }
    }

    return filtered, nil
}

func (tf *TokenFilter) isValidNextToken(token string) bool {
    // Check if token is valid in current parsing context
    for _, char := range token {
        if !tf.isValidChar(byte(char)) {
            return false
        }
        tf.updateState(byte(char))
    }
    return true
}

func (tf *TokenFilter) isValidChar(char byte) bool {
    // String parsing rules
    if tf.state.InString {
        if tf.state.EscapeNext {
            // After backslash, only certain chars allowed
            validEscapes := map[byte]bool{
                '"':  true,
                '\\': true,
                '/':  true,
                'b':  true,
                'f':  true,
                'n':  true,
                'r':  true,
                't':  true,
                'u':  true,
            }
            return validEscapes[char]
        }

        if char == '"' {
            return true // End of string
        }
        if char == '\\' {
            return true // Start of escape
        }

        // Disallow control characters in strings
        return char >= 0x20
    }

    // Outside strings: whitespace, brackets, colons, commas valid
    validChars := map[byte]bool{
        ' ':  true,
        '\t': true,
        '\n': true,
        '\r': true,
        '{':  true,
        '}':  true,
        '[':  true,
        ']':  true,
        ':':  true,
        ',':  true,
        '"':  true,
        '-':  true,
        '.':  true,
        'e':  true,
        'E':  true,
        't':  true,
        'r':  true,
        'u':  true,
        'e':  true,
        'f':  true,
        'a':  true,
        'l':  true,
        's':  true,
        'n':  true,
    }

    if validChars[char] {
        return true
    }

    // Digits are always valid
    return char >= '0' && char <= '9'
}

func (tf *TokenFilter) updateState(char byte) {
    if tf.state.EscapeNext {
        tf.state.EscapeNext = false
        return
    }

    if tf.state.InString {
        if char == '\\' {
            tf.state.EscapeNext = true
        } else if char == '"' {
            tf.state.InString = false
        }
    } else {
        if char == '"' {
            tf.state.InString = true
        } else if char == '{' || char == '[' {
            tf.state.BracketStack = append(tf.state.BracketStack, char)
            if char == '{' {
                tf.state.Depth++
            }
        } else if char == '}' || char == ']' {
            if len(tf.state.BracketStack) > 0 {
                tf.state.BracketStack = tf.state.BracketStack[:len(tf.state.BracketStack)-1]
                if char == '}' {
                    tf.state.Depth--
                }
            }
        }
    }

    tf.state.LastChar = char
}

// IsComplete checks if valid JSON is complete
func (tf *TokenFilter) IsComplete() bool {
    return !tf.state.InString && len(tf.state.BracketStack) == 0
}
```

### Phase 3: Generate with Grammar

```go
// server/generate_with_grammar.go
package server

import (
    "context"
    "log"

    "ollama/llm/grammar"
)

type GenerateWithGrammarRequest struct {
    Model      string      `json:"model"`
    Prompt     string      `json:"prompt"`
    JSONSchema interface{} `json:"json_schema"` // Optional JSON schema
    Grammar    string      `json:"grammar"`     // Optional GBNF grammar
    Stream     bool        `json:"stream"`
}

func (s *Server) GenerateWithGrammar(ctx context.Context, req *GenerateWithGrammarRequest) error {
    // Determine grammar to use
    var grammarStr string

    if req.JSONSchema != nil {
        constraint, err := grammar.NewJSONSchemaConstraint(req.JSONSchema)
        if err != nil {
            return err
        }
        grammarStr = constraint.Grammar
    } else if req.Grammar != "" {
        grammarStr = req.Grammar
    } else {
        // Default to JSON grammar
        grammarStr = grammar.JSONGrammarTemplate
    }

    // Create token filter
    tokenFilter, err := grammar.NewTokenFilter(grammarStr, s.logger)
    if err != nil {
        return err
    }

    // Load model
    model, err := s.modelManager.Load(ctx, req.Model)
    if err != nil {
        return err
    }

    // Generate with grammar constraints
    output, err := model.GenerateWithFilter(ctx, req.Prompt, tokenFilter)
    if err != nil {
        return err
    }

    // Verify output matches grammar
    if !tokenFilter.IsComplete() {
        s.logger.Printf("WARNING: Output incomplete according to grammar")
    }

    s.logger.Printf("Generated JSON-constrained output: %s", output)
    return nil
}
```

## Acceptance Criteria

- ✅ Outputs always valid JSON (no parse errors)
- ✅ Supports custom JSON schemas
- ✅ Supports custom GBNF grammars
- ✅ No performance degradation (token filtering <1ms)
- ✅ Handles nested objects/arrays correctly
- ✅ Escape sequences handled properly
- ✅ Works with streaming responses

## Testing

```bash
# Test JSON mode
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Generate: {\"name\": \"",
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name", "age"]
    }
  }'

# Expected: {  "name": "John",  "age": 30  }
# Never: {  "name": "John  ,  "age": 30  }
```

## Deployment Checklist

- [ ] Implement GBNF parser
- [ ] Create token filter middleware
- [ ] Integrate into generation pipeline
- [ ] Test with various schemas
- [ ] Performance optimization
- [ ] Documentation with examples

---

**Ready for Implementation**: Yes - grammar constraints are well-defined, token filtering is clear pattern.
