package sample

import (
	"encoding/json"
	"strings"
)

type jsonState int

const (
	stateExpectValue jsonState = iota
	stateExpectObjectKeyOrEnd
	stateExpectObjectColon
	stateExpectObjectValue
	stateExpectObjectCommaOrEnd
	stateExpectArrayValueOrEnd
	stateExpectArrayCommaOrEnd
	stateParsingString
	stateParsingObjectKey
	stateParsingLiteral
)

type Schema struct {
	Type       string             `json:"type"`
	Properties map[string]*Schema `json:"properties"`
	Required   []string           `json:"required"`
	Items      *Schema            `json:"items"`
}

type GrammarConstraint struct {
	rootSchema *Schema
}

type StateTracker struct {
	State       jsonState
	Stack       []rune
	SchemaStack []*Schema
	CurrentKey  string
	Escape      bool
	LiteralBuf  string
	Valid       bool
}

func NewGrammarConstraint(format json.RawMessage, grammar string) (*GrammarConstraint, error) {
	var s *Schema
	var isJSON bool
	if len(format) > 0 {
		switch string(format) {
		case `null`, `""`:
			// noop
		case `"json"`:
			isJSON = true
		default:
			if format[0] == '{' {
				s = &Schema{}
				if err := json.Unmarshal(format, s); err != nil {
					return nil, err
				}
			}
		}
	} else if grammar != "" {
		isJSON = true
	}

	if s == nil && !isJSON {
		return nil, nil
	}

	return &GrammarConstraint{rootSchema: s}, nil
}

func (gc *GrammarConstraint) IsValidPrefix(s string) bool {
	st := gc.GetState(s)
	return st.Valid
}

func (gc *GrammarConstraint) IsComplete(s string) bool {
	if len(s) == 0 {
		return false
	}
	st := gc.GetState(s)
	if !st.Valid {
		return false
	}
	return len(st.Stack) == 0 && st.State != stateParsingLiteral && st.State != stateExpectObjectColon && st.State != stateExpectObjectValue
}

func (gc *GrammarConstraint) GetState(s string) StateTracker {
	st := StateTracker{
		State:       stateExpectValue,
		Stack:       []rune{},
		SchemaStack: []*Schema{gc.rootSchema},
		Valid:       true,
	}

	for _, r := range s {
		st = st.Transition(r)
		if !st.Valid {
			break
		}
	}
	return st
}

func (st StateTracker) Transition(r rune) StateTracker {
	if !st.Valid {
		return st
	}

	next := StateTracker{
		State:      st.State,
		Stack:      append([]rune{}, st.Stack...),
		Escape:     st.Escape,
		CurrentKey: st.CurrentKey,
		LiteralBuf: st.LiteralBuf,
		Valid:      true,
	}
	next.SchemaStack = append([]*Schema{}, st.SchemaStack...)

	var currentSchema *Schema
	if len(next.SchemaStack) > 0 {
		currentSchema = next.SchemaStack[len(next.SchemaStack)-1]
	}

	// 1. Handle string/key parsing
	if next.State == stateParsingString || next.State == stateParsingObjectKey {
		if next.Escape {
			next.Escape = false
		} else if r == '\\' {
			next.Escape = true
		} else if r == '"' {
			if next.State == stateParsingObjectKey {
				next.State = stateExpectObjectColon
				keyStr := next.CurrentKey
				if currentSchema != nil && currentSchema.Properties != nil {
					if _, exists := currentSchema.Properties[keyStr]; !exists {
						next.Valid = false
						return next
					}
				}
			} else {
				if len(next.Stack) > 0 && next.Stack[len(next.Stack)-1] == '{' {
					next.State = stateExpectObjectCommaOrEnd
				} else {
					if len(next.Stack) == 0 {
						next.State = stateExpectValue
					} else if next.Stack[len(next.Stack)-1] == '[' {
						next.State = stateExpectArrayCommaOrEnd
					}
				}
			}
		} else {
			if next.State == stateParsingObjectKey {
				next.CurrentKey += string(r)
				if currentSchema != nil && currentSchema.Properties != nil {
					prefixMatches := false
					for k := range currentSchema.Properties {
						if strings.HasPrefix(k, next.CurrentKey) {
							prefixMatches = true
							break
						}
					}
					if !prefixMatches {
						next.Valid = false
						return next
					}
				}
			}
		}
		return next
	}

	// Ignore spaces outside of strings
	if isSpace(r) {
		return next
	}

	switch r {
	case '{':
		if currentSchema != nil && currentSchema.Type != "object" && currentSchema.Type != "" {
			next.Valid = false
			return next
		}
		if next.State != stateExpectValue && next.State != stateExpectObjectValue && next.State != stateExpectArrayValueOrEnd {
			next.Valid = false
			return next
		}

		if next.State == stateExpectObjectValue {
			if currentSchema != nil && currentSchema.Properties != nil {
				next.SchemaStack = append(next.SchemaStack, currentSchema.Properties[next.CurrentKey])
			} else {
				next.SchemaStack = append(next.SchemaStack, nil)
			}
		} else if next.State == stateExpectArrayValueOrEnd {
			if currentSchema != nil && currentSchema.Items != nil {
				next.SchemaStack = append(next.SchemaStack, currentSchema.Items)
			} else {
				next.SchemaStack = append(next.SchemaStack, nil)
			}
		}

		next.Stack = append(next.Stack, '{')
		next.State = stateExpectObjectKeyOrEnd
		next.CurrentKey = ""

	case '}':
		if len(next.Stack) == 0 || next.Stack[len(next.Stack)-1] != '{' {
			next.Valid = false
			return next
		}
		if next.State != stateExpectObjectKeyOrEnd && next.State != stateExpectObjectCommaOrEnd && next.State != stateParsingLiteral {
			next.Valid = false
			return next
		}

		next.Stack = next.Stack[:len(next.Stack)-1]
		if len(next.SchemaStack) > 1 {
			next.SchemaStack = next.SchemaStack[:len(next.SchemaStack)-1]
		}
		if len(next.Stack) == 0 {
			next.State = stateExpectValue
		} else if next.Stack[len(next.Stack)-1] == '{' {
			next.State = stateExpectObjectCommaOrEnd
		} else {
			next.State = stateExpectArrayCommaOrEnd
		}

	case '[':
		if currentSchema != nil && currentSchema.Type != "array" && currentSchema.Type != "" {
			next.Valid = false
			return next
		}
		if next.State != stateExpectValue && next.State != stateExpectObjectValue && next.State != stateExpectArrayValueOrEnd {
			next.Valid = false
			return next
		}

		if next.State == stateExpectObjectValue {
			if currentSchema != nil && currentSchema.Properties != nil {
				next.SchemaStack = append(next.SchemaStack, currentSchema.Properties[next.CurrentKey])
			} else {
				next.SchemaStack = append(next.SchemaStack, nil)
			}
		} else if next.State == stateExpectArrayValueOrEnd {
			if currentSchema != nil && currentSchema.Items != nil {
				next.SchemaStack = append(next.SchemaStack, currentSchema.Items)
			} else {
				next.SchemaStack = append(next.SchemaStack, nil)
			}
		}

		next.Stack = append(next.Stack, '[')
		next.State = stateExpectArrayValueOrEnd

	case ']':
		if len(next.Stack) == 0 || next.Stack[len(next.Stack)-1] != '[' {
			next.Valid = false
			return next
		}
		if next.State != stateExpectArrayValueOrEnd && next.State != stateExpectArrayCommaOrEnd && next.State != stateParsingLiteral {
			next.Valid = false
			return next
		}
		next.Stack = next.Stack[:len(next.Stack)-1]
		if len(next.SchemaStack) > 1 {
			next.SchemaStack = next.SchemaStack[:len(next.SchemaStack)-1]
		}
		if len(next.Stack) == 0 {
			next.State = stateExpectValue
		} else if next.Stack[len(next.Stack)-1] == '{' {
			next.State = stateExpectObjectCommaOrEnd
		} else {
			next.State = stateExpectArrayCommaOrEnd
		}

	case '"':
		if next.State != stateExpectValue && next.State != stateExpectObjectValue && next.State != stateExpectObjectKeyOrEnd && next.State != stateExpectArrayValueOrEnd {
			next.Valid = false
			return next
		}

		if next.State == stateExpectObjectKeyOrEnd {
			next.State = stateParsingObjectKey
			next.CurrentKey = ""
		} else {
			// Expecting value, so we must be starting a string
			var valSchema *Schema
			if next.State == stateExpectObjectValue {
				if currentSchema != nil && currentSchema.Properties != nil {
					valSchema = currentSchema.Properties[next.CurrentKey]
				}
			} else if next.State == stateExpectArrayValueOrEnd {
				if currentSchema != nil {
					valSchema = currentSchema.Items
				}
			} else {
				valSchema = currentSchema
			}
			if valSchema != nil && valSchema.Type != "string" && valSchema.Type != "" {
				next.Valid = false
				return next
			}
			next.State = stateParsingString
		}

	case ':':
		if next.State != stateExpectObjectColon {
			next.Valid = false
			return next
		}
		next.State = stateExpectObjectValue

	case ',':
		if len(next.Stack) == 0 {
			next.Valid = false
			return next
		}
		if next.Stack[len(next.Stack)-1] == '{' {
			if next.State != stateExpectObjectCommaOrEnd && next.State != stateParsingLiteral {
				next.Valid = false
				return next
			}
			next.State = stateExpectObjectKeyOrEnd
			next.CurrentKey = ""
		} else {
			if next.State != stateExpectArrayCommaOrEnd && next.State != stateParsingLiteral {
				next.Valid = false
				return next
			}
			next.State = stateExpectArrayValueOrEnd
		}

	default:
		if isLiteralChar(r) {
			if next.State == stateExpectValue || next.State == stateExpectObjectValue || next.State == stateExpectArrayValueOrEnd {
				next.State = stateParsingLiteral
				next.LiteralBuf = string(r)
			} else if next.State == stateParsingLiteral {
				next.LiteralBuf += string(r)
			} else {
				next.Valid = false
				return next
			}

			var valSchema *Schema
			if len(next.Stack) > 0 && next.Stack[len(next.Stack)-1] == '{' {
				if currentSchema != nil && currentSchema.Properties != nil {
					valSchema = currentSchema.Properties[next.CurrentKey]
				}
			} else if len(next.Stack) > 0 && next.Stack[len(next.Stack)-1] == '[' {
				if currentSchema != nil {
					valSchema = currentSchema.Items
				}
			} else {
				valSchema = currentSchema
			}

			if valSchema != nil && valSchema.Type != "" {
				lit := r
				switch valSchema.Type {
				case "integer", "number":
					if !(lit >= '0' && lit <= '9') && lit != '.' && lit != '-' && lit != '+' && lit != 'e' && lit != 'E' {
						next.Valid = false
						return next
					}
				case "boolean":
					prefix := next.LiteralBuf
					if !strings.HasPrefix("true", prefix) && !strings.HasPrefix("false", prefix) {
						next.Valid = false
						return next
					}
				case "null":
					prefix := next.LiteralBuf
					if !strings.HasPrefix("null", prefix) {
						next.Valid = false
						return next
					}
				default:
					next.Valid = false
					return next
				}
			}
		} else {
			next.Valid = false
			return next
		}
	}
	return next
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

func isLiteralChar(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '-' || r == '+'
}

func ValidateJSONPrefix(s string) bool {
	st := StateTracker{
		State: stateExpectValue,
		Stack: []rune{},
		Valid: true,
	}
	for _, r := range s {
		st = st.Transition(r)
		if !st.Valid {
			return false
		}
	}
	return true
}
