//go:build mlx

// Package schema converts OpenAI-compatible JSON Schema into constrained grammars.
package schema

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/grammar"
)

// schemaNode represents OpenAI-compatible JSON Schema for structured outputs.
// See: https://platform.openai.com/docs/guides/structured-outputs
type schemaNode struct {
	// Core types
	Type interface{} `json:"type"` // string, []string, or nil

	// Object properties
	Properties           map[string]*schemaNode `json:"properties"`
	Required             []string               `json:"required"`
	AdditionalProperties interface{}            `json:"additionalProperties"`

	// Array properties
	Items    *schemaNode `json:"items"`
	MinItems *int        `json:"minItems"`
	MaxItems *int        `json:"maxItems"`

	// String properties
	Pattern string `json:"pattern"` // Regex pattern
	Format  string `json:"format"`  // date-time, email, uuid, etc.

	// Number properties (noted but not enforced in grammar - validated post-generation)
	Minimum          *float64 `json:"minimum"`
	Maximum          *float64 `json:"maximum"`
	ExclusiveMinimum *float64 `json:"exclusiveMinimum"`
	ExclusiveMaximum *float64 `json:"exclusiveMaximum"`
	MultipleOf       *float64 `json:"multipleOf"`

	// Enum and const
	Enum  []interface{} `json:"enum"`
	Const interface{}   `json:"const"`

	// Composition
	AnyOf []*schemaNode `json:"anyOf"`
	OneOf []*schemaNode `json:"oneOf"` // Treated same as anyOf for grammar

	// References and definitions
	Ref  string                 `json:"$ref"`
	Defs map[string]*schemaNode `json:"$defs"`

	// Description (ignored for grammar but useful for docs)
	Description string `json:"description"`
}

// converter handles JSON Schema to EBNF conversion with state.
type converter struct {
	schema      *schemaNode
	definitions map[string]*schemaNode // Resolved $defs
	usedTypes   map[string]bool
	rules       []string
	ruleNum     int
	definedRefs map[string]bool // Track which refs we've already defined as rules
}

// EBNF converts a JSON Schema to EBNF grammar
func EBNF(schemaJSON string) (string, error) {
	var schema schemaNode
	if err := json.Unmarshal([]byte(schemaJSON), &schema); err != nil {
		return "", fmt.Errorf("failed to parse JSON Schema: %w", err)
	}

	conv := &converter{
		schema:      &schema,
		definitions: schema.Defs,
		usedTypes:   make(map[string]bool),
		definedRefs: make(map[string]bool),
	}

	return conv.convert()
}

func (c *converter) convert() (string, error) {
	var b strings.Builder

	// Generate root rule
	rootExpr := c.schemaToExpr(c.schema, "root")
	b.WriteString("root = ")
	b.WriteString(rootExpr)
	b.WriteString(" .\n")

	// Add generated rules (refs, items, etc.)
	for _, rule := range c.rules {
		b.WriteString(rule)
		b.WriteString("\n")
	}

	// Add primitives based on usage
	c.addPrimitives(&b)

	return b.String(), nil
}

func (c *converter) addPrimitives(b *strings.Builder) {
	if c.usedTypes["string"] {
		b.WriteString(`
string = "\"" { character } "\"" .
`)
	}

	if c.usedTypes["string"] || c.usedTypes["character"] {
		b.WriteString(`
character = unescaped | escaped .
unescaped = " " | "!" | "#" … "[" | "]" … "~" .
escaped = "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode ) .
unicode = "u" hex hex hex hex .
`)
	}

	if c.usedTypes["number"] {
		b.WriteString(`
number = [ "-" ] integer [ fraction ] [ exponent ] .
integer = "0" | onenine { digit } .
fraction = "." digit { digit } .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
`)
	}

	if c.usedTypes["integer"] {
		b.WriteString(`
int = [ "-" ] ( "0" | onenine { digit } ) .
`)
	}

	if c.usedTypes["number"] || c.usedTypes["integer"] || c.usedTypes["digit"] {
		b.WriteString(`
digit = "0" … "9" .
`)
	}

	// onenine only needed for number/integer, not for digit-only formats
	if c.usedTypes["number"] || c.usedTypes["integer"] {
		b.WriteString(`onenine = "1" … "9" .
`)
	}

	if c.usedTypes["string"] || c.usedTypes["character"] || c.usedTypes["hex"] {
		b.WriteString(`
hex = "0" … "9" | "A" … "F" | "a" … "f" .
`)
	}

	if c.usedTypes["ws"] {
		b.WriteString(`
ws = { " " | "\t" | "\n" | "\r" } .
`)
	}
}

func (c *converter) schemaToExpr(schema *schemaNode, name string) string {
	if schema == nil {
		c.usedTypes["string"] = true
		c.usedTypes["number"] = true
		return "( string | number | object | array | \"true\" | \"false\" | \"null\" )"
	}

	// Handle $ref first
	if schema.Ref != "" {
		return c.resolveRef(schema.Ref)
	}

	// Handle const
	if schema.Const != nil {
		return c.constToExpr(schema.Const)
	}

	// Handle enum
	if len(schema.Enum) > 0 {
		return c.enumToExpr(schema.Enum)
	}

	// Handle anyOf / oneOf
	if len(schema.AnyOf) > 0 {
		return c.anyOfToExpr(schema.AnyOf, name)
	}
	if len(schema.OneOf) > 0 {
		return c.anyOfToExpr(schema.OneOf, name)
	}

	// Handle type
	types := c.getTypes(schema.Type)
	if len(types) == 0 {
		// No type specified, could be anything
		c.usedTypes["string"] = true
		c.usedTypes["number"] = true
		return "( string | number | \"true\" | \"false\" | \"null\" )"
	}

	if len(types) == 1 {
		return c.typeToExpr(types[0], schema, name)
	}

	// Multiple types (e.g., ["string", "null"])
	var parts []string
	for _, t := range types {
		parts = append(parts, c.typeToExpr(t, schema, name))
	}
	return "( " + strings.Join(parts, " | ") + " )"
}

func (c *converter) typeToExpr(typeName string, schema *schemaNode, name string) string {
	switch typeName {
	case "object":
		return c.objectToExpr(schema, name)
	case "array":
		return c.arrayToExpr(schema, name)
	case "string":
		return c.stringToExpr(schema, name)
	case "number":
		c.usedTypes["number"] = true
		return "number"
	case "integer":
		c.usedTypes["integer"] = true
		c.usedTypes["digit"] = true
		return "int"
	case "boolean":
		return `( "true" | "false" )`
	case "null":
		return `"null"`
	default:
		c.usedTypes["string"] = true
		c.usedTypes["number"] = true
		return "string"
	}
}

func (c *converter) objectToExpr(schema *schemaNode, name string) string {
	c.usedTypes["ws"] = true

	if len(schema.Properties) == 0 {
		return `"{" ws "}"`
	}

	// Sort properties for deterministic output
	// Required properties come first, in their required order
	var propOrder []string
	requiredSet := make(map[string]bool)
	for _, r := range schema.Required {
		requiredSet[r] = true
		propOrder = append(propOrder, r)
	}

	// Add any non-required properties (though OpenAI requires all to be required)
	var optionalProps []string
	for propName := range schema.Properties {
		if !requiredSet[propName] {
			optionalProps = append(optionalProps, propName)
		}
	}
	sort.Strings(optionalProps)
	propOrder = append(propOrder, optionalProps...)

	var propExprs []string
	first := true

	for _, propName := range propOrder {
		propSchema, exists := schema.Properties[propName]
		if !exists {
			continue
		}

		propExpr := c.schemaToExpr(propSchema, propName)

		prefix := ""
		if !first {
			prefix = `"," ws `
		}
		first = false

		propExprs = append(propExprs, fmt.Sprintf(`%s"\"%s\"" ws ":" ws %s`, prefix, propName, propExpr))
	}

	if len(propExprs) == 0 {
		return `"{" ws "}"`
	}

	return `"{" ws ` + strings.Join(propExprs, " ") + ` ws "}"`
}

func (c *converter) arrayToExpr(schema *schemaNode, name string) string {
	c.usedTypes["ws"] = true

	itemExpr := "value"
	if schema.Items != nil {
		itemExpr = c.schemaToExpr(schema.Items, name+"_item")
	} else {
		c.usedTypes["string"] = true
		c.usedTypes["number"] = true
	}

	// Create item rule
	c.ruleNum++
	itemRule := fmt.Sprintf("item%d", c.ruleNum)
	c.rules = append(c.rules, fmt.Sprintf("%s = %s .", itemRule, itemExpr))

	// Handle minItems/maxItems
	if schema.MinItems != nil || schema.MaxItems != nil {
		return c.arrayWithBounds(itemRule, schema.MinItems, schema.MaxItems)
	}

	// Default: zero or more items
	return fmt.Sprintf(`( "[" ws "]" | "[" ws %s { "," ws %s } ws "]" )`, itemRule, itemRule)
}

func (c *converter) arrayWithBounds(itemRule string, minItems, maxItems *int) string {
	min := 0
	max := -1 // unlimited

	if minItems != nil {
		min = *minItems
	}
	if maxItems != nil {
		max = *maxItems
	}

	if min == 0 && max < 0 {
		// No constraints
		return fmt.Sprintf(`( "[" ws "]" | "[" ws %s { "," ws %s } ws "]" )`, itemRule, itemRule)
	}

	if min == 0 && max == 0 {
		return `"[" ws "]"`
	}

	// Build pattern for bounded array
	// For min=2, max=4: item "," item [ "," item ] [ "," item ]
	var parts []string

	// Required items
	for i := 0; i < min; i++ {
		if i > 0 {
			parts = append(parts, `"," ws`)
		}
		parts = append(parts, itemRule)
	}

	// Optional items up to max
	if max > min {
		for i := min; i < max; i++ {
			if i == 0 {
				parts = append(parts, fmt.Sprintf(`[ %s`, itemRule))
			} else {
				parts = append(parts, fmt.Sprintf(`[ "," ws %s`, itemRule))
			}
		}
		// Close all optional brackets
		for i := min; i < max; i++ {
			parts = append(parts, "]")
		}
	} else if max < 0 {
		// Unlimited after min
		if min > 0 {
			parts = append(parts, fmt.Sprintf(`{ "," ws %s }`, itemRule))
		} else {
			parts = append(parts, fmt.Sprintf(`[ %s { "," ws %s } ]`, itemRule, itemRule))
		}
	}

	if min == 0 {
		return fmt.Sprintf(`( "[" ws "]" | "[" ws %s ws "]" )`, strings.Join(parts, " "))
	}
	return fmt.Sprintf(`"[" ws %s ws "]"`, strings.Join(parts, " "))
}

func (c *converter) stringToExpr(schema *schemaNode, name string) string {
	// Handle format
	if schema.Format != "" {
		return c.formatToExpr(schema.Format)
	}

	// Handle pattern (regex)
	if schema.Pattern != "" {
		return c.patternToExpr(schema.Pattern, name)
	}

	// Default string
	c.usedTypes["string"] = true
	if name == "root" {
		c.usedTypes["character"] = true
		return `"\"" { character } "\""`
	}
	return "string"
}

func (c *converter) formatToExpr(format string) string {
	switch format {
	case "date":
		// YYYY-MM-DD
		c.ruleNum++
		c.usedTypes["digit"] = true
		ruleName := fmt.Sprintf("date%d", c.ruleNum)
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" digit digit digit digit "-" digit digit "-" digit digit "\"" .`, ruleName))
		return ruleName

	case "time":
		// HH:MM:SS
		c.ruleNum++
		c.usedTypes["digit"] = true
		ruleName := fmt.Sprintf("time%d", c.ruleNum)
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" digit digit ":" digit digit ":" digit digit "\"" .`, ruleName))
		return ruleName

	case "date-time":
		// YYYY-MM-DDTHH:MM:SSZ or with offset
		c.ruleNum++
		c.usedTypes["digit"] = true
		ruleName := fmt.Sprintf("datetime%d", c.ruleNum)
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" digit digit digit digit "-" digit digit "-" digit digit "T" digit digit ":" digit digit ":" digit digit ( "Z" | ( "+" | "-" ) digit digit ":" digit digit ) "\"" .`, ruleName))
		return ruleName

	case "email":
		// Simplified email pattern
		c.ruleNum++
		ruleName := fmt.Sprintf("email%d", c.ruleNum)
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" emailchar { emailchar } "@" emailchar { emailchar } "." emailchar { emailchar } "\"" .`, ruleName))
		c.rules = append(c.rules, `emailchar = "a" … "z" | "A" … "Z" | "0" … "9" | "." | "-" | "_" .`)
		return ruleName

	case "uuid":
		// 8-4-4-4-12 hex pattern
		c.ruleNum++
		ruleName := fmt.Sprintf("uuid%d", c.ruleNum)
		c.usedTypes["hex"] = true
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" hex hex hex hex hex hex hex hex "-" hex hex hex hex "-" hex hex hex hex "-" hex hex hex hex "-" hex hex hex hex hex hex hex hex hex hex hex hex "\"" .`, ruleName))
		return ruleName

	case "ipv4":
		c.ruleNum++
		c.usedTypes["digit"] = true
		ruleName := fmt.Sprintf("ipv4_%d", c.ruleNum)
		c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" digit { digit } "." digit { digit } "." digit { digit } "." digit { digit } "\"" .`, ruleName))
		return ruleName

	case "uri", "hostname":
		// Fallback to general string for complex formats
		c.usedTypes["string"] = true
		return "string"

	default:
		c.usedTypes["string"] = true
		return "string"
	}
}

func (c *converter) patternToExpr(pattern string, name string) string {
	// Try to convert simple regex patterns to EBNF
	// This handles common cases; complex regex falls back to string

	// Remove anchors
	pattern = strings.TrimPrefix(pattern, "^")
	pattern = strings.TrimSuffix(pattern, "$")

	// Try to parse and convert
	expr, ok := c.regexToEBNF(pattern)
	if !ok {
		// Fallback to general string
		c.usedTypes["string"] = true
		return "string"
	}

	c.ruleNum++
	ruleName := fmt.Sprintf("pattern%d", c.ruleNum)
	c.rules = append(c.rules, fmt.Sprintf(`%s = "\"" %s "\"" .`, ruleName, expr))
	return ruleName
}

func (c *converter) regexToEBNF(pattern string) (string, bool) {
	// Simple regex to EBNF converter
	// Handles: literals, [a-z], [A-Z], [0-9], +, *, ?, basic groups

	var result strings.Builder
	i := 0

	for i < len(pattern) {
		ch := pattern[i]

		switch ch {
		case '[':
			// Character class
			end := strings.Index(pattern[i:], "]")
			if end == -1 {
				return "", false
			}
			class := pattern[i+1 : i+end]
			ebnfClass, ok := c.charClassToEBNF(class)
			if !ok {
				return "", false
			}
			result.WriteString(ebnfClass)
			i += end + 1

		case '(':
			// Group - find matching )
			depth := 1
			start := i + 1
			j := start
			for j < len(pattern) && depth > 0 {
				if pattern[j] == '(' {
					depth++
				} else if pattern[j] == ')' {
					depth--
				}
				j++
			}
			if depth != 0 {
				return "", false
			}
			groupContent := pattern[start : j-1]
			groupExpr, ok := c.regexToEBNF(groupContent)
			if !ok {
				return "", false
			}
			result.WriteString("( ")
			result.WriteString(groupExpr)
			result.WriteString(" )")
			i = j

		case '|':
			result.WriteString(" | ")
			i++

		case '+':
			// One or more - wrap previous in { } and add one required
			// This is a simplification
			return "", false // TODO: handle properly

		case '*':
			// Zero or more - need to wrap previous
			return "", false // TODO: handle properly

		case '?':
			// Optional - need to wrap previous in [ ]
			return "", false // TODO: handle properly

		case '\\':
			// Escape sequence
			if i+1 >= len(pattern) {
				return "", false
			}
			next := pattern[i+1]
			switch next {
			case 'd':
				result.WriteString("digit")
				c.usedTypes["digit"] = true
			case 'w':
				result.WriteString(`( "a" … "z" | "A" … "Z" | "0" … "9" | "_" )`)
			case 's':
				result.WriteString(`( " " | "\t" )`)
			default:
				result.WriteString(fmt.Sprintf(`"%c"`, next))
			}
			i += 2

		default:
			// Literal character
			if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' {
				result.WriteString(fmt.Sprintf(`"%c" `, ch))
			} else {
				// Special char, try to escape
				result.WriteString(fmt.Sprintf(`"%c" `, ch))
			}
			i++
		}
	}

	return strings.TrimSpace(result.String()), true
}

func (c *converter) charClassToEBNF(class string) (string, bool) {
	// Handle character classes like a-z, A-Z, 0-9
	if class == "a-zA-Z0-9_" || class == "a-zA-Z_" {
		return `( "a" … "z" | "A" … "Z" | "0" … "9" | "_" )`, true
	}
	if class == "a-zA-Z0-9" {
		return `( "a" … "z" | "A" … "Z" | "0" … "9" )`, true
	}
	if class == "a-z" {
		return `"a" … "z"`, true
	}
	if class == "A-Z" {
		return `"A" … "Z"`, true
	}
	if class == "0-9" {
		c.usedTypes["digit"] = true
		return "digit", true
	}

	// Try to parse range patterns
	if matched, _ := regexp.MatchString(`^[a-zA-Z]-[a-zA-Z]$`, class); matched {
		return fmt.Sprintf(`"%c" … "%c"`, class[0], class[2]), true
	}
	if matched, _ := regexp.MatchString(`^[0-9]-[0-9]$`, class); matched {
		return fmt.Sprintf(`"%c" … "%c"`, class[0], class[2]), true
	}

	return "", false
}

func (c *converter) anyOfToExpr(schemas []*schemaNode, name string) string {
	var parts []string
	for i, s := range schemas {
		expr := c.schemaToExpr(s, fmt.Sprintf("%s_opt%d", name, i))
		parts = append(parts, expr)
	}
	return "( " + strings.Join(parts, " | ") + " )"
}

func (c *converter) enumToExpr(values []interface{}) string {
	var parts []string
	for _, v := range values {
		parts = append(parts, c.constToExpr(v))
	}
	return "( " + strings.Join(parts, " | ") + " )"
}

func (c *converter) constToExpr(v interface{}) string {
	switch val := v.(type) {
	case string:
		return fmt.Sprintf(`"\"%s\""`, c.escapeString(val))
	case float64:
		if val == float64(int(val)) {
			return fmt.Sprintf(`"%d"`, int(val))
		}
		return fmt.Sprintf(`"%v"`, val)
	case bool:
		if val {
			return `"true"`
		}
		return `"false"`
	case nil:
		return `"null"`
	default:
		c.usedTypes["string"] = true
		return "string"
	}
}

func (c *converter) resolveRef(ref string) string {
	// Handle #/$defs/name references
	if strings.HasPrefix(ref, "#/$defs/") {
		defName := strings.TrimPrefix(ref, "#/$defs/")
		return c.resolveDefRef(defName)
	}

	// Handle root recursion #
	if ref == "#" {
		return "root"
	}

	// Unknown ref format
	c.usedTypes["string"] = true
	return "string"
}

func (c *converter) resolveDefRef(defName string) string {
	// Check if we've already defined this as a rule
	ruleName := "def_" + defName
	if c.definedRefs[defName] {
		return ruleName
	}

	// Mark as defined to prevent infinite recursion
	c.definedRefs[defName] = true

	// Look up the definition
	if c.definitions == nil {
		c.usedTypes["string"] = true
		return "string"
	}

	defSchema, ok := c.definitions[defName]
	if !ok {
		c.usedTypes["string"] = true
		return "string"
	}

	// Generate the rule
	expr := c.schemaToExpr(defSchema, ruleName)
	c.rules = append(c.rules, fmt.Sprintf("%s = %s .", ruleName, expr))

	return ruleName
}

func (c *converter) getTypes(t interface{}) []string {
	switch v := t.(type) {
	case string:
		return []string{v}
	case []interface{}:
		var types []string
		for _, item := range v {
			if s, ok := item.(string); ok {
				types = append(types, s)
			}
		}
		return types
	}
	return nil
}

func (c *converter) escapeString(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}

// Grammar converts a JSON Schema string into a compiled grammar.
func Grammar(schemaJSON string) (*grammar.Grammar, error) {
	ebnf, err := EBNF(schemaJSON)
	if err != nil {
		return nil, err
	}
	return grammar.ParseEBNF(ebnf, "root")
}
