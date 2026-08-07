package structured

import (
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// This file ports llama.cpp b10091's common/json-schema-to-grammar.cpp —
// the converter the fork's llama-server path runs when a request carries a
// JSON Schema — so safetensors/MLX models constrain to the same language
// GGUF models do. The port emits the same GBNF rule bodies and feeds them
// through gbnfToGrammar. Behaviour differences are deliberate and named:
// "pattern" returns an error instead of compiling regexes, and external
// (https://) $refs return an error instead of fetching.

const spaceRule = `| " " | "\n"{1,2} [ \t]{0,20}`

type builtinRule struct {
	content string
	deps    []string
}

var primitiveRules = map[string]builtinRule{
	"boolean":       {`("true" | "false") space`, nil},
	"decimal-part":  {`[0-9]{1,16}`, nil},
	"integral-part": {`[0] | [1-9] [0-9]{0,15}`, nil},
	"number":        {`("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space`, []string{"integral-part", "decimal-part"}},
	"integer":       {`("-"? integral-part) space`, []string{"integral-part"}},
	"value":         {`object | array | string | number | boolean | null`, []string{"object", "array", "string", "number", "boolean", "null"}},
	"object":        {`"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space`, []string{"string", "value"}},
	"array":         {`"[" space ( value ("," space value)* )? "]" space`, []string{"value"}},
	"uuid":          {`"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space`, nil},
	"char":          {`[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`, nil},
	"string":        {`"\"" char* "\"" space`, []string{"char"}},
	"null":          {`"null" space`, nil},
}

var stringFormatRules = map[string]builtinRule{
	"date":             {`[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )`, nil},
	"time":             {`([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )`, nil},
	"date-time":        {`date "T" time`, []string{"date", "time"}},
	"date-string":      {`"\"" date "\"" space`, []string{"date"}},
	"time-string":      {`"\"" time "\"" space`, []string{"time"}},
	"date-time-string": {`"\"" date-time "\"" space`, []string{"date-time"}},
}

var reservedNames = func() map[string]bool {
	s := map[string]bool{"root": true}
	for k := range primitiveRules {
		s[k] = true
	}
	for k := range stringFormatRules {
		s[k] = true
	}
	return s
}()

var (
	invalidRuleChars = regexp.MustCompile(`[^a-zA-Z0-9-]+`)
	uuidFormatRe     = regexp.MustCompile(`^uuid[1-5]?$`)
)

// formatLiteral renders s as a GBNF literal, escaping per b10091's
// GRAMMAR_LITERAL_ESCAPE_RE ([\r\n"\\]).
func formatLiteral(s string) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; c {
		case '\r':
			sb.WriteString(`\r`)
		case '\n':
			sb.WriteString(`\n`)
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		default:
			sb.WriteByte(c)
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

// buildRepetition emits itemRule{minItems,maxItems}, optionally interleaved
// with separatorRule. maxItems < 0 means unbounded.
func buildRepetition(itemRule string, minItems, maxItems int, separatorRule string) string {
	hasMax := maxItems >= 0
	if maxItems == 0 {
		return ""
	}
	if minItems == 0 && maxItems == 1 {
		return itemRule + "?"
	}
	if separatorRule == "" {
		switch {
		case minItems == 1 && !hasMax:
			return itemRule + "+"
		case minItems == 0 && !hasMax:
			return itemRule + "*"
		case hasMax:
			return itemRule + "{" + strconv.Itoa(minItems) + "," + strconv.Itoa(maxItems) + "}"
		default:
			return itemRule + "{" + strconv.Itoa(minItems) + ",}"
		}
	}

	innerMin := 0
	if minItems > 0 {
		innerMin = minItems - 1
	}
	innerMax := maxItems
	if hasMax {
		innerMax = maxItems - 1
	}
	result := itemRule + " " + buildRepetition("("+separatorRule+" "+itemRule+")", innerMin, innerMax, "")
	if minItems == 0 {
		result = "(" + result + ")?"
	}
	return result
}

// buildMinMaxInt ports b10091's _build_min_max_int digit-decomposition of
// integer ranges. hasMin/hasMax replace the C++ int64 sentinels.
func buildMinMaxInt(minValue, maxValue int64, hasMin, hasMax bool, decimalsLeft int, topLevel bool, out *strings.Builder) error {
	digitRange := func(from, to byte) {
		out.WriteByte('[')
		if from == to {
			out.WriteByte(from)
		} else {
			out.WriteByte(from)
			out.WriteByte('-')
			out.WriteByte(to)
		}
		out.WriteByte(']')
	}
	moreDigits := func(minDigits, maxDigits int) {
		out.WriteString("[0-9]")
		if minDigits == maxDigits && minDigits == 1 {
			return
		}
		fmt.Fprintf(out, "{%d", minDigits)
		if maxDigits != minDigits {
			out.WriteByte(',')
			fmt.Fprintf(out, "%d", maxDigits)
		}
		out.WriteByte('}')
	}
	var uniformRange func(from, to string)
	uniformRange = func(from, to string) {
		i := 0
		for i < len(from) && i < len(to) && from[i] == to[i] {
			i++
		}
		if i > 0 {
			out.WriteString("\"" + from[:i] + "\"")
		}
		if i < len(from) && i < len(to) {
			if i > 0 {
				out.WriteString(" ")
			}
			subLen := len(from) - i - 1
			if subLen > 0 {
				fromSub := from[i+1:]
				toSub := to[i+1:]
				subZeros := strings.Repeat("0", subLen)
				subNines := strings.Repeat("9", subLen)

				toReached := false
				out.WriteString("(")
				if fromSub == subZeros {
					digitRange(from[i], to[i]-1)
					out.WriteString(" ")
					moreDigits(subLen, subLen)
				} else {
					out.WriteString("[" + string(from[i]) + "] ")
					out.WriteString("(")
					uniformRange(fromSub, subNines)
					out.WriteString(")")
					if from[i] < to[i]-1 {
						out.WriteString(" | ")
						if toSub == subNines {
							digitRange(from[i]+1, to[i])
							toReached = true
						} else {
							digitRange(from[i]+1, to[i]-1)
						}
						out.WriteString(" ")
						moreDigits(subLen, subLen)
					}
				}
				if !toReached {
					out.WriteString(" | ")
					digitRange(to[i], to[i])
					out.WriteString(" ")
					uniformRange(subZeros, toSub)
				}
				out.WriteString(")")
			} else {
				out.WriteString("[" + string(from[i]) + "-" + string(to[i]) + "]")
			}
		}
	}

	const minInt64 = int64(-1) << 63
	negate := func(v int64) (int64, error) {
		if v == minInt64 {
			return 0, fmt.Errorf("integer bound %d is out of the supported range", v)
		}
		return -v, nil
	}

	if hasMin && hasMax {
		if minValue < 0 && maxValue < 0 {
			negMax, err := negate(maxValue)
			if err != nil {
				return err
			}
			negMin, err := negate(minValue)
			if err != nil {
				return err
			}
			out.WriteString(`"-" (`)
			if err := buildMinMaxInt(negMax, negMin, true, true, decimalsLeft, true, out); err != nil {
				return err
			}
			out.WriteString(")")
			return nil
		}

		if minValue < 0 {
			negMin, err := negate(minValue)
			if err != nil {
				return err
			}
			out.WriteString(`"-" (`)
			if err := buildMinMaxInt(0, negMin, true, true, decimalsLeft, true, out); err != nil {
				return err
			}
			out.WriteString(") | ")
			minValue = 0
		}

		minS := strconv.FormatInt(minValue, 10)
		maxS := strconv.FormatInt(maxValue, 10)
		for digits := len(minS); digits < len(maxS); digits++ {
			uniformRange(minS, strings.Repeat("9", digits))
			minS = "1" + strings.Repeat("0", digits)
			out.WriteString(" | ")
		}
		uniformRange(minS, maxS)
		return nil
	}

	lessDecimals := max(decimalsLeft-1, 1)

	if hasMin {
		if minValue < 0 {
			negMin, err := negate(minValue)
			if err != nil {
				return err
			}
			out.WriteString(`"-" (`)
			if err := buildMinMaxInt(0, negMin, false, true, decimalsLeft, false, out); err != nil {
				return err
			}
			out.WriteString(") | [0] | [1-9] ")
			moreDigits(0, decimalsLeft-1)
		} else if minValue == 0 {
			if topLevel {
				out.WriteString("[0] | [1-9] ")
				moreDigits(0, lessDecimals)
			} else {
				moreDigits(1, decimalsLeft)
			}
		} else if minValue <= 9 {
			c := byte('0' + minValue)
			rangeStart := byte('0')
			if topLevel {
				rangeStart = '1'
			}
			if c > rangeStart {
				digitRange(rangeStart, c-1)
				out.WriteString(" ")
				moreDigits(1, lessDecimals)
				out.WriteString(" | ")
			}
			digitRange(c, '9')
			out.WriteString(" ")
			moreDigits(0, lessDecimals)
		} else {
			minS := strconv.FormatInt(minValue, 10)
			length := len(minS)
			c := minS[0]

			if c > '1' {
				rangeStart := byte('0')
				if topLevel {
					rangeStart = '1'
				}
				digitRange(rangeStart, c-1)
				out.WriteString(" ")
				moreDigits(length, lessDecimals)
				out.WriteString(" | ")
			}
			digitRange(c, c)
			out.WriteString(" (")
			rest, err := strconv.ParseInt(minS[1:], 10, 64)
			if err != nil {
				return err
			}
			if err := buildMinMaxInt(rest, 0, true, false, lessDecimals, false, out); err != nil {
				return err
			}
			out.WriteString(")")
			if c < '9' {
				out.WriteString(" | ")
				digitRange(c+1, '9')
				out.WriteString(" ")
				moreDigits(length-1, lessDecimals)
			}
		}
		return nil
	}

	if hasMax {
		if maxValue >= 0 {
			if topLevel {
				out.WriteString(`"-" [1-9] `)
				moreDigits(0, lessDecimals)
				out.WriteString(" | ")
			}
			return buildMinMaxInt(0, maxValue, true, true, decimalsLeft, true, out)
		}
		negMax, err := negate(maxValue)
		if err != nil {
			return err
		}
		out.WriteString(`"-" (`)
		if err := buildMinMaxInt(negMax, 0, true, false, decimalsLeft, false, out); err != nil {
			return err
		}
		out.WriteString(")")
		return nil
	}

	return fmt.Errorf("at least one of minimum or maximum must be set")
}

type schemaConverter struct {
	rules             map[string]string
	refs              map[string]*jval
	refsBeingResolved map[string]bool
	errors            []string
	doc               *jval
}

// schemaGrammar compiles a JSON Schema object into a Grammar.
func schemaGrammar(schema []byte) (*Grammar, error) {
	doc, err := parseOrdered(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON Schema: %w", err)
	}
	c := &schemaConverter{
		rules:             map[string]string{"space": spaceRule},
		refs:              make(map[string]*jval),
		refsBeingResolved: make(map[string]bool),
		doc:               doc,
	}
	c.resolveRefs(doc)
	c.visit(doc, "")
	if len(c.errors) > 0 {
		return nil, fmt.Errorf("JSON schema conversion failed: %s", strings.Join(c.errors, "; "))
	}
	g, err := gbnfToGrammar(c.rules, "root")
	if err != nil {
		// A parse failure here means the converter emitted something the
		// GBNF subset cannot express; surface it rather than guessing.
		return nil, fmt.Errorf("JSON schema conversion failed: %w", err)
	}
	return g, nil
}

func (c *schemaConverter) errf(format string, args ...any) {
	c.errors = append(c.errors, fmt.Sprintf(format, args...))
}

func (c *schemaConverter) addRule(name, rule string) string {
	escName := invalidRuleChars.ReplaceAllString(name, "-")
	if existing, ok := c.rules[escName]; !ok || existing == rule {
		c.rules[escName] = rule
		return escName
	}
	i := 0
	for {
		key := escName + strconv.Itoa(i)
		if existing, ok := c.rules[key]; !ok || existing == rule {
			c.rules[key] = rule
			return key
		}
		i++
	}
}

func (c *schemaConverter) addPrimitive(name string, rule builtinRule) string {
	n := c.addRule(name, rule.content)
	for _, dep := range rule.deps {
		depRule, ok := primitiveRules[dep]
		if !ok {
			depRule, ok = stringFormatRules[dep]
		}
		if !ok {
			c.errf("rule %s not known", dep)
			continue
		}
		if _, defined := c.rules[dep]; !defined {
			c.addPrimitive(dep, depRule)
		}
	}
	return n
}

// resolveRefs walks the schema and resolves every internal "#/..." $ref to
// its target subschema. External refs are rejected: the llama-server path
// refuses to fetch remote schemas at runtime and so does this one.
func (c *schemaConverter) resolveRefs(node *jval) {
	if node == nil {
		return
	}
	switch node.kind {
	case jArr:
		for _, e := range node.arr {
			c.resolveRefs(e)
		}
	case jObj:
		if refVal := node.get("$ref"); refVal != nil && refVal.kind == jStr {
			ref := refVal.str
			if _, done := c.refs[ref]; done {
				return
			}
			if strings.HasPrefix(ref, "https://") || strings.HasPrefix(ref, "http://") {
				c.errf("unsupported external $ref: %s", ref)
				return
			}
			if !strings.HasPrefix(ref, "#/") {
				c.errf("unsupported $ref: %s", ref)
				return
			}
			target := c.doc
			tokens := strings.Split(ref[strings.Index(ref, "#")+1:], "/")
			for _, sel := range tokens[1:] {
				switch {
				case target != nil && target.kind == jObj && target.has(sel):
					target = target.get(sel)
				case target != nil && target.kind == jArr:
					idx, err := strconv.Atoi(sel)
					if err != nil || idx < 0 || idx >= len(target.arr) {
						c.errf("error resolving $ref %s: %s not found", ref, sel)
						return
					}
					target = target.arr[idx]
				default:
					c.errf("error resolving $ref %s: %s not found", ref, sel)
					return
				}
			}
			c.refs[ref] = target
			return
		}
		for _, kv := range node.obj {
			c.resolveRefs(kv.v)
		}
	}
}

func (c *schemaConverter) resolveRef(ref string) string {
	fragment := ref
	if i := strings.Index(ref, "#"); i >= 0 {
		fragment = ref[i+1:]
	}
	refName := "ref" + invalidRuleChars.ReplaceAllString(fragment, "-")
	if _, defined := c.rules[refName]; !defined && !c.refsBeingResolved[ref] {
		c.refsBeingResolved[ref] = true
		resolved, ok := c.refs[ref]
		if !ok {
			c.errf("unresolved $ref: %s", ref)
		} else {
			refName = c.visit(resolved, refName)
		}
		delete(c.refsBeingResolved, ref)
	}
	return refName
}

func (c *schemaConverter) generateConstantRule(v *jval) string {
	return formatLiteral(v.dump())
}

func (c *schemaConverter) generateUnionRule(name string, altSchemas []*jval) string {
	rules := make([]string, 0, len(altSchemas))
	for i, alt := range altSchemas {
		sep := "-"
		if name == "" {
			sep = "alternative-"
		}
		rules = append(rules, c.visit(alt, name+sep+strconv.Itoa(i)))
	}
	return strings.Join(rules, " | ")
}

// notStrings emits a rule matching any JSON string except the given ones,
// via the same character-trie construction as b10091 (including its quirk
// that proper prefixes of the given strings are unmatchable too).
func (c *schemaConverter) notStrings(strs []string) string {
	type trieNode struct {
		children map[byte]*trieNode
		end      bool
	}
	newNode := func() *trieNode { return &trieNode{children: make(map[byte]*trieNode)} }
	trie := newNode()
	for _, s := range strs {
		node := trie
		for i := 0; i < len(s); i++ {
			next, ok := node.children[s[i]]
			if !ok {
				next = newNode()
				node.children[s[i]] = next
			}
			node = next
		}
		node.end = true
	}

	charRule := c.addPrimitive("char", primitiveRules["char"])
	var out strings.Builder
	out.WriteString(`["] ( `)
	var visitNode func(node *trieNode)
	visitNode = func(node *trieNode) {
		var rejects []byte
		first := true
		for _, b := range slices.Sorted(maps.Keys(node.children)) {
			child := node.children[b]
			rejects = append(rejects, b)
			if first {
				first = false
			} else {
				out.WriteString(" | ")
			}
			out.WriteString("[" + string(b) + "]")
			if len(child.children) > 0 {
				out.WriteString(" (")
				visitNode(child)
				out.WriteString(")")
			} else if child.end {
				out.WriteString(" " + charRule + "+")
			}
		}
		if len(node.children) > 0 {
			if !first {
				out.WriteString(" | ")
			}
			out.WriteString(`[^"` + string(rejects) + `] ` + charRule + `*`)
		}
	}
	visitNode(trie)
	out.WriteString(" )")
	if !trie.end {
		out.WriteString("?")
	}
	out.WriteString(` ["] space`)
	return out.String()
}

func (c *schemaConverter) buildObjectRule(properties []jkv, required map[string]bool, name string, additionalProperties *jval) string {
	var requiredProps, optionalProps, propNames []string
	propKvRuleNames := make(map[string]string)
	for _, kv := range properties {
		propName, propSchema := kv.k, kv.v
		sep := "-"
		if name == "" {
			sep = ""
		}
		propRuleName := c.visit(propSchema, name+sep+propName)
		var lit strings.Builder
		dumpJSONString(&lit, propName)
		propKvRuleNames[propName] = c.addRule(
			name+sep+propName+"-kv",
			formatLiteral(lit.String())+` space ":" space `+propRuleName,
		)
		if required[propName] {
			requiredProps = append(requiredProps, propName)
		} else {
			optionalProps = append(optionalProps, propName)
		}
		propNames = append(propNames, propName)
	}
	if (additionalProperties != nil && additionalProperties.kind == jBool && additionalProperties.b) ||
		(additionalProperties != nil && additionalProperties.kind == jObj) {
		sep := "-"
		if name == "" {
			sep = ""
		}
		subName := name + sep + "additional"
		var valueRule string
		if additionalProperties.kind == jObj {
			valueRule = c.visit(additionalProperties, subName+"-value")
		} else {
			valueRule = c.addPrimitive("value", primitiveRules["value"])
		}
		var keyRule string
		if len(propNames) == 0 {
			keyRule = c.addPrimitive("string", primitiveRules["string"])
		} else {
			keyRule = c.addRule(subName+"-k", c.notStrings(propNames))
		}
		kvRule := c.addRule(subName+"-kv", keyRule+` ":" space `+valueRule)
		propKvRuleNames["*"] = kvRule
		optionalProps = append(optionalProps, "*")
	}

	rule := `"{" space `
	for i, propName := range requiredProps {
		if i > 0 {
			rule += ` "," space `
		}
		rule += propKvRuleNames[propName]
	}

	if len(optionalProps) > 0 {
		rule += " ("
		if len(requiredProps) > 0 {
			rule += ` "," space ( `
		}

		var getRecursiveRefs func(ks []string, firstIsOptional bool) string
		getRecursiveRefs = func(ks []string, firstIsOptional bool) string {
			if len(ks) == 0 {
				return ""
			}
			k := ks[0]
			kvRuleName := propKvRuleNames[k]
			commaRef := `( "," space ` + kvRuleName + ` )`
			var res string
			if firstIsOptional {
				if k == "*" {
					res = commaRef + "*"
				} else {
					res = commaRef + "?"
				}
			} else {
				res = kvRuleName
				if k == "*" {
					res += " " + commaRef + "*"
				}
			}
			if len(ks) > 1 {
				sep := "-"
				if name == "" {
					sep = ""
				}
				res += " " + c.addRule(name+sep+k+"-rest", getRecursiveRefs(ks[1:], true))
			}
			return res
		}

		for i := range optionalProps {
			if i > 0 {
				rule += " | "
			}
			rule += getRecursiveRefs(optionalProps[i:], false)
		}
		if len(requiredProps) > 0 {
			rule += " )"
		}
		rule += " )?"
	}

	rule += ` "}" space`
	return rule
}

// visit compiles one (sub)schema and returns the name of the rule that
// matches it. Branch order mirrors b10091's visit() exactly.
func (c *schemaConverter) visit(schema *jval, name string) string {
	schemaType := schema.get("type")
	typeIs := func(s string) bool {
		return schemaType != nil && schemaType.kind == jStr && schemaType.str == s
	}
	typeNull := schemaType == nil
	schemaFormat := ""
	if f := schema.get("format"); f != nil && f.kind == jStr {
		schemaFormat = f.str
	}
	ruleName := name
	switch {
	case reservedNames[name]:
		ruleName = name + "-"
	case name == "":
		ruleName = "root"
	}

	switch {
	case schema.has("$ref"):
		refVal := schema.get("$ref")
		if refVal.kind != jStr {
			c.errf("invalid $ref: %s", refVal.dump())
			return ruleName
		}
		return c.addRule(ruleName, c.resolveRef(refVal.str))

	case schema.has("oneOf") || schema.has("anyOf"):
		alts := schema.get("oneOf")
		if alts == nil {
			alts = schema.get("anyOf")
		}
		if alts.kind != jArr {
			c.errf("oneOf/anyOf must be an array: %s", alts.dump())
			return ruleName
		}
		return c.addRule(ruleName, c.generateUnionRule(name, alts.arr))

	case schemaType != nil && schemaType.kind == jArr:
		altSchemas := make([]*jval, 0, len(schemaType.arr))
		for _, t := range schemaType.arr {
			if t.kind != jStr {
				c.errf("invalid type entry: %s", t.dump())
				return ruleName
			}
			altSchemas = append(altSchemas, schema.copyReplaceType(t.str))
		}
		return c.addRule(ruleName, c.generateUnionRule(name, altSchemas))

	case schema.has("const"):
		return c.addRule(ruleName, c.generateConstantRule(schema.get("const"))+" space")

	case schema.has("enum"):
		enum := schema.get("enum")
		if enum.kind != jArr {
			c.errf("enum must be an array: %s", enum.dump())
			return ruleName
		}
		enumValues := make([]string, 0, len(enum.arr))
		for _, v := range enum.arr {
			enumValues = append(enumValues, c.generateConstantRule(v))
		}
		return c.addRule(ruleName, "("+strings.Join(enumValues, " | ")+") space")

	case (typeNull || typeIs("object")) &&
		(schema.has("properties") ||
			(schema.has("additionalProperties") && !isTrueVal(schema.get("additionalProperties")))):
		required := make(map[string]bool)
		if req := schema.get("required"); req != nil && req.kind == jArr {
			for _, item := range req.arr {
				if item.kind == jStr {
					required[item.str] = true
				}
			}
		}
		var properties []jkv
		if props := schema.get("properties"); props != nil && props.kind == jObj {
			properties = props.obj
		}
		return c.addRule(ruleName, c.buildObjectRule(properties, required, name, schema.get("additionalProperties")))

	case (typeNull || typeIs("object") || typeIs("string")) && schema.has("allOf"):
		allOf := schema.get("allOf")
		if allOf.kind != jArr {
			c.errf("allOf must be an array: %s", allOf.dump())
			return ruleName
		}
		required := make(map[string]bool)
		var properties []jkv
		enumValues := make(map[string]int)
		var addComponent func(comp *jval, isRequired bool)
		addComponent = func(comp *jval, isRequired bool) {
			switch {
			case comp.has("$ref"):
				if refVal := comp.get("$ref"); refVal.kind == jStr {
					addComponent(c.refs[refVal.str], isRequired)
				}
			case comp.has("properties"):
				if props := comp.get("properties"); props.kind == jObj {
					for _, kv := range props.obj {
						properties = append(properties, kv)
						if isRequired {
							required[kv.k] = true
						}
					}
				}
			case comp.has("enum"):
				if e := comp.get("enum"); e.kind == jArr {
					for _, v := range e.arr {
						enumValues[c.generateConstantRule(v)]++
					}
				}
			}
		}
		for _, t := range allOf.arr {
			if anyOf := t.get("anyOf"); anyOf != nil && anyOf.kind == jArr {
				for _, tt := range anyOf.arr {
					addComponent(tt, false)
				}
			} else {
				addComponent(t, true)
			}
		}
		if len(enumValues) > 0 {
			var enumIntersection []string
			for _, rule := range slices.Sorted(maps.Keys(enumValues)) {
				if enumValues[rule] == len(allOf.arr) {
					enumIntersection = append(enumIntersection, rule)
				}
			}
			if len(enumIntersection) > 0 {
				return c.addRule(ruleName, "("+strings.Join(enumIntersection, " | ")+") space")
			}
		}
		return c.addRule(ruleName, c.buildObjectRule(properties, required, name, nil))

	case (typeNull || typeIs("array")) && (schema.has("items") || schema.has("prefixItems")):
		items := schema.get("items")
		if items == nil {
			items = schema.get("prefixItems")
		}
		if items.kind == jArr {
			rule := `"[" space `
			for i, item := range items.arr {
				if i > 0 {
					rule += ` "," space `
				}
				sep := "-"
				if name == "" {
					sep = ""
				}
				rule += c.visit(item, name+sep+"tuple-"+strconv.Itoa(i))
			}
			rule += ` "]" space`
			return c.addRule(ruleName, rule)
		}
		sep := "-"
		if name == "" {
			sep = ""
		}
		itemRuleName := c.visit(items, name+sep+"item")
		minItems := 0
		if v := schema.get("minItems"); v != nil {
			if n, err := v.intValue(); err == nil {
				minItems = n
			}
		}
		maxItems := -1
		if v := schema.get("maxItems"); v != nil && v.kind == jNum && !strings.ContainsAny(v.num, ".eE") {
			if n, err := v.intValue(); err == nil {
				maxItems = n
			}
		}
		return c.addRule(ruleName, `"[" space `+buildRepetition(itemRuleName, minItems, maxItems, `"," space`)+` "]" space`)

	case (typeNull || typeIs("string")) && schema.has("pattern"):
		c.errf("pattern is not supported for structured output on this path")
		return ruleName

	case (typeNull || typeIs("string")) && uuidFormatRe.MatchString(schemaFormat):
		primName := schemaFormat
		if ruleName == "root" {
			primName = "root"
		}
		return c.addPrimitive(primName, primitiveRules["uuid"])

	case func() bool {
		_, ok := stringFormatRules[schemaFormat+"-string"]
		return (typeNull || typeIs("string")) && ok
	}():
		primName := schemaFormat + "-string"
		return c.addRule(ruleName, c.addPrimitive(primName, stringFormatRules[primName]))

	case typeIs("string") && (schema.has("minLength") || schema.has("maxLength")):
		charRule := c.addPrimitive("char", primitiveRules["char"])
		minLen := 0
		if v := schema.get("minLength"); v != nil {
			if n, err := v.intValue(); err == nil {
				minLen = n
			}
		}
		maxLen := -1
		if v := schema.get("maxLength"); v != nil {
			if n, err := v.intValue(); err == nil {
				maxLen = n
			}
		}
		return c.addRule(ruleName, `"\"" `+buildRepetition(charRule, minLen, maxLen, "")+` "\"" space`)

	case typeIs("integer") && (schema.has("minimum") || schema.has("exclusiveMinimum") || schema.has("maximum") || schema.has("exclusiveMaximum")):
		var minValue, maxValue int64
		var hasMin, hasMax bool
		var err error
		if v := schema.get("minimum"); v != nil {
			minValue, err = v.int64Value()
			hasMin = true
		} else if v := schema.get("exclusiveMinimum"); v != nil {
			minValue, err = v.int64Value()
			minValue++
			hasMin = true
		}
		if err != nil {
			c.errf("invalid integer minimum: %v", err)
			return ruleName
		}
		if v := schema.get("maximum"); v != nil {
			maxValue, err = v.int64Value()
			hasMax = true
		} else if v := schema.get("exclusiveMaximum"); v != nil {
			maxValue, err = v.int64Value()
			maxValue--
			hasMax = true
		}
		if err != nil {
			c.errf("invalid integer maximum: %v", err)
			return ruleName
		}
		var out strings.Builder
		out.WriteString("(")
		if err := buildMinMaxInt(minValue, maxValue, hasMin, hasMax, 16, true, &out); err != nil {
			c.errf("%v", err)
			return ruleName
		}
		out.WriteString(") space")
		return c.addRule(ruleName, out.String())

	case cppEmpty(schema) || typeIs("object"):
		return c.addRule(ruleName, c.addPrimitive("object", primitiveRules["object"]))

	default:
		if schemaType == nil || schemaType.kind != jStr {
			c.errf("unrecognized schema: %s", schema.dump())
			return ""
		}
		prim, ok := primitiveRules[schemaType.str]
		if !ok {
			c.errf("unrecognized schema: %s", schema.dump())
			return ""
		}
		primName := schemaType.str
		if ruleName == "root" {
			primName = "root"
		}
		return c.addPrimitive(primName, prim)
	}
}

func isTrueVal(v *jval) bool { return v != nil && v.kind == jBool && v.b }

// cppEmpty mirrors nlohmann json's empty(): true for null, empty
// containers, and the empty string.
func cppEmpty(v *jval) bool {
	if v == nil {
		return true
	}
	switch v.kind {
	case jNull:
		return true
	case jObj:
		return len(v.obj) == 0
	case jArr:
		return len(v.arr) == 0
	case jStr:
		return v.str == ""
	}
	return false
}
