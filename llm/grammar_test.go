package llm

import (
	"testing"
)

/* Basic parsing */

func TestIsDigit(t *testing.T) {
	if !isDigit('0') {
		t.Errorf("Expected '0' to be a digit")
	}
	if !isDigit('9') {
		t.Errorf("Expected '9' to be a digit")
	}
	if isDigit('a') {
		t.Errorf("Expected 'a' to not be a digit")
	}
	if isDigit('Z') {
		t.Errorf("Expected 'Z' to not be a digit")
	}
}

func TestIsValidIdCharacter(t *testing.T) {
	if !isValidIdCharacter('0') {
		t.Errorf("Expected '0' to be a valid character")
	}
	if !isValidIdCharacter('9') {
		t.Errorf("Expected '9' to be a valid character")
	}
	if !isValidIdCharacter('a') {
		t.Errorf("Expected 'a' to be a valid character")
	}
	if !isValidIdCharacter('Z') {
		t.Errorf("Expected 'Z' to be a valid character")
	}
}

func TestIsValidRuleName(t *testing.T) {
	valid, err := isValidRuleName("foo")
	if !valid {
		t.Errorf("Expected 'foo' to be a valid rule name")
	}
	if err != nil {
		t.Errorf("Expected 'foo' to be a valid rule name, got error: %v", err)
	}
}

func TestIsValidRuleNameEmpty(t *testing.T) {
	valid, err := isValidRuleName("")
	if valid {
		t.Errorf("Expected '' to be an invalid rule name")
	}
	if err == nil {
		t.Errorf("Expected '' to be an invalid rule name, got nil error")
	}
}

func TestIsValidRuleNameStartsWithNumber(t *testing.T) {
	valid, err := isValidRuleName("1foo")
	if valid {
		t.Errorf("Expected '1foo' to be an invalid rule name")
	}
	if err == nil {
		t.Errorf("Expected '1foo' to be an invalid rule name, got nil error")
	}
}

func TestRuleParseNonTerminal(t *testing.T) {
	rule, err := parseRule("expr")
	if err != nil {
		t.Errorf("Error lexing rule: %v", err)
		return
	}
	if len(rule) != 1 {
		t.Errorf("Expected 1 token, got %d", len(rule))
		return
	}
	if rule[0].Type != TokenNonTerminal {
		t.Errorf("Expected non-terminal token, got %v", rule[0].Type)
		return
	}
	if rule[0].Value != "expr" {
		t.Errorf("Expected value 'expr', got %v", rule[0].Value)
		return
	}
}

func TestRuleParseTerminalRule(t *testing.T) {
	rule, err := parseRule(`"foo"`)
	if err != nil {
		t.Errorf("Error lexing rule: %v", err)
		return
	}
	if len(rule) != 1 {
		t.Errorf("Expected 1 token, got %d", len(rule))
		return
	}
	if rule[0].Type != TokenTerminal {
		t.Errorf("Expected terminal token, got %v", rule[0].Type)
		return
	}
	if rule[0].Value != `"foo"` {
		t.Errorf("Expected value '\"foo\"', got %v", rule[0].Value)
		return
	}
}

func TestStringInvalidEscape(t *testing.T) {
	input := `"ab\bc"`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

func TestRuleParseTerminalRuleWithSpaces(t *testing.T) {
	rule, err := parseRule(`"foo bar"`)
	if err != nil {
		t.Errorf("Error lexing rule: %v", err)
		return
	}
	if len(rule) != 1 {
		t.Errorf("Expected 1 token, got %d", len(rule))
		return
	}
	if rule[0].Type != TokenTerminal {
		t.Errorf("Expected terminal token, got %v", rule[0].Type)
		return
	}
	if rule[0].Value != `"foo bar"` {
		t.Errorf("Expected value '\"foo bar\"', got %v", rule[0].Value)
		return
	}
}

func TestRuleParseTerminalRuleWithEscapedQuotes(t *testing.T) {
	rule, err := parseRule(`"foo \"bar\" baz"`)
	if err != nil {
		t.Errorf("Error lexing rule: %v", err)
		return
	}
	if len(rule) != 1 {
		t.Errorf("Expected 1 token, got %d", len(rule))
		return
	}
	if rule[0].Type != TokenTerminal {
		t.Errorf("Expected terminal token, got %v", rule[0].Type)
		return
	}
	if rule[0].Value != `"foo \"bar\" baz"` {
		t.Errorf("Expected value '\"foo \"bar\" baz\"', got %v", rule[0].Value)
		return
	}
}

func TestRuleParseTerminalRuleWithEscapedQuotesAndSpaces(t *testing.T) {
	_, err := parseRule(`foo b"r baz"`)
	if err == nil {
		t.Errorf("Expected error lexing rule, got nil")
		return
	}
}

func TestRuleParseComplexRule(t *testing.T) {
	input := "(pawn | nonpawn | castle) [+#]?"
	rule, err := parseRule(input)
	if err != nil {
		t.Errorf("Error lexing rule: %v", err)
		return
	}
	if len(rule) != 9 {
		t.Errorf("Expected 9 tokens, got %d", len(rule))
		return
	}
	if rule[0].Type != TokenLParen {
		t.Errorf("Expected LParen token, got %v", rule[0].Type)
		return
	}
	if rule[1].Type != TokenNonTerminal {
		t.Errorf("Expected non-terminal token, got %v", rule[1].Type)
		return
	}
	if rule[1].Value != "pawn" {
		t.Errorf("Expected value 'pawn', got %v", rule[1].Value)
		return
	}
	if rule[2].Type != TokenPipe {
		t.Errorf("Expected pipe token, got %v", rule[2].Type)
		return
	}
	if rule[3].Type != TokenNonTerminal {
		t.Errorf("Expected non-terminal token, got %v", rule[3].Type)
		return
	}
	if rule[3].Value != "nonpawn" {
		t.Errorf("Expected value 'nonpawn', got %v", rule[3].Value)
		return
	}
	if rule[4].Type != TokenPipe {
		t.Errorf("Expected pipe token, got %v", rule[4].Type)
		return
	}
	if rule[5].Type != TokenNonTerminal {
		t.Errorf("Expected non-terminal token, got %v", rule[5].Type)
		return
	}
	if rule[5].Value != "castle" {
		t.Errorf("Expected value 'castle', got %v", rule[5].Value)
		return
	}
	if rule[6].Type != TokenRParen {
		t.Errorf("Expected RParen token, got %v", rule[6].Type)
		return
	}
	if rule[7].Type != TokenCharacterClass {
		t.Errorf("Expected character class token, got %v", rule[7].Type)
		return
	}
	if rule[7].Value != "[+#]" {
		t.Errorf("Expected value '[+#]', got %v", rule[7].Value)
		return
	}
	if rule[8].Type != TokenQuestion {
		t.Errorf("Expected question token, got %v", rule[8].Type)
		return
	}
}

func TestNonExistantRule(t *testing.T) {
	input := `root ::= missing`
	err := ValidateGrammar(input)
	if err == nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

/* Weird cases */

func TestNoRoot(t *testing.T) {
	// this is a common typo
	input := `llama ::= "yes" | "no"`
	err := ValidateGrammar(input)
	if err == nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

func TestInvalidRoot(t *testing.T) {
	// this is a common typo
	input := `root ::= "yes`
	err := ValidateGrammar(input)
	if err == nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

func TestBadLlama(t *testing.T) {
	// this is a common typo
	input := `root :== "yes" | "no"`
	err := ValidateGrammar(input)
	if err == nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

func TestLlama(t *testing.T) {
	input := `root ::= "yes" | "no"`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestBadRuleName(t *testing.T) {
	input := `not good ::= "yes" | "no"`
	err := ValidateGrammar(input)
	if err == nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

func TestRuleOfRule(t *testing.T) {
	input := `root ::= "::="`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Expected error validating grammar, got nil")
	}
}

func TestLlamaRuleSeperatorAlone(t *testing.T) {
	input := `root ::=
"::="`
	lines, err := breakIntoArrayOfRules(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
	if len(lines) != 1 {
		t.Errorf("Expected 1 rule, got %d", len(lines))
	}
}

func TestNonCommentHash(t *testing.T) {
	// this is a common typo
	input := `root ::= "y#s" # this is the # real comment`
	lines := removeComments(input)
	if lines != `root ::= "y#s"` {
		t.Errorf("Expected 'root ::= \"y#s\"', got %v", lines)
	}
}

func TestNonCommentHash2(t *testing.T) {
	// this is a common typo
	input := `root ::= [#] # this is the # real comment`
	lines := removeComments(input)
	if lines != `root ::= [#]` {
		t.Errorf("Expected 'root ::= [#]', got %v", lines)
	}
}

/* Regex */
func TestRegexParserBadFollower(t *testing.T) {
	input := `[^\r\n\x0b\x0c\x85\u2028\u2029]-`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

func TestRegexParserBadFollower2(t *testing.T) {
	input := `[abc]&`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

func TestRegexParserBadHex(t *testing.T) {
	input := `[\x0q]`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

func TestRegexParserBadUnicode(t *testing.T) {
	input := `[\u20q8]`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

func TestInverseOnlyAtBeginning(t *testing.T) {
	input := `[abc^123]`
	_, err := parseRule(input)
	if err == nil {
		t.Errorf("Parser should have failed: %v", err)
	}
}

/* Full grammars */

func TestChessGrammar(t *testing.T) {
	input := `
		# Specifies chess moves as a list in algebraic notation, using PGN conventions

		# Force first move to "1. ", then any 1-2 digit number after, relying on model to follow the pattern
		root    ::= "1. " move " " move "\n" ([1-9] [0-9]? ". " move " " move "\n")+
		move    ::= (pawn | nonpawn | castle) [+#]?

		# piece type, optional file/rank, optional capture, dest file & rank
		nonpawn ::= [NBKQR] [a-h]? [1-8]? "x"? [a-h] [1-8]

		# optional file & capture, dest file & rank, optional promotion
		pawn    ::= ([a-h] "x")? [a-h] [1-8] ("=" [NBKQR])?

		castle  ::= "O-O" "-O"?`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestInternationalGrammar(t *testing.T) {
	input := `# A probably incorrect grammar for Japanese
	root        ::= jp-char+ ([ \t\n] jp-char+)*
	jp-char     ::= hiragana | katakana | punctuation | cjk
	hiragana    ::= [ぁ-ゟ]
	katakana    ::= [ァ-ヿ]
	punctuation ::= [、-〾]
	cjk         ::= [一-鿿]
	`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestArithmaticGrammar(t *testing.T) {
	input := `root  ::= (expr "=" ws term "\n")+
	expr  ::= term ([-+*/] term)*
	term  ::= ident | num | "(" ws expr ")" ws
	ident ::= [a-z] [a-z0-9_]* ws
	num   ::= [0-9]+ ws
	ws    ::= [ \t\n]*
	`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestCGrammar(t *testing.T) {
	input := `root ::= (declaration)*

	declaration ::= dataType identifier "(" parameter? ")" "{" statement* "}"
	
	dataType  ::= "int" ws | "float" ws | "char" ws
	identifier ::= [a-zA-Z_] [a-zA-Z_0-9]*
	
	parameter ::= dataType identifier
	
	statement ::=
		( dataType identifier ws "=" ws expression ";" ) |
		( identifier ws "=" ws expression ";" ) |
		( identifier ws "(" argList? ")" ";" ) |
		( "return" ws expression ";" ) |
		( "while" "(" condition ")" "{" statement* "}" ) |
		( "for" "(" forInit ";" ws condition ";" ws forUpdate ")" "{" statement* "}" ) |
		( "if" "(" condition ")" "{" statement* "}" ("else" "{" statement* "}")? ) |
		( singleLineComment ) |
		( multiLineComment )
	
	forInit ::= dataType identifier ws "=" ws expression | identifier ws "=" ws expression
	forUpdate ::= identifier ws "=" ws expression
	
	condition ::= expression relationOperator expression
	relationOperator ::= ("<=" | "<" | "==" | "!=" | ">=" | ">")
	
	expression ::= term (("+" | "-") term)*
	term ::= factor(("*" | "/") factor)*
	
	factor ::= identifier | number | unaryTerm | funcCall | parenExpression
	unaryTerm ::= "-" factor
	funcCall ::= identifier "(" argList? ")"
	parenExpression ::= "(" ws expression ws ")"
	
	argList ::= expression ("," ws expression)*
	
	number ::= [0-9]+
	
	singleLineComment ::= "//" [^\n]* "\n"
	multiLineComment ::= "/*" ( [^*] | ("*" [^/]) )* "*/"
	
	ws ::= ([ \t\n]+)`
	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestJsonGrammar(t *testing.T) {
	input := `root   ::= object
	value  ::= object | array | string | number | ("true" | "false" | "null") ws
	
	object ::=
	  "{" ws (
				string ":" ws value
		("," ws string ":" ws value)*
	  )? "}" ws
	
	array  ::=
	  "[" ws (
				value
		("," ws value)*
	  )? "]" ws
	
	string ::=
	  "\"" (
		[^"\\\x7F\x00-\x1F] |
		"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
	  )* "\"" ws
	
	number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
	
	# Optional space: by convention, applied in this grammar after literal chars when allowed
	ws ::= ([ \t\n] ws)?`

	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestJsonArrayGrammar(t *testing.T) {
	input := `# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
	# Useful for generating JSON arrays
	
	root   ::= arr
	value  ::= object | array | string | number | ("true" | "false" | "null") ws
	
	arr  ::=
	  "[\n" ws (
				value
		(",\n" ws value)*
	  )? "]"
	
	object ::=
	  "{" ws (
				string ":" ws value
		("," ws string ":" ws value)*
	  )? "}" ws
	
	array  ::=
	  "[" ws (
				value
		("," ws value)*
	  )? "]" ws
	
	string ::=
	  "\"" (
		[^"\\\x7F\x00-\x1F] |
		"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
	  )* "\"" ws
	
	number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
	
	# Optional space: by convention, applied in this grammar after literal chars when allowed
	ws ::= ([ \t\n] ws)?`

	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}

func TestJsonList(t *testing.T) {
	input := `root ::= item+

	# Excludes various line break characters
	item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"`

	err := ValidateGrammar(input)
	if err != nil {
		t.Errorf("Error validating grammar: %v", err)
	}
}
