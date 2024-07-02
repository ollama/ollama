package llm

import (
	"bufio"
	"fmt"
	"strings"

	"github.com/ollama/ollama/format"
)

var maxGrammarSize = 32 * format.KiloByte

// a cache that stores max 100 grammars
var grammarValidationCache = make(map[string]error)

func findIndexOfTextNotInQuotesOrCharacterSet(input string, text string) int {
	quoteBalance := 0
	bracketBalance := 0
	for i, c := range input {
		if c == '"' && (i == 0 || (i > 0 && input[i-1] != '\\')) {
			quoteBalance++
		} else if c == '[' && (i == 0 || (i > 0 && input[i-1] != '\\')) {
			bracketBalance++
		} else if c == ']' && (i == 0 || (i > 0 && input[i-1] != '\\')) {
			bracketBalance--
		} else if quoteBalance%2 == 0 && bracketBalance == 0 && strings.HasPrefix(input[i:], text) {
			return i
		}
	}
	return -1
}

func removeComments(input string) string {
	var output strings.Builder
	scanner := bufio.NewScanner(strings.NewReader(input))

	for scanner.Scan() {
		line := scanner.Text()

		// remove comment in a line by finding the first hash that is not inside a quoted string
		indexFirstCommentHash := findIndexOfTextNotInQuotesOrCharacterSet(line, "#")

		// if there is a comment hash, remove everything after it
		if indexFirstCommentHash != -1 {
			line = line[:indexFirstCommentHash]
		}

		// Trim any trailing spaces from the cleaned line
		line = strings.TrimSpace(line)

		// Append the clean line to the output if it's not empty
		if line != "" {
			output.WriteString(line)
			output.WriteString("\n")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading input:", err)
	}

	return strings.TrimSpace(output.String())
}

func breakIntoArrayOfRules(input string) (map[string]string, error) {
	var rules []string
	scanner := bufio.NewScanner(strings.NewReader(input))

	var currentRule strings.Builder
	for scanner.Scan() {
		line := scanner.Text()

		indexFirstSeperator := findIndexOfTextNotInQuotesOrCharacterSet(line, "::=")
		// if the line matches the pattern, we start a new rule
		if indexFirstSeperator != -1 {
			if currentRule.Len() > 0 {
				rules = append(rules, currentRule.String())
				currentRule.Reset()
			}
		}

		// append the line to the current rule
		currentRule.WriteString(line)
		currentRule.WriteString("\n")
	}

	if currentRule.Len() > 0 {
		rules = append(rules, currentRule.String())
	}

	// remove all new lines from all rules
	for i, rule := range rules {
		rules[i] = strings.ReplaceAll(rule, "\n", " ")
	}

	// put rules into a map
	rulesMap := make(map[string]string)

	for _, rule := range rules {
		// split the rule into the key and value
		parts := strings.Split(rule, "::=")
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid rule did not contain exactly one ::= separator: %s", rule)
		}
		key := strings.TrimSpace(parts[0])
		if _, err := isValidRuleName(key); err != nil {
			return nil, fmt.Errorf("invalid rule name: %s", key)
		}

		// gather the rest of the rule after the key there may be more than 1 ::= separator (weird cases like a string with "::=" in it)
		valueBuilder := strings.Builder{}
		for i := 1; i < len(parts); i++ {
			valueBuilder.WriteString(parts[i])
		}
		value := strings.TrimSpace(valueBuilder.String())

		// add the key and value to the map
		rulesMap[key] = value
	}

	return rulesMap, nil
}

type TokenType int

const (
	TokenUnknown TokenType = iota
	TokenSpace
	TokenPipe
	TokenLParen
	TokenRParen
	TokenLBracket
	TokenRBracket
	TokenAsterisk
	TokenPlus
	TokenQuestion
	TokenTerminal
	TokenCharacterClass
	TokenNonTerminal
)

type Token struct {
	Type  TokenType
	Value string
}

func isDigit(c rune) bool {
	return c >= '0' && c <= '9'
}

func isLetter(c rune) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
}

func isHexDigit(c rune) bool {
	return isDigit(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')
}

func isValidIdCharacter(c rune) bool {
	return c == '-' || c == '_' || isLetter(c) || (c >= '0' && c <= '9')
}

func isValidRuleName(name string) (bool, error) {
	if len(name) == 0 {
		return false, fmt.Errorf("empty rule name")
	}

	// can have A-Z, a-z, 0-9, -, _ and cannot start with number
	for i, c := range name {
		if i == 0 {
			if !isLetter(c) {
				return false, fmt.Errorf("rule name cannot start with a number '%s'", name)
			}
		}
		if !isValidIdCharacter(c) {
			return false, fmt.Errorf("invalid character '%c' in rule name '%s'", c, name)
		}
	}

	return true, nil
}

func validateCharacterClass(charClass string) error {
	// Check if the character class starts with '[' and ends with ']'
	if len(charClass) < 2 || charClass[0] != '[' || charClass[len(charClass)-1] != ']' {
		return fmt.Errorf("character class must start with '[' and end with ']'")
	}

	// Check for negation
	i := 1
	if charClass[1] == '^' {
		i++
	}

	classCharacters := charClass[i : len(charClass)-1]

	validEscapeCharacters := "\\ntrxu"

	i = 0
	for i < len(classCharacters) {
		if classCharacters[i] == '\\' {
			// Check if the escape character is valid
			if i+1 >= len(classCharacters) {
				return fmt.Errorf("incomplete escape character")
			}
			escapeCharacter := classCharacters[i+1]
			if !strings.Contains(validEscapeCharacters, string(escapeCharacter)) {
				return fmt.Errorf("invalid escape character")
			}
			// if the escape character is 'x' or 'u', check if the following characters are hexadecimal digits
			if escapeCharacter == 'x' {
				// grab the following two characters
				if i+3 >= len(classCharacters) {
					return fmt.Errorf("incomplete hexadecimal escape character")
				}

				nextTwoCharacters := classCharacters[i+2 : i+4]

				if !isHexDigit(rune(nextTwoCharacters[0])) || !isHexDigit(rune(nextTwoCharacters[1])) {
					return fmt.Errorf("invalid hexadecimal escape character")
				}

				i += 4
			} else if escapeCharacter == 'u' {
				// grab the following four characters
				if i+5 >= len(classCharacters) {
					return fmt.Errorf("incomplete unicode escape character")
				}

				nextFourCharacters := classCharacters[i+2 : i+6]

				if !isHexDigit(rune(nextFourCharacters[0])) || !isHexDigit(rune(nextFourCharacters[1])) || !isHexDigit(rune(nextFourCharacters[2])) || !isHexDigit(rune(nextFourCharacters[3])) {
					return fmt.Errorf("invalid unicode escape character")
				}

				i += 6
			} else {
				i += 2
			}
		} else {
			if classCharacters[i] == '^' {
				return fmt.Errorf("^ should only be at the beginning of a character class")
			}
			i++
		}
	}

	return nil
}

func validateStringLiteral(strLiteral string) error {
	validEscapeCharacters := "\\\"ntrxu"

	// make sure the string literal starts and ends with a quote
	if len(strLiteral) < 2 || strLiteral[0] != '"' || strLiteral[len(strLiteral)-1] != '"' {
		return fmt.Errorf("string literal must start and end with a quote")
	}

	i := 0
	for i < len(strLiteral) {
		if strLiteral[i] == '\\' {
			// Check if the escape character is valid
			if i+1 >= len(strLiteral) {
				return fmt.Errorf("incomplete escape character")
			}
			escapeCharacter := strLiteral[i+1]
			if !strings.Contains(validEscapeCharacters, string(escapeCharacter)) {
				return fmt.Errorf("invalid escape character")
			}
			// if the escape character is 'x' or 'u', check if the following characters are hexadecimal digits
			if escapeCharacter == 'x' {
				// grab the following two characters
				if i+3 >= len(strLiteral) {
					return fmt.Errorf("incomplete hexadecimal escape character")
				}

				nextTwoCharacters := strLiteral[i+2 : i+4]

				if !isHexDigit(rune(nextTwoCharacters[0])) || !isHexDigit(rune(nextTwoCharacters[1])) {
					return fmt.Errorf("invalid hexadecimal escape character")
				}

				i += 4
			} else if escapeCharacter == 'u' {
				// grab the following four characters
				if i+5 >= len(strLiteral) {
					return fmt.Errorf("incomplete unicode escape character")
				}

				nextFourCharacters := strLiteral[i+2 : i+6]

				if !isHexDigit(rune(nextFourCharacters[0])) || !isHexDigit(rune(nextFourCharacters[1])) || !isHexDigit(rune(nextFourCharacters[2])) || !isHexDigit(rune(nextFourCharacters[3])) {
					return fmt.Errorf("invalid unicode escape character")
				}

				i += 6
			} else {
				i += 2
			}
		} else {
			i++
		}
	}

	return nil
}

func parseRule(rule string) ([]Token, error) {
	var tokens []Token
	rule = strings.TrimSpace(rule)
	i := 0
	n := len(rule)

	for i < n {
		switch {
		case rule[i] == ' ':
			// Skip spaces
			i++
		case rule[i] == '|':
			tokens = append(tokens, Token{Type: TokenPipe, Value: string(rule[i])})
			i++
		case rule[i] == '(':
			tokens = append(tokens, Token{Type: TokenLParen, Value: string(rule[i])})
			i++
		case rule[i] == ')':
			tokens = append(tokens, Token{Type: TokenRParen, Value: string(rule[i])})
			i++
		case rule[i] == '[':
			// Character class
			start := i
			i++
			for i < n {
				if rule[i] == ']' && rule[i-1] != '\\' {
					i++
					break
				}
				i++
			}
			if i <= n {
				charClass := rule[start:i]

				if err := validateCharacterClass(charClass); err == nil {
					tokens = append(tokens, Token{Type: TokenCharacterClass, Value: rule[start:i]})
				} else {
					return nil, fmt.Errorf("invalid character class: %s", err)
				}
			} else {
				return nil, fmt.Errorf("unclosed character class")
			}
		case rule[i] == ']':
			tokens = append(tokens, Token{Type: TokenRBracket, Value: string(rule[i])})
			i++
		case rule[i] == '*':
			tokens = append(tokens, Token{Type: TokenAsterisk, Value: string(rule[i])})
			i++
		case rule[i] == '+':
			tokens = append(tokens, Token{Type: TokenPlus, Value: string(rule[i])})
			i++
		case rule[i] == '?':
			tokens = append(tokens, Token{Type: TokenQuestion, Value: string(rule[i])})
			i++
		case rule[i] == '"':
			// Terminal sequence with escaped quotes handling
			start := i
			i++
			for i < n {
				if rule[i] == '\\' && i+1 < n {
					i += 2
				} else if rule[i] == '"' {
					i++
					break
				} else {
					i++
				}
			}

			strValue := rule[start:i]

			err := validateStringLiteral(strValue)

			if err == nil {
				tokens = append(tokens, Token{Type: TokenTerminal, Value: strValue})
			} else {
				return nil, fmt.Errorf("invalid string literal: %s", err)
			}
		default:
			// Non-terminal or literal character
			start := i
			for i < n && !strings.ContainsRune(" |()[]*+?", rune(rule[i])) {
				i++
			}
			// Check if it is a valid rule name or character class
			if isValid, err := isValidRuleName(rule[start:i]); !isValid {
				return nil, err
			}
			strValue := rule[start:i]

			tokens = append(tokens, Token{Type: TokenNonTerminal, Value: strValue})
		}
	}

	return tokens, nil
}

func validateRule(tokens []Token, definedRules map[string]bool) error {
	if len(tokens) == 0 {
		return fmt.Errorf("empty rule")
	}

	// Stack to track the balance of parentheses and brackets
	var stack []TokenType

	for i, token := range tokens {
		switch token.Type {
		case TokenLParen:
			stack = append(stack, TokenLParen)
		case TokenRParen:
			if len(stack) == 0 || stack[len(stack)-1] != TokenLParen {
				return fmt.Errorf("unmatched closing parenthesis")
			}
			stack = stack[:len(stack)-1]
		case TokenLBracket:
			stack = append(stack, TokenLBracket)
		case TokenRBracket:
			if len(stack) == 0 || stack[len(stack)-1] != TokenLBracket {
				return fmt.Errorf("unmatched closing bracket")
			}
			stack = stack[:len(stack)-1]
		case TokenAsterisk, TokenPlus, TokenQuestion:
			// These tokens should follow either a terminal, non-terminal, or character class
			if i == 0 {
				return fmt.Errorf("unexpected %s at the beginning of the rule", token.Value)
			}
			prevToken := tokens[i-1]
			if prevToken.Type != TokenTerminal && prevToken.Type != TokenNonTerminal && prevToken.Type != TokenCharacterClass && prevToken.Type != TokenRParen && prevToken.Type != TokenRBracket {
				return fmt.Errorf("unexpected %s following %s", token.Value, prevToken.Value)
			}
		case TokenPipe:
			// Pipe must be within a sequence, not at the beginning or end
			if i == 0 || i == len(tokens)-1 {
				return fmt.Errorf("unexpected pipe '|' at the beginning or end of the rule")
			}
		case TokenNonTerminal:
			// Check if the non-terminal is defined
			if !definedRules[token.Value] {
				return fmt.Errorf("undefined rule: %s", token.Value)
			}
		}
	}

	if len(stack) > 0 {
		return fmt.Errorf("unclosed parentheses or brackets")
	}

	return nil
}

func parseGrammar(grammar string) (map[string]([]Token), error) {
	cleanedInput := removeComments(grammar)
	rules, err := breakIntoArrayOfRules(cleanedInput)
	if err != nil {
		return nil, err
	}

	ruleTokens := make(map[string]([]Token))

	for key, value := range rules {
		tokens, err := parseRule(value)
		if err != nil {
			return nil, fmt.Errorf("error parsing rule \"%s\": %v", key, err)
		}
		ruleTokens[key] = tokens
	}

	return ruleTokens, nil
}

func addToCache(grammar string, err error) {
	if len(grammarValidationCache) >= 100 {
		// remove the first element
		for key := range grammarValidationCache {
			delete(grammarValidationCache, key)
			break
		}
	}
	grammarValidationCache[grammar] = err
}

func ValidateGrammar(grammar string) error {
	// check to see if we've cached this before and if so return it
	if err, ok := grammarValidationCache[grammar]; ok {
		return err
	}

	if len(grammar) > maxGrammarSize {
		err := fmt.Errorf("grammar size exceeds maximum size of %d bytes", maxGrammarSize)
		addToCache(grammar, err)
		return err
	}

	// Since GBNF is essentially just a list of rules, we can validate the grammar by
	// removing all comments, removing all non-essential white space
	// and then breaking the input into an array of rules
	// and validating each rule one by one. This will hopefully
	// keep the function simple and easy to understand
	// while still being able to validate the grammar
	// with some help to tell where something is wrong

	ruleTokens, err := parseGrammar(grammar)
	if err != nil {
		addToCache(grammar, err)
		return err
	}

	definedRules := make(map[string]bool)
	for key := range ruleTokens {
		definedRules[key] = true
	}

	// check that it has root rule
	if _, ok := definedRules["root"]; !ok {
		err := fmt.Errorf("no root rule defined")
		addToCache(grammar, err)
		return err
	}

	for key, value := range ruleTokens {
		if err := validateRule(value, definedRules); err != nil {
			err = fmt.Errorf("error in rule \"%s\": %v", key, err)
			addToCache(grammar, err)
			return err
		}
	}

	addToCache(grammar, nil)
	return nil
}
