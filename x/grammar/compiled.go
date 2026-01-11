//go:build mlx

package grammar

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Grammar is the compiled form of an EBNF grammar.
// It contains terminals, parse tables, and the start state.
// Use ParseEBNF or JSONGrammar to create a Grammar.
type Grammar struct {
	// The underlying pda
	pda *pda

	// Compiled terminal matcher
	matcher *terminalMatcher
}

// ParseEBNF compiles an EBNF grammar string into a Grammar.
// startRule is the name of the start rule (e.g., "root", "json").
func ParseEBNF(ebnf string, startRule string) (*Grammar, error) {
	pda, err := compileString(ebnf, startRule)
	if err != nil {
		return nil, fmt.Errorf("failed to compile EBNF: %w", err)
	}

	matcher, err := compileTerminalsStrict(pda)
	if err != nil {
		return nil, fmt.Errorf("failed to compile terminals: %w", err)
	}

	return &Grammar{
		pda:     pda,
		matcher: matcher,
	}, nil
}

// JSONGrammar returns the compiled JSON grammar.
// This is a convenience wrapper for ParseEBNF(JSONGrammarEBNF, "json").
func JSONGrammar() (*Grammar, error) {
	return ParseEBNF(JSONGrammarEBNF, "json")
}

// JSONObjectGrammar returns a JSON grammar that only allows objects at the top level.
// Use this when you want to ensure the output is a JSON object (starts with {).
func JSONObjectGrammar() (*Grammar, error) {
	return ParseEBNF(JSONObjectGrammarEBNF, "json")
}

// compileTerminalsStrict builds a matcher that properly handles:
// - Escaped literals ("\n", \"", \uXXXX)
// - Unicode ranges (rune-based, not byte-based)
// - Rejects unsupported patterns with an error (no silent fallback)
func compileTerminalsStrict(pda *pda) (*terminalMatcher, error) {
	m := &terminalMatcher{
		literalTrie: &trieNode{terminalID: -1},
		ranges:      make([]terminal, 0),
		terminals:   make([]terminal, 0, len(pda.Terminals)),
		patternToID: make(map[string]int),
	}

	// Track which pattern produced each unescaped value for collision detection
	unescapedSource := make(map[string]string) // unescaped -> original pattern

	for i, pattern := range pda.Terminals {
		terminal, err := parseTerminalPattern(pattern, i)
		if err != nil {
			return nil, fmt.Errorf("terminal %q: %w", pattern, err)
		}

		if terminal.Type == terminalLiteral {
			// Use the unescaped pattern for trie matching
			m.addLiteralToTrie(terminal.Unescaped, i)

			// Detect collisions between literals that unescape to the same value
			if existingPattern, exists := unescapedSource[terminal.Unescaped]; exists {
				if existingPattern != pattern {
					return nil, fmt.Errorf("collision: patterns %q and %q both unescape to %q",
						existingPattern, pattern, terminal.Unescaped)
				}
			} else {
				unescapedSource[terminal.Unescaped] = pattern
			}
		} else if terminal.Type == terminalRange {
			m.ranges = append(m.ranges, terminal)
		}

		m.terminals = append(m.terminals, terminal)
		m.patternToID[pattern] = i
	}

	return m, nil
}

// parseTerminalPattern parses a terminal pattern and returns a terminal.
// Supports:
// - Literal strings (with escape sequences)
// - Character ranges [X-Y] (unicode-aware)
func parseTerminalPattern(pattern string, id int) (terminal, error) {
	if len(pattern) == 0 {
		return terminal{}, fmt.Errorf("empty pattern")
	}

	// Check for range pattern: [X-Y]
	if isUnicodeRangePattern(pattern) {
		lowRune, highRune, err := parseUnicodeRange(pattern)
		if err != nil {
			return terminal{}, err
		}
		return terminal{
			ID:        id,
			Type:      terminalRange,
			Pattern:   pattern,
			Unescaped: pattern,
			LowRune:   lowRune,
			HighRune:  highRune,
		}, nil
	}

	// It's a literal - unescape it
	unescaped, err := unescapeLiteral(pattern)
	if err != nil {
		return terminal{}, fmt.Errorf("invalid escape sequence: %w", err)
	}

	return terminal{
		ID:        id,
		Type:      terminalLiteral,
		Pattern:   pattern,
		Unescaped: unescaped,
	}, nil
}

// isUnicodeRangePattern checks if pattern is a character range like [a-z] or [\u0000-\uFFFF]
func isUnicodeRangePattern(pattern string) bool {
	if len(pattern) < 5 || pattern[0] != '[' || pattern[len(pattern)-1] != ']' {
		return false
	}
	// Find the dash that separates low-high
	inner := pattern[1 : len(pattern)-1]
	dashIdx := strings.Index(inner, "-")
	// Handle escaped dash at start
	if dashIdx <= 0 {
		return false
	}
	return true
}

// parseUnicodeRange parses [X-Y] into low and high runes
func parseUnicodeRange(pattern string) (rune, rune, error) {
	if len(pattern) < 5 || pattern[0] != '[' || pattern[len(pattern)-1] != ']' {
		return 0, 0, fmt.Errorf("invalid range pattern")
	}

	inner := pattern[1 : len(pattern)-1]

	// Simple case: [a-z] where a and z are single chars
	if len(inner) == 3 && inner[1] == '-' {
		return rune(inner[0]), rune(inner[2]), nil
	}

	// Handle escaped characters like [\u0000-\uFFFF]
	dashIdx := findRangeDash(inner)
	if dashIdx < 0 {
		return 0, 0, fmt.Errorf("no dash in range")
	}

	lowStr := inner[:dashIdx]
	highStr := inner[dashIdx+1:]

	lowRune, err := parseRune(lowStr)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid low bound: %w", err)
	}

	highRune, err := parseRune(highStr)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid high bound: %w", err)
	}

	if lowRune > highRune {
		return 0, 0, fmt.Errorf("low bound > high bound")
	}

	return lowRune, highRune, nil
}

// findRangeDash finds the dash separating low-high in a range pattern
func findRangeDash(inner string) int {
	i := 0
	for i < len(inner) {
		if inner[i] == '\\' && i+1 < len(inner) {
			// Skip escape sequence
			if inner[i+1] == 'u' && i+6 <= len(inner) {
				i += 6 // \uXXXX
			} else {
				i += 2 // \n, \t, etc.
			}
			continue
		}
		if inner[i] == '-' && i > 0 {
			return i
		}
		i++
	}
	return -1
}

// parseRune parses a single rune from a string (handles escapes)
func parseRune(s string) (rune, error) {
	if len(s) == 0 {
		return 0, fmt.Errorf("empty rune")
	}

	// Handle escape sequences
	if s[0] == '\\' {
		if len(s) < 2 {
			return 0, fmt.Errorf("incomplete escape")
		}
		switch s[1] {
		case 'n':
			return '\n', nil
		case 't':
			return '\t', nil
		case 'r':
			return '\r', nil
		case '\\':
			return '\\', nil
		case '"':
			return '"', nil
		case '\'':
			return '\'', nil
		case 'u':
			if len(s) < 6 {
				return 0, fmt.Errorf("incomplete unicode escape")
			}
			val, err := strconv.ParseInt(s[2:6], 16, 32)
			if err != nil {
				return 0, fmt.Errorf("invalid unicode escape: %w", err)
			}
			return rune(val), nil
		default:
			return 0, fmt.Errorf("unknown escape: \\%c", s[1])
		}
	}

	// Plain character
	r, _ := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError {
		return 0, fmt.Errorf("invalid utf8")
	}
	return r, nil
}

// unescapeLiteral unescapes a literal pattern string
func unescapeLiteral(pattern string) (string, error) {
	// Try strconv.Unquote if it looks quoted
	if len(pattern) >= 2 && pattern[0] == '"' && pattern[len(pattern)-1] == '"' {
		unquoted, err := strconv.Unquote(pattern)
		if err != nil {
			return "", err
		}
		return unquoted, nil
	}

	// If no backslashes, return as-is
	if !strings.Contains(pattern, "\\") {
		return pattern, nil
	}

	// Manual unescape
	var result strings.Builder
	i := 0
	for i < len(pattern) {
		if pattern[i] == '\\' && i+1 < len(pattern) {
			switch pattern[i+1] {
			case 'n':
				result.WriteByte('\n')
				i += 2
			case 't':
				result.WriteByte('\t')
				i += 2
			case 'r':
				result.WriteByte('\r')
				i += 2
			case '\\':
				result.WriteByte('\\')
				i += 2
			case '"':
				result.WriteByte('"')
				i += 2
			case '\'':
				result.WriteByte('\'')
				i += 2
			case 'u':
				if i+6 <= len(pattern) {
					val, err := strconv.ParseInt(pattern[i+2:i+6], 16, 32)
					if err != nil {
						return "", fmt.Errorf("invalid unicode escape at %d", i)
					}
					result.WriteRune(rune(val))
					i += 6
				} else {
					return "", fmt.Errorf("incomplete unicode escape at %d", i)
				}
			default:
				// Reject unknown escape sequences
				return "", fmt.Errorf("unknown escape sequence: \\%c at position %d", pattern[i+1], i)
			}
		} else {
			result.WriteByte(pattern[i])
			i++
		}
	}
	return result.String(), nil
}
