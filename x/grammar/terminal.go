//go:build mlx

package grammar

import "unicode/utf8"

// terminalType distinguishes different kinds of grammar terminals
type terminalType int

const (
	terminalLiteral terminalType = iota // Exact string: "true", "{"
	terminalRange                       // Character range: [a-z], [0-9]
)

// terminal represents a compiled grammar terminal
type terminal struct {
	ID        int
	Type      terminalType
	Pattern   string // Original pattern from grammar
	Unescaped string // Unescaped literal (for terminalLiteral)
	LowRune   rune   // For unicode ranges: low bound
	HighRune  rune   // For unicode ranges: high bound
}

// terminalMatch represents a terminal that matched at a position
type terminalMatch struct {
	TerminalID int
	Length     int // Number of bytes consumed
}

// trieNode is a node in the literal matching trie
type trieNode struct {
	children   [256]*trieNode // Byte-indexed children
	terminalID int            // -1 if not accepting, else terminal ID
}

// terminalMatcher tests which terminals match at a position in a byte slice
type terminalMatcher struct {
	// Trie for literal matching (fast path)
	literalTrie *trieNode

	// Range terminals (single-byte matches)
	ranges []terminal

	// All terminals for enumeration
	terminals []terminal

	// Pattern to terminal ID map for fast lookup (keyed by raw pattern)
	patternToID map[string]int
}

// addLiteralToTrie adds a literal pattern to the trie
func (m *terminalMatcher) addLiteralToTrie(pattern string, terminalID int) {
	node := m.literalTrie
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		if node.children[c] == nil {
			node.children[c] = &trieNode{terminalID: -1}
		}
		node = node.children[c]
	}
	node.terminalID = terminalID
}

// matchesAt returns all terminals that match at pos in data
func (m *terminalMatcher) matchesAt(data []byte, pos int) []terminalMatch {
	if pos >= len(data) {
		return nil
	}

	var matches []terminalMatch

	// Check literal matches via trie
	node := m.literalTrie
	for i := pos; i < len(data) && node != nil; i++ {
		c := data[i]
		node = node.children[c]
		if node != nil && node.terminalID >= 0 {
			matches = append(matches, terminalMatch{
				TerminalID: node.terminalID,
				Length:     i - pos + 1,
			})
		}
	}

	// Check range matches (unicode-aware)
	r, runeLen := utf8.DecodeRune(data[pos:])
	if r != utf8.RuneError {
		for _, rng := range m.ranges {
			if r >= rng.LowRune && r <= rng.HighRune {
				matches = append(matches, terminalMatch{
					TerminalID: rng.ID,
					Length:     runeLen,
				})
			}
		}
	}

	return matches
}

// terminalCount returns the number of terminals
func (m *terminalMatcher) terminalCount() int {
	return len(m.terminals)
}
