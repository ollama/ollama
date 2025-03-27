package sample

import (
	"bytes"
	"strings"

	"github.com/ollama/ollama/model"
)

type Node struct {
	TransitionEdges map[rune]*Node
}

type Graph struct {
	proc        model.TextProcessor
	decodedToks []string
	curNode     *Node
	grammar     []byte
	rules       map[string]string
}

// baseRules is the set of rules that are used to parse the grammar
// JSON grammar from RFC 7159
var baseRules = map[string]string{
	"object":  "\"{\" (kv (\",\" kv)*)? \"}\"",
	"array":   "\"[\" (value (\",\" value)*)? \"]\"",
	"string":  "\"\\\"\" char* \"\\\"\"",
	"number":  "\"-\"? integer frac? exp?",
	"kv":      "string \":\" value",
	"integer": "\"0\" | [1-9] [0-9]*",
	"frac":    "\".\" [0-9]+",
	"exp":     "(\"e\" | \"E\") (\"+\" | \"-\") [0-9]+",
	"escape":  "[\"/\" | \"b\" | \"f\" | \"n\" | \"r\" | \"t\" | unicode]",
	"char":    "[^\"\\\\] | escape",
	"space":   "(\" \" | \"\\t\" | \"\\n\" | \"\\r\")*",
	"hex":     "[0-9] | [a-f] | [A-F]",
	"boolean": "\"true\" | \"false\"",
	"value":   "object | array | string | number | boolean | \"null\"",
	"null":    "\"null\"",
}

func (g *Graph) BuildGraph(node *Node) error {
	vocab := g.proc.Vocab()
	decodedToks := make([]string, len(vocab.Values))
	for i := range vocab.Values {
		token, err := g.proc.Decode([]int32{int32(i)})
		if err != nil {
			return err
		}
		decodedToks[i] = token
	}

	g.decodedToks = decodedToks
	g.rules = baseRules
	g.rootPrefixes()
	rootNode := &Node{
		TransitionEdges: make(map[rune]*Node),
	}
	g.parseRule(g.rules["root"], rootNode)

	return nil
}

// rootPrefixes extracts all root prefixes from the grammar
// and parses the grammar string to extract root prefixes
func (g *Graph) rootPrefixes() {
	lines := bytes.Split(g.grammar, []byte("\n"))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 || bytes.HasPrefix(line, []byte("#")) {
			continue
		}

		parts := bytes.SplitN(line, []byte("::="), 2)
		if len(parts) != 2 {
			continue
		}

		ruleName := string(bytes.TrimSpace(parts[0]))
		if strings.HasPrefix(ruleName, "root") {
			g.rules[ruleName] = string(bytes.TrimSpace(parts[1]))
		}
	}
}

// parseRule parses a grammar rule and returns a Node
func (g *Graph) parseRule(rule string, curNode *Node) *Node {
	/*
		Here are the special characters in BNF grammar and their functions:
		::= - Definition operator, means "is defined as"
		| - Alternation, means "or"
		* - Zero or more repetitions of preceding element
		+ - One or more repetitions
		? - Optional (zero or one occurrence)
		[] - Character class, matches any single character within brackets
		[^] - Negated character class, matches any character NOT listed
		() - Grouping of elements
		- - Range operator in character classes (e.g., [a-z])
		"" - Literal string match
	*/

	// Split rule into tokens by whitespace
	tokens := strings.Fields(rule)
	if len(tokens) == 0 {
		return &Node{
			TransitionEdges: make(map[rune]*Node),
		}
	}

	// Handle integer rule
	if strings.Contains(rule, "[0-9]+") {
		// Create node for first digit 1-9
		firstDigitNode := &Node{
			TransitionEdges: make(map[rune]*Node),
		}
		for r := '1'; r <= '9'; r++ {
			curNode.TransitionEdges[r] = firstDigitNode
		}

		// Create node for subsequent digits 0-9
		zeroToNineNode := &Node{
			TransitionEdges: make(map[rune]*Node),
		}
		for r := '0'; r <= '9'; r++ {
			// Loop back to same node for * operator
			zeroToNineNode.TransitionEdges[r] = zeroToNineNode
		}

		// Connect first digit to subsequent digits
		firstDigitNode.TransitionEdges = zeroToNineNode.TransitionEdges

		// Also handle the "0" case
		if strings.Contains(rule, "\"0\"") {
			zeroNode := &Node{
				TransitionEdges: make(map[rune]*Node),
			}
			curNode.TransitionEdges['0'] = zeroNode
		}

		return curNode
	}

	// recursive case
	// grammar options
	// TODO: handle left recursion
	if strings.Contains(rule, "|") {
		parts := strings.Split(rule, "|")
		savedNode := curNode
		for _, part := range parts {
			// TODO: add correct transitions
			g.parseRule(part, savedNode)
		}
	}

	for _, token := range tokens {
		if strings.HasPrefix(token, "\"") && strings.HasSuffix(token, "\"") {
			token = strings.Trim(token, "\"")

			for _, r := range token {
				newNode := &Node{
					TransitionEdges: make(map[rune]*Node),
				}
				curNode.TransitionEdges[r] = newNode
				curNode = newNode
			}
			// strNode := &Node{
			// 	TransitionEdges: make(map[rune]*Node),
			// }

			// TODO: length constraint
			// to self
		}
	}

	return curNode
}
