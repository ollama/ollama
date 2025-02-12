package sample

import (
	"slices"

	"github.com/ollama/ollama/model"
)

/*
Key JSON rules to consider:

1. Whitespace handling:
   - Need to handle all valid JSON whitespace characters (\r, spaces between tokens)
   - Current code only handles some whitespace cases

2. Number validation:
   - Need proper validation for special number cases like -0
   - Should handle .5 style decimals
   - Need limits on scientific notation (e, E)

3. String escaping:
   - Currently marks \ as invalid but should allow escaped sequences:
     - \"
     - \n
     - \u1234 unicode escapes

4. Empty object/array transitions:
   - Direct {} and [] cases could be more explicit
   - Need clear transitions for these edge cases

5. Nested depth limits:
   - No protection against excessive nesting
   - Could cause stack overflow with deeply nested structures
*/

// TODO: / should be valid but an escape character
var stringInvalidRunes = []rune{'\n', '\t', '{', '}', ':', ',', '/'}

var intInvalidRunes = []rune{'e', 'E', ' ', '\n', '\t', '{', '}', ':', ',', '"'}
var validIntRunes = []rune{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'}

var validNumberRunes = []rune{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e', 'E'}

var validBoolRunes = []rune{'t', 'r', 'u', 'e', 'f', 'a', 'l', 's', 'e'}

var validNullRunes = []rune{'n', 'u', 'l', 'l'}

type PDA struct {
	State             JSONState
	TransitionEdges   map[rune]*PDA
	MaskTokenIDToNode map[int32]*PDA
}

func NewPDANode(state JSONState) *PDA {
	return &PDA{
		State:             state,
		TransitionEdges:   make(map[rune]*PDA),
		MaskTokenIDToNode: make(map[int32]*PDA),
	}
}

type PDAGraphBuilder struct {
	proc           model.TextProcessor
	decodedToks    []string
	stateToNodeMap map[JSONState]*PDA
}

func (b *PDAGraphBuilder) BuildGraph() error {
	stateToNodeMap := make(map[JSONState]*PDA)
	for _, state := range JSONStates {
		stateToNodeMap[state] = NewPDANode(state)
	}

	stateToNodeMap[StateStart].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateStart].TransitionEdges['['] = stateToNodeMap[StateInList]

	stateToNodeMap[StateInObject].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObject].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]
	stateToNodeMap[StateInObject].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	// new line
	stateToNodeMap[StateInNewline].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInNewline].TransitionEdges['\t'] = stateToNodeMap[StateInTab]
	stateToNodeMap[StateInNewline].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	stateToNodeMap[StateInNewline].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	// new line end value
	stateToNodeMap[StateInNewlineEndValue].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInNewlineEndValue].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	stateToNodeMap[StateInNewlineEndValue].TransitionEdges[']'] = stateToNodeMap[StateInListEnd]

	stateToNodeMap[StateInObjSpace].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObjSpace].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]
	// TODO: see if this is needed for formatting
	stateToNodeMap[StateInObjSpace].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	stateToNodeMap[StateInTab].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]

	stateToNodeMap[StateInObjectKey].TransitionEdges[rune(-1)] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObjectKey].TransitionEdges['"'] = stateToNodeMap[StateInObjectKeyEnd]

	stateToNodeMap[StateInObjectKeyEnd].TransitionEdges[':'] = stateToNodeMap[StateInColon]

	stateToNodeMap[StateInObjectEnd].TransitionEdges[','] = stateToNodeMap[StateInComma]
	stateToNodeMap[StateInObjectEnd].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]

	// where values should be
	// this could be combined but the probl might change, we're alr doing a skip ahead
	stateToNodeMap[StateInColon].TransitionEdges[' '] = stateToNodeMap[StateInSpaceToValue]
	stateToNodeMap[StateInColon].TransitionEdges['['] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInColon].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	b.addValueConnections(stateToNodeMap[StateInColon])

	// Leads to a value
	stateToNodeMap[StateInSpaceToValue].TransitionEdges['['] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInSpaceToValue].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	b.addValueConnections(stateToNodeMap[StateInSpaceToValue])
	stateToNodeMap[StateInSpaceToValue].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]

	// Values
	// string node
	stateToNodeMap[StateInString].TransitionEdges[rune(-1)] = stateToNodeMap[StateInString]
	stateToNodeMap[StateInString].TransitionEdges['"'] = stateToNodeMap[StateInStringEnd]

	// String end node
	b.addEnds(stateToNodeMap[StateInStringEnd])
	stateToNodeMap[StateInStringEnd].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInStringEnd].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// TODO: add counters for allowable number of decimals, e, E, etc
	// number node
	for _, r := range validNumberRunes {
		stateToNodeMap[StateInNumber].TransitionEdges[r] = stateToNodeMap[StateInNumber]
	}
	b.addEnds(stateToNodeMap[StateInNumber])
	stateToNodeMap[StateInNumber].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInNumber].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// list node
	stateToNodeMap[StateInList].TransitionEdges[','] = stateToNodeMap[StateInComma]
	stateToNodeMap[StateInList].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInList].TransitionEdges[' '] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInList].TransitionEdges['\n'] = stateToNodeMap[StateInList]

	// list end node
	stateToNodeMap[StateInListEnd].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	stateToNodeMap[StateInListEnd].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInListEnd].TransitionEdges[','] = stateToNodeMap[StateInComma]
	stateToNodeMap[StateInListEnd].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// empty list
	stateToNodeMap[StateInList].TransitionEdges[']'] = stateToNodeMap[StateInListEnd]
	b.addValueConnections(stateToNodeMap[StateInList])

	// null node
	for _, r := range validNullRunes {
		stateToNodeMap[StateInNull].TransitionEdges[r] = stateToNodeMap[StateInNull]
	}
	b.addEnds(stateToNodeMap[StateInNull])
	stateToNodeMap[StateInNull].TransitionEdges[' '] = stateToNodeMap[StateInSpaceToValue]
	stateToNodeMap[StateInNull].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// list comma
	// should point to values
	stateToNodeMap[StateInListComma].TransitionEdges[' '] = stateToNodeMap[StateInListComma]
	stateToNodeMap[StateInListComma].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInListComma].TransitionEdges['\n'] = stateToNodeMap[StateInList]
	b.addValueConnections(stateToNodeMap[StateInListComma])

	// list object end
	stateToNodeMap[StateInListObjectEnd].TransitionEdges[','] = stateToNodeMap[StateInListComma]
	stateToNodeMap[StateInListObjectEnd].TransitionEdges[']'] = stateToNodeMap[StateInListEnd]
	// TODO: not sure if this is needed
	stateToNodeMap[StateInListObjectEnd].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// bool node
	for _, r := range validBoolRunes {
		stateToNodeMap[StateInBool].TransitionEdges[r] = stateToNodeMap[StateInBool]
	}
	stateToNodeMap[StateInBool].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]
	b.addEnds(stateToNodeMap[StateInBool])
	stateToNodeMap[StateInBool].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInBool].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	// comma node
	stateToNodeMap[StateInComma].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInComma].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]
	stateToNodeMap[StateInComma].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInComma].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	// space end value
	stateToNodeMap[StateInSpaceEndValue].TransitionEdges[' '] = stateToNodeMap[StateInSpaceEndValue]
	stateToNodeMap[StateInSpaceEndValue].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	stateToNodeMap[StateInSpaceEndValue].TransitionEdges[']'] = stateToNodeMap[StateInListEnd]
	stateToNodeMap[StateInSpaceEndValue].TransitionEdges['\n'] = stateToNodeMap[StateInNewlineEndValue]

	b.stateToNodeMap = stateToNodeMap
	if err := b.preComputeValidStates(); err != nil {
		return err
	}
	return nil
}

func (b *PDAGraphBuilder) addEnds(node *PDA) {
	node.TransitionEdges[','] = b.stateToNodeMap[StateInComma]
	node.TransitionEdges['}'] = b.stateToNodeMap[StateInObjectEnd]
	node.TransitionEdges[']'] = b.stateToNodeMap[StateInListEnd]
}

func (b *PDAGraphBuilder) addValueConnections(node *PDA) {
	node.TransitionEdges['"'] = b.stateToNodeMap[StateInString]
	for _, r := range validNumberRunes {
		node.TransitionEdges[r] = b.stateToNodeMap[StateInNumber]
	}
	// TODO(parthsareen): force the output and shift similar to structured outputs
	node.TransitionEdges['t'] = b.stateToNodeMap[StateInBool]
	node.TransitionEdges['f'] = b.stateToNodeMap[StateInBool]
	node.TransitionEdges['n'] = b.stateToNodeMap[StateInNull]
}

func (b *PDAGraphBuilder) preComputeValidStates() error {
	for _, node := range b.stateToNodeMap {
		if err := b.CreateMask(node); err != nil {
			return err
		}
	}
	return nil
}

func (b *PDAGraphBuilder) CreateMask(node *PDA) error {
	for i := range b.decodedToks {
		token := b.decodedToks[i]
		// Skip EOS/BOS tokens and empty tokens since they are not valid in JSON
		if b.proc.Is(uint32(i), model.SpecialEOS) || b.proc.Is(uint32(i), model.SpecialBOS) || token == "" || token == "\"\"" {
			continue
		}
		curNode := node
		valid := true
		consumedSpecialRunes := make(map[rune]bool)
		for _, r := range token {
			curNode, valid = isRuneValid(r, curNode, consumedSpecialRunes)
			if curNode == nil || !valid {
				break
			}
		}
		if valid {
			node.MaskTokenIDToNode[int32(i)] = curNode
		}
	}
	return nil
}

func isRuneValid(r rune, curNode *PDA, consumedSpecialRunes map[rune]bool) (*PDA, bool) {
	if consumedSpecialRunes[r] {
		return nil, false
	}

	specialRune := slices.Contains(stringInvalidRunes, r)
	if specialRune {
		if curNode.State == StateInString || curNode.State == StateInObjectKey {
			return nil, false
		}
	}

	// Check for specific rune transition
	if nextNode, ok := curNode.TransitionEdges[r]; ok {
		if specialRune {
			if curNode.State == nextNode.State {
				return nil, false
			}
			consumedSpecialRunes[r] = true
		}
		return nextNode, true
	}

	// Check for sentinel value - if present, any rune is valid
	if nextNode, ok := curNode.TransitionEdges[rune(-1)]; ok {
		return nextNode, true
	}

	return nil, false
}
