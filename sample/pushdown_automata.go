package sample

import (
	"slices"

	"github.com/ollama/ollama/model"
)

// TODO: / should be valid but an escape character

var stringInvalidRunes = []rune{'\\', '\n', '\t', '{', '}', ':', ',', '/'}

var intInvalidRunes = []rune{'e', 'E', ' ', '\n', '\t', '{', '}', ':', ',', '"'}
var validIntRunes = []rune{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'}

var validNumberRunes = []rune{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e', 'E'}

var validBoolRunes = []rune{'t', 'r', 'u', 'e', 'f', 'a', 'l', 's', 'e'}

var validNullRunes = []rune{'n', 'u', 'l', 'l'}

type PDANode struct {
	State             JSONState
	TransitionEdges   map[rune]*PDANode
	MaskTokenIDToNode map[int32]JSONState
}

func NewPDANode(state JSONState) *PDANode {
	return &PDANode{
		State:             state,
		TransitionEdges:   make(map[rune]*PDANode),
		MaskTokenIDToNode: make(map[int32]JSONState),
	}
}

func BuildGraph(proc model.TextProcessor) (*PDANode, map[JSONState]*PDANode, error) {
	stateToNodeMap := make(map[JSONState]*PDANode)

	// TODO: make this a loop
	startNode := NewPDANode(StateStart)
	stateToNodeMap[StateStart] = startNode

	objNode := NewPDANode(StateInObject)
	stateToNodeMap[StateInObject] = objNode

	objEndNode := NewPDANode(StateInObjectEnd)
	stateToNodeMap[StateInObjectEnd] = objEndNode

	objKeyNode := NewPDANode(StateInObjectKey)
	stateToNodeMap[StateInObjectKey] = objKeyNode

	objKeyEndNode := NewPDANode(StateInObjectKeyEnd)
	stateToNodeMap[StateInObjectKeyEnd] = objKeyEndNode

	colonNode := NewPDANode(StateInColon)
	stateToNodeMap[StateInColon] = colonNode

	commaNode := NewPDANode(StateInComma)
	stateToNodeMap[StateInComma] = commaNode

	newlineNode := NewPDANode(StateInNewline)
	stateToNodeMap[StateInNewline] = newlineNode

	spaceNode := NewPDANode(StateInSpace)
	stateToNodeMap[StateInSpace] = spaceNode

	spaceObjNode := NewPDANode(StateInObjSpace)
	stateToNodeMap[StateInObjSpace] = spaceObjNode

	tabNode := NewPDANode(StateInTab)
	stateToNodeMap[StateInTab] = tabNode

	stringNode := NewPDANode(StateInString)
	stateToNodeMap[StateInString] = stringNode

	stringEndNode := NewPDANode(StateInStringEnd)
	stateToNodeMap[StateInStringEnd] = stringEndNode

	listNode := NewPDANode(StateInList)
	stateToNodeMap[StateInList] = listNode

	listCommaNode := NewPDANode(StateInListComma)
	stateToNodeMap[StateInListComma] = listCommaNode

	listEndNode := NewPDANode(StateListEnd)
	stateToNodeMap[StateListEnd] = listEndNode

	numberNode := NewPDANode(StateInNumber)
	stateToNodeMap[StateInNumber] = numberNode

	boolNode := NewPDANode(StateInBool)
	stateToNodeMap[StateInBool] = boolNode

	nullNode := NewPDANode(StateInNull)
	stateToNodeMap[StateInNull] = nullNode

	// Defined with structured outputs only
	intNode := NewPDANode(StateInInt)
	stateToNodeMap[StateInInt] = intNode

	listObjEndNode := NewPDANode(StateInListObjectEnd)
	stateToNodeMap[StateInListObjectEnd] = listObjEndNode

	// TODO:
	// consider adding a node to just point to values, could be good to compute that
	// mask rather than many different nodes

	// Connect nodes
	// TODO: if all are single tokens then this can just be connected instead of defining the token
	startNode.TransitionEdges['{'] = objNode

	objNode.TransitionEdges['"'] = objKeyNode
	objNode.TransitionEdges['\n'] = newlineNode
	objNode.TransitionEdges[' '] = spaceObjNode

	//new line
	newlineNode.TransitionEdges['"'] = objKeyNode
	newlineNode.TransitionEdges['\t'] = tabNode

	tabNode.TransitionEdges['"'] = objKeyNode

	objKeyNode.TransitionEdges[rune(-1)] = objKeyNode
	objKeyNode.TransitionEdges['"'] = objKeyEndNode

	objKeyEndNode.TransitionEdges[':'] = colonNode

	objEndNode.TransitionEdges[','] = commaNode
	objEndNode.TransitionEdges['}'] = objEndNode

	// where values should be
	// this could be combined but the probs might change, we're alr doing a skip ahead
	colonNode.TransitionEdges[' '] = spaceNode
	colonNode.TransitionEdges['['] = listNode
	colonNode.TransitionEdges['{'] = objNode
	addValueConnections(colonNode, stateToNodeMap)

	// Leads to a value
	spaceNode.TransitionEdges['['] = listNode
	spaceNode.TransitionEdges['{'] = objNode
	addValueConnections(spaceNode, stateToNodeMap)

	// Values
	// string node
	stringNode.TransitionEdges[rune(-1)] = stringNode
	stringNode.TransitionEdges['"'] = stringEndNode

	// String end node
	addEnds(stringEndNode, stateToNodeMap)

	// TODO: add counters for allowable number of decimals, e, E, etc
	// number node
	for _, r := range validNumberRunes {
		numberNode.TransitionEdges[r] = numberNode
	}
	addEnds(numberNode, stateToNodeMap)

	// bool node
	for _, r := range validBoolRunes {
		boolNode.TransitionEdges[r] = boolNode
	}
	addEnds(boolNode, stateToNodeMap)

	// list node
	listNode.TransitionEdges[','] = commaNode
	listNode.TransitionEdges['{'] = objNode
	listNode.TransitionEdges[' '] = listNode
	listNode.TransitionEdges['\n'] = listNode
	addValueConnections(listNode, stateToNodeMap)

	// null node
	for _, r := range validNullRunes {
		nullNode.TransitionEdges[r] = nullNode
	}
	addEnds(nullNode, stateToNodeMap)

	// list comma
	// should point to values
	listCommaNode.TransitionEdges[' '] = listCommaNode
	listCommaNode.TransitionEdges['{'] = objNode
	listCommaNode.TransitionEdges['\n'] = newlineNode
	addValueConnections(listCommaNode, stateToNodeMap)

	// list object end
	listObjEndNode.TransitionEdges[','] = listCommaNode
	listObjEndNode.TransitionEdges[']'] = listEndNode

	// bool node
	for _, r := range validBoolRunes {
		boolNode.TransitionEdges[r] = boolNode
	}
	addEnds(boolNode, stateToNodeMap)

	listEndNode.TransitionEdges['}'] = objEndNode
	listEndNode.TransitionEdges[','] = commaNode

	commaNode.TransitionEdges['{'] = objNode
	commaNode.TransitionEdges['\n'] = newlineNode
	commaNode.TransitionEdges['\t'] = tabNode
	commaNode.TransitionEdges['"'] = objKeyNode
	commaNode.TransitionEdges[' '] = spaceObjNode

	spaceObjNode.TransitionEdges['"'] = objKeyNode
	spaceObjNode.TransitionEdges['\n'] = newlineNode

	return startNode, stateToNodeMap, nil
}

func addEnds(node *PDANode, stateToNodeMap map[JSONState]*PDANode) {
	node.TransitionEdges[','] = stateToNodeMap[StateInComma]
	node.TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	node.TransitionEdges[']'] = stateToNodeMap[StateListEnd]
}

func addValueConnections(node *PDANode, stateToNodeMap map[JSONState]*PDANode) {
	node.TransitionEdges['"'] = stateToNodeMap[StateInString]
	for _, r := range validNumberRunes {
		node.TransitionEdges[r] = stateToNodeMap[StateInNumber]
	}
	node.TransitionEdges['t'] = stateToNodeMap[StateInBool]
	node.TransitionEdges['f'] = stateToNodeMap[StateInBool]
	node.TransitionEdges['n'] = stateToNodeMap[StateInNull]
}

func PreComputeValidStates(stateToNodeMap map[JSONState]*PDANode, proc model.TextProcessor) error {

	vocab := proc.GetVocabulary()

	decodedToks := make([]string, len(vocab.Values))
	for i := range vocab.Values {
		token, err := proc.Decode([]int32{int32(i)})
		if err != nil {
			return err
		}
		decodedToks[i] = token
	}

	var err error
	for _, node := range stateToNodeMap {
		for i := range vocab.Values {
			token := decodedToks[i]
			// Skip EOS/BOS tokens and empty tokens since they are not valid in JSON
			if proc.Is(uint32(i), model.SpecialEOS) || proc.Is(uint32(i), model.SpecialBOS) || token == "" || token == "\"\"" {
				continue
			}
			valid := true
			curNode := node
			consumedSpecialRunes := make(map[rune]bool)
			for _, r := range token {
				valid, curNode, err = isRuneValid(r, curNode, consumedSpecialRunes)
				if err != nil {
					return err
				}
				if !valid {
					break
				}
			}
			if valid {
				node.MaskTokenIDToNode[int32(i)] = curNode.State
			}
		}
	}
	return nil
}

// garbage interface plz fix
func isRuneValid(r rune, curNode *PDANode, consumedSpecialRunes map[rune]bool) (bool, *PDANode, error) {
	if consumedSpecialRunes[r] {
		return false, nil, nil
	}

	specialRune := slices.Contains(stringInvalidRunes, r)
	if specialRune {
		if curNode.State == StateInString || curNode.State == StateInObjectKey {
			return false, nil, nil
		}
	}

	// Check for specific rune transition
	if nextNode, ok := curNode.TransitionEdges[r]; ok {
		if specialRune {
			if curNode.State == nextNode.State {
				return false, nil, nil
			}
			consumedSpecialRunes[r] = true
		}
		return true, nextNode, nil
	}

	// Check for sentinel value - if present, any rune is valid
	if nextNode, ok := curNode.TransitionEdges[rune(-1)]; ok {
		return true, nextNode, nil
	}

	return false, nil, nil
}
