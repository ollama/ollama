package sample

import (
	"slices"

	"github.com/ollama/ollama/model"
)

var stringInvalidRunes = []rune{'\\', '\n', '\t', '{', '}', ':', ','}

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

	// TODO:
	// consider adding a node to just point to values, could be good to compute that
	// mask rather than many different nodes

	// Connect nodes
	// TODO: if all are single tokens then this can just be connected instead of defining the token
	startNode.TransitionEdges['{'] = objNode

	objNode.TransitionEdges['"'] = objKeyNode
	objNode.TransitionEdges['\n'] = newlineNode
	// objNode.TransitionEdges['\t'] = tabNode

	newlineNode.TransitionEdges['"'] = objKeyNode
	newlineNode.TransitionEdges['\t'] = tabNode

	tabNode.TransitionEdges['"'] = objKeyNode
	// tabNode.TransitionEdges['\t'] = tabNode

	objKeyNode.TransitionEdges[rune(-1)] = objKeyNode
	objKeyNode.TransitionEdges['"'] = objKeyEndNode

	objKeyEndNode.TransitionEdges[':'] = colonNode
	objEndNode.TransitionEdges[' '] = spaceNode

	// where values should be
	// this could be combined but the probs might change, we're alr doing a skip ahead
	colonNode.TransitionEdges[' '] = spaceNode

	// Leads to a value
	spaceNode.TransitionEdges['"'] = stringNode
	spaceNode.TransitionEdges['['] = listNode
	spaceNode.TransitionEdges['{'] = objNode

	for _, r := range validNumberRunes {
		spaceNode.TransitionEdges[r] = numberNode
	}
	for _, r := range validBoolRunes {
		spaceNode.TransitionEdges[r] = boolNode
	}

	for _, r := range validNullRunes {
		spaceNode.TransitionEdges[r] = nullNode
	}

	// Values
	// string node
	stringNode.TransitionEdges[rune(-1)] = stringNode
	stringNode.TransitionEdges['"'] = stringEndNode

	stringEndNode.TransitionEdges[','] = commaNode
	stringEndNode.TransitionEdges['}'] = objEndNode
	stringEndNode.TransitionEdges[']'] = listEndNode

	// TODO: add counters for allowable number of decimals, e, E, etc
	// number node
	for _, r := range validNumberRunes {
		numberNode.TransitionEdges[r] = numberNode
	}
	numberNode.TransitionEdges[','] = commaNode
	numberNode.TransitionEdges['}'] = objEndNode
	numberNode.TransitionEdges[']'] = listEndNode

	for _, r := range validBoolRunes {
		boolNode.TransitionEdges[r] = boolNode
	}

	// list node
	listNode.TransitionEdges[','] = commaNode
	listNode.TransitionEdges['"'] = stringNode
	// squash states to a value
	for _, r := range validNumberRunes {
		listNode.TransitionEdges[r] = numberNode
	}
	for _, r := range validBoolRunes {
		listNode.TransitionEdges[r] = boolNode
	}
	for _, r := range validNullRunes {
		listNode.TransitionEdges[r] = nullNode
	}

	// null node
	for _, r := range validNullRunes {
		nullNode.TransitionEdges[r] = nullNode
	}
	nullNode.TransitionEdges[','] = commaNode
	nullNode.TransitionEdges['}'] = objEndNode
	nullNode.TransitionEdges[']'] = listEndNode

	// list comma
	// should point to values
	listCommaNode.TransitionEdges['"'] = stringNode
	listCommaNode.TransitionEdges[' '] = listCommaNode
	listCommaNode.TransitionEdges['{'] = objNode
	listCommaNode.TransitionEdges['\n'] = newlineNode

	for _, r := range validNumberRunes {
		listCommaNode.TransitionEdges[r] = numberNode
	}
	for _, r := range validBoolRunes {
		listCommaNode.TransitionEdges[r] = boolNode
	}
	for _, r := range validNullRunes {
		listCommaNode.TransitionEdges[r] = nullNode
	}

	// bool node
	for _, r := range validBoolRunes {
		boolNode.TransitionEdges[r] = boolNode
	}
	boolNode.TransitionEdges['}'] = objEndNode
	boolNode.TransitionEdges[']'] = listEndNode
	boolNode.TransitionEdges[','] = commaNode

	listEndNode.TransitionEdges['}'] = objEndNode
	listEndNode.TransitionEdges[','] = commaNode

	commaNode.TransitionEdges['{'] = objNode
	commaNode.TransitionEdges['\n'] = newlineNode
	commaNode.TransitionEdges['\t'] = tabNode
	commaNode.TransitionEdges['"'] = objKeyNode
	commaNode.TransitionEdges[' '] = spaceObjNode

	spaceObjNode.TransitionEdges['"'] = objKeyNode

	return startNode, stateToNodeMap, nil
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
			if proc.Is(uint32(i), model.SpecialEOS) || proc.Is(uint32(i), model.SpecialBOS) || token == "" {
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
			// fmt.Println("special rune", r, "consumed")
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
