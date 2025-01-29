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

	for _, state := range JSONStates {
		stateToNodeMap[state] = NewPDANode(state)
	}
	// TODO:
	// consider adding a node to just point to values, could be good to compute that
	// mask rather than many different nodes

	// Connect nodes
	// TODO: if all are single tokens then this can just be connected instead of defining the token
	stateToNodeMap[StateStart].TransitionEdges['{'] = stateToNodeMap[StateInObject]

	stateToNodeMap[StateInObject].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObject].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]
	stateToNodeMap[StateInObject].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	//new line
	stateToNodeMap[StateInNewline].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInNewline].TransitionEdges['\t'] = stateToNodeMap[StateInTab]

	stateToNodeMap[StateInTab].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]

	stateToNodeMap[StateInObjectKey].TransitionEdges[rune(-1)] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObjectKey].TransitionEdges['"'] = stateToNodeMap[StateInObjectKeyEnd]

	stateToNodeMap[StateInObjectKeyEnd].TransitionEdges[':'] = stateToNodeMap[StateInColon]

	stateToNodeMap[StateInObjectEnd].TransitionEdges[','] = stateToNodeMap[StateInComma]
	stateToNodeMap[StateInObjectEnd].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]

	// where values should be
	// this could be combined but the probl might change, we're alr doing a skip ahead
	stateToNodeMap[StateInColon].TransitionEdges[' '] = stateToNodeMap[StateInSpace]
	stateToNodeMap[StateInColon].TransitionEdges['['] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInColon].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	addValueConnections(stateToNodeMap[StateInColon], stateToNodeMap)

	// Leads to a value
	stateToNodeMap[StateInSpace].TransitionEdges['['] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInSpace].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	addValueConnections(stateToNodeMap[StateInSpace], stateToNodeMap)

	// Values
	// string node
	stateToNodeMap[StateInString].TransitionEdges[rune(-1)] = stateToNodeMap[StateInString]
	stateToNodeMap[StateInString].TransitionEdges['"'] = stateToNodeMap[StateInStringEnd]

	// String end node
	addEnds(stateToNodeMap[StateInStringEnd], stateToNodeMap)

	// TODO: add counters for allowable number of decimals, e, E, etc
	// number node
	for _, r := range validNumberRunes {
		stateToNodeMap[StateInNumber].TransitionEdges[r] = stateToNodeMap[StateInNumber]
	}
	addEnds(stateToNodeMap[StateInNumber], stateToNodeMap)

	// bool node
	for _, r := range validBoolRunes {
		stateToNodeMap[StateInBool].TransitionEdges[r] = stateToNodeMap[StateInBool]
	}
	addEnds(stateToNodeMap[StateInBool], stateToNodeMap)

	// list node
	stateToNodeMap[StateInList].TransitionEdges[','] = stateToNodeMap[StateInComma]
	stateToNodeMap[StateInList].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInList].TransitionEdges[' '] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInList].TransitionEdges['\n'] = stateToNodeMap[StateInList]
	addValueConnections(stateToNodeMap[StateInList], stateToNodeMap)

	// null node
	for _, r := range validNullRunes {
		stateToNodeMap[StateInNull].TransitionEdges[r] = stateToNodeMap[StateInNull]
	}
	addEnds(stateToNodeMap[StateInNull], stateToNodeMap)

	// list comma
	// should point to values
	stateToNodeMap[StateInListComma].TransitionEdges[' '] = stateToNodeMap[StateInListComma]
	stateToNodeMap[StateInListComma].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInListComma].TransitionEdges['\n'] = stateToNodeMap[StateInList]
	addValueConnections(stateToNodeMap[StateInListComma], stateToNodeMap)

	// list object end
	stateToNodeMap[StateInListObjectEnd].TransitionEdges[','] = stateToNodeMap[StateInListComma]
	stateToNodeMap[StateInListObjectEnd].TransitionEdges[']'] = stateToNodeMap[StateListEnd]

	// bool node
	for _, r := range validBoolRunes {
		stateToNodeMap[StateInBool].TransitionEdges[r] = stateToNodeMap[StateInBool]
	}
	addEnds(stateToNodeMap[StateInBool], stateToNodeMap)

	stateToNodeMap[StateListEnd].TransitionEdges['}'] = stateToNodeMap[StateInObjectEnd]
	stateToNodeMap[StateListEnd].TransitionEdges[','] = stateToNodeMap[StateInComma]

	stateToNodeMap[StateInComma].TransitionEdges['{'] = stateToNodeMap[StateInObject]
	stateToNodeMap[StateInComma].TransitionEdges['\n'] = stateToNodeMap[StateInList]
	stateToNodeMap[StateInComma].TransitionEdges['\t'] = stateToNodeMap[StateInTab]
	stateToNodeMap[StateInComma].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInComma].TransitionEdges[' '] = stateToNodeMap[StateInObjSpace]

	stateToNodeMap[StateInObjSpace].TransitionEdges['"'] = stateToNodeMap[StateInObjectKey]
	stateToNodeMap[StateInObjSpace].TransitionEdges['\n'] = stateToNodeMap[StateInNewline]

	return stateToNodeMap[StateStart], stateToNodeMap, nil
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
