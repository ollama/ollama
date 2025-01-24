package sample

import (
	"slices"

	"github.com/ollama/ollama/model"
)

var stringInvalidRunes = []rune{'\\', '\n', '\t', '{', '}', ':', ','}

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

	tabNode := NewPDANode(StateInTab)
	stateToNodeMap[StateInTab] = tabNode

	stringNode := NewPDANode(StateInString)
	stateToNodeMap[StateInString] = stringNode

	stringEndNode := NewPDANode(StateInStringEnd)
	stateToNodeMap[StateInStringEnd] = stringEndNode

	// terminateNode := NewNode(StateTerminate)

	// Connect nodes
	// TODO: if all are single tokens then this can just be connected instead of defining the token
	startNode.TransitionEdges['{'] = objNode

	objNode.TransitionEdges['"'] = objKeyNode
	objNode.TransitionEdges['\n'] = newlineNode

	newlineNode.TransitionEdges['"'] = objKeyNode
	newlineNode.TransitionEdges['\t'] = tabNode

	tabNode.TransitionEdges['"'] = objKeyNode

	spaceNode.TransitionEdges['"'] = stringNode

	objKeyNode.TransitionEdges[rune(-1)] = objKeyNode
	objKeyNode.TransitionEdges['"'] = objKeyEndNode
	objKeyNode.TransitionEdges[' '] = spaceNode
	// objKeyNode.TransitionEdges['\t'] = tabNode

	objKeyEndNode.TransitionEdges[':'] = colonNode

	colonNode.TransitionEdges['"'] = stringNode
	colonNode.TransitionEdges[' '] = spaceNode

	stringNode.TransitionEdges[rune(-1)] = stringNode
	stringNode.TransitionEdges['"'] = stringEndNode

	stringEndNode.TransitionEdges[','] = commaNode
	stringEndNode.TransitionEdges['}'] = objEndNode

	commaNode.TransitionEdges['{'] = objNode
	commaNode.TransitionEdges['\n'] = newlineNode
	commaNode.TransitionEdges['\t'] = tabNode
	commaNode.TransitionEdges['"'] = objKeyNode

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
