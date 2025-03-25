package sample

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/model"
)

type PythonState int

const (
	PythonStateStart PythonState = iota
	StateInFunction
	StateInFunctionArgs
	StateInFunctionArgsType
	StateInFunctionEnd
	PStateInString
	PStateInStringEnd
	PStateInNumber
	PStateInList
	PStateInListEnd
	PStateInDict
	PStateInDictEnd
	PStateInTuple
	PStateInTupleEnd
	PStateTerminate
)

func (s PythonState) String() string {
	switch s {
	case PythonStateStart:
		return "PythonStateStart"
	case StateInFunction:
		return "StateInFunction"
	case StateInFunctionArgs:
		return "StateInFunctionArgs"
	case StateInFunctionArgsType:
		return "StateInFunctionArgsType"
	case StateInFunctionEnd:
		return "StateInFunctionEnd"
	case PStateInString:
		return "PStateInString"
	case PStateInStringEnd:
		return "PStateInStringEnd"
	case PStateInNumber:
		return "PStateInNumber"
	case PStateInList:
		return "PStateInList"
	case PStateInListEnd:
		return "PStateInListEnd"
	case PStateInDict:
		return "PStateInDict"
	case PStateInDictEnd:
		return "PStateInDictEnd"
	case PStateInTuple:
		return "PStateInTuple"
	case PStateInTupleEnd:
		return "PStateInTupleEnd"
	case PStateTerminate:
		return "PStateTerminate"
	default:
		return fmt.Sprintf("PythonState(%d)", s)
	}
}

var PythonStates = []PythonState{
	PythonStateStart,
	StateInFunction,
	StateInFunctionArgs,
	StateInFunctionArgsType,
	StateInFunctionEnd,
	PStateInString,
	PStateInStringEnd,
	PStateInNumber,
	PStateInList,
	PStateInListEnd,
	PStateInDict,
	PStateInDictEnd,
	PStateInTuple,
	PStateInTupleEnd,
	PStateTerminate,
}

type Node struct {
	State             PythonState
	TransitionEdges   map[rune]*Node
	MaskTokenIDToNode map[int32]*Node
}

func NewNode(state PythonState) *Node {
	return &Node{
		State:             state,
		TransitionEdges:   make(map[rune]*Node),
		MaskTokenIDToNode: make(map[int32]*Node),
	}
}

type PythonFunction struct {
	Name  string
	Args  []string
	Types []string
}

type PythonSampler struct {
	stateToNodes map[PythonState]*Node
	proc         model.TextProcessor
	decodedToks  []string
	curNode      *Node
	completed    int
	functions    []PythonFunction
}

func (s *PythonSampler) Init(functions []PythonFunction, proc model.TextProcessor) error {
	s.proc = proc
	s.functions = functions
	decodedToks := make([]string, len(proc.Vocab().Values))
	for i := range proc.Vocab().Values {
		token, err := proc.Decode([]int32{int32(i)})
		if err != nil {
			return err
		}
		decodedToks[i] = token
	}
	s.decodedToks = decodedToks
	s.BuildGraph()
	for _, function := range functions {
		prevNode := s.stateToNodes[PythonStateStart]

		for _, r := range function.Name {
			nextNode := NewNode(StateInFunction)
			prevNode.TransitionEdges[r] = nextNode
			if err := s.CreateMask(nextNode); err != nil {
				return err
			}
			fmt.Println("prevNode", prevNode.State)
			fmt.Printf("transition edge: %q\n", r)
			fmt.Println("nextNode", nextNode.State)
			prevNode = nextNode
		}
		prevNode.TransitionEdges['('] = s.stateToNodes[StateInFunctionArgs]
		s.CreateMask(prevNode)
		prevNode = s.stateToNodes[StateInFunctionArgs]
		for i, arg := range function.Args {
			for _, r := range arg {
				nextNode := NewNode(StateInFunctionArgs)
				prevNode.TransitionEdges[r] = nextNode
				s.CreateMask(prevNode)
				prevNode = nextNode
			}
			prevNode.TransitionEdges[','] = s.stateToNodes[StateInFunctionArgs]
			// prevNode = s.stateToNodes[StateInFunctionArgs]
			prevNode.TransitionEdges['='] = NewNode(StateInFunctionArgsType)
			s.CreateMask(prevNode)
			prevNode = prevNode.TransitionEdges['=']
			switch function.Types[i] {
			case "string":
				prevNode.TransitionEdges['"'] = s.stateToNodes[PStateInString]
				s.CreateMask(prevNode.TransitionEdges['"'])
			case "number":
				prevNode.TransitionEdges['"'] = s.stateToNodes[PStateInNumber]
				s.CreateMask(prevNode.TransitionEdges['"'])
			}
		}

	}
	s.curNode = s.stateToNodes[PythonStateStart]
	fmt.Println("curNode", s.curNode.State)
	fmt.Println("transition edges", s.curNode.TransitionEdges)
	if err := s.CreateMask(s.curNode); err != nil {
		return err
	}
	fmt.Println("maskTokenIDToNode", s.curNode.MaskTokenIDToNode)
	for tokenID, node := range s.curNode.MaskTokenIDToNode {
		fmt.Printf("tokenID: %d, node: %v\n", s.decodedToks[tokenID], node.State)
	}

	return nil
}

func (s *PythonSampler) BuildGraph() error {
	s.stateToNodes = make(map[PythonState]*Node)
	for _, state := range PythonStates {
		s.stateToNodes[state] = NewNode(state)
	}

	for _, state := range s.stateToNodes {
		if err := s.CreateMask(state); err != nil {
			return err
		}
	}

	// String
	s.stateToNodes[PStateInString].TransitionEdges[rune(-1)] = s.stateToNodes[PStateInString]
	s.stateToNodes[PStateInString].TransitionEdges['"'] = s.stateToNodes[PStateInStringEnd]

	// String end
	s.stateToNodes[PStateInStringEnd].TransitionEdges[','] = s.stateToNodes[StateInFunctionArgs]
	// s.stateToNodes[PStateInStringEnd].TransitionEdges[')'] = s.stateToNodes[PStateTerminate]
	// Number
	for _, r := range validNumberRunes {
		s.stateToNodes[PStateInNumber].TransitionEdges[r] = s.stateToNodes[PStateInNumber]
	}
	s.stateToNodes[PStateInNumber].TransitionEdges[')'] = s.stateToNodes[PStateTerminate]
	s.stateToNodes[PStateInNumber].TransitionEdges[','] = s.stateToNodes[StateInFunctionArgs]
	s.stateToNodes[PStateInNumber].TransitionEdges[' '] = s.stateToNodes[StateInFunctionArgs]

	return nil
}

func (s *PythonSampler) ApplyMask(logits []float32) ([]float32, error) {
	if s.curNode.State == PStateTerminate {
		logits, err := finish(s, logits)
		if err != nil {
			return nil, err
		}
		return logits, nil
	}
	logits, err := s.maskLogits(logits, s.curNode)
	if err != nil {
		return nil, err
	}
	return logits, nil
}

func (s *PythonSampler) UpdateState(token int32) error {
	mappedString, err := s.proc.Decode([]int32{token})
	if err != nil {
		return err
	}
	fmt.Printf(">>> mappedString: %q\n", mappedString)

	if s.curNode.State == PStateTerminate {
		if s.proc.Is(token, model.SpecialEOS) {
			return nil
		}
	}
	nextNode, ok := s.curNode.MaskTokenIDToNode[token]
	if !ok {
		return fmt.Errorf("invalid token: %q", mappedString)
	}

	if mappedString == "\"" {
		if s.curNode.State == PStateInStringEnd {
			s.completed++
		}
		if s.completed == len(s.functions) {
			s.curNode.TransitionEdges[')'] = s.stateToNodes[PStateTerminate]
			s.CreateMask(s.curNode)
		}
	}
	s.curNode = nextNode
	fmt.Println("curNode", s.curNode.State)
	for r, node := range s.curNode.TransitionEdges {
		fmt.Printf("transition edge: %q -> %v\n", r, node.State)
	}
	if err := s.CreateMask(s.curNode); err != nil {
		return err
	}
	return nil
}

func (s *PythonSampler) CreateMask(node *Node) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}
	for i := range s.decodedToks {
		token := s.decodedToks[i]
		// Skip EOS/BOS tokens and empty tokens since they are not valid in JSON
		if s.proc.Is(int32(i), model.SpecialEOS) || s.proc.Is(int32(i), model.SpecialBOS) || token == "" || token == "\"\"" {
			continue
		}
		curNode := node
		valid := true
		consumedSpecialRunes := make(map[rune]bool)
		for _, r := range token {
			curNode, valid = isRValid(r, curNode, consumedSpecialRunes)
			if curNode == nil || !valid {
				break
			}
		}
		if valid {
			if curNode.State == StateInFunction {
				// fmt.Println("cm curNode", curNode.State)
				// fmt.Println("cm token", s.decodedToks[i])
			}
			node.MaskTokenIDToNode[int32(i)] = curNode
		}
	}
	return nil
}

func isRValid(r rune, curNode *Node, consumedSpecialRunes map[rune]bool) (*Node, bool) {
	if consumedSpecialRunes[r] {
		return nil, false
	}

	specialRune := slices.Contains(stringInvalidRunes, r)
	if specialRune {
		if curNode.State == PStateInString || curNode.State == PStateInStringEnd {
			return nil, false
		}
	}

	// Check for specific rune transition
	if nextNode, ok := curNode.TransitionEdges[r]; ok {
		// fmt.Println("next node", nextNode)
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

func (s *PythonSampler) maskLogits(logits []float32, node *Node) ([]float32, error) {
	// Create a new slice with same length as logits, initialized to -Inf
	maskedLogits := make([]float32, len(logits))
	for i := range maskedLogits {
		maskedLogits[i] = float32(math.Inf(-1))
	}

	// Only update values for valid token IDs from the mask map
	for tokenID := range node.MaskTokenIDToNode {
		if int(tokenID) < len(logits) {
			maskedLogits[tokenID] = logits[tokenID]
		}
	}

	return maskedLogits, nil
}

func finish(s *PythonSampler, logits []float32) ([]float32, error) {
	for i := range logits {
		if s.proc.Is(int32(i), model.SpecialEOS) {
			logits[i] = 1.0
		} else {
			logits[i] = float32(math.Inf(-1))
		}
	}
	return logits, nil
}
