package sample

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/ollama/ollama/model"
)

type PushdownSampler struct {
	// stateful
	curNode        *PDANode
	proc           model.TextProcessor
	stateToNodeMap map[JSONState]*PDANode
	braceStack     []rune
	stateCounter   uint32
}

func NewPushdownSampler(proc model.TextProcessor) *PushdownSampler {
	start := time.Now()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc
	fmt.Printf("Alloc = %.2f MB\n", float64(before)/(1024*1024))

	startNode, stateToNodeMap, err := BuildGraph(proc)
	if err != nil {
		panic(err)
	}
	err = PreComputeValidStates(stateToNodeMap, proc)
	if err != nil {
		panic(err)
	}
	runtime.ReadMemStats(&m)
	after := m.Alloc
	fmt.Printf("Alloc = %.2f MB\n", float64(after)/(1024*1024))
	fmt.Printf("Graph memory usage = %.2f MB\n", float64(after-before)/(1024*1024))
	fmt.Printf("Graph build time = %v\n", time.Since(start))
	// for id, node := range stateToNodeMap[StateInComma].MaskTokenIDToNode {
	// 	token, err := proc.Decode([]int32{int32(id)})
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fmt.Println("id", id, "node", node, "token", token)
	// }
	// time.Sleep(10 * time.Second)
	return &PushdownSampler{
		curNode:        startNode,
		proc:           proc,
		stateToNodeMap: stateToNodeMap,
		braceStack:     []rune{},
		stateCounter:   0,
	}
}

func (s *PushdownSampler) Sample(logits []float64) ([]float64, error) {
	fmt.Println("sample:", s.curNode.State)

	switch s.curNode.State {
	case StateInObjectEnd:
		// force finish if no braces left
		if len(s.braceStack) == 0 {
			s.curNode = NewPDANode(StateTerminate)
			for i := range logits {
				if s.proc.Is(uint32(i), model.SpecialEOS) {
					logits[i] = 1.0
				} else {
					logits[i] = math.NaN()
				}
			}
			return logits, nil
		}
		valid, err := s.proc.Encode("}")
		if err != nil {
			return nil, err
		}
		for i := range logits {
			for _, token := range valid {
				if i != int(token) {
					logits[i] = math.NaN()
				}
			}
		}
		return logits, nil

	case StateInComma:
		peek := s.braceStack[len(s.braceStack)-1]
		if peek == rune('[') {
			s.curNode = s.stateToNodeMap[StateInListComma]
			fmt.Println("switching to list comma", s.curNode.State)
		}
		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil

	case StateTerminate:
		for i := range logits {
			if s.proc.Is(uint32(i), model.SpecialEOS) {
				logits[i] = 1.0
			} else {
				logits[i] = math.NaN()
			}
		}
		return logits, nil

	default:
		fmt.Println("masking logits current state", s.curNode.State)
		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil
	}
}

func (s *PushdownSampler) UpdateState(tokenSlice []int32) error {
	fmt.Println("update state", s.curNode.State)

	// TODO: need to handle end states and entering object case, and list case
	if s.curNode.State == StateInObjectEnd {
		fmt.Println("in object end")
		if len(s.braceStack) > 0 {
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
			return nil
		}
		s.curNode = NewPDANode(StateTerminate)
		// TODO: return here?
	}
	// need this cause there could be multiple transitions
	mappedString, err := s.proc.Decode(tokenSlice)
	if err != nil {
		return err
	}
	// TODO: should force closing for all braces
	for _, r := range mappedString {
		if r == rune('{') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('[') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('}') {
			if len(s.braceStack) == 0 || s.braceStack[len(s.braceStack)-1] != rune('{') {
				return fmt.Errorf("unmatched closing brace")
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
			fmt.Println("popping brace stack", s.braceStack)
		}

		if r == rune(']') {
			if len(s.braceStack) == 0 || s.braceStack[len(s.braceStack)-1] != rune('[') {
				return fmt.Errorf("unmatched closing brace")
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
			fmt.Println("popping brace stack", s.braceStack)
		}
	}
	for _, tokenID := range tokenSlice {
		// transition to the next node
		nextNodeState, ok := s.curNode.MaskTokenIDToNode[tokenID]
		if !ok {
			return fmt.Errorf("invalid token: %q", mappedString)
		}
		fmt.Println("transitioning to", nextNodeState)

		// TODO: add a penalty for staying in the same state too long
		if nextNodeState == s.curNode.State {
			s.stateCounter++
		} else {
			s.stateCounter = 0
		}
		s.curNode = s.stateToNodeMap[nextNodeState]
	}
	return nil
}

func (s *PushdownSampler) maskLogits(logits []float64, node *PDANode) ([]float64, error) {
	for i := range logits {
		_, exists := node.MaskTokenIDToNode[int32(i)]
		if !exists {
			logits[i] = math.NaN()
		}
	}
	return logits, nil
}

// TODO: add penalties for string \n stuff
