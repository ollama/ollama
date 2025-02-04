package sample

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/ollama/ollama/model"
)

// TODO: safety in case of invalid json
// TODO: interfaces to cleanup with return values
type PushdownSampler struct {
	// stateful
	curNode        *PDANode
	proc           model.TextProcessor
	stateToNodeMap map[JSONState]*PDANode
	braceStack     []rune
	stateCounter   uint32
}

// graph should be built once and reused per tokenizer
func NewPushdownSampler(proc model.TextProcessor) *PushdownSampler {
	start := time.Now()

	fmt.Println("--------------------------------")
	fmt.Println("PDA sampler")
	fmt.Println("--------------------------------")
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

	return &PushdownSampler{
		curNode:        startNode,
		proc:           proc,
		stateToNodeMap: stateToNodeMap,
		braceStack:     []rune{},
		stateCounter:   0,
	}
}

// TODO: need to add resampling logic if the first sample was not good
// greedy sample + backtrack?
func (s *PushdownSampler) Apply(logits []float64) ([]float64, error) {
	switch s.curNode.State {
	case StateInString:
		return s.maskLogits(logits, s.curNode)

	case StateInListEnd:
		// force finish if no braces left
		if len(s.braceStack) == 0 {
			s.curNode = NewPDANode(StateTerminate)
			for i := range logits {
				if s.proc.Is(uint32(i), model.SpecialEOS) {
					logits[i] = 1.0
				} else {
					logits[i] = math.Inf(-1)
				}
			}
			return logits, nil
		}

		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil

	case StateInObjectEnd:
		// force finish if no braces left
		if len(s.braceStack) == 0 {
			s.curNode = NewPDANode(StateTerminate)
			for i := range logits {
				if s.proc.Is(uint32(i), model.SpecialEOS) {
					logits[i] = 1.0
				} else {
					logits[i] = math.Inf(-1)
				}
			}
			return logits, nil
		}

		peek := s.braceStack[len(s.braceStack)-1]
		if peek == rune('[') {
			s.curNode = s.stateToNodeMap[StateInListObjectEnd]
		}

		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil

	case StateInComma:
		peek := s.braceStack[len(s.braceStack)-1]
		if peek == rune('[') {
			s.curNode = s.stateToNodeMap[StateInListComma]
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
				logits[i] = math.Inf(-1)
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
	fmt.Println("current state - updating", s.curNode.State)
	mappedString, err := s.proc.Decode(tokenSlice)
	if err != nil {
		return err
	}
	fmt.Println(">>> mappedString", mappedString)

	// TODO: should force closing for all braces - not doing square yet
	for _, r := range mappedString {
		if r == rune('{') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('[') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('}') {
			if len(s.braceStack) == 0 {
				return fmt.Errorf("stack is empty, extra closing brace %c", r)
			}
			top := s.braceStack[len(s.braceStack)-1]
			if top != rune('{') {
				return fmt.Errorf("unmatched closing brace, got%c, want%c", top, '{')
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
		}

		if r == rune(']') {
			if len(s.braceStack) == 0 {
				return fmt.Errorf("stack is empty, extra closing brace %c", r)
			}
			top := s.braceStack[len(s.braceStack)-1]
			if top != rune('[') {
				return fmt.Errorf("unmatched closing brace, got%c, want%c", top, '[')
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
		}
	}

	for _, tokenID := range tokenSlice {
		// transition to the next node
		nextNode, ok := s.curNode.MaskTokenIDToNode[tokenID]
		if !ok {
			return fmt.Errorf("invalid token: %q", mappedString)
		}
		fmt.Println("transitioning to", nextNode.State)

		// TODO: add a penalty for staying in the same state too long
		if nextNode.State == s.curNode.State {
			s.stateCounter++
		} else {
			s.stateCounter = 0
		}
		s.curNode = nextNode
		fmt.Println("updated curNode state", s.curNode.State)
	}
	return nil
}

// greedy sample + backtrack?
func (s *PushdownSampler) maskLogits(logits []float64, node *PDANode) ([]float64, error) {
	// Create a new slice with same length as logits, initialized to -Inf
	maskedLogits := make([]float64, len(logits))
	for i := range maskedLogits {
		maskedLogits[i] = math.Inf(-1)
	}

	// Only update values for valid token IDs from the mask map
	for tokenID := range node.MaskTokenIDToNode {
		if int(tokenID) < len(logits) {
			maskedLogits[tokenID] = logits[tokenID]
		}
	}

	return maskedLogits, nil
}

// TODO: add penalties for string \n stuff
