package sample

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/ollama/ollama/model"
)

// TODO: safety in case of invalid json
// TODO: partial JSON matching?
// TODO: interfaces to cleanup with return values
// TODO this interface shouldn't be the sampler - should just use Sampler
// TODO: add penalties for string \n stuff
// TODO: minimize number of fwd passes if there is only one match
// TODO: greedy sample initially and then backtrack if no match

type PushdownSampler struct {
	PDAGraphBuilder
	curNode      *PDA
	braceStack   []rune
	stateCounter uint32
}

// graph should be built once and reused per tokenizer
func NewPushdownSampler(proc model.TextProcessor) (*PushdownSampler, error) {
	start := time.Now()

	fmt.Println("--------------------------------")
	fmt.Println("PDA sampler")
	fmt.Println("--------------------------------")
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc
	fmt.Printf("Alloc = %.2f MB\n", float64(before)/(1024*1024))

	vocab := proc.Vocab()
	decodedToks := make([]string, len(vocab.Values))
	for i := range vocab.Values {
		token, err := proc.Decode([]int32{int32(i)})
		if err != nil {
			return nil, err
		}
		decodedToks[i] = token
	}

	gb := &PDAGraphBuilder{
		proc:        proc,
		decodedToks: decodedToks,
	}

	if err := gb.BuildGraph(); err != nil {
		return nil, err
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	fmt.Printf("Alloc = %.2f MB\n", float64(after)/(1024*1024))
	fmt.Printf("Graph memory usage = %.2f MB\n", float64(after-before)/(1024*1024))
	fmt.Printf("Graph build time = %v\n", time.Since(start))

	// TODO: this can be simplified
	return &PushdownSampler{
		curNode:         gb.stateToNodeMap[StateStart],
		PDAGraphBuilder: *gb,
		braceStack:      []rune{},
		stateCounter:    0,
	}, nil
}

// TODO: need to add resampling logic if the first sample was not good
// greedy sample + backtrack?
func (s *PushdownSampler) Apply(logits []float32) ([]float32, error) {
	switch s.curNode.State {
	case StateInString:
		return s.maskLogits(logits, s.curNode)

	case StateInListEnd:
		// force finish if no braces left
		if len(s.braceStack) == 0 {
			s.curNode = NewPDANode(StateTerminate)
			return forceFinish(s, logits)
		}

		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil

	case StateTerminate:
		return forceFinish(s, logits)

	case StateInObjectEnd:
		// force finish if no braces left
		if len(s.braceStack) == 0 {
			s.curNode = NewPDANode(StateTerminate)
			return forceFinish(s, logits)
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

	default:
		fmt.Println("masking logits current state", s.curNode.State)
		logits, err := s.maskLogits(logits, s.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil
	}
}

func forceFinish(s *PushdownSampler, logits []float32) ([]float32, error) {
	for i := range logits {
		if s.proc.Is(int32(i), model.SpecialEOS) {
			logits[i] = 1.0
		} else {
			logits[i] = float32(math.Inf(-1))
		}
	}
	return logits, nil
}

func (s *PushdownSampler) UpdateState(tokenSlice []int32) ([]int32, error) {
	fmt.Println("current state - updating", s.curNode.State)
	mappedString, err := s.proc.Decode(tokenSlice)
	if err != nil {
		return nil, err
	}
	fmt.Printf(">>> mappedString: %q\n", mappedString)

	// Special handling for EOS token in terminate state
	if s.curNode.State == StateTerminate {
		for _, tokenID := range tokenSlice {
			if s.proc.Is(tokenID, model.SpecialEOS) {
				return tokenSlice, nil
			}
		}
	}

	// flag := -1
	// endBraceRunes := []rune{'}', ']'}
	for _, r := range mappedString {
		// TODO: if this is enabled again, make sure to appropriately handle the state transitions
		// if slices.Contains(endBraceRunes, r) && len(s.braceStack) == 0 {
		// 	fmt.Printf("stack is empty, extra closing brace %c\n", r)
		// 	// flag = i
		// 	break

		// }
		if r == rune('{') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('[') {
			s.braceStack = append(s.braceStack, r)
		}
		if r == rune('}') {
			if len(s.braceStack) == 0 {
				return nil, fmt.Errorf("stack is empty, extra closing brace %c", r)
			}
			top := s.braceStack[len(s.braceStack)-1]
			if top != rune('{') {
				return nil, fmt.Errorf("unmatched closing brace, got%c, want%c", top, '{')
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
		}

		if r == rune(']') {
			if len(s.braceStack) == 0 {
				return nil, fmt.Errorf("stack is empty, extra closing brace %c", r)
			}
			top := s.braceStack[len(s.braceStack)-1]
			if top != rune('[') {
				return nil, fmt.Errorf("unmatched closing brace, got%c, want%c", top, '[')
			}
			s.braceStack = s.braceStack[:len(s.braceStack)-1]
		}
	}

	// if flag != -1 {
	// 	tokenSlice = tokenSlice[:flag]
	// }
	// fmt.Println("flag!", flag)
	for _, tokenID := range tokenSlice {
		// transition to the next node
		nextNode, ok := s.curNode.MaskTokenIDToNode[tokenID]
		if !ok {
			return nil, fmt.Errorf("invalid token: %q", mappedString)
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
	return tokenSlice, nil
}

// greedy sample + backtrack?
func (s *PushdownSampler) maskLogits(logits []float32, node *PDA) ([]float32, error) {
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

func (s *PushdownSampler) fastMaskLogits(logits []float32, node *PDA) ([]float32, error) {
	maxLogit := float32(math.Inf(-1))
	maxIndex := -1

	// Find the maximum logit value among valid tokens
	for tokenID := range node.MaskTokenIDToNode {
		if int(tokenID) < len(logits) && logits[tokenID] > maxLogit {
			maxLogit = logits[tokenID]
			maxIndex = int(tokenID)
		}
	}

	if maxIndex == -1 {
		return nil, fmt.Errorf("no valid tokens found in mask")
	}

	logits[0] = float32(maxIndex)
	return logits, nil
	// return maxIndex, nil
}
