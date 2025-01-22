package sample

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/model"
)

type JSONState int

const (
	StateStart JSONState = iota
	StateInObject
	StateInObjectKey
	StateNewline
	StateTab
	StateSpace
	StateInString
	StateInInt
	StateInFloat
	StateInBool
	StateInNull
	StateInArray
	StateInColon
	StateInComma
	StateInStringEnd
	StateInObjectKeyEnd
	StateTerminate
	StateEnd
)

func (s JSONState) String() string {
	switch s {
	case StateStart:
		return "StateStart"
	case StateInObject:
		return "StateInObject"
	case StateInObjectKey:
		return "StateInObjectKey"
	case StateInString:
		return "StateInString"
	case StateNewline:
		return "StateNewline"
	case StateTab:
		return "StateTab"
	case StateSpace:
		return "StateSpace"
	case StateInInt:
		return "StateInInt"
	case StateInFloat:
		return "StateInFloat"
	case StateInColon:
		return "StateInColon"
	case StateInBool:
		return "StateInBool"
	case StateInNull:
		return "StateInNull"
	case StateInArray:
		return "StateInArray"
	case StateEnd:
		return "StateEnd"
	case StateInComma:
		return "StateInComma"
	case StateInObjectKeyEnd:
		return "StateInObjectKeyEnd"
	case StateTerminate:
		return "StateTerminate"
	case StateInStringEnd:
		return "StateInStringEnd"
	default:
		return fmt.Sprintf("Unknown state: %d", s)
	}
}

type JSONSampler struct {
	curNode *Node
	proc    model.TextProcessor
	stack   []*Node
}

func NewJSONSampler(proc model.TextProcessor) (*JSONSampler, error) {
	// fmt.Println("Creating new JSON sampler")
	startNode, err := buildStateMachine(proc)
	if err != nil {
		return nil, err
	}
	js := &JSONSampler{
		curNode: startNode,
		proc:    proc,
	}

	return js, nil
}

func (s *JSONSampler) UpdateState(tokenSlice []int32) error {
	// fmt.Printf("Updating state with token: %v\n", tokenSlice)
	// fmt.Printf("Current state: %s\n", s.curNode.State)

	// fmt.Println("tokenSlice", tokenSlice)
	// todo: account for strings here
	for node, edge := range s.curNode.TransitionEdges {
		for _, validToken := range edge {
			if slices.Equal(tokenSlice, validToken) {
				s.curNode = node
				// fmt.Printf("Transitioned to state: %s\n", node.State)
				return nil
			}
		}
	}
	for node, edge := range s.curNode.TransitionEdges {
		for _, validToken := range edge {
			if len(validToken) == 1 && validToken[0] == -1 || validToken[0] == -2 {
				s.curNode = node
				// fmt.Printf("Accepting any token, staying in state: %s\n", node.State)
				return nil
			}
		}
	}
	fmt.Println("invalid token ", tokenSlice)
	return errors.New("invalid token")
}

func (s *JSONSampler) Sample(logits []float64) ([]float64, error) {
	fmt.Printf("Sampling in state: %s\n", s.curNode.State)
	var err error

	switch s.curNode.State {
	case StateTerminate:
		for i := range logits {
			if s.proc.Is(uint32(i), model.SpecialEOS) {
				logits[i] = 1.0
			} else {
				logits[i] = math.NaN()
			}
		}
		return logits, nil

	case StateInInt:
		validStates := []int32{}
		minus, err := s.proc.Encode("-")
		if err != nil {
			return nil, err
		}
		digits := make([][]int32, 10)
		for i := 0; i < 10; i++ {
			digits[i], err = s.proc.Encode(fmt.Sprintf("%d", i))
			if err != nil {
				return nil, err
			}
		}
		// Allow "-" and digits 0-9 at start
		for i := range logits {
			for _, d := range digits {
				if len(d) == 1 && int32(i) == d[0] {
					validStates = append(validStates, int32(i))
				}
			}
			if len(minus) == 1 && int32(i) == minus[0] {
				validStates = append(validStates, int32(i))
			}
		}
		return logits, nil

	default:
		validStates := getValidStates(s.curNode)
		logits, err = s.maskLogits(logits, validStates)
		if err != nil {
			return nil, err
		}
		return logits, nil
	}
}

func getValidStates(node *Node) []int32 {
	validStates := []int32{}
	for _, edge := range node.TransitionEdges {
		for _, token := range edge {
			validStates = append(validStates, token...)
		}
	}
	return validStates
}

func (s *JSONSampler) maskLogits(logits []float64, validStates []int32) ([]float64, error) {
	// fmt.Printf("Masking logits with valid states: %v\n", validStates)
	for i := range logits {
		isValid := false
		for _, token := range validStates {
			if token == -1 {
				// fmt.Println("Found sentinel token, returning unmasked logits")
				return logits, nil
			}
			if i == int(token) {
				// fmt.Printf("Found valid token: %d\n", token)
				isValid = true
				break
			}
		}
		if !isValid {
			logits[i] = math.NaN()
		}
	}
	return logits, nil
}
