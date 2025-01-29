package sample

import (
	"errors"
	"fmt"
	"math"

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
	StateInColon
	StateInComma
	StateInTab
	StateInSpace
	StateInObjSpace
	StateInList
	StateInListComma
	StateListEnd
	StateInValue
	StateInValueEnd
	StateInListEnd
	StateInListObjectEnd
	StateInNewline
	StateInNumber
	StateInNumberEnd
	StateInStringEnd
	StateInObjectKeyEnd
	StateTerminate
	StateInObjectEnd
	StateTransitioningToTerminate
)

func (s JSONState) String() string {
	switch s {
	case StateStart:
		return "StateStart"
	case StateInObject:
		return "StateInObject"
	case StateInObjectKey:
		return "StateInObjectKey"
	case StateNewline:
		return "StateNewline"
	case StateTab:
		return "StateTab"
	case StateSpace:
		return "StateSpace"
	case StateInString:
		return "StateInString"
	case StateInInt:
		return "StateInInt"
	case StateInFloat:
		return "StateInFloat"
	case StateInBool:
		return "StateInBool"
	case StateInNull:
		return "StateInNull"
	case StateInColon:
		return "StateInColon"
	case StateInComma:
		return "StateInComma"
	case StateInTab:
		return "StateInTab"
	case StateInSpace:
		return "StateInSpace"
	case StateInObjSpace:
		return "StateInObjSpace"
	case StateInList:
		return "StateInList"
	case StateInListObjectEnd:
		return "StateInListObjectEnd"
	case StateInListComma:
		return "StateInListComma"
	case StateListEnd:
		return "StateListEnd"
	case StateInListEnd:
		return "StateInListEnd"
	case StateInNewline:
		return "StateInNewline"
	case StateInNumber:
		return "StateInNumber"
	case StateInNumberEnd:
		return "StateInNumberEnd"
	case StateInStringEnd:
		return "StateInStringEnd"
	case StateInObjectKeyEnd:
		return "StateInObjectKeyEnd"
	case StateTerminate:
		return "StateTerminate"
	case StateInObjectEnd:
		return "StateInObjectEnd"
	default:
		return fmt.Sprintf("Unknown state: %d", s)
	}
}

type JSONSampler struct {
	curNode        *Node
	proc           model.TextProcessor
	stack          []*Node
	bracketCounter int
}

func NewJSONSampler(proc model.TextProcessor) (*JSONSampler, error) {
	// fmt.Println("Creating new JSON sampler")
	startNode, err := buildStateMachine(proc)
	if err != nil {
		return nil, err
	}
	js := &JSONSampler{
		curNode:        startNode,
		proc:           proc,
		stack:          []*Node{},
		bracketCounter: 0,
	}

	return js, nil
}

func isTokenSubset(subset, superset []int32) bool {
	freq1 := make(map[int32]int)
	freq2 := make(map[int32]int)

	for _, v := range subset {
		freq1[v]++
	}
	for _, v := range superset {
		freq2[v]++
	}
	isSubset := true
	for k, count1 := range freq1 {
		count2 := freq2[k]
		if count1 > count2 {
			isSubset = false
			break
		}
	}
	return isSubset
}

func (s *JSONSampler) UpdateState(tokenSlice []int32) error {
	// fmt.Printf("Updating state with token: %v\n", tokenSlice)
	// fmt.Printf("Current state: %s\n", s.curNode.State)

	// fmt.Println("tokenSlice", tokenSlice)
	// todo: account for strings here

	objectTokens, err := ComputeTokenVariants([]string{"{", " {", "{\n", " {\n"}, s.proc)
	if err != nil {
		return err
	}

	// only move to terminate state if stack is empty
	if s.curNode.State == StateInObjectEnd {
		fmt.Println("debug: node.State", s.curNode.State)
		if len(s.stack) > 0 {
			s.stack = s.stack[:len(s.stack)-1]
			fmt.Println("popped and cur state", s.curNode.State)
			return nil
		}
		return nil
	}

	for node, edge := range s.curNode.TransitionEdges {
		for _, validToken := range edge {
			if isTokenSubset(tokenSlice, validToken) {
				s.curNode = node
				for _, token := range objectTokens {
					if isTokenSubset(tokenSlice, token) {
						fmt.Println("Appending to stack", s.curNode.State)
						s.stack = append(s.stack, s.curNode)
					}
				}
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
	dec, err := s.proc.Decode(tokenSlice)
	if err != nil {
		return err
	}
	fmt.Println("decoded token ", dec)
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

	case StateInString:
		penalizeNewlineVariants := []string{"\n", " \"\n"}
		penalizeNewlineToks, err := ComputeTokenVariants(penalizeNewlineVariants, s.proc)
		if err != nil {
			return nil, err
		}
		penalizeNewlineToks = append(penalizeNewlineToks, []int32{702})
		logits, err = s.maskSpecificLogits(logits, penalizeNewlineToks)
		if err != nil {
			return nil, err
		}
		validStates := getValidStates(s.curNode)
		logits, err = s.maskLogits(logits, validStates)
		if err != nil {
			return nil, err
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
	// todo: this can prob be more efficient
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

func (s *JSONSampler) maskSpecificLogits(logits []float64, tokensToMask []token) ([]float64, error) {
	// fmt.Printf("Masking specific logits: %v\n", tokensToMask)
	for i := range logits {
		for _, token := range tokensToMask {
			for _, chunked := range token {
				if int(chunked) == i {
					logits[i] = math.NaN()
				}
			}
		}
	}
	return logits, nil
}
