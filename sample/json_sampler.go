package sample

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/model"
)

type JSONState int

const (
	StateStart      JSONState = iota // Initial state
	StateInObject                    // Inside an object {}
	StateInArray                     // Inside an array []
	StateInString                    // Inside a string ""
	StateAfterKey                    // After object key, expecting :
	StateAfterColon                  // After :, expecting value
	StateAfterValue                  // After value, expecting , or closing bracket
	StateDone                        // JSON parsing complete
)

type JSONSampler struct {
	state JSONState
	stack []string
	proc  model.TextProcessor
}

func NewJSONSampler(proc model.TextProcessor) *JSONSampler {
	return &JSONSampler{
		state: StateStart,
		proc:  proc,
	}
}

func (s *JSONSampler) Sample(logits []float64) ([]float64, error) {
	// Pre-decode valid tokens for current state
	validTokens := make(map[uint32]bool)

	// Always allow EOS token in any state
	// TODO: Check for other special tokens if needed
	for i := range logits {
		if s.proc.Is(uint32(i), model.SpecialEOS) {
			validTokens[uint32(i)] = true
		}
	}

	// Build set of valid tokens based on current state
	switch s.state {
	case StateStart:
		// Only allow opening brace
		for i := range logits {
			text, err := s.proc.Decode([]int32{int32(i)})
			if err == nil && text == "{" {
				validTokens[uint32(i)] = true
			}
		}
	case StateInObject, StateInArray:
		// Allow any token
		for i := range logits {
			validTokens[uint32(i)] = true
		}
	case StateInString:
		// Allow any token except closing brace
		for i := range logits {
			text, err := s.proc.Decode([]int32{int32(i)})
			if err == nil && text != "}" {
				validTokens[uint32(i)] = true
			}
		}
	case StateDone:
		// No tokens allowed
	}

	// Mark invalid tokens as NaN in one pass
	for i := range logits {
		if !validTokens[uint32(i)] {
			logits[i] = math.NaN()
		}
	}
	return logits, nil
}

func (s *JSONSampler) UpdateState(tokenID int) error {
	text, err := s.proc.Decode([]int32{int32(tokenID)})
	if err != nil {
		return fmt.Errorf("failed to decode token: %w", err)
	}

	switch s.state {
	case StateStart:
		if text != "{" {
			return fmt.Errorf("expected {, got %s", text)
		}
		s.state = StateInObject
	case StateInObject:
		if text == "}" {
			s.state = StateDone
		}
	case StateDone:
		return fmt.Errorf("unexpected token after closing bracket: %s", text)
	}
	return nil
}
