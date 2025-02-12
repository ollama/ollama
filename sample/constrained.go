package sample

import (
	"github.com/ollama/ollama/model"
)

type ConstrainedSampler struct {
	schema        *Schema
	propIdx       int
	propToNodeMap map[string]*PDA
	pdaSampler    *PushdownSampler
	decodedToks   []string
}

func NewConstrainedSampler(proc model.TextProcessor, schema *Schema) (*ConstrainedSampler, error) {
	pdaSampler, err := NewPushdownSampler(proc)
	if err != nil {
		return nil, err
	}

	// if schema == nil {
	return &ConstrainedSampler{
		schema:        nil,
		propIdx:       -1,
		propToNodeMap: nil,
		pdaSampler:    pdaSampler,
	}, nil

}

func (s *ConstrainedSampler) Apply(logits []float64) ([]float64, error) {
	if s.schema == nil {
		return s.pdaSampler.Apply(logits)
	}

	return nil, nil
}

func (s *ConstrainedSampler) UpdateState(tokenSlice []int32) error {
	if err := s.pdaSampler.UpdateState(tokenSlice); err != nil {
		return err
	}

	if s.schema == nil {
		return nil
	}

	return nil
}
