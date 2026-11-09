package d1

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
)

// calibration is the part of a checkpoint's calibration.json that noul,
// choice and score answers use.
type calibration struct {
	Default float64 // softmax temperature
}

// loadCalibration parses calibration.json. A missing "default" is 1.
func loadCalibration(data []byte) (calibration, error) {
	var file struct {
		Default *float64 `json:"default"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return calibration{}, fmt.Errorf("d1 calibration: %w", err)
	}
	c := calibration{Default: 1}
	if file.Default != nil {
		c.Default = *file.Default
	}
	return c, nil
}

// softmax at temperature t over float64 scores (max-subtracted).
func softmax(scores []float64, t float64) []float64 {
	if len(scores) == 0 {
		return nil
	}
	s := scores
	if t != 1 {
		s = make([]float64, len(scores))
		for i, x := range scores {
			s[i] = x / t
		}
	}
	m := slices.Max(s)
	p := make([]float64, len(s))
	var total float64
	for i, x := range s {
		p[i] = math.Exp(x - m)
		total += p[i]
	}
	for i := range p {
		p[i] /= total
	}
	return p
}
