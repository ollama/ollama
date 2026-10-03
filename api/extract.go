package api

import (
	"fmt"
	"math"
	"strings"
	"time"
	"unicode/utf8"
)

// ExtractRequest finds entities of the supplied types in text.
type ExtractRequest struct {
	Model     string    `json:"model"`
	Input     string    `json:"input"`
	Labels    []string  `json:"labels"`
	Threshold *float32  `json:"threshold,omitempty"`
	KeepAlive *Duration `json:"keep_alive,omitempty"`
}

// Entity offsets count Unicode code points; End is exclusive.
type Entity struct {
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"`
	End   int     `json:"end"`
	Score float32 `json:"score"`
}

type ExtractResponse struct {
	Model           string        `json:"model"`
	Entities        []Entity      `json:"entities"`
	TotalDuration   time.Duration `json:"total_duration,omitempty"`
	LoadDuration    time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
}

func (r ExtractRequest) ScoreThreshold() float32 {
	if r.Threshold == nil {
		return 0.5
	}
	return *r.Threshold
}

// Validate bounds request work before tokenization or model loading.
func (r ExtractRequest) Validate() error {
	if !utf8.ValidString(r.Input) {
		return fmt.Errorf("input must be valid UTF-8")
	}
	if len(r.Input) > 1<<20 {
		return fmt.Errorf("input exceeds 1 MiB")
	}
	if len(r.Labels) == 0 || len(r.Labels) > 128 {
		return fmt.Errorf("labels must contain between 1 and 128 entity types")
	}
	seen := make(map[string]bool, len(r.Labels))
	for _, label := range r.Labels {
		if !utf8.ValidString(label) || strings.TrimSpace(label) == "" || len(label) > 256 {
			return fmt.Errorf("labels must be nonempty UTF-8 strings of at most 256 bytes")
		}
		if seen[label] {
			return fmt.Errorf("duplicate label: %q", label)
		}
		seen[label] = true
	}
	t := r.ScoreThreshold()
	if math.IsNaN(float64(t)) || t < 0 || t > 1 {
		return fmt.Errorf("threshold must be between 0 and 1")
	}
	return nil
}
