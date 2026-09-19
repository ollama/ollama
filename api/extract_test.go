package api

import (
	"math"
	"strings"
	"testing"
)

func TestExtractValidation(t *testing.T) {
	valid := ExtractRequest{Input: "", Labels: []string{"person"}}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	if valid.ScoreThreshold() != .5 {
		t.Fatal("default threshold")
	}
	for _, r := range []ExtractRequest{
		{Labels: nil}, {Labels: []string{" "}}, {Labels: []string{"person", "person"}},
		{Labels: []string{strings.Repeat("x", 257)}}, {Labels: make([]string, 129)},
		{Input: strings.Repeat("x", 1<<20+1), Labels: valid.Labels},
		{Input: string([]byte{0xff}), Labels: valid.Labels},
	} {
		if err := r.Validate(); err == nil {
			t.Fatal("accepted invalid request")
		}
	}
	for _, threshold := range []float32{-1, 1.1, float32(math.NaN()), float32(math.Inf(1))} {
		r := valid
		r.Threshold = &threshold
		if err := r.Validate(); err == nil {
			t.Fatalf("accepted %v", threshold)
		}
	}
}
