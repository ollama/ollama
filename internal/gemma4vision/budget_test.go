package gemma4vision

import "testing"

func TestSnapGemma4VisualTokens(t *testing.T) {
	tests := []struct {
		in   int
		want int
	}{
		{69, 70},
		{70, 70},
		{105, 70}, // tie 70 vs 140 → lower
		{106, 140},
		{200, 140},
		{1120, 1120},
		{2000, 1120},
		{0, DefaultMinVisualTokens},
		{-5, DefaultMinVisualTokens},
	}
	for _, tt := range tests {
		if got := SnapGemma4VisualTokens(tt.in); got != tt.want {
			t.Errorf("SnapGemma4VisualTokens(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestNormalizeGemma4ImageBudgets(t *testing.T) {
	min, max := NormalizeGemma4ImageBudgets(0, 0)
	if min != DefaultMinVisualTokens || max != DefaultMaxVisualTokens {
		t.Fatalf("defaults: got (%d,%d)", min, max)
	}
	min, max = NormalizeGemma4ImageBudgets(100, 200)
	if min != 70 || max != 140 {
		t.Fatalf("100,200: got (%d,%d) want (70,140)", min, max)
	}
	// min > max after snap: max raised
	min, max = NormalizeGemma4ImageBudgets(560, 70)
	if min != 560 || max != 560 {
		t.Fatalf("560,70: got (%d,%d) want (560,560)", min, max)
	}
}
