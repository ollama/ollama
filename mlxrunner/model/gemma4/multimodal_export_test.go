package gemma4

import "testing"

// VisionSoftTokenBudget: explicit override wins, otherwise the default 280.
func TestVisionSoftTokenBudgetFreeFunction(t *testing.T) {
	cfg := &VisionConfig{ModelType: "gemma4_vision"}
	if got := VisionSoftTokenBudget(0, cfg); got != 280 {
		t.Errorf("default: %d, want 280", got)
	}
	if got := VisionSoftTokenBudget(70, cfg); got != 70 {
		t.Errorf("override: %d, want 70", got)
	}
	if got := VisionSoftTokenBudget(1120, cfg); got != 1120 {
		t.Errorf("override: %d, want 1120", got)
	}
}

// ValidateVisionSoftTokenBudget accepts only the reference processor's set.
func TestValidateVisionSoftTokenBudget(t *testing.T) {
	for _, want := range []int32{70, 140, 280, 560, 1120} {
		if err := ValidateVisionSoftTokenBudget(want); err != nil {
			t.Errorf("budget %d rejected", want)
		}
	}
	for _, bad := range []int32{0, 71, 300, 1121} {
		if err := ValidateVisionSoftTokenBudget(bad); err == nil {
			t.Errorf("budget %d accepted", bad)
		}
	}
}
