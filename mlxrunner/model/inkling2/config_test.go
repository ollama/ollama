package inkling2

import "testing"

func TestLoopedExecutionPlan(t *testing.T) {
	config := textConfig{
		NumLayers:               24,
		ResidualBlockSize:       6,
		NormalizeResidualValues: true,
		Loop:                    &loopConfig{Start: 6, End: 18, Passes: 3, GateRank: 64},
	}
	for i := range config.NumLayers {
		if i%6 != 5 {
			config.LocalLayers = append(config.LocalLayers, i)
		}
	}
	plan, err := config.executionPlan()
	if err != nil {
		t.Fatal(err)
	}
	if len(plan) != 48 {
		t.Fatalf("decoder visits = %d, want 48", len(plan))
	}
	for _, span := range []struct{ firstVisit, firstLayer, length, pass int }{
		{0, 0, 18, 0},
		{18, 6, 12, 1},
		{30, 6, 12, 2},
		{42, 18, 6, 0},
	} {
		for offset := range span.length {
			index := span.firstVisit + offset
			layer := span.firstLayer + offset
			want := layerVisit{layer: layer, pass: span.pass, local: layer%6 != 5}
			if got := plan[index]; got != want {
				t.Errorf("decoder visit %d = %+v, want %+v", index, got, want)
			}
		}
	}
}

func TestExecutionPlanRejectsLoopInsideResidualBlock(t *testing.T) {
	config := textConfig{
		NumLayers:               24,
		ResidualBlockSize:       6,
		NormalizeResidualValues: true,
		Loop:                    &loopConfig{Start: 7, End: 18, Passes: 3, GateRank: 64},
	}
	if _, err := config.executionPlan(); err == nil {
		t.Fatal("expected an error for a loop starting inside a residual block")
	}
}

func TestExecutionPlanRejectsVariableLoop(t *testing.T) {
	cfg := textConfig{
		NumLayers: 4, ResidualBlockSize: 2, NormalizeResidualValues: true,
		Loop: &loopConfig{Start: 0, End: 2, Passes: 3, GateRank: 1, Variable: true},
	}
	if _, err := cfg.executionPlan(); err == nil {
		t.Fatal("accepted variable looping with a fixed execution plan")
	}
}

func TestExecutionPlanBoundsExpandedDepth(t *testing.T) {
	config := textConfig{
		NumLayers:               2,
		ResidualBlockSize:       1,
		NormalizeResidualValues: true,
		Loop:                    &loopConfig{Start: 0, End: 2, Passes: maxLayerCount / 2, GateRank: 1},
	}
	plan, err := config.executionPlan()
	if err != nil || len(plan) != maxLayerCount {
		t.Fatalf("plan at visit limit: %d visits, error %v", len(plan), err)
	}
	config.Loop.Passes++
	if _, err := config.executionPlan(); err == nil {
		t.Fatal("accepted a loop exceeding the visit limit")
	}
	config.Loop = nil
	config.NumLayers = maxLayerCount + 1
	if _, err := config.executionPlan(); err == nil {
		t.Fatal("accepted a decoder exceeding the layer limit")
	}
}
