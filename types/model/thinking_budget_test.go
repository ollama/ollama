package model

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestThinkBudgetTokens(t *testing.T) {
	for _, tt := range []struct {
		name   string
		budget *ThinkBudget
		window int
		want   int
	}{
		{"nil", nil, 32768, 0},
		{"count", &ThinkBudget{Value: 4096}, 32768, 4096},
		{"count ignores the window", &ThinkBudget{Value: 4096}, 0, 4096},
		{"minimal", &ThinkBudget{Value: "minimal"}, 32768, 2048},
		{"low", &ThinkBudget{Value: "low"}, 32768, 4096},
		{"medium", &ThinkBudget{Value: "medium"}, 32768, 8192},
		{"high", &ThinkBudget{Value: "high"}, 32768, 16384},
		{"max", &ThinkBudget{Value: "max"}, 32768, 26214},
		{"xhigh is max", &ThinkBudget{Value: "xhigh"}, 32768, 26214},
		{"share without a window", &ThinkBudget{Value: "medium"}, 0, 0},
		{"unknown word", &ThinkBudget{Value: "exhaustive"}, 32768, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.budget.Tokens(tt.window); got != tt.want {
				t.Fatalf("Tokens(%d) = %d, want %d", tt.window, got, tt.want)
			}
		})
	}

	for share := range thinkBudgetFraction {
		if !slices.Contains(thinkBudgetShares, share) {
			t.Errorf("fraction for an unlisted share %q", share)
		}
	}
}

func TestThinkBudgetJSON(t *testing.T) {
	for _, tt := range []struct {
		in      string
		want    any
		wantErr bool
	}{
		{`4096`, 4096, false},
		{`"4096"`, 4096, false},
		{`"medium"`, "medium", false},
		{`"xhigh"`, "xhigh", false},
		{`null`, nil, false},
		{`0`, nil, true},
		{`-3`, nil, true},
		{`0.5`, nil, true},
		{`"0"`, nil, true},
		{`"exhaustive"`, nil, true},
		{`true`, nil, true},
		{`{}`, nil, true},
	} {
		t.Run(tt.in, func(t *testing.T) {
			var budget ThinkBudget
			err := json.Unmarshal([]byte(tt.in), &budget)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Unmarshal(%s) error = %v, wantErr %v", tt.in, err, tt.wantErr)
			}
			if err == nil && budget.Value != tt.want {
				t.Fatalf("Unmarshal(%s) = %#v, want %#v", tt.in, budget.Value, tt.want)
			}
		})
	}

	out, err := json.Marshal(&ThinkBudget{Value: "medium"})
	if err != nil || string(out) != `"medium"` {
		t.Fatalf("Marshal = %s, %v", out, err)
	}
	out, err = json.Marshal(&ThinkBudget{Value: 4096})
	if err != nil || string(out) != `4096` {
		t.Fatalf("Marshal = %s, %v", out, err)
	}
}

// A thinking level is the model's control and never a bound: the two types do
// not convert into each other, and a level has no way to yield a token count.
func TestThinkLevelCarriesNoBudget(t *testing.T) {
	for _, level := range []string{"low", "medium", "high", "max", "exhaustive"} {
		think := &ThinkValue{Value: level}
		if !think.IsValid() || !think.Bool() || think.String() != level {
			t.Errorf("level %q: valid=%v bool=%v string=%q", level, think.IsValid(), think.Bool(), think.String())
		}
	}
	if (&ThinkValue{Value: 4096}).IsValid() {
		t.Error("a number is not a think value; a token budget is think_budget")
	}
}
