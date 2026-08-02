package model

import (
	"slices"
	"testing"
)

func TestThinkLevelsAreIndependentOfBudgets(t *testing.T) {
	// A level a model understands does not have to carry a budget. Tying the
	// two together would reject any level that exists only to be handed to the
	// model.
	for _, level := range thinkLevels {
		think := &ThinkValue{Value: level}
		if !think.IsValid() || !think.Bool() || think.String() != level {
			t.Errorf("level %q: valid=%v bool=%v string=%q", level, think.IsValid(), think.Bool(), think.String())
		}
	}

	for level := range thinkBudgetFraction {
		if !slices.Contains(thinkLevels, level) {
			t.Errorf("budget fraction for an unknown level %q", level)
		}
	}

	budgetless := &ThinkValue{Value: "exhaustive"}
	if !budgetless.IsValid() || !budgetless.Bool() {
		t.Error("a model-defined level must be valid and request thinking")
	}
	if got := budgetless.Level(); got != "exhaustive" {
		t.Errorf("Level() = %q, the level still reaches the model", got)
	}
	if got := budgetless.BudgetTokens(32768); got != 0 {
		t.Errorf("BudgetTokens() = %d, a level without a share carries no budget", got)
	}
}
