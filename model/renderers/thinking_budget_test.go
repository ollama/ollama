package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestResolveThinkingTreatsABudgetAsThinkingOn(t *testing.T) {
	budget := &api.ThinkValue{Value: 4096}
	for _, tt := range []struct {
		name     string
		thinking *model.Thinking
		want     any
	}{
		{"states true", &model.Thinking{Values: []any{false, true}, Default: false}, true},
		{"named levels, default thinks", &model.Thinking{Values: []any{false, "low", "high"}, Default: "high"}, "high"},
		{"named levels, default off", &model.Thinking{Values: []any{false, "low", "high"}, Default: false}, "low"},
		{"cannot think", &model.Thinking{Values: []any{false}, Default: false}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := ResolveThinking(budget, tt.thinking); got == nil || got.Value != tt.want {
				t.Fatalf("ResolveThinking(4096) = %#v, want %#v", got, tt.want)
			}
		})
	}
	if got := ResolveThinking(budget, nil); got != budget {
		t.Fatalf("unknown metadata must keep the budget, got %#v", got)
	}
}
