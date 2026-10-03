package server

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

// The MLX client copies api.Options wholesale, marshals it and the runner
// decodes it back into the same struct. A budget that cannot survive that trip
// would fail the request rather than be ignored.
func TestThinkBudgetSurvivesMLXOptionsRoundTrip(t *testing.T) {
	for _, v := range []any{8192, "medium", "max", true, false} {
		opts := api.Options{}
		opts.ThinkBudget = &api.ThinkValue{Value: v}
		opts.ThinkBudgetMessage = "\n\nwrap up now\n"
		opts.NumCtx = 32768

		b, err := json.Marshal(opts)
		if err != nil {
			t.Fatalf("marshal %v: %v", v, err)
		}
		var got api.Options
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal %v: %v (json=%s)", v, err, b)
		}
		if got.ThinkBudget == nil {
			t.Fatalf("%v: budget lost in transit", v)
		}
		if got.ThinkBudgetMessage != opts.ThinkBudgetMessage {
			t.Fatalf("%v: message lost", v)
		}
	}
}
