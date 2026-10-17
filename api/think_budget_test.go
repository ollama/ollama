package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestThinkBudgetOption(t *testing.T) {
	// think_budget is set as a model default with `PARAMETER think_budget`,
	// which reaches Options through the generic parameter map. It takes either
	// a token count or a share of the response.
	tests := []struct {
		name     string
		param    string // as written in a Modelfile
		fromMap  []any  // equivalent values arriving through FromMap
		expected any    // ThinkBudget.Value once decoded
		budget   int    // resolved against a 32768 token window
	}{
		{
			name:     "token count",
			param:    "8192",
			fromMap:  []any{int64(8192), float64(8192), "8192"},
			expected: 8192,
			budget:   8192,
		},
		{
			name:     "share",
			param:    "high",
			fromMap:  []any{"high"},
			expected: "high",
			budget:   16384,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := FormatParams(map[string][]string{"think_budget": {tt.param}})
			require.NoError(t, err)

			// what FormatParams produces must itself be decodable
			values := append([]any{params["think_budget"]}, tt.fromMap...)
			for _, value := range values {
				opts := DefaultOptions()
				require.NoError(t, opts.FromMap(map[string]any{"think_budget": value}), "value %#v", value)
				require.NotNil(t, opts.ThinkBudget)
				assert.Equal(t, tt.expected, opts.ThinkBudget.Value)
				assert.Equal(t, tt.budget, opts.ThinkBudget.Tokens(32768))
			}
		})
	}

	assert.Nil(t, DefaultOptions().ThinkBudget)
	assert.Equal(t, 0, DefaultOptions().ThinkBudget.Tokens(32768))

	// A word that names no share bounds nothing, so it is refused rather than
	// accepted and ignored.
	opts := DefaultOptions()
	assert.Error(t, opts.FromMap(map[string]any{"think_budget": "enormous"}))
	assert.Error(t, opts.FromMap(map[string]any{"think_budget": 0}))
	assert.Error(t, opts.FromMap(map[string]any{"think_budget": true}))
}

func TestThinkBudgetMessageOption(t *testing.T) {
	// think_budget_message ships with a model the same way the budget does, and
	// is free text: whatever wording works for that model.
	const message = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."

	params, err := FormatParams(map[string][]string{"think_budget_message": {message}})
	require.NoError(t, err)

	for _, value := range []any{params["think_budget_message"], message} {
		opts := DefaultOptions()
		require.NoError(t, opts.FromMap(map[string]any{"think_budget_message": value}), "value %#v", value)
		assert.Equal(t, message, opts.ThinkBudgetMessage)
	}

	// unset means the bare closing tag is forced, which is what every runner
	// does today
	assert.Empty(t, DefaultOptions().ThinkBudgetMessage)
}

func TestThinkBudgetWindow(t *testing.T) {
	tests := []struct {
		name       string
		numCtx     int
		numPredict int
		want       int
	}{
		{name: "no cap on the response", numCtx: 128000, numPredict: -1, want: 128000},
		{name: "unset num_predict", numCtx: 128000, numPredict: 0, want: 128000},
		// the case this exists for: a share resolved against the context would
		// have matched num_predict exactly and bounded nothing
		{name: "response capped below the context", numCtx: 128000, numPredict: 32000, want: 32000},
		{name: "response cap above the context", numCtx: 8192, numPredict: 32000, want: 8192},
		{name: "no context length known", numCtx: 0, numPredict: 4096, want: 4096},
		{name: "neither known", numCtx: 0, numPredict: 0, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ThinkBudgetWindow(tt.numCtx, tt.numPredict); got != tt.want {
				t.Errorf("ThinkBudgetWindow(%d, %d) = %d, want %d", tt.numCtx, tt.numPredict, got, tt.want)
			}
		})
	}
}

func TestThinkBudgetSharesTheResponseNotTheContext(t *testing.T) {
	// A client sends num_ctx 128000 with num_predict 32000, where "medium" as a
	// quarter of the context is exactly the response cap: the model can spend
	// every token it is allowed to emit inside the thinking block and stop with
	// no answer at all.
	const numCtx, numPredict = 128000, 32000

	budget := &ThinkBudget{Value: "medium"}
	if got := budget.Tokens(numCtx); got != numPredict {
		t.Fatalf("premise changed: a quarter of %d is %d, not the response cap %d", numCtx, got, numPredict)
	}

	got := budget.Tokens(ThinkBudgetWindow(numCtx, numPredict))
	if want := numPredict / 4; got != want {
		t.Errorf("budget = %d, want %d (a quarter of the response)", got, want)
	}
}

// The request's think value is the model's control and nothing else: a number
// there is refused, and a level decodes to itself with no bound attached.
func TestThinkIsNotABudget(t *testing.T) {
	var req ChatRequest
	assert.Error(t, json.Unmarshal([]byte(`{"think": 8192}`), &req))

	require.NoError(t, json.Unmarshal([]byte(`{"think": "high", "options": {"think_budget": 2000}}`), &req))
	assert.Equal(t, "high", req.Think.Value)

	opts := DefaultOptions()
	require.NoError(t, opts.FromMap(req.Options))
	assert.Equal(t, 2000, opts.ThinkBudget.Tokens(32000), "the budget is the option, whatever the level says")
}
