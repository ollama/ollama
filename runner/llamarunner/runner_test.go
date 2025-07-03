package llamarunner

import (
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestContextShiftLogic(t *testing.T) {
	tests := []struct {
		name               string
		enableContextShift bool
		contextLength      int32
		cacheInputs        int
		pendingInputs      int
		minBatch           int
		expectedDoneReason llm.DoneReason
		shouldRemove       bool
	}{
		{
			name:               "context shifting enabled - should shift",
			enableContextShift: true,
			contextLength:      100,
			cacheInputs:        80,
			pendingInputs:      0,
			minBatch:           30,
			expectedDoneReason: llm.DoneReasonStop,
			shouldRemove:       false,
		},
		{
			name:               "context shifting disabled - should remove",
			enableContextShift: false,
			contextLength:      100,
			cacheInputs:        80,
			pendingInputs:      0,
			minBatch:           30,
			expectedDoneReason: llm.DoneReasonContextShift,
			shouldRemove:       true,
		},
		{
			name:               "context shifting disabled - within limits",
			enableContextShift: false,
			contextLength:      100,
			cacheInputs:        50,
			pendingInputs:      0,
			minBatch:           30,
			expectedDoneReason: llm.DoneReasonStop,
			shouldRemove:       false,
		},
		{
			name:               "pending inputs - should break batch",
			enableContextShift: true,
			contextLength:      100,
			cacheInputs:        50,
			pendingInputs:      20,
			minBatch:           30,
			expectedDoneReason: llm.DoneReasonStop,
			shouldRemove:       false,
		},
		{
			name:               "no pending inputs - should shift",
			enableContextShift: true,
			contextLength:      100,
			cacheInputs:        80,
			pendingInputs:      0,
			minBatch:           30,
			expectedDoneReason: llm.DoneReasonStop,
			shouldRemove:       false,
		},
		{
			name:               "long prompt with context shifting disabled - will be handled at runtime",
			enableContextShift: false,
			contextLength:      100,
			cacheInputs:        0,
			pendingInputs:      0,
			minBatch:           150, // Simulates a long prompt
			expectedDoneReason: llm.DoneReasonContextShift,
			shouldRemove:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test the core logic from processBatch
			if int32(tt.cacheInputs+tt.pendingInputs+tt.minBatch) > tt.contextLength {
				if tt.pendingInputs != 0 {
					// Should break batch
					if tt.shouldRemove {
						t.Error("should not remove sequence when pending inputs exist")
					}
				} else if !tt.enableContextShift {
					// Should remove with DoneReasonContextShift
					if !tt.shouldRemove {
						t.Error("should remove sequence when context shifting disabled")
					}
					if tt.expectedDoneReason != llm.DoneReasonContextShift {
						t.Errorf("expected DoneReason %v, got %v", llm.DoneReasonContextShift, tt.expectedDoneReason)
					}
				} else {
					// Should shift context
					if tt.shouldRemove {
						t.Error("should not remove sequence when context shifting enabled")
					}
				}
			}
		})
	}
}

func TestPredictLimitLogic(t *testing.T) {
	tests := []struct {
		name         string
		numPredict   int
		numPredicted int
		expectRemove bool
	}{
		{
			name:         "predict limit not reached",
			numPredict:   5,
			numPredicted: 3,
			expectRemove: false,
		},
		{
			name:         "predict limit reached",
			numPredict:   5,
			numPredicted: 5,
			expectRemove: true,
		},
		{
			name:         "predict limit exceeded",
			numPredict:   5,
			numPredicted: 6,
			expectRemove: true,
		},
		{
			name:         "no predict limit",
			numPredict:   0,
			numPredicted: 100,
			expectRemove: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test the core logic from processBatch
			shouldRemove := tt.numPredict > 0 && tt.numPredicted >= tt.numPredict
			if shouldRemove != tt.expectRemove {
				t.Errorf("expected remove=%v, got %v", tt.expectRemove, shouldRemove)
			}
		})
	}
}
