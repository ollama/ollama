package newrunner

import (
	"testing"
	"time"
)

func TestCountCommon(t *testing.T) {
	tests := []struct {
		name     string
		t1       []input
		t2       []input
		expected int
	}{
		{
			name:     "Equal",
			t1:       []input{{token: 1}, {token: 2}, {token: 3}},
			t2:       []input{{token: 1}, {token: 2}, {token: 3}},
			expected: 3,
		},
		{
			name:     "Prefix",
			t1:       []input{{token: 1}},
			t2:       []input{{token: 1}, {token: 2}, {token: 3}},
			expected: 1,
		},
		{
			name:     "Embeddings Prefix",
			t1:       []input{{embed: []float32{0.1, 0.2, 0.3}}},
			t2:       []input{{embed: []float32{0.1, 0.2, 0.3}}, {embed: []float32{0.4, 0.5, 0.6}}, {embed: []float32{0.7}}},
			expected: 1,
		},
		{
			name:     "Embeddings Prefix Partial",
			t1:       []input{{embed: []float32{0.1, 0.2, 0.3}}},
			t2:       []input{{embed: []float32{0.1, 0.2}}, {embed: []float32{0.4, 0.5, 0.6}}, {embed: []float32{0.7}}},
			expected: 0,
		},
		{
			name:     "Mixed",
			t1:       []input{{token: 1}, {embed: []float32{0.2, 0.3, 0.4}}},
			t2:       []input{{token: 1}, {embed: []float32{0.2, 0.3, 0.4}}, {token: 5}},
			expected: 2,
		},
		{
			name:     "Empty",
			t1:       []input{},
			t2:       []input{{token: 1}, {token: 2}, {token: 3}},
			expected: 0,
		},
		{
			name:     "Both Empty",
			t1:       []input{},
			t2:       []input{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := countCommonPrefix(tt.t1, tt.t2)
			if result != tt.expected {
				t.Errorf("countCommonPrefix(%v, %v): have %v; want %v", tt.t1, tt.t2, result, tt.expected)
			}
		})
	}
}

func TestFindCacheSlot(t *testing.T) {
	type expected struct {
		result int
		len    int
	}

	tests := []struct {
		name    string
		cache   InputCache
		prompt  []input
		longest expected
		best    expected
	}{
		{
			name: "Empty",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
				{
					Id:       1,
					Inputs:   []input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:  []input{{token: 1}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 0, len: 0},
		},
		{
			name: "Extend",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []input{{token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []input{{token: 1}, {token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []input{{token: 1}, {token: 2}},
			longest: expected{result: 1, len: 2},
			best:    expected{result: 1, len: 2},
		},
		{
			name: "New",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []input{{token: 1}, {token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:  []input{{token: 2}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 1, len: 0},
		},
		{
			name: "Fork",
			cache: InputCache{
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []input{{token: 1}, {token: 2}},
						InUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						Id:       1,
						Inputs:   []input{},
						InUse:    false,
						lastUsed: time.Time{},
					},
				},
			},
			prompt:  []input{{token: 1}},
			longest: expected{result: 0, len: 1},
			best:    expected{result: 1, len: 1},
		},
		{
			name: "Evict",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []input{{token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []input{{token: 1}, {token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []input{{token: 2}, {token: 3}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 1, len: 0},
		},
		{
			name: "In use",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []input{{token: 1}, {token: 2}},
					InUse:    true,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []input{{token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []input{{token: 1}, {token: 2}},
			longest: expected{result: 1, len: 1},
			best:    expected{result: 1, len: 2},
		},
	}

	for _, tt := range tests {
		t.Run("Longest-"+tt.name, func(t *testing.T) {
			result, resultLen, err := tt.cache.findLongestCacheSlot(tt.prompt)
			if err != nil {
				t.Errorf("findLongestCacheSlot: err %v", err)
			} else if result.Id != tt.longest.result || resultLen != tt.longest.len {
				t.Errorf("findLongestCacheSlot: slot have %v, want %v len have %v, want %v",
					result.Id, tt.longest.result, resultLen, tt.longest.len)
			}
		})
	}

	for _, tt := range tests {
		t.Run("Best-"+tt.name, func(t *testing.T) {
			result, resultLen, err := tt.cache.findBestCacheSlot(tt.prompt)
			if err != nil {
				t.Errorf("findBestCacheSlot: err %v", err)
			} else if result.Id != tt.best.result || resultLen != tt.best.len {
				t.Errorf("findBestCacheSlot: slot have %v, want %v len have %v, want %v",
					result.Id, tt.best.result, resultLen, tt.best.len)
			}
		})
	}
}

func TestShiftDiscard(t *testing.T) {
	tests := []struct {
		name     string
		numCtx   int
		numKeep  int
		inputLen int
		expected int
	}{
		{
			name:     "Shift",
			numCtx:   2048,
			numKeep:  5,
			inputLen: 2048,
			expected: 1021,
		},
		{
			name:     "Max Keep",
			numCtx:   2048,
			numKeep:  2047,
			inputLen: 2048,
			expected: 1,
		},
		{
			name:     "No Keep",
			numCtx:   2048,
			numKeep:  0,
			inputLen: 2048,
			expected: 1024,
		},
		{
			name:     "Truncate",
			numCtx:   2048,
			numKeep:  5,
			inputLen: 5000,
			expected: 3973,
		},
		{
			name:     "Truncate Keep",
			numCtx:   2048,
			numKeep:  2047,
			inputLen: 5000,
			expected: 2953,
		},
		{
			name:     "No Op",
			numCtx:   2048,
			numKeep:  5,
			inputLen: 512,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := InputCache{numCtx: tt.numCtx}
			result := c.ShiftDiscard(tt.inputLen, tt.numKeep)
			if result != tt.expected {
				t.Errorf("shiftDiscard(ctx: %v, keep: %v input: %v): have %v; want %v", tt.numCtx, tt.numKeep, tt.inputLen, result, tt.expected)
			}
		})
	}
}
