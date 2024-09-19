package main

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
	tests := []struct {
		name        string
		cache       InputCache
		prompt      []input
		expected    int
		expectedLen int
	}{
		{
			name: "Empty",
			cache: InputCache{slots: []InputCacheSlot{
				{
					id:       0,
					inputs:   []input{},
					inUse:    false,
					lastUsed: time.Time{},
				},
				{
					id:       1,
					inputs:   []input{},
					inUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:      []input{{token: 1}},
			expected:    0,
			expectedLen: 0,
		},
		{
			name: "Extend",
			cache: InputCache{slots: []InputCacheSlot{
				{
					id:       0,
					inputs:   []input{{token: 1}},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					inputs:   []input{{token: 1}, {token: 2}},
					inUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []input{{token: 1}, {token: 2}},
			expected:    1,
			expectedLen: 2,
		},
		{
			name: "New",
			cache: InputCache{slots: []InputCacheSlot{
				{
					id:       0,
					inputs:   []input{{token: 1}, {token: 2}},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					inputs:   []input{},
					inUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:      []input{{token: 2}},
			expected:    1,
			expectedLen: 0,
		},
		{
			name: "Fork",
			cache: InputCache{
				slots: []InputCacheSlot{
					{
						id:       0,
						inputs:   []input{{token: 1}, {token: 2}},
						inUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						id:       1,
						inputs:   []input{},
						inUse:    false,
						lastUsed: time.Time{},
					},
				},
			},
			prompt:      []input{{token: 1}},
			expected:    1,
			expectedLen: 1,
		},
		{
			name: "Evict",
			cache: InputCache{slots: []InputCacheSlot{
				{
					id:       0,
					inputs:   []input{{token: 1}},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					inputs:   []input{{token: 1}, {token: 2}},
					inUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []input{{token: 2}, {token: 3}},
			expected:    1,
			expectedLen: 0,
		},
		{
			name: "In use",
			cache: InputCache{slots: []InputCacheSlot{
				{
					id:       0,
					inputs:   []input{{token: 1}},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					inputs:   []input{{token: 1}, {token: 2}},
					inUse:    true,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []input{{token: 1}, {token: 2}},
			expected:    0,
			expectedLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultLen, err := tt.cache.findCacheSlot(tt.prompt)
			if err != nil {
				t.Errorf("findCacheSlot: err %v", err)
			} else if result.id != tt.expected || resultLen != tt.expectedLen {
				t.Errorf("findCacheSlot: slot have %v, want %v len have %v, want %v",
					result.id, tt.expected, resultLen, tt.expectedLen)
			}
		})
	}
}
