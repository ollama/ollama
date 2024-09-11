package main

import (
	"testing"
	"time"
)

func TestCountCommon(t *testing.T) {
	tests := []struct {
		name     string
		t1       []int
		t2       []int
		expected int
	}{
		{
			name:     "Equal",
			t1:       []int{1, 2, 3},
			t2:       []int{1, 2, 3},
			expected: 3,
		},
		{
			name:     "Prefix",
			t1:       []int{1},
			t2:       []int{1, 2, 3},
			expected: 1,
		},
		{
			name:     "Empty",
			t1:       []int{},
			t2:       []int{1, 2, 3},
			expected: 0,
		},
		{
			name:     "Both Empty",
			t1:       []int{},
			t2:       []int{},
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
		cache       TokenCache
		prompt      []int
		expected    int
		expectedLen int
	}{
		{
			name: "Empty",
			cache: TokenCache{slots: []TokenCacheSlot{
				{
					id:       0,
					tokens:   []int{},
					inUse:    false,
					lastUsed: time.Time{},
				},
				{
					id:       1,
					tokens:   []int{},
					inUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:      []int{1},
			expected:    0,
			expectedLen: 0,
		},
		{
			name: "Extend",
			cache: TokenCache{slots: []TokenCacheSlot{
				{
					id:       0,
					tokens:   []int{1},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					tokens:   []int{1, 2},
					inUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []int{1, 2},
			expected:    1,
			expectedLen: 2,
		},
		{
			name: "New",
			cache: TokenCache{slots: []TokenCacheSlot{
				{
					id:       0,
					tokens:   []int{1, 2},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					tokens:   []int{},
					inUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:      []int{2},
			expected:    1,
			expectedLen: 0,
		},
		{
			name: "Fork",
			cache: TokenCache{
				slots: []TokenCacheSlot{
					{
						id:       0,
						tokens:   []int{1, 2},
						inUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						id:       1,
						tokens:   []int{},
						inUse:    false,
						lastUsed: time.Time{},
					},
				},
			},
			prompt:      []int{1},
			expected:    1,
			expectedLen: 1,
		},
		{
			name: "Evict",
			cache: TokenCache{slots: []TokenCacheSlot{
				{
					id:       0,
					tokens:   []int{1},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					tokens:   []int{1, 2},
					inUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []int{2, 3},
			expected:    1,
			expectedLen: 0,
		},
		{
			name: "In use",
			cache: TokenCache{slots: []TokenCacheSlot{
				{
					id:       0,
					tokens:   []int{1},
					inUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					id:       1,
					tokens:   []int{1, 2},
					inUse:    true,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:      []int{1, 2},
			expected:    0,
			expectedLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultLen := tt.cache.findCacheSlot(tt.prompt)
			if result.id != tt.expected || resultLen != tt.expectedLen {
				t.Errorf("findCacheSlot: slot have %v, want %v len have %v, want %v",
					result.id, tt.expected, resultLen, tt.expectedLen)
			}
		})
	}
}
