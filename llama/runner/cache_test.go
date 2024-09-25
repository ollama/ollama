package main

import (
	"reflect"
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

func TestImageCache(t *testing.T) {
	cache := NewInputCache(nil, 2048, 4, false)

	valA := [][]float32{{0.1, 0.2}, {0.3}}
	valB := [][]float32{{0.4}, {0.5}, {0.6}}
	valC := [][]float32{{0.7}}
	valD := [][]float32{{0.8}}
	valE := [][]float32{{0.9}}

	// Empty cache
	result, err := cache.FindImage(0x5adb61d31933a946)
	if err != ErrImageNotFound {
		t.Errorf("found result in empty cache: result %v, err %v", result, err)
	}

	// Insert A
	cache.AddImage(0x5adb61d31933a946, valA)

	result, err = cache.FindImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Insert B
	cache.AddImage(0x011551369a34a901, valB)

	result, err = cache.FindImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Replace B with C
	cache.AddImage(0x011551369a34a901, valC)

	result, err = cache.FindImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Evict A
	cache.AddImage(0x756b218a517e7353, valB)
	cache.AddImage(0x75e5e8d35d7e3967, valD)
	cache.AddImage(0xd96f7f268ca0646e, valE)

	result, err = cache.FindImage(0x5adb61d31933a946)
	if reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0x756b218a517e7353)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0x75e5e8d35d7e3967)
	if !reflect.DeepEqual(result, valD) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.FindImage(0xd96f7f268ca0646e)
	if !reflect.DeepEqual(result, valE) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
}
