// Copyright (c) Tailscale Inc & AUTHORS
// SPDX-License-Identifier: BSD-3-Clause

package stringsx

import (
	"cmp"
	"strings"
	"testing"
)

func TestCompareFold(t *testing.T) {
	tests := []struct {
		a, b string
	}{
		// Basic ASCII cases
		{"", ""},
		{"a", "a"},
		{"a", "A"},
		{"A", "a"},
		{"a", "b"},
		{"b", "a"},
		{"abc", "ABC"},
		{"ABC", "abc"},
		{"abc", "abd"},
		{"abd", "abc"},

		// Length differences
		{"abc", "ab"},
		{"ab", "abc"},

		// Unicode cases
		{"世界", "世界"},
		{"Hello世界", "hello世界"},
		{"世界Hello", "世界hello"},
		{"世界", "世界x"},
		{"世界x", "世界"},

		// Special case folding examples
		{"ß", "ss"},      // German sharp s
		{"ﬁ", "fi"},      // fi ligature
		{"Σ", "σ"},       // Greek sigma
		{"İ", "i\u0307"}, // Turkish dotted I

		// Mixed cases
		{"HelloWorld", "helloworld"},
		{"HELLOWORLD", "helloworld"},
		{"helloworld", "HELLOWORLD"},
		{"HelloWorld", "helloworld"},
		{"helloworld", "HelloWorld"},

		// Edge cases
		{" ", " "},
		{"1", "1"},
		{"123", "123"},
		{"!@#", "!@#"},
	}

	wants := []int{}
	for _, tt := range tests {
		got := CompareFold(tt.a, tt.b)
		want := cmp.Compare(strings.ToLower(tt.a), strings.ToLower(tt.b))
		if got != want {
			t.Errorf("CompareFold(%q, %q) = %v, want %v", tt.a, tt.b, got, want)
		}
		wants = append(wants, want)
	}

	if n := testing.AllocsPerRun(1000, func() {
		for i, tt := range tests {
			if CompareFold(tt.a, tt.b) != wants[i] {
				panic("unexpected")
			}
		}
	}); n > 0 {
		t.Errorf("allocs = %v; want 0", int(n))
	}
}
