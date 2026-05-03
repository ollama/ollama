package cmputil

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareBool(t *testing.T) {
	testCases := []struct {
		a        bool
		b        bool
		expected int
	}{
		{true, true, 0},
		{false, false, 0},
		{true, false, -1},
		{false, true, 1},
	}

	for _, tc := range testCases {
		require.Equal(t, tc.expected, CompareBool(tc.a, tc.b))
	}
}
