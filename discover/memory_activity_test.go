//go:build linux || windows

package discover

import (
	"math"
	"testing"
)

func TestMemoryPressureFeedback(t *testing.T) {
	var history memoryHistory
	for _, tc := range []struct {
		name          string
		activity      memoryActivity
		reclaim       int64
		active, known bool
	}{
		{"existing compression is a baseline", memoryActivity{compressed: 100, compressions: 200, pageOut: 50}, 0, false, false},
		{"same observation cannot replay", memoryActivity{compressed: 100, compressions: 200, pageOut: 50}, 0, false, true},
		{"new compression", memoryActivity{compressed: 150, compressions: 210, pageOut: 50}, 50, true, true},
		{"overlapping paging", memoryActivity{compressed: 200, compressions: 220, pageOut: 90}, 50, true, true},
		{"compression shrinks", memoryActivity{compressed: 20, compressions: 220, pageOut: 90}, 0, false, true},
		{"compression grows again", memoryActivity{compressed: 30, compressions: 221, pageOut: 90}, 10, true, true},
		{"steady store still compresses", memoryActivity{compressed: 30, compressions: 230, pageOut: 90}, 0, true, true},
		{"counter reset establishes baseline", memoryActivity{compressed: 200, compressions: 1, pageOut: 0}, 0, false, false},
		{"paging after reset", memoryActivity{compressed: 200, compressions: 1, pageOut: 40}, 40, true, true},
		{"large feedback saturates", memoryActivity{compressed: math.MaxUint64, compressions: 1, pageOut: 40}, math.MaxInt64, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reclaim, active, known := history.observe(tc.activity)
			if reclaim != tc.reclaim || active != tc.active || known != tc.known {
				t.Fatalf("feedback (%d, %v, %v), want (%d, %v, %v)", reclaim, active, known, tc.reclaim, tc.active, tc.known)
			}
		})
	}
}
