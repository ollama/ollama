package backoff

import (
	"testing"
)

func TestLoopAllocs(t *testing.T) {
	for i := range 3 {
		got := testing.AllocsPerRun(1000, func() {
			for tick := range Loop(t.Context(), 1) {
				if tick >= i {
					break
				}
			}
		})
		want := float64(0)
		if i > 0 {
			want = 3 // due to time.NewTimer
		}
		if got > want {
			t.Errorf("[%d ticks]: allocs = %v, want 0", i, want)
		}
	}
}
