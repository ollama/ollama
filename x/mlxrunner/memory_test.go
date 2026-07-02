package mlxrunner

import "testing"

const gib = 1 << 30

func TestPlanWired(t *testing.T) {
	if got, want := planWired(64*gib), 64*gib*4/5; got != want {
		t.Errorf("planWired(64 GiB) = %d, want %d", got, want)
	}
}

func TestPlanCache(t *testing.T) {
	cases := []struct {
		name            string
		modelSize, free int
		want            int
	}{
		// free=16 GiB -> wired cap 12.8 GiB.
		{"model fits leaves default", 10 * gib, 16 * gib, 0},
		{"model at cap leaves default", 12 * gib, 16 * gib, 0},
		{"model overflows tightens", 14 * gib, 16 * gib, 16 * gib / 5},
		{"unknown free leaves default", 14 * gib, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := planCache(tc.modelSize, tc.free); got != tc.want {
				t.Errorf("planCache(%d, %d) = %d, want %d", tc.modelSize, tc.free, got, tc.want)
			}
		})
	}
}
