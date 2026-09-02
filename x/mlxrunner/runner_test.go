package mlxrunner

import "testing"

func TestEffectiveContextLength(t *testing.T) {
	cases := []struct {
		name            string
		archMax, numCtx int
		want            int
	}{
		{"num_ctx unset uses arch max", 262144, 0, 262144},
		{"negative num_ctx uses arch max", 262144, -1, 262144},
		{"num_ctx smaller than arch max is enforced", 262144, 65536, 65536},
		{"num_ctx larger than arch max is clamped to arch max", 4096, 65536, 4096},
		{"num_ctx equal to arch max is a no-op", 65536, 65536, 65536},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := effectiveContextLength(tt.archMax, tt.numCtx); got != tt.want {
				t.Errorf("effectiveContextLength(%d, %d) = %d, want %d", tt.archMax, tt.numCtx, got, tt.want)
			}
		})
	}
}
