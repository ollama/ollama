package mlxrunner

import "testing"

func TestEffectiveContextLength(t *testing.T) {
	for _, tt := range []struct {
		name      string
		requested int
		maximum   int
		want      int
	}{
		{name: "unset uses model maximum", maximum: 262144, want: 262144},
		{name: "smaller request is enforced", requested: 65536, maximum: 262144, want: 65536},
		{name: "native maximum is accepted", requested: 262144, maximum: 262144, want: 262144},
		{name: "unsupported extension is capped", requested: 524288, maximum: 262144, want: 262144},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := effectiveContextLength(tt.requested, tt.maximum); got != tt.want {
				t.Fatalf("effectiveContextLength(%d, %d) = %d, want %d", tt.requested, tt.maximum, got, tt.want)
			}
		})
	}
}
