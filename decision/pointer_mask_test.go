package decision

import "testing"

func TestPointerAllow(t *testing.T) {
	cases := []struct {
		name   string
		seg    []int32
		bidir  bool
		want   []int
	}{
		{
			name:  "bidir",
			seg:   []int32{0, 0, 0, 1, 1, 2, 2},
			bidir: true,
			want: []int{
				1, 1, 1, 0, 0, 0, 0,
				1, 1, 1, 0, 0, 0, 0,
				1, 1, 1, 0, 0, 0, 0,
				1, 1, 1, 1, 0, 0, 0,
				1, 1, 1, 1, 1, 0, 0,
				1, 1, 1, 0, 0, 1, 0,
				1, 1, 1, 0, 0, 1, 1,
			},
		},
		{
			name:  "causal_state",
			seg:   []int32{0, 0, 0, 1, 1},
			bidir: false,
			want: []int{
				1, 0, 0, 0, 0,
				1, 1, 0, 0, 0,
				1, 1, 1, 0, 0,
				1, 1, 1, 1, 0,
				1, 1, 1, 1, 1,
			},
		},
		{
			name:  "pad",
			seg:   []int32{0, 0, -1},
			bidir: true,
			want: []int{
				1, 1, 0,
				1, 1, 0,
				1, 1, 1,
			},
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			n := len(tt.seg)
			if len(tt.want) != n*n {
				t.Fatalf("want length %d, have %d", n*n, len(tt.want))
			}
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					got := PointerAllow(tt.seg, i, j, tt.bidir)
					exp := tt.want[i*n+j] != 0
					if got != exp {
						t.Fatalf("mask[%d][%d] = %v, want %v", i, j, got, exp)
					}
				}
			}
		})
	}
}
