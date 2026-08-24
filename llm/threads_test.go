package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestNumThreadsForQuota(t *testing.T) {
	cases := []struct {
		name      string
		available int
		quota     int
		want      int
	}{
		{"no quota clamps to available", 8, 0, 8},
		{"quota below available clamps down", 8, 4, 4},
		{"quota above available keeps available", 4, 8, 4},
		{"quota equal to available", 4, 4, 4},
		{"never returns less than one", 1, 0, 1},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := numThreadsForQuota(tt.available, tt.quota); got != tt.want {
				t.Errorf("numThreadsForQuota(%d, %d) = %d, want %d", tt.available, tt.quota, got, tt.want)
			}
		})
	}
}

func TestAppendThreadArgsExplicit(t *testing.T) {
	opts := api.DefaultOptions()
	opts.NumThread = 6
	got := appendThreadArgs(nil, opts)
	want := []string{"-t", "6"}
	if !slices.Equal(got, want) {
		t.Errorf("appendThreadArgs with explicit NumThread = %v, want %v", got, want)
	}
}

func TestAppendThreadArgsDefaultAlwaysEmitsFlag(t *testing.T) {
	// Regression guard: previously NumThread == 0 meant no -t flag was ever
	// passed, leaving llama-server to auto-detect from the raw host core
	// count regardless of cgroup CPU limits. The derived default must
	// always be a positive thread count, so -t must always be present now.
	opts := api.DefaultOptions()
	opts.NumThread = 0
	got := appendThreadArgs(nil, opts)
	if len(got) != 2 || got[0] != "-t" {
		t.Fatalf("appendThreadArgs with NumThread == 0 = %v, want [-t <n>]", got)
	}
}
