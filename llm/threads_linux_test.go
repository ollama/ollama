package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCgroupCPUQuota(t *testing.T) {
	cases := []struct {
		name    string
		content string
		exists  bool
		want    int
	}{
		{"unlimited quota", "max 100000\n", true, 0},
		{"quota below one period", "400000 100000\n", true, 4},
		{"quota not a multiple of period rounds up", "250000 100000\n", true, 3},
		{"missing file (no cgroup v2, e.g. bare metal or macOS)", "", false, 0},
		{"malformed content", "garbage\n", true, 0},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "cpu.max")
			if tt.exists {
				if err := os.WriteFile(path, []byte(tt.content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if got := cgroupCPUQuota(path); got != tt.want {
				t.Errorf("cgroupCPUQuota(%q) = %d, want %d", tt.content, got, tt.want)
			}
		})
	}
}
