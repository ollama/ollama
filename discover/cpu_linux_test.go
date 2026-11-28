package discover

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCPUMemCgroupHeadroom(t *testing.T) {
	const gib = uint64(1 << 30)
	tests := []struct {
		name      string
		max       string
		current   string
		hostFree  uint64
		wantTotal uint64
		wantFree  uint64
	}{
		{"unlimited", "max", "1073741824", 4 * gib, 16 * gib, 4 * gib},
		{"host_pressure", "8589934592", "1073741824", 2 * gib, 8 * gib, 2 * gib},
		{"cgroup_pressure", "8589934592", "7516192768", 4 * gib, 8 * gib, gib},
		{"limit_above_host", "34359738368", "1073741824", 4 * gib, 16 * gib, 4 * gib},
		{"usage_above_limit", "8589934592", "9663676416", 4 * gib, 8 * gib, 0},
		{"missing_limit", "", "1073741824", 4 * gib, 16 * gib, 4 * gib},
		{"invalid_limit", "invalid", "1073741824", 4 * gib, 16 * gib, 4 * gib},
		{"missing_usage", "8589934592", "", 12 * gib, 8 * gib, 8 * gib},
		{"invalid_usage", "8589934592", "invalid", 12 * gib, 8 * gib, 8 * gib},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, value := range map[string]string{"memory.max": tt.max, "memory.current": tt.current} {
				if value != "" {
					if err := os.WriteFile(filepath.Join(dir, name), []byte(value+"\n"), 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			mem := memInfo{TotalMemory: 16 * gib, FreeMemory: tt.hostFree, FreeSwap: gib}
			got := getCPUMemByCgroupsDir(mem, dir)
			if got.TotalMemory != tt.wantTotal || got.FreeMemory != tt.wantFree || got.FreeSwap != mem.FreeSwap {
				t.Fatalf("got %+v; want total=%d free=%d swap=%d", got, tt.wantTotal, tt.wantFree, mem.FreeSwap)
			}
		})
	}
}
