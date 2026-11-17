package discover

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func linuxMemoryFixture(t *testing.T) *MemoryMonitor {
	t.Helper()
	root := t.TempDir()
	for path, value := range map[string]string{
		"proc/meminfo":                  pressureMeminfo + "Zswapped: 100 kB\n",
		"proc/sys/vm/overcommit_memory": "0\n",
		"proc/self/cgroup":              "1:cpu:/\n",
		"proc/vmstat":                   "pswpout 50\nzswpout 20\n",
		"proc/swaps":                    "Filename Type Size Used Priority\n",
	} {
		dir, name := filepath.Split(filepath.Join(root, path))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFakeSysfsFile(t, dir, name, value)
	}
	return &MemoryMonitor{memoryPlatform: memoryPlatform{proc: filepath.Join(root, "proc"), root: root}}
}

func TestLinuxMemoryReadFailurePreservesBaseline(t *testing.T) {
	m := linuxMemoryFixture(t)
	if _, err := m.Read(); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, m.proc, "meminfo", "MemTotal: invalid\n")
	if _, err := m.Read(); err == nil {
		t.Fatal("malformed required capacity did not fail")
	}
	writeFakeSysfsFile(t, m.proc, "meminfo", pressureMeminfo+"Zswapped: 108 kB\n")
	got, err := m.Read()
	if err != nil {
		t.Fatal(err)
	}
	if got.ReclaimBytes != 8*1024 {
		t.Fatalf("failed read lost compression baseline: got %d, want %d", got.ReclaimBytes, 8*1024)
	}
}

func TestLinuxMemoryActivity(t *testing.T) {
	m := linuxMemoryFixture(t)
	for _, step := range []struct {
		name            string
		meminfo, vmstat string
		pressure        PressureLevel
		reclaim         int64
	}{
		{name: "baseline", pressure: PressureUnknown},
		{name: "quiet", pressure: PressureNormal},
		{name: "compression growth", meminfo: pressureMeminfo + "Zswapped: 108 kB\n", vmstat: "pswpout 50\nzswpout 22\n", pressure: PressureElevated, reclaim: 8 * 1024},
		{name: "no duplicate", pressure: PressureNormal},
		{name: "compressor churn", vmstat: "pswpout 50\nzswpout 23\n", pressure: PressureElevated},
		{name: "page out", vmstat: "pswpout 53\nzswpout 23\n", pressure: PressureElevated, reclaim: 3 * int64(os.Getpagesize())},
		{name: "counter unavailable", vmstat: "unavailable\n", pressure: PressureUnknown},
		{name: "recovery baseline", vmstat: "pswpout 70\nzswpout 23\n", pressure: PressureUnknown},
		{name: "recovered", pressure: PressureNormal},
		{name: "counter reset", vmstat: "pswpout 0\nzswpout 0\n", pressure: PressureUnknown},
	} {
		t.Run(step.name, func(t *testing.T) {
			if step.meminfo != "" {
				writeFakeSysfsFile(t, m.proc, "meminfo", step.meminfo)
			}
			if step.vmstat != "" {
				writeFakeSysfsFile(t, m.proc, "vmstat", step.vmstat)
			}
			got, err := m.Read()
			if err != nil {
				t.Fatal(err)
			}
			if got.Pressure != step.pressure || got.ReclaimBytes != step.reclaim || got.AvailableRAM != 16<<30 {
				t.Fatalf("sample=%+v, want pressure=%v reclaim=%d with intact capacity", got, step.pressure, step.reclaim)
			}
		})
	}
}

func TestLinuxZramSwap(t *testing.T) {
	m := linuxMemoryFixture(t)
	dir := filepath.Join(m.root, "sys/block/zram0")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, dir, "mm_stat", "40960 16384 20480\n")
	writeFakeSysfsFile(t, dir, "stat", "0 0 0 0 10 0 80 0 0 0 0\n")
	writeFakeSysfsFile(t, m.proc, "swaps", "Filename Type Size Used Priority\n/dev/zram0 partition 4096 40 100\n")
	if _, err := m.Read(); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, dir, "mm_stat", "81920 32768 40960\n")
	writeFakeSysfsFile(t, dir, "stat", "0 0 0 0 20 0 160 0 0 0 0\n")
	writeFakeSysfsFile(t, m.proc, "vmstat", "pswpout 55\nzswpout 20\n")
	got, err := m.Read()
	if err != nil {
		t.Fatal(err)
	}
	if got.ReclaimBytes != max(40960, 5*int64(os.Getpagesize())) || got.Pressure != PressureElevated {
		t.Fatalf("overlapping compression and paging charged incorrectly: %+v", got)
	}
	writeFakeSysfsFile(t, m.proc, "swaps", "Filename Type Size Used Priority\n")
	got, err = m.Read()
	if err != nil || got.ReclaimBytes != 0 || got.Pressure != PressureUnknown {
		t.Fatalf("removed compression source must establish a fresh baseline: %+v, %v", got, err)
	}
}

func TestLinuxSwapFileIsNotZram(t *testing.T) {
	m := linuxMemoryFixture(t)
	writeFakeSysfsFile(t, m.proc, "swaps", "Filename Type Size Used Priority\n/var/zram0 file 4096 40 100\n")
	if _, err := m.Read(); err != nil {
		t.Fatal(err)
	}
	got, err := m.Read()
	if err != nil || got.Pressure != PressureNormal {
		t.Fatalf("ordinary swap file was mistaken for a zram device: %+v, %v", got, err)
	}
}

func TestLinuxMemoryStalls(t *testing.T) {
	m := linuxMemoryFixture(t)
	dir := filepath.Join(m.proc, "pressure")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, dir, "memory", "some avg10=0 total=10\nfull avg10=0 total=10\n")
	if _, err := m.Read(); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, dir, "memory", "some avg10=0 total=20\nfull avg10=0 total=20\n")
	got, err := m.Read()
	if err != nil || got.Pressure != PressureNormal {
		t.Fatalf("stalls without scarce RAM: %+v, %v", got, err)
	}
	writeFakeSysfsFile(t, m.proc, "meminfo", strings.Replace(pressureMeminfo, "MemAvailable: 16777216", "MemAvailable: 65536", 1))
	writeFakeSysfsFile(t, dir, "memory", "some avg10=0 total=30\nfull avg10=0 total=30\n")
	got, err = m.Read()
	if err != nil || got.Pressure != PressureElevated || got.ReclaimBytes != 0 {
		t.Fatalf("corroborated stalls: %+v, %v", got, err)
	}
}

func TestLinuxUnavailableCompression(t *testing.T) {
	for _, tc := range []struct{ name, field string }{
		{"missing enabled compressor", ""},
		{"malformed compressed size", "Zswapped: unknown kB\n"},
		{"compressed size overflow", "Zswapped: 18446744073709551615 kB\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := linuxMemoryFixture(t)
			writeFakeSysfsFile(t, m.proc, "meminfo", pressureMeminfo+tc.field)
			dir := filepath.Join(m.root, "sys/module/zswap/parameters")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			writeFakeSysfsFile(t, dir, "enabled", "Y\n")
			for range 2 {
				got, err := m.Read()
				if err != nil || got.Pressure != PressureUnknown || got.AvailableRAM != 16<<30 {
					t.Fatalf("unavailable compression must preserve capacity and hold growth: %+v, %v", got, err)
				}
			}
		})
	}
}

const pressureMeminfo = "MemTotal: 67108864 kB\nMemAvailable: 16777216 kB\nSwapFree: 33554432 kB\nCommitLimit: 33554432 kB\nCommitted_AS: 16777216 kB\n"

func TestLinuxMemoryPressure(t *testing.T) {
	const gib = uint64(1 << 30)
	for _, tc := range []struct {
		name      string
		files     map[string]string
		remove    []string
		total     uint64
		available uint64
		commit    uint64
		wantError bool
	}{
		{name: "host capacity excludes swap", total: 64 * gib, available: 16 * gib, commit: math.MaxInt64},
		{name: "zero available stays zero", files: map[string]string{"proc/meminfo": "MemTotal: 67108864 kB\nMemAvailable: 0 kB\nMemFree: 1024 kB\nCached: 4096 kB\nCommitLimit: 33554432 kB\nCommitted_AS: 16777216 kB\n"}, total: 64 * gib, commit: math.MaxInt64},
		{name: "strict commit", files: map[string]string{"proc/sys/vm/overcommit_memory": "2\n"}, total: 64 * gib, available: 16 * gib, commit: 16 * gib},
		{name: "always overcommit", files: map[string]string{"proc/sys/vm/overcommit_memory": "1\n"}, total: 64 * gib, available: 16 * gib, commit: math.MaxInt64},
		{name: "nested parent includes siblings", files: map[string]string{"sys/fs/cgroup/team/memory.max": strconv.FormatUint(12*gib, 10), "sys/fs/cgroup/team/memory.current": strconv.FormatUint(10*gib, 10)}, total: 12 * gib, available: 2 * gib, commit: math.MaxInt64},
		{name: "high throttling boundary", files: map[string]string{"sys/fs/cgroup/team/job/memory.high": strconv.FormatUint(8*gib, 10), "sys/fs/cgroup/team/job/memory.current": strconv.FormatUint(7*gib, 10)}, total: 8 * gib, available: gib, commit: math.MaxInt64},
		{name: "usage over limit does not wrap", files: map[string]string{"sys/fs/cgroup/team/job/memory.max": strconv.FormatUint(gib, 10), "sys/fs/cgroup/team/job/memory.current": strconv.FormatUint(2*gib, 10)}, total: gib, commit: math.MaxInt64},
		{name: "missing available", files: map[string]string{"proc/meminfo": "MemTotal: 67108864 kB\n"}, wantError: true},
		{name: "invalid units", files: map[string]string{"proc/meminfo": "MemTotal: 64 GB\n"}, wantError: true},
		{name: "size overflow", files: map[string]string{"proc/meminfo": "MemTotal: 18446744073709551615 kB\n"}, wantError: true},
		{name: "invalid overcommit mode", files: map[string]string{"proc/sys/vm/overcommit_memory": "unknown"}, wantError: true},
		{name: "strict commit exhausted", files: map[string]string{"proc/sys/vm/overcommit_memory": "2", "proc/meminfo": "MemTotal: 67108864 kB\nMemAvailable: 16777216 kB\nCommitLimit: 1024 kB\nCommitted_AS: 2048 kB\n"}, total: 64 * gib, available: 16 * gib},
		{name: "malformed limit", files: map[string]string{"sys/fs/cgroup/team/memory.high": "broken"}, wantError: true},
		{name: "v1 fails closed", files: map[string]string{"proc/self/cgroup": "5:memory:/team\n"}, wantError: true},
		{name: "empty membership fails closed", files: map[string]string{"proc/self/cgroup": ""}, wantError: true},
		{name: "empty unified path fails closed", files: map[string]string{"proc/self/cgroup": "0::\n"}, wantError: true},
		{name: "partial controller fails closed", remove: []string{"sys/fs/cgroup/team/job/memory.high"}, wantError: true},
		{name: "parent limit without child controller", files: map[string]string{"sys/fs/cgroup/team/memory.max": strconv.FormatUint(12*gib, 10), "sys/fs/cgroup/team/memory.current": strconv.FormatUint(10*gib, 10)}, remove: []string{"sys/fs/cgroup/team/job/memory.max", "sys/fs/cgroup/team/job/memory.high", "sys/fs/cgroup/team/job/memory.current"}, total: 12 * gib, available: 2 * gib, commit: math.MaxInt64},
		{name: "unmapped hierarchy fails closed", files: map[string]string{"proc/self/mountinfo": "29 1 0:26 /elsewhere /sys/fs/cgroup rw - cgroup2 cgroup rw\n"}, wantError: true},
		{name: "subtree hides ancestors", files: map[string]string{"proc/self/cgroup": "0::/hidden/team/job\n", "proc/self/mountinfo": "29 1 0:26 /hidden /sys/fs/cgroup rw - cgroup2 cgroup rw\n"}, wantError: true},
		{name: "full hierarchy after subtree mount", files: map[string]string{"proc/self/mountinfo": "28 1 0:26 /team /sys/fs/subtree rw - cgroup2 cgroup rw\n29 1 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"}, total: 64 * gib, available: 16 * gib, commit: math.MaxInt64},
		{name: "namespace root hides ancestors", files: map[string]string{"proc/self/cgroup": "0::/\n", "sys/fs/cgroup/cgroup.events": "populated 1\n", "sys/fs/cgroup/memory.max": "max", "sys/fs/cgroup/memory.high": "max"}, wantError: true},
		{name: "namespace without memory controller hides ancestors", files: map[string]string{"proc/self/cgroup": "0::/\n", "sys/fs/cgroup/cgroup.events": "populated 1\n"}, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			files := map[string]string{
				"proc/meminfo":                  pressureMeminfo,
				"proc/sys/vm/overcommit_memory": "0\n",
				"proc/self/cgroup":              "0::/team/job\n",
				"proc/self/mountinfo":           "29 1 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
			}
			for _, dir := range []string{"team", "team/job"} {
				for _, name := range []string{"memory.max", "memory.high"} {
					files["sys/fs/cgroup/"+dir+"/"+name] = "max\n"
				}
				files["sys/fs/cgroup/"+dir+"/memory.current"] = "0\n"
			}
			for path, value := range tc.files {
				files[path] = value
			}
			for _, path := range tc.remove {
				delete(files, path)
			}
			if err := os.MkdirAll(filepath.Join(root, "sys/fs/cgroup/team/job"), 0o755); err != nil {
				t.Fatal(err)
			}
			for path, value := range files {
				path = filepath.Join(root, path)
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(value), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			got, err := readLinuxMemory(filepath.Join(root, "proc"), root)
			if tc.wantError {
				if err == nil {
					t.Fatal("invalid observation was accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			commitMatches := tc.commit == math.MaxInt64 && got.AvailableCommit == nil || got.AvailableCommit != nil && uint64(*got.AvailableCommit) == tc.commit
			if got.RAMLimit != int64(tc.total) || got.AvailableRAM != int64(tc.available) || !commitMatches {
				t.Fatalf("observation = %+v; want total=%d available=%d commit=%d", got, tc.total, tc.available, tc.commit)
			}
		})
	}
}

func FuzzLinuxMemoryPressure(f *testing.F) {
	f.Add(pressureMeminfo, "2", "max", "0")
	f.Add(pressureMeminfo, "0", "1073741824", "2147483648")
	f.Add("MemTotal: 18446744073709551615 kB\n", "1", "broken", "-1")
	f.Fuzz(func(t *testing.T, meminfo, mode, limit, current string) {
		root := t.TempDir()
		for path, value := range map[string]string{
			"proc/meminfo":                     meminfo,
			"proc/sys/vm/overcommit_memory":    mode,
			"proc/self/cgroup":                 "0::/job\n",
			"proc/self/mountinfo":              "29 1 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
			"sys/fs/cgroup/job/memory.max":     limit,
			"sys/fs/cgroup/job/memory.high":    "max\n",
			"sys/fs/cgroup/job/memory.current": current,
		} {
			dir, name := filepath.Split(filepath.Join(root, path))
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			writeFakeSysfsFile(t, dir, name, value)
		}
		m, err := readLinuxMemory(filepath.Join(root, "proc"), root)
		if err == nil && (m.AvailableRAM > m.RAMLimit || m.AvailableRAM < 0 || m.RAMLimit < 0 || m.AvailableCommit != nil && *m.AvailableCommit < 0) {
			t.Fatalf("successful observation escaped byte bounds: %+v", m)
		}
	})
}

func FuzzLinuxMemoryActivity(f *testing.F) {
	f.Add(uint8(0), "pswpout 55\nzswpout 21\n")
	f.Add(uint8(0), "pswpout -1\n")
	f.Add(uint8(1), "Filename Type Size Used Priority\n/var/zram0 file 4096 40 100\n")
	f.Add(uint8(2), "81920 32768 40960\n")
	f.Add(uint8(2), "18446744073709551615 0 0\n")
	f.Add(uint8(3), "0 0 0 0 20 0 160 0 0 0 0\n")
	f.Add(uint8(4), "some avg10=0 total=30\nfull avg10=0 total=30\n")
	f.Fuzz(func(t *testing.T, field uint8, data string) {
		m := linuxMemoryFixture(t)
		files := []struct{ path, value string }{
			{"proc/vmstat", "pswpout 50\nzswpout 20\n"},
			{"proc/swaps", "Filename Type Size Used Priority\n/dev/zram0 partition 4096 40 100\n"},
			{"sys/block/zram0/mm_stat", "40960 16384 20480\n"},
			{"sys/block/zram0/stat", "0 0 0 0 10 0 80 0 0 0 0\n"},
			{"proc/pressure/memory", "some avg10=0 total=10\nfull avg10=0 total=10\n"},
		}
		for _, file := range files {
			dir, name := filepath.Split(filepath.Join(m.root, file.path))
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			writeFakeSysfsFile(t, dir, name, file.value)
		}
		if _, err := m.Read(); err != nil {
			t.Fatal(err)
		}
		dir, name := filepath.Split(filepath.Join(m.root, files[int(field)%len(files)].path))
		writeFakeSysfsFile(t, dir, name, data)
		got, err := m.Read()
		if err != nil || got.AvailableRAM != 16<<30 || got.RAMLimit != 64<<30 || got.ReclaimBytes < 0 {
			t.Fatalf("optional activity corrupted capacity or feedback: %+v, %v", got, err)
		}
		again, err := m.Read()
		if err != nil || again.ReclaimBytes != 0 {
			t.Fatalf("unchanged activity replayed feedback: %+v, %v", again, err)
		}
	})
}
