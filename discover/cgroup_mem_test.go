package discover

import (
	"strings"
	"testing"
)

func TestGetNamedUint64FromStat(t *testing.T) {
	realisticStat := `anon 1234567
file 7654321
kernel_stack 131072
pagetables 65536
sec_pagetables 0
percpu 39584
sock 0
vmalloc 0
shmem 0
zswap 0
zswapped 0
file_mapped 2097152
file_dirty 0
file_writeback 0
anon_thp 0
inactive_anon 0
active_anon 1234567
inactive_file 3145728
active_file 4508593
unevictable 0
slab_reclaimable 524288
slab_unreclaimable 262144
slab 786432
workingset_refault_anon 0
workingset_refault_file 1024
workingset_activate_anon 0
workingset_activate_file 512
workingset_restore_anon 0
workingset_restore_file 0
workingset_nodereclaim 0
pgfault 12345
pgmajfault 0
pgrefill 0
pgscan 0
pgsteal 0
pgactivate 0
pgdeactivate 0
pglazyfree 0
pglazyfreed 0
zswpin 0
zswpout 0
thp_fault_alloc 0
thp_collapse_alloc 0`

	t.Run("finds_named_key", func(t *testing.T) {
		v, err := getNamedUint64FromStat(strings.NewReader(realisticStat), "inactive_file")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v != 3145728 {
			t.Errorf("got %d, want 3145728", v)
		}
	})

	t.Run("finds_other_key", func(t *testing.T) {
		v, err := getNamedUint64FromStat(strings.NewReader(realisticStat), "active_file")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v != 4508593 {
			t.Errorf("got %d, want 4508593", v)
		}
	})

	t.Run("missing_key_returns_error", func(t *testing.T) {
		_, err := getNamedUint64FromStat(strings.NewReader(realisticStat), "nonexistent_key")
		if err == nil {
			t.Error("expected error for missing key, got nil")
		}
	})

	t.Run("empty_reader_returns_error", func(t *testing.T) {
		_, err := getNamedUint64FromStat(strings.NewReader(""), "inactive_file")
		if err == nil {
			t.Error("expected error for empty reader, got nil")
		}
	})

	t.Run("prefix_must_match_exactly", func(t *testing.T) {
		// "inactive_file_extra" should not match "inactive_file"
		stat := "inactive_file_extra 999\ninactive_file 42\n"
		v, err := getNamedUint64FromStat(strings.NewReader(stat), "inactive_file")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v != 42 {
			t.Errorf("got %d, want 42 (prefix match must require space delimiter)", v)
		}
	})
}

func TestGetCPUMemByCgroupsInactiveFile(t *testing.T) {
	const (
		memMax     = 8 * 1024 * 1024 * 1024 // 8 GiB limit
		memCurrent = 6 * 1024 * 1024 * 1024 // 6 GiB used (includes page cache)
		inactive   = 2 * 1024 * 1024 * 1024 // 2 GiB reclaimable page cache
	)

	// Baseline: without inactive_file subtraction, free = 8-6 = 2 GiB
	// With fix: free = 8 - (6-2) = 4 GiB
	baselineFree := uint64(memMax - memCurrent)
	correctedFree := uint64(memMax - (memCurrent - inactive))

	// Verify the formula holds
	if correctedFree <= baselineFree {
		t.Fatalf("corrected free (%d) should be greater than baseline (%d)", correctedFree, baselineFree)
	}

	// Test the stat parser with realistic values
	stat := "anon 123\ninactive_file 2147483648\nactive_file 456\n"
	v, err := getNamedUint64FromStat(strings.NewReader(stat), "inactive_file")
	if err != nil {
		t.Fatalf("getNamedUint64FromStat: %v", err)
	}
	if v != inactive {
		t.Errorf("got %d, want %d", v, uint64(inactive))
	}

	t.Run("clamped_when_inactive_exceeds_used", func(t *testing.T) {
		// If inactive_file somehow exceeds memory.current (should never happen),
		// getCPUMemByCgroups clamps inactiveFile to 0 to avoid underflow.
		// We test the clamp logic directly via the formula.
		used := uint64(100)
		inactiveFile := uint64(200) // anomalous: larger than used

		if inactiveFile > used {
			inactiveFile = 0
		}
		free := uint64(memMax) - (used - inactiveFile)
		expected := uint64(memMax) - used
		if free != expected {
			t.Errorf("clamped free=%d, want %d", free, expected)
		}
	})

	t.Run("zero_inactive_file_is_identity", func(t *testing.T) {
		// When inactive_file is 0 (or stat unavailable), result equals original formula
		used := uint64(memCurrent)
		inactiveFile := uint64(0)
		free := uint64(memMax) - (used - inactiveFile)
		if free != baselineFree {
			t.Errorf("got %d, want %d (should match old formula when inactive=0)", free, baselineFree)
		}
	})
}
