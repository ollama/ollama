package cmd

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestEnumPIDs(t *testing.T) {
	for _, tc := range []struct {
		name          string
		pids          []uint32
		bytesReturned uint32
		wantLen       int
		wantFull      bool
	}{
		{name: "no processes", pids: make([]uint32, 4), bytesReturned: 0, wantLen: 0, wantFull: false},
		{name: "three of four", pids: make([]uint32, 4), bytesReturned: 12, wantLen: 3, wantFull: false},
		{name: "buffer full", pids: make([]uint32, 4), bytesReturned: 16, wantLen: 4, wantFull: true},
		{name: "byte count exceeds buffer", pids: make([]uint32, 4), bytesReturned: 40, wantLen: 4, wantFull: true},
		{name: "empty buffer", pids: nil, bytesReturned: 0, wantLen: 0, wantFull: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			used, full := enumPIDs(tc.pids, tc.bytesReturned)
			if len(used) != tc.wantLen {
				t.Errorf("len(used) = %d, want %d", len(used), tc.wantLen)
			}
			if full != tc.wantFull {
				t.Errorf("full = %t, want %t", full, tc.wantFull)
			}
		})
	}
}

func TestEnumerateProcessPIDsGrowsWhenFull(t *testing.T) {
	calls := 0
	enum := func(pids []uint32, bytesReturned *uint32) error {
		calls++
		if calls == 1 {
			// Simulate a full buffer. EnumProcesses reports bytes written,
			// which is four times the number of process IDs on Windows.
			for i := range pids {
				pids[i] = uint32(i + 1)
			}
			*bytesReturned = uint32(len(pids)) * 4
			return nil
		}

		pids[0], pids[1], pids[2] = 11, 22, 33
		*bytesReturned = 12 // 3 process IDs
		return nil
	}

	got, err := enumerateProcessPIDs(enum)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("enum called %d times, want 2 (grow and retry)", calls)
	}
	if want := []uint32{11, 22, 33}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestEnumerateProcessPIDsError(t *testing.T) {
	want := errors.New("enum failed")
	_, err := enumerateProcessPIDs(func([]uint32, *uint32) error { return want })
	if !errors.Is(err, want) {
		t.Errorf("got %v, want %v", err, want)
	}
}

func TestIsProcRunningFindsCurrentProcess(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	name := filepath.Base(exe)
	if got := isProcRunning(name); !slices.Contains(got, uint32(os.Getpid())) {
		t.Errorf("isProcRunning(%q) = %v, want it to contain the current pid %d", name, got, os.Getpid())
	}
}
