package discover

import (
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestMemoryMonitor(t *testing.T) {
	if unsafe.Sizeof(MEMORYSTATUSEX{}) != 64 {
		t.Fatal("unexpected MEMORYSTATUSEX layout")
	}
	var store windowsCompressionStatistics
	if unsafe.Sizeof(store) != 48 || unsafe.Offsetof(store.resident) != 8 || unsafe.Offsetof(store.original) != 16 || unsafe.Offsetof(store.stored) != 24 {
		t.Fatal("unexpected Windows compression statistics layout")
	}
	for range 4 {
		m, err := NewMemoryMonitor()
		if err != nil {
			t.Fatal(err)
		}
		if m.history.valid {
			m.Close()
			t.Fatal("constructor consumed the first pressure sample")
		}
		memory, err := m.Read()
		if err != nil {
			m.Close()
			t.Fatal(err)
		}
		if memory.RAMLimit == 0 || memory.AvailableRAM > memory.RAMLimit || memory.SampledAt.IsZero() || memory.AvailableCommit == nil {
			t.Errorf("invalid physical memory observation: %+v", memory)
		}
		done := make(chan struct{})
		go func() {
			m.Close()
			m.Close()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("notification waiter did not shut down")
		}
	}
}

func TestMemoryMonitorTransitions(t *testing.T) {
	// Use private manual-reset events to exercise the real Windows waiter
	// without exhausting system memory or changing the OS notification state.
	m := &MemoryMonitor{memoryPlatform: memoryPlatform{changes: make(chan struct{}, 1), done: make(chan struct{})}}
	for _, handle := range []*windows.Handle{&m.low, &m.high, &m.stop} {
		value, err := windows.CreateEvent(nil, 1, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		*handle = value
		t.Cleanup(func() { windows.CloseHandle(value) })
	}
	go m.watch()
	t.Cleanup(func() {
		if err := windows.SetEvent(m.stop); err != nil {
			t.Fatal(err)
		}
		select {
		case <-m.done:
		case <-time.After(5 * time.Second):
			t.Error("notification waiter did not stop")
		}
	})
	for range 2 {
		for _, handle := range []windows.Handle{m.low, m.high} {
			if err := windows.SetEvent(handle); err != nil {
				t.Fatal(err)
			}
			select {
			case <-m.Changes():
			case <-time.After(5 * time.Second):
				t.Fatal("memory transition did not wake the caller")
			}
			// Leaving this level signaled must not spin or repeatedly wake the
			// caller while the waiter is expecting the opposite transition.
			select {
			case <-m.Changes():
				t.Fatal("sustained memory level produced a duplicate notification")
			case <-time.After(20 * time.Millisecond):
			}
			if err := windows.ResetEvent(handle); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestWindowsMemoryActivity(t *testing.T) {
	m, err := NewMemoryMonitor()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for range 2 {
		memory, err := m.Read()
		if err != nil {
			t.Fatal(err)
		}
		if m.activityError != "" || !m.history.valid {
			t.Fatalf("compression/paging collection did not run: %s", m.activityError)
		}
		t.Logf("compressed_estimate=%d paging_total=%d reclaim=%d pressure=%d", m.history.previous.compressed, m.history.previous.pageOut, memory.ReclaimBytes, memory.Pressure)
	}
}

func TestWindowsUnavailablePressurePreservesCapacity(t *testing.T) {
	// Invalid notification handles and no paging query exercise the real API
	// failure path without disturbing a live monitor's watcher or resources.
	m := &MemoryMonitor{}
	got, err := m.Read()
	if err != nil || got.RAMLimit <= 0 || got.AvailableRAM < 0 || got.AvailableCommit == nil || got.Pressure != PressureUnknown {
		t.Fatalf("optional pressure failure lost capacity: %+v, %v", got, err)
	}
	if m.notificationError == "" || m.activityError == "" {
		t.Fatal("the unavailable-observation paths did not execute")
	}
}
