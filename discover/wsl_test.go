package discover

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestWSLMemoryBudget(t *testing.T) {
	const gib = format.GibiByte

	for _, tt := range []struct {
		name                     string
		backend, dedicated, held uint64
		shared                   bool
		want                     uint64
	}{
		{"dedicated plus shared less guest and reserve", 110 * gib, 80 * gib, 16 * gib, true, 90 * gib},
		{"nothing held", 110 * gib, 80 * gib, 0, true, 106 * gib},
		{"guest and reserve fill the shared part", 110 * gib, 80 * gib, 26 * gib, true, 80 * gib},
		{"one byte of shared room", 110 * gib, 80 * gib, 26*gib - 1, true, 80*gib + 1},
		{"guest leaves less than the reserve", 110 * gib, 80 * gib, 28 * gib, true, 80 * gib},
		{"guest holds more than the shared part", 110 * gib, 80 * gib, 40 * gib, true, 80 * gib},
		{"shared memory off", 110 * gib, 80 * gib, 16 * gib, false, 80 * gib},
		{"backend total at dedicated", 80 * gib, 80 * gib, 0, true, 80 * gib},
		{"backend total below dedicated", 70 * gib, 80 * gib, 0, true, 70 * gib},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := wslMemoryBudget(tt.backend, tt.dedicated, tt.held, tt.shared); got != tt.want {
				t.Fatalf("wslMemoryBudget = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestParseGuestHeld(t *testing.T) {
	const meminfo = "MemTotal:        1024 kB\nMemFree:          900 kB\nMemAvailable:     950 kB\n"
	const buddyinfo = "Node 0, zone      DMA      1      2      3\nNode 0, zone   Normal     10     20     30\n"
	parse := func(meminfo, buddyinfo string, order int) (uint64, error) {
		return parseGuestHeld(strings.NewReader(meminfo), strings.NewReader(buddyinfo), order, 4096)
	}

	// Blocks of order 1 and 2 are 2 and 4 pages: (2+20)*2 + (3+30)*4 pages are returned.
	if got, err := parse(meminfo, buddyinfo, 1); err != nil || got != 1024*format.KibiByte-176*4096 {
		t.Fatalf("parseGuestHeld = %d, %v", got, err)
	}
	if got, err := parse(meminfo, buddyinfo, 0); err != nil || got != 1024*format.KibiByte-187*4096 {
		t.Fatalf("parseGuestHeld at order 0 = %d, %v", got, err)
	}
	if got, err := parse(meminfo, buddyinfo, 3); err != nil || got != 1024*format.KibiByte {
		t.Fatalf("parseGuestHeld above the largest order = %d, %v", got, err)
	}
	// page_reporting_order is -1 when no reporting device registered.
	if got, err := parse(meminfo, buddyinfo, -1); err == nil {
		t.Fatalf("parseGuestHeld with page reporting off = %d, want an error", got)
	}

	for name, input := range map[string][2]string{
		"no MemTotal":           {"MemFree: 900 kB\n", buddyinfo},
		"unparseable MemTotal":  {"MemTotal: x kB\n", buddyinfo},
		"no zones":              {meminfo, ""},
		"unparseable count":     {meminfo, "Node 0, zone Normal 10 x\n"},
		"line without counts":   {meminfo, "Node 0, zone Normal\n"},
		"not a zone line":       {meminfo, "Node 0, area Normal 10\n"},
		"more free than memory": {"MemTotal: 4 kB\n", buddyinfo},
	} {
		if got, err := parse(input[0], input[1], 1); err == nil {
			t.Errorf("%s: parseGuestHeld = %d, want an error", name, got)
		}
	}

	if _, err := os.Stat("/sys/module/page_reporting/parameters/page_reporting_order"); err == nil {
		if held, err := readGuestHeld(); err != nil || held == 0 {
			t.Fatalf("readGuestHeld = %d, %v", held, err)
		}
	}
}

func TestApplyWSLBudget(t *testing.T) {
	const gib = format.GibiByte

	integrated := func(free uint64) ml.DeviceInfo {
		return ml.DeviceInfo{
			DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
			PCIID:       "000f:01:00.0",
			Integrated:  true,
			TotalMemory: 110 * gib,
			FreeMemory:  free,
		}
	}
	reading := func(total, used uint64) func(string) (uint64, uint64, error) {
		return func(string) (uint64, uint64, error) { return total, used, nil }
	}
	held := func(n uint64) func() (uint64, error) {
		return func() (uint64, error) { return n, nil }
	}
	host := func(nvml func(string) (uint64, uint64, error), guest func() (uint64, error)) wslHost {
		return wslHost{passthrough: true, nvml: nvml, guestHeld: guest, shared: true}
	}
	apply := func(devices []ml.DeviceInfo, h wslHost) []ml.DeviceInfo {
		return applyWSLBudget(devices, nil, h)
	}
	check := func(t *testing.T, got ml.DeviceInfo, total, free, backend uint64, budgeted bool) {
		t.Helper()
		if got.TotalMemory != total || got.FreeMemory != free || got.BackendFreeMemory != backend || got.MemoryBudget != budgeted {
			t.Fatalf("total/free/backend/budgeted = %d/%d/%d/%v GiB, want %d/%d/%d/%v",
				got.TotalMemory/gib, got.FreeMemory/gib, got.BackendFreeMemory/gib, got.MemoryBudget,
				total/gib, free/gib, backend/gib, budgeted)
		}
		if got.HostMemoryBounded() == budgeted {
			t.Fatalf("HostMemoryBounded = %v with budget %v", got.HostMemoryBounded(), budgeted)
		}
	}
	reset := func() {
		wslModes = map[string]string{}
		wslTotals = map[string]uint64{}
	}

	t.Run("dedicated and shared memory", func(t *testing.T) {
		reset()
		nvml := func(pci string) (uint64, uint64, error) {
			if pci != "000f:01:00.0" {
				t.Fatalf("pci = %q", pci)
			}
			return 80 * gib, 2 * gib, nil
		}
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(nvml, held(16*gib)))[0], 90*gib, 88*gib, 103*gib, true)
	})

	t.Run("budget follows the memory the guest holds", func(t *testing.T) {
		reset()
		h := host(reading(80*gib, 2*gib), held(16*gib))
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 90*gib, 88*gib, 103*gib, true)
		h.guestHeld = held(22 * gib)
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 84*gib, 82*gib, 103*gib, true)
		h.guestHeld = held(40 * gib)
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 80*gib, 78*gib, 103*gib, true)
		h.guestHeld = held(16 * gib)
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 90*gib, 88*gib, 103*gib, true)
	})

	t.Run("memory in use at startup returns once freed", func(t *testing.T) {
		reset()
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(reading(80*gib, 30*gib), held(16*gib)))[0], 90*gib, 60*gib, 103*gib, true)
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(reading(80*gib, 2*gib), held(16*gib)))[0], 90*gib, 88*gib, 103*gib, true)
	})

	t.Run("backend total below dedicated bounds the budget", func(t *testing.T) {
		reset()
		low := integrated(60 * gib)
		low.TotalMemory = 50 * gib
		check(t, apply([]ml.DeviceInfo{low}, host(reading(80*gib, 2*gib), held(0)))[0], 50*gib, 48*gib, 60*gib, true)
		check(t, apply([]ml.DeviceInfo{low}, host(reading(80*gib, 60*gib), held(0)))[0], 50*gib, 0, 60*gib, true)
	})

	// An 18 GiB runner loads after the first pass. Each pass keeps that pass's
	// backend figure, whether or not the driver lowers it for the runner.
	t.Run("backend figure follows each pass", func(t *testing.T) {
		for _, tt := range []struct {
			name    string
			backend uint64
		}{
			{"driver counts other processes", 85 * gib},
			{"driver does not count other processes", 103 * gib},
		} {
			t.Run(tt.name, func(t *testing.T) {
				reset()
				check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(reading(80*gib, 2*gib), held(16*gib)))[0], 90*gib, 88*gib, 103*gib, true)
				check(t, apply([]ml.DeviceInfo{integrated(tt.backend)}, host(reading(80*gib, 20*gib), held(16*gib)))[0], 90*gib, 70*gib, tt.backend, true)
			})
		}
	})

	t.Run("shared memory off keeps dedicated memory", func(t *testing.T) {
		reset()
		h := host(reading(80*gib, 2*gib), func() (uint64, error) {
			t.Fatal("guest memory read with shared memory off")
			return 0, nil
		})
		h.shared = false
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 80*gib, 78*gib, 103*gib, true)
	})

	t.Run("unreadable guest memory keeps dedicated memory", func(t *testing.T) {
		reset()
		h := host(reading(80*gib, 2*gib), func() (uint64, error) { return 0, errors.New("no meminfo") })
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 80*gib, 78*gib, 103*gib, true)
	})

	t.Run("leaves devices alone outside WSL", func(t *testing.T) {
		reset()
		h := host(reading(80*gib, 0), held(0))
		h.passthrough = false
		got := apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0]
		if got.TotalMemory != 110*gib || got.FreeMemory != 103*gib || got.BackendFreeMemory != 0 || got.MemoryBudget {
			t.Fatalf("total/free/backend/budgeted = %d/%d/%d/%v", got.TotalMemory, got.FreeMemory, got.BackendFreeMemory, got.MemoryBudget)
		}
	})

	t.Run("leaves discrete and non-CUDA devices alone", func(t *testing.T) {
		reset()
		discrete := integrated(20 * gib)
		discrete.Integrated = false
		vulkan := integrated(103 * gib)
		vulkan.Library = "Vulkan"
		got := apply([]ml.DeviceInfo{discrete, vulkan}, host(reading(16*gib, 0), held(0)))
		for i, d := range got {
			if d.FreeMemory != []uint64{20 * gib, 103 * gib}[i] || d.BackendFreeMemory != 0 || d.MemoryBudget {
				t.Fatalf("device %d free/backend/budgeted = %d/%d/%v", i, d.FreeMemory, d.BackendFreeMemory, d.MemoryBudget)
			}
		}
		if !got[1].HostMemoryBounded() {
			t.Fatal("Vulkan integrated GPU not bounded by host memory")
		}
	})

	t.Run("NVML failure keeps the backend figures and the host memory cap", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			nvml func(string) (uint64, uint64, error)
		}{
			{"error", func(string) (uint64, uint64, error) { return 0, 0, errors.New("no device") }},
			{"zero total", reading(0, 0)},
			{"no NVML", nil},
		} {
			t.Run(tt.name, func(t *testing.T) {
				reset()
				check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(tt.nvml, held(16*gib)))[0], 110*gib, 103*gib, 103*gib, false)
			})
		}
	})

	t.Run("each pass reads NVML again", func(t *testing.T) {
		reset()
		fail := true
		nvml := func(string) (uint64, uint64, error) {
			if fail {
				return 0, 0, errors.New("busy")
			}
			return 80 * gib, 2 * gib, nil
		}
		h := host(nvml, held(16*gib))
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 110*gib, 103*gib, 103*gib, false)
		fail = false
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 90*gib, 88*gib, 103*gib, true)
		fail = true
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 110*gib, 103*gib, 103*gib, false)
	})

	// A device whose bootstrap probe timed out keeps its last bootstrap figure,
	// which llama-server fits against, so it stays the backend figure.
	t.Run("stale free figure keeps the host memory cap and the backend figure", func(t *testing.T) {
		reset()
		calls := 0
		nvml := func(string) (uint64, uint64, error) {
			calls++
			return 80 * gib, 2 * gib, nil
		}
		second := integrated(103 * gib)
		second.ID = "1"
		got := applyWSLBudget([]ml.DeviceInfo{integrated(103 * gib), second}, []bool{false, true}, host(nvml, held(16*gib)))
		check(t, got[0], 110*gib, 103*gib, 103*gib, false)
		check(t, got[1], 90*gib, 88*gib, 103*gib, true)
		if calls != 1 {
			t.Fatalf("NVML read %d times, want 1", calls)
		}
		if _, ok := wslModes["0"]; ok {
			t.Fatal("stale device recorded a mode")
		}
	})

	t.Run("hung NVML read counts as a failure", func(t *testing.T) {
		reset()
		defer func(old time.Duration) { nvmlReadTimeout = old }(nvmlReadTimeout)
		nvmlReadTimeout = 20 * time.Millisecond

		release := make(chan struct{})
		hung := func(string) (uint64, uint64, error) {
			<-release
			return 80 * gib, 2 * gib, nil
		}
		start := time.Now()
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(hung, held(16*gib)))[0], 110*gib, 103*gib, 103*gib, false)
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Fatalf("pass took %s", elapsed)
		}

		// A read that overlaps the hung one fails at once and spawns nothing.
		nvmlReadTimeout = time.Minute
		var calls atomic.Int32
		counting := func(string) (uint64, uint64, error) {
			calls.Add(1)
			return 80 * gib, 2 * gib, nil
		}
		goroutines := runtime.NumGoroutine()
		start = time.Now()
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(counting, held(16*gib)))[0], 110*gib, 103*gib, 103*gib, false)
		if elapsed := time.Since(start); elapsed > time.Second {
			t.Fatalf("overlapping read took %s", elapsed)
		}
		if n := runtime.NumGoroutine(); n > goroutines {
			t.Fatalf("goroutines = %d, want at most %d", n, goroutines)
		}
		if calls.Load() != 0 {
			t.Fatal("overlapping read called NVML")
		}

		close(release)
		for !nvmlReadMu.TryLock() {
			time.Sleep(time.Millisecond)
		}
		nvmlReadMu.Unlock()
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, host(reading(80*gib, 2*gib), held(16*gib)))[0], 90*gib, 88*gib, 103*gib, true)
	})

	t.Run("stale pass keeps the last budgeted total", func(t *testing.T) {
		reset()
		h := host(reading(80*gib, 2*gib), held(16*gib))
		check(t, apply([]ml.DeviceInfo{integrated(103 * gib)}, h)[0], 90*gib, 88*gib, 103*gib, true)

		// Uncapped backend free is above the budget.
		got := applyWSLBudget([]ml.DeviceInfo{integrated(100 * gib)}, []bool{false}, h)[0]
		check(t, got, 90*gib, 90*gib, 100*gib, false)

		got = applyWSLBudget([]ml.DeviceInfo{integrated(60 * gib)}, []bool{false}, h)[0]
		check(t, got, 90*gib, 60*gib, 60*gib, false)
	})

	t.Run("does not need the device lock", func(t *testing.T) {
		reset()
		deviceMu.Lock()
		defer deviceMu.Unlock()
		done := make(chan ml.DeviceInfo, 1)
		go func() {
			done <- apply([]ml.DeviceInfo{integrated(103 * gib)}, host(reading(80*gib, 2*gib), held(16*gib)))[0]
		}()
		select {
		case got := <-done:
			check(t, got, 90*gib, 88*gib, 103*gib, true)
		case <-time.After(5 * time.Second):
			t.Fatal("applyWSLBudget blocked on deviceMu")
		}
	})
}

func TestRunnerCanRefresh(t *testing.T) {
	igpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Integrated: true}
	dgpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}}
	devices := []ml.DeviceInfo{igpu, dgpu}

	for _, tt := range []struct {
		name        string
		ids         []ml.DeviceID
		updated     []bool
		passthrough bool
		want        bool
	}{
		{"WSL integrated GPU only", []ml.DeviceID{igpu.DeviceID}, []bool{false, false}, true, false},
		{"integrated GPU outside WSL", []ml.DeviceID{igpu.DeviceID}, []bool{false, false}, false, true},
		{"discrete GPU", []ml.DeviceID{dgpu.DeviceID}, []bool{false, false}, true, true},
		{"discrete GPU already refreshed", []ml.DeviceID{dgpu.DeviceID}, []bool{false, true}, true, false},
		{"both on WSL", []ml.DeviceID{igpu.DeviceID, dgpu.DeviceID}, []bool{false, false}, true, true},
		{"unknown device", []ml.DeviceID{{ID: "9", Library: "CUDA"}}, []bool{false, false}, true, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := runnerCanRefresh(tt.ids, devices, tt.updated, tt.passthrough); got != tt.want {
				t.Fatalf("runnerCanRefresh = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRefreshFromRunnerSkipsWSLIntegratedGPU(t *testing.T) {
	igpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Integrated: true, FreeMemory: 87}
	dgpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 20}
	reported := []ml.DeviceInfo{igpu, dgpu}
	reported[0].FreeMemory = 60
	reported[1].FreeMemory = 10

	for _, tt := range []struct {
		passthrough bool
		wantUpdated []bool
		wantFree    []uint64
	}{
		{true, []bool{false, true}, []uint64{87, 10}},
		{false, []bool{true, true}, []uint64{60, 10}},
	} {
		devices := []ml.DeviceInfo{igpu, dgpu}
		updated := make([]bool, len(devices))
		refreshFromRunner(devices, updated, reported, tt.passthrough)
		for i := range devices {
			if updated[i] != tt.wantUpdated[i] || devices[i].FreeMemory != tt.wantFree[i] {
				t.Fatalf("passthrough %v device %d updated/free = %v/%d, want %v/%d",
					tt.passthrough, i, updated[i], devices[i].FreeMemory, tt.wantUpdated[i], tt.wantFree[i])
			}
		}
	}
}
