package discover

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// nvmlDeviceMemory returns NVML's total and used memory for a PCI bus ID. It
// is nil on platforms without NVML support.
var nvmlDeviceMemory func(pciBusID string) (total, used uint64, err error)

var wslGPUPassthrough = sync.OnceValue(func() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	_, err := os.Stat("/dev/dxg")
	return err == nil
})

// wslHostReserve is Windows memory a WSL2 integrated GPU's budget leaves
// unused on top of the memory the guest holds.
const wslHostReserve = 4 * format.GibiByte

// wslHost is what applyWSLBudget reads about the host in a pass.
type wslHost struct {
	passthrough bool
	nvml        func(pciBusID string) (total, used uint64, err error)
	guestHeld   func() (uint64, error)
	// shared allows Windows shared GPU memory (OLLAMA_WSL_SHARED_GPU_MEMORY).
	shared bool
}

// wslModes records each device's budget mode from its last fresh pass, so
// changes are logged once, and wslTotals each device's total from its last
// budgeted pass. Guarded by wslMu.
var (
	wslMu     sync.Mutex
	wslModes  = map[string]string{}
	wslTotals = map[string]uint64{}
)

// nvmlReadTimeout bounds one NVML read; a longer one counts as a failure.
var nvmlReadTimeout = 2 * time.Second

// nvmlReadMu is held from the start of an NVML read until it returns. A read
// that never returns keeps it, so later reads fail at once instead of piling
// goroutines onto NVML.
var nvmlReadMu sync.Mutex

func readNVMLMemory(nvml func(string) (uint64, uint64, error), pciID string) (total, used uint64, err error) {
	if nvml == nil {
		return 0, 0, errors.New("NVML is not available")
	}
	if !nvmlReadMu.TryLock() {
		return 0, 0, errors.New("an earlier NVML read has not returned")
	}
	type reading struct {
		total, used uint64
		err         error
	}
	ch := make(chan reading, 1)
	go func() {
		defer nvmlReadMu.Unlock()
		var r reading
		r.total, r.used, r.err = nvml(pciID)
		ch <- r
	}()
	timer := time.NewTimer(nvmlReadTimeout)
	defer timer.Stop()
	select {
	case r := <-ch:
		return r.total, r.used, r.err
	case <-timer.C:
		return 0, 0, fmt.Errorf("NVML read timed out after %s", nvmlReadTimeout)
	}
}

// readGuestHeld returns the Windows memory the WSL2 guest holds: MemTotal
// less its free blocks of page_reporting_order or larger, which free page
// reporting hands back to Windows. Smaller free blocks stay in Windows memory.
func readGuestHeld() (uint64, error) {
	order, err := os.ReadFile("/sys/module/page_reporting/parameters/page_reporting_order")
	if err != nil {
		return 0, err
	}
	o, err := strconv.Atoi(strings.TrimSpace(string(order)))
	if err != nil {
		return 0, err
	}
	meminfo, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	defer meminfo.Close()
	buddyinfo, err := os.Open("/proc/buddyinfo")
	if err != nil {
		return 0, err
	}
	defer buddyinfo.Close()
	return parseGuestHeld(meminfo, buddyinfo, o, uint64(os.Getpagesize()))
}

func parseGuestHeld(meminfo, buddyinfo io.Reader, order int, pageSize uint64) (uint64, error) {
	// The order reads -1 until a free page reporting device registers.
	if order < 0 {
		return 0, fmt.Errorf("free page reporting is off (page_reporting_order %d)", order)
	}
	var total uint64
	s := bufio.NewScanner(meminfo)
	for s.Scan() {
		if _, err := fmt.Sscanf(s.Text(), "MemTotal:%d", &total); err == nil {
			break
		}
	}
	if err := s.Err(); err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, errors.New("no MemTotal in /proc/meminfo")
	}
	total *= format.KibiByte

	// Each line is "Node 0, zone Normal" and the zone's free block count per order.
	var returned uint64
	zones := 0
	s = bufio.NewScanner(buddyinfo)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) < 5 || fields[0] != "Node" || fields[2] != "zone" {
			return 0, fmt.Errorf("unexpected /proc/buddyinfo line %q", s.Text())
		}
		for o, field := range fields[4:] {
			n, err := strconv.ParseUint(field, 10, 64)
			if err != nil {
				return 0, err
			}
			if o >= order {
				returned += n * pageSize << o
			}
		}
		zones++
	}
	if err := s.Err(); err != nil {
		return 0, err
	}
	if zones == 0 || returned > total {
		return 0, fmt.Errorf("no usable /proc/buddyinfo (%d zones, %d of %d bytes free)", zones, returned, total)
	}
	return total - returned, nil
}

func wslBudgetCandidate(d ml.DeviceInfo, passthrough bool) bool {
	return passthrough && d.Integrated && strings.EqualFold(d.Library, "CUDA")
}

// wslMemoryBudget returns the memory a WSL2 integrated GPU can plan against.
// backendTotal is the adapter's Windows video memory budget; the part past
// dedicated memory is Windows RAM, which also backs the guest's memory.
func wslMemoryBudget(backendTotal, dedicated, guestHeld uint64, shared bool) uint64 {
	if backendTotal <= dedicated {
		return backendTotal
	}
	if spill := backendTotal - dedicated; shared && spill > guestHeld+wslHostReserve {
		return backendTotal - guestHeld - wslHostReserve
	}
	return dedicated
}

// applyWSLBudget bounds integrated CUDA devices under WSL2 by wslMemoryBudget,
// since CUDA's free figure there ignores memory held outside the process.
// fresh[i] reports whether device i was refreshed in this pass (nil: all were).
func applyWSLBudget(devices []ml.DeviceInfo, fresh []bool, host wslHost) []ml.DeviceInfo {
	for i := range devices {
		d := &devices[i]
		if !wslBudgetCandidate(*d, host.passthrough) {
			continue
		}
		// On a stale pass this is the last bootstrap figure, which a new
		// process also sees: it does not drop for other processes' memory.
		d.BackendFreeMemory = d.FreeMemory
		if fresh != nil && !fresh[i] {
			slog.Debug("WSL GPU free memory not refreshed, keeping the host memory limit", "id", d.ID)
			wslMu.Lock()
			total, ok := wslTotals[d.ID]
			wslMu.Unlock()
			if ok {
				d.TotalMemory = min(d.TotalMemory, total)
				d.FreeMemory = min(d.FreeMemory, d.TotalMemory)
			}
			continue
		}

		// NVML's total and used count dedicated memory only; updateFreeSpace
		// also caps free memory at the budget less loaded runners' predictions.
		dedicated, used, err := readNVMLMemory(host.nvml, d.PCIID)
		if err == nil && dedicated == 0 {
			err = errors.New("NVML reported no dedicated memory")
		}
		if err != nil {
			if wslSetMode(d.ID, "host memory limit", 0) {
				slog.Warn("unable to read dedicated memory of WSL GPU, keeping the host memory limit", "id", d.ID, "pci_id", d.PCIID, "error", err)
			}
			continue
		}

		shared := host.shared
		var held uint64
		var heldErr error
		if shared {
			if held, heldErr = host.guestHeld(); heldErr != nil {
				shared = false
			}
		}
		total := wslMemoryBudget(d.TotalMemory, dedicated, held, shared)
		used = min(used, total)
		mode := "dedicated"
		if total > dedicated {
			mode = "dedicated and shared"
		}
		attrs := []any{
			"id", d.ID, "mode", mode,
			"budget", format.HumanBytes2(total),
			"dedicated", format.HumanBytes2(dedicated),
			"in_use", format.HumanBytes2(used),
			"backend_total", format.HumanBytes2(d.TotalMemory),
			"backend_free", format.HumanBytes2(d.FreeMemory),
			"guest_held", format.HumanBytes2(held),
		}
		if heldErr != nil {
			attrs = append(attrs, "error", heldErr)
		}
		if wslSetMode(d.ID, mode, total) {
			slog.Info("WSL GPU memory budget", attrs...)
		} else {
			slog.Debug("WSL GPU memory budget", attrs...)
		}
		d.TotalMemory = total
		d.FreeMemory = min(d.FreeMemory, total-used)
		d.MemoryBudget = true
	}
	return devices
}

// wslSetMode records a device's mode, and its total when nonzero, and
// reports whether the mode changed.
func wslSetMode(id, mode string, total uint64) bool {
	wslMu.Lock()
	defer wslMu.Unlock()
	if total > 0 {
		wslTotals[id] = total
	}
	changed := wslModes[id] != mode
	wslModes[id] = mode
	return changed
}
