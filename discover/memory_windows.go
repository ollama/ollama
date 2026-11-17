package discover

import (
	"fmt"
	"log/slog"
	"math"
	"math/bits"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	createMemoryResourceNotification = k32.NewProc("CreateMemoryResourceNotification")
	queryMemoryResourceNotification  = k32.NewProc("QueryMemoryResourceNotification")
	pdh                              = windows.NewLazySystemDLL("pdh.dll")
	pdhOpenQuery                     = pdh.NewProc("PdhOpenQueryW")
	pdhAddCounter                    = pdh.NewProc("PdhAddEnglishCounterW")
	pdhCollect                       = pdh.NewProc("PdhCollectQueryData")
	pdhRawValue                      = pdh.NewProc("PdhGetRawCounterValue")
	pdhCloseQuery                    = pdh.NewProc("PdhCloseQuery")
)

type memoryPlatform struct {
	history                    memoryHistory
	low, high, stop            windows.Handle
	changes                    chan struct{}
	done                       chan struct{}
	once                       sync.Once
	pagingQuery, pagingCounter uintptr
	activityError              string
	notificationError          string
}

func newMemoryMonitor() (*MemoryMonitor, error) {
	m := &MemoryMonitor{memoryPlatform: memoryPlatform{changes: make(chan struct{}, 1), done: make(chan struct{})}}
	for i, handle := range []*windows.Handle{&m.low, &m.high} {
		value, _, err := createMemoryResourceNotification.Call(uintptr(i))
		if value == 0 {
			if m.low != 0 {
				windows.CloseHandle(m.low)
			}
			return nil, fmt.Errorf("CreateMemoryResourceNotification: %w", err)
		}
		*handle = windows.Handle(value)
	}
	var err error
	m.stop, err = windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		windows.CloseHandle(m.low)
		windows.CloseHandle(m.high)
		return nil, err
	}
	if err := m.openPagingCounter(); err != nil {
		m.activityError = err.Error()
		slog.Debug("memory paging observation unavailable", "error", err)
	}
	go m.watch()
	return m, nil
}

func (m *memoryPlatform) watch() {
	defer close(m.done)
	// These are level-triggered events. Alternate low/high so sustained low
	// memory wakes the caller once instead of spinning on a signaled handle.
	next := m.low
	for {
		result, err := windows.WaitForMultipleObjects([]windows.Handle{m.stop, next}, false, windows.INFINITE)
		if err != nil {
			slog.Warn("memory pressure notifications stopped", "error", err)
			return
		}
		if result == windows.WAIT_OBJECT_0 {
			return
		}
		select {
		case m.changes <- struct{}{}:
		default:
		}
		if next == m.low {
			next = m.high
		} else {
			next = m.low
		}
	}
}

// Changes coalesces native low/high transitions; callers must also poll.
func (m *memoryPlatform) Changes() <-chan struct{} { return m.changes }

func (m *memoryPlatform) read() (memorySample, error) {
	started := time.Now()
	status, err := getMemoryStatus()
	if err != nil {
		return memorySample{}, err
	}
	if max(status.TotalPhys, status.AvailPhys, status.AvailPageFile) > math.MaxInt64 {
		return memorySample{}, fmt.Errorf("host memory exceeds signed byte range")
	}
	commit := int64(status.AvailPageFile)
	result := memorySample{MemoryHeadroom: MemoryHeadroom{
		SampledAt: started, RAMLimit: int64(status.TotalPhys), AvailableRAM: int64(status.AvailPhys),
		AvailableCommit: &commit,
	}}
	var low, high bool
	notificationsKnown := true
	for _, notification := range []struct {
		handle windows.Handle
		value  *bool
	}{{m.low, &low}, {m.high, &high}} {
		var signaled int32
		ok, _, err := queryMemoryResourceNotification.Call(uintptr(notification.handle), uintptr(unsafe.Pointer(&signaled)))
		if ok == 0 {
			// Capacity remains useful if optional notification state is lost.
			notificationsKnown = false
			if message := err.Error(); message != m.notificationError {
				slog.Debug("memory pressure notification unavailable", "error", err)
				m.notificationError = message
			}
			continue
		}
		*notification.value = signaled != 0
	}
	if notificationsKnown {
		m.notificationError = ""
	}
	if low {
		result.Pressure = PressureCritical
	} else if high && notificationsKnown {
		result.Pressure = PressureNormal
	}
	result.compressed, err = readWindowsCompression()
	if err == nil {
		result.pageOut, err = m.readPagingCounter()
	}
	if err != nil {
		if message := err.Error(); message != m.activityError {
			slog.Debug("memory pressure observation unavailable", "error", err)
			m.activityError = message
		}
		return result, nil
	}
	m.activityError = ""
	result.activityKnown = true
	// The system counter also includes file-backed page writes. Only treat
	// its growth as reclaim feedback when scarce RAM or compression confirms
	// pressure; otherwise establish a new page-out baseline quietly.
	if !low && result.AvailableRAM >= max(1<<30, result.RAMLimit/20) && (!m.history.valid || result.compressed <= m.history.previous.compressed) {
		m.history.previous.pageOut = result.pageOut
	}
	return result, nil
}

// Close joins the notification watcher and releases OS resources. Calls to Read
// must finish first. Close may be called repeatedly.
func (m *memoryPlatform) Close() {
	m.once.Do(func() {
		windows.SetEvent(m.stop)
		<-m.done
		windows.CloseHandle(m.stop)
		windows.CloseHandle(m.low)
		windows.CloseHandle(m.high)
		if m.pagingQuery != 0 {
			pdhCloseQuery.Call(m.pagingQuery)
		}
	})
}

type windowsCompressionStatistics struct {
	version, processID                            uint32
	resident, original, stored, unique, partition uint64
}

func readWindowsCompression() (uint64, error) {
	// The store query is a versioned native ABI. Revision 4 uses six 64-bit
	// words; the first contains revision/PID, followed by resident bytes,
	// original data size, stored data size, unique data size, and partition.
	// System Task Manager on Windows 11 uses this read-only query as well.
	data := windowsCompressionStatistics{version: 4}
	query := struct {
		version, operation uint32
		data               unsafe.Pointer
		length             uint32
	}{1, 22, unsafe.Pointer(&data), uint32(unsafe.Sizeof(data))}
	err := windows.NtQuerySystemInformation(windows.SystemStoreInformation, unsafe.Pointer(&query), uint32(unsafe.Sizeof(query)), nil)
	runtime.KeepAlive(&data)
	if err != nil {
		return 0, fmt.Errorf("reading compression store: %w", err)
	}
	if data.resident == 0 {
		return 0, nil
	}
	if data.stored == 0 || data.original > math.MaxInt64 || data.stored > math.MaxInt64 || data.resident > math.MaxInt64 {
		return 0, fmt.Errorf("invalid compression store sizes")
	}
	// Estimate the original size represented by the resident store. The
	// store's overall data/size ratio is an estimate, not extra available RAM.
	hi, lo := bits.Mul64(data.resident, data.original)
	if hi >= data.stored {
		return 0, fmt.Errorf("compression store estimate overflows")
	}
	original, _ := bits.Div64(hi, lo, data.stored)
	return min(original, math.MaxInt64), nil
}

func (m *memoryPlatform) openPagingCounter() error {
	for _, proc := range []*windows.LazyProc{pdhOpenQuery, pdhAddCounter, pdhCollect, pdhRawValue, pdhCloseQuery} {
		if err := proc.Find(); err != nil {
			return err
		}
	}
	if status, _, _ := pdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&m.pagingQuery))); status != 0 {
		return fmt.Errorf("opening memory counter: PDH status %#x", status)
	}
	path, err := windows.UTF16PtrFromString(`\Memory\Pages Output/sec`)
	if err != nil {
		return err
	}
	if status, _, _ := pdhAddCounter.Call(m.pagingQuery, uintptr(unsafe.Pointer(path)), 0, uintptr(unsafe.Pointer(&m.pagingCounter))); status != 0 {
		pdhCloseQuery.Call(m.pagingQuery)
		m.pagingQuery = 0
		return fmt.Errorf("adding memory counter: PDH status %#x", status)
	}
	return nil
}

func (m *memoryPlatform) readPagingCounter() (uint64, error) {
	if m.pagingQuery == 0 {
		return 0, fmt.Errorf("memory paging counter unavailable")
	}
	if status, _, _ := pdhCollect.Call(m.pagingQuery); status != 0 {
		return 0, fmt.Errorf("collecting memory counter: PDH status %#x", status)
	}
	var raw struct {
		status        uint32
		timestamp     windows.Filetime
		first, second int64
		count         uint32
	}
	if status, _, _ := pdhRawValue.Call(m.pagingCounter, 0, uintptr(unsafe.Pointer(&raw))); status != 0 {
		return 0, fmt.Errorf("reading memory counter: PDH status %#x", status)
	}
	// PDH reports both valid and newly collected data as successful samples.
	if raw.status > 1 || raw.first < 0 || uint64(raw.first) > math.MaxInt64/uint64(os.Getpagesize()) {
		return 0, fmt.Errorf("invalid memory paging counter: status %#x, value %d", raw.status, raw.first)
	}
	return uint64(raw.first) * uint64(os.Getpagesize()), nil
}
