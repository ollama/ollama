package discover

import (
	"bufio"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
)

type memoryPlatform struct {
	history       memoryHistory
	proc, root    string
	zramSources   string
	stallTotal    uint64
	stallTime     time.Time
	activityError string
}

func newMemoryMonitor() (*MemoryMonitor, error) {
	return &MemoryMonitor{memoryPlatform: memoryPlatform{proc: "/proc", root: "/"}}, nil
}

// Changes is nil because Linux observations are sampled periodically.
func (m *memoryPlatform) Changes() <-chan struct{} { return nil }

// Close releases no resources on Linux and may be called repeatedly.
func (m *memoryPlatform) Close() {}

func (m *memoryPlatform) read() (memorySample, error) {
	sample, err := readLinuxMemory(m.proc, m.root)
	if err != nil {
		return sample, err
	}
	err = m.readActivity(&sample)
	if err == nil && !sample.activityKnown {
		err = fmt.Errorf("zswap contents unavailable")
	}
	if err != nil {
		sample.activityKnown = false
		if message := err.Error(); message != m.activityError {
			slog.Debug("memory pressure observation unavailable", "error", err)
			m.activityError = message
		}
	} else {
		m.activityError = ""
	}
	return sample, nil
}

// The full cgroup hierarchy must be visible: siblings can consume capacity
// below a shared parent limit. Swap is not room to retain resident history.
func readLinuxMemory(proc, root string) (memorySample, error) {
	m := memorySample{MemoryHeadroom: MemoryHeadroom{SampledAt: time.Now(), Pressure: PressureNormal}}
	f, err := os.Open(filepath.Join(proc, "meminfo"))
	if err != nil {
		return m, err
	}
	defer f.Close()
	values := map[string]uint64{}
	var compressionReported bool
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "MemTotal:", "MemAvailable:", "CommitLimit:", "Committed_AS:", "Zswapped:":
			if fields[0] == "Zswapped:" {
				compressionReported = true
			}
			if len(fields) != 3 || fields[2] != "kB" {
				if fields[0] == "Zswapped:" {
					continue
				}
				return m, fmt.Errorf("invalid meminfo field %q", s.Text())
			}
			value, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil || value > math.MaxInt64/1024 {
				if fields[0] == "Zswapped:" {
					continue
				}
				return m, fmt.Errorf("invalid meminfo size %q", s.Text())
			}
			values[fields[0]] = value * 1024
		}
	}
	if err := s.Err(); err != nil {
		return m, err
	}
	for _, key := range []string{"MemTotal:", "MemAvailable:", "CommitLimit:", "Committed_AS:"} {
		if _, ok := values[key]; !ok {
			return m, fmt.Errorf("meminfo missing %s", key)
		}
	}
	m.RAMLimit, m.AvailableRAM = int64(values["MemTotal:"]), int64(values["MemAvailable:"])
	if m.RAMLimit == 0 || m.AvailableRAM > m.RAMLimit {
		return m, fmt.Errorf("invalid physical memory capacity")
	}
	mode, err := os.ReadFile(filepath.Join(proc, "sys/vm/overcommit_memory"))
	if err != nil {
		return m, err
	}
	switch strings.TrimSpace(string(mode)) {
	case "0", "1":
		// CommitLimit is not an allocation ceiling in these modes.
	case "2":
		commit := int64(values["CommitLimit:"] - min(values["CommitLimit:"], values["Committed_AS:"]))
		m.AvailableCommit = &commit
	default:
		return m, fmt.Errorf("invalid overcommit mode %q", mode)
	}
	if err := constrainMemoryByCgroup(&m.MemoryHeadroom, proc, root); err != nil {
		return m, err
	}
	m.compressed, m.activityKnown = values["Zswapped:"]
	if !m.activityKnown && !compressionReported {
		// Disabled zswap can still have stored pages; use Zswapped whenever it
		// exists. Without that field, establish that the feature is absent/off.
		enabled, err := os.ReadFile(filepath.Join(root, "sys/module/zswap/parameters/enabled"))
		m.activityKnown = errors.Is(err, os.ErrNotExist) || err == nil && strings.TrimSpace(string(enabled)) == "N"
	}
	return m, nil
}

func constrainMemoryByCgroup(m *MemoryHeadroom, proc, root string) error {
	data, err := os.ReadFile(filepath.Join(proc, "self/cgroup"))
	if err != nil {
		return err
	}
	var group string
	var unified, populated bool
	for line := range strings.SplitSeq(string(data), "\n") {
		if line == "" {
			continue
		}
		populated = true
		fields := strings.SplitN(line, ":", 3)
		if len(fields) != 3 {
			return fmt.Errorf("invalid process cgroup entry %q", line)
		}
		if fields[0] == "0" && fields[1] == "" {
			group = fields[2]
			unified = true
		}
		for controller := range strings.SplitSeq(fields[1], ",") {
			if controller == "memory" {
				return fmt.Errorf("adaptive retention requires cgroup v2 memory accounting")
			}
		}
	}
	if !populated {
		return fmt.Errorf("empty process cgroup membership")
	}
	if !unified {
		return nil // No cgroup memory controller.
	}
	if !filepath.IsAbs(group) || filepath.Clean(group) != group {
		return fmt.Errorf("invalid cgroup path %q", group)
	}
	mounts, err := os.ReadFile(filepath.Join(proc, "self/mountinfo"))
	if err != nil {
		return err
	}
	for line := range strings.SplitSeq(string(mounts), "\n") {
		before, after, ok := strings.Cut(line, " - ")
		if !ok || !strings.HasPrefix(after, "cgroup2 ") {
			continue
		}
		fields := strings.Fields(before)
		if len(fields) < 6 {
			return fmt.Errorf("invalid cgroup mount %q", line)
		}
		// mountinfo escapes whitespace and backslashes in path fields.
		unescape := strings.NewReplacer(`\040`, " ", `\011`, "\t", `\012`, "\n", `\134`, `\`)
		mountRoot, mountPoint := unescape.Replace(fields[3]), unescape.Replace(fields[4])
		if mountRoot != "/" {
			continue // A subtree mount can hide stricter ancestor limits.
		}
		rel, err := filepath.Rel(mountRoot, group)
		if err != nil || rel == ".." || strings.HasPrefix(rel, "../") {
			continue
		}
		mount := filepath.Join(root, mountPoint)
		// A cgroup namespace can present a non-root group as "/". Only
		// non-root groups have cgroup.events, independently of controllers.
		if _, err := os.Stat(filepath.Join(mount, "cgroup.events")); err == nil {
			continue
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		for dir := filepath.Join(mount, rel); ; dir = filepath.Dir(dir) {
			limit := uint64(math.MaxInt64)
			missing := 0
			for _, name := range []string{"memory.max", "memory.high"} {
				data, err := os.ReadFile(filepath.Join(dir, name))
				if errors.Is(err, os.ErrNotExist) {
					missing++
					continue
				}
				if err != nil {
					return err
				}
				value := strings.TrimSpace(string(data))
				if value == "max" {
					continue
				}
				n, err := strconv.ParseUint(value, 10, 64)
				if err != nil {
					return fmt.Errorf("invalid %s: %w", filepath.Join(dir, name), err)
				}
				limit = min(limit, n)
			}
			if missing == 1 {
				return fmt.Errorf("incomplete memory controller in %s", dir)
			}
			if missing == 2 {
				// The root and groups without the memory controller have no
				// limits. Their ancestors may still constrain this process.
				if _, err := os.Stat(dir); err != nil {
					return err
				}
			}
			if limit < math.MaxInt64 {
				used, err := getUint64ValueFromFile(filepath.Join(dir, "memory.current"))
				if err != nil {
					return err
				}
				m.RAMLimit = min(m.RAMLimit, int64(limit))
				m.AvailableRAM = min(m.AvailableRAM, int64(limit-min(limit, used)))
			}
			if dir == mount {
				return nil
			}
		}
	}
	return fmt.Errorf("cannot locate cgroup v2 memory hierarchy")
}

func (m *memoryPlatform) readActivity(sample *memorySample) error {
	data, err := os.ReadFile(filepath.Join(m.proc, "vmstat"))
	if err != nil {
		return err
	}
	var swapOut bool
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0] != "pswpout" && fields[0] != "zswpout" {
			continue
		}
		if len(fields) != 2 {
			return fmt.Errorf("invalid memory activity counter %q", line)
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil || value > math.MaxInt64/uint64(os.Getpagesize()) {
			return fmt.Errorf("invalid memory activity count %q", line)
		}
		if fields[0] == "pswpout" {
			sample.pageOut, swapOut = value*uint64(os.Getpagesize()), true
		} else {
			sample.compressions = value
		}
	}
	if !swapOut {
		return fmt.Errorf("vmstat missing pswpout")
	}

	// Only zram devices used as swap represent reclaimed anonymous memory.
	// A zram filesystem or cache must not look like host memory pressure.
	data, err = os.ReadFile(filepath.Join(m.proc, "swaps"))
	if err != nil {
		return err
	}
	var devices []string
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0] == "Filename" {
			continue
		}
		if len(fields) != 5 || !filepath.IsAbs(fields[0]) {
			return fmt.Errorf("invalid swap device %q", line)
		}
		if fields[1] != "partition" || !strings.HasPrefix(fields[0], "/dev/") {
			continue
		}
		name := filepath.Base(fields[0])
		if strings.HasPrefix(fields[0], "/dev/") && !strings.HasPrefix(name, "zram") {
			resolved, err := filepath.EvalSymlinks(filepath.Join(m.root, fields[0]))
			if err != nil {
				return err
			}
			name = filepath.Base(resolved)
		}
		index, zram := strings.CutPrefix(name, "zram")
		if !zram {
			continue
		}
		if _, err := strconv.ParseUint(index, 10, 32); err != nil {
			return fmt.Errorf("invalid zram device %q", name)
		}
		devices = append(devices, name)
	}
	slices.Sort(devices)
	devices = slices.Compact(devices)
	if sources := strings.Join(devices, ","); sources != m.zramSources {
		m.zramSources = sources
		m.history.valid = false
	}
	for _, name := range devices {
		dir := filepath.Join(m.root, "sys/block", name)
		data, err := os.ReadFile(filepath.Join(dir, "mm_stat"))
		if err != nil {
			return err
		}
		fields := strings.Fields(string(data))
		if len(fields) < 3 {
			return fmt.Errorf("invalid %s/mm_stat", name)
		}
		original, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil || original > math.MaxInt64-sample.compressed {
			return fmt.Errorf("invalid %s original size", name)
		}
		sample.compressed += original
		data, err = os.ReadFile(filepath.Join(dir, "stat"))
		if err != nil {
			return err
		}
		fields = strings.Fields(string(data))
		if len(fields) < 11 {
			return fmt.Errorf("invalid %s/stat", name)
		}
		writes, err := strconv.ParseUint(fields[4], 10, 64)
		if err != nil || writes > math.MaxInt64-sample.compressions {
			return fmt.Errorf("invalid %s write count", name)
		}
		sample.compressions += writes
	}

	data, err = os.ReadFile(filepath.Join(m.proc, "pressure/memory"))
	if errors.Is(err, os.ErrNotExist) {
		return nil // PSI is an optional kernel capability.
	}
	if err != nil {
		return err
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0] != "full" {
			continue
		}
		for _, field := range fields[1:] {
			value, ok := strings.CutPrefix(field, "total=")
			if !ok {
				continue
			}
			total, err := strconv.ParseUint(value, 10, 64)
			if err != nil {
				return fmt.Errorf("invalid memory stall total %q", field)
			}
			// Stall time alone also reflects reclaim during ordinary file I/O.
			// Corroborate it with scarce capacity before calling it pressure.
			if !m.stallTime.IsZero() && total > m.stallTotal && sample.AvailableRAM < max(1<<30, sample.RAMLimit/20) {
				sample.Pressure = PressureElevated
			}
			m.stallTotal, m.stallTime = total, sample.SampledAt
			return nil
		}
	}
	return fmt.Errorf("memory pressure missing full stall total")
}
