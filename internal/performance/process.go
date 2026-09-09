package performance

import (
	"strings"

	"github.com/shirou/gopsutil/v3/process"
)

type ProcessInfo struct {
	PID           int32
	Name          string
	MemoryPercent float32 // percentage of total RAM (0-100)
	MemoryBytes   uint64
	proc          *process.Process
}

// GetHighMemoryProcesses returns processes consuming more than a threshold percentage of total memory (threshold in 0-100)
func GetHighMemoryProcesses(thresholdPercent float32, totalRAM uint64) ([]ProcessInfo, error) {
	procs, err := process.Processes()
	if err != nil {
		return nil, err
	}

	var results []ProcessInfo
	for _, p := range procs {
		if p.Pid <= 4 {
			continue
		}

		name, err := p.Name()
		if err != nil {
			continue
		}

		if IsWhitelisted(name) {
			continue
		}

		percent, err := p.MemoryPercent()
		if err != nil {
			continue
		}

		if percent > thresholdPercent {
			var memBytes uint64
			memInfo, err := p.MemoryInfo()
			if err == nil && memInfo != nil {
				memBytes = memInfo.RSS
			} else {
				memBytes = uint64(float64(totalRAM) * float64(percent) / 100.0)
			}

			results = append(results, ProcessInfo{
				PID:           p.Pid,
				Name:          name,
				MemoryPercent: percent,
				MemoryBytes:   memBytes,
				proc:          p,
			})
		}
	}

	return results, nil
}

// IsWhitelisted returns true if the process should never be killed
func IsWhitelisted(name string) bool {
	lowerName := strings.ToLower(name)
	lowerName = strings.TrimSuffix(lowerName, ".exe")

	whitelist := []string{
		"ollama",
		"systemd",
		"launchd",
		"explorer",
		"system idle process",
		"system",
		"registry",
		"smss",
		"csrss",
		"wininit",
		"services",
		"lsass",
		"svchost",
		"winlogon",
	}

	for _, item := range whitelist {
		if lowerName == item {
			return true
		}
	}
	return false
}

// KillProcess terminates a high-memory process
func KillProcess(p ProcessInfo) error {
	if p.proc == nil {
		// Try to find the process if nil
		proc, err := process.NewProcess(p.PID)
		if err != nil {
			return err
		}
		p.proc = proc
	}
	return p.proc.Kill()
}
