//go:build windows

package network

import (
	"context"
	"os/exec"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var (
	wininet                   = syscall.NewLazyDLL("wininet.dll")
	internetGetConnectedState = wininet.NewProc("InternetGetConnectedState")
)

const INTERNET_CONNECTION_OFFLINE = 0x20

func (m *Monitor) startPlatformMonitor(ctx context.Context) {
	go m.watchNetworkChanges(ctx)
}

func (m *Monitor) checkPlatformConnectivity() bool {
	// First check Windows Internet API
	if internetGetConnectedState.Find() == nil {
		var flags uint32
		r, _, _ := internetGetConnectedState.Call(
			uintptr(unsafe.Pointer(&flags)),
			0,
		)

		if r == 1 && (flags&INTERNET_CONNECTION_OFFLINE) == 0 {
			// Also verify with netsh that interfaces are actually connected
			return m.checkWindowsInterfaces()
		}
	}

	return false
}

func (m *Monitor) checkWindowsInterfaces() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "netsh", "interface", "show", "interface")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	for line := range strings.SplitSeq(string(output), "\n") {
		line = strings.ToLower(strings.TrimSpace(line))

		// Look for a “connected” interface that isn’t “disconnected” or “loopback”
		if strings.Contains(line, "connected") &&
			!strings.Contains(line, "disconnected") &&
			!strings.Contains(line, "loopback") {
			return true
		}
	}

	return false
}

func (m *Monitor) watchNetworkChanges(ctx context.Context) {
	// Windows doesn't have a simple built-in tool like scutil,
	// so poll frequently to detect changes
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	// Initial check
	m.checkConnectivity()

	var lastState bool = m.checkPlatformConnectivity()

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopChan:
			return
		case <-ticker.C:
			currentState := m.checkPlatformConnectivity()
			if currentState != lastState {
				lastState = currentState
				m.checkConnectivity()
			}
		}
	}
}
