//go:build darwin

package network

import (
	"bufio"
	"context"
	"os/exec"
	"strings"
	"time"
)

func (m *Monitor) startPlatformMonitor(ctx context.Context) {
	go m.watchNetworkChanges(ctx)
}

func (m *Monitor) checkPlatformConnectivity() bool {
	// Check if we have active network interfaces
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "scutil", "--nwi")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	outputStr := string(output)

	// Check for active interfaces with IP addresses
	hasIPv4 := strings.Contains(outputStr, "IPv4") &&
		!strings.Contains(outputStr, "IPv4 : No addresses")
	hasIPv6 := strings.Contains(outputStr, "IPv6") &&
		!strings.Contains(outputStr, "IPv6 : No addresses")

	if !hasIPv4 && !hasIPv6 {
		return false
	}

	// Check for active network interfaces
	lines := strings.Split(outputStr, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Look for active ethernet (en) or VPN (utun) interfaces
		if strings.HasPrefix(line, "en") || strings.HasPrefix(line, "utun") {
			if strings.Contains(line, "flags") && !strings.Contains(line, "inactive") {
				return true
			}
		}
	}

	return false
}

func (m *Monitor) watchNetworkChanges(ctx context.Context) {
	// Use scutil to watch for network changes
	cmd := exec.CommandContext(ctx, "scutil")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return
	}

	if err := cmd.Start(); err != nil {
		return
	}
	defer cmd.Wait()

	// Watch for network state changes
	stdin.Write([]byte("n.add State:/Network/Global/IPv4\n"))
	stdin.Write([]byte("n.add State:/Network/Global/IPv6\n"))
	stdin.Write([]byte("n.add State:/Network/Interface\n"))
	stdin.Write([]byte("n.watch\n"))

	// Trigger initial check
	m.checkConnectivity()

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return
		case <-m.stopChan:
			return
		default:
			// Any output from scutil indicates a network change
			// Trigger connectivity check
			m.checkConnectivity()
		}
	}
}
