//go:build windows || darwin

package version

import (
	"os/exec"
	"runtime/debug"
	"strings"
)

var Version string = "0.0.0"

// GetVersion returns the version, with fallback to git or build info
func GetVersion() string {
	// If version is set via ldflags, use it
	if Version != "" && Version != "0.0.0" {
		return Version
	}

	// Try to get from build info
	if buildinfo, ok := debug.ReadBuildInfo(); ok {
		if buildinfo.Main.Version != "" && buildinfo.Main.Version != "(devel)" {
			return buildinfo.Main.Version
		}
	}

	// In development, try to get from git
	if cmd := exec.Command("git", "describe", "--tags", "--first-parent", "--abbrev=7", "--long", "--dirty", "--always"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			version := strings.TrimSpace(string(output))
			version = strings.TrimPrefix(version, "v")
			if version != "" {
				return version
			}
		}
	}

	// Fallback
	return "dev"
}
