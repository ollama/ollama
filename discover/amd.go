// AMD discovery needs a small amount of backend-specific handling beyond the
// generic llama-server device list. ROCm devices expose their real capability
// as gfx targets, and the shipped rocBLAS kernels define which of those
// targets are actually usable. On Windows, older HIP driver installs can also
// leave ROCm libraries present but too old to support GPU inference. These
// helpers keep that extra validation and warning logic in one place.
package discover

import (
	"bufio"
	"log/slog"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
)

// gfxTargetRegex matches ROCm stderr lines like:
//
//	Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
//	Device 1: AMD Radeon Pro VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 16368 MiB
var gfxTargetRegex = regexp.MustCompile(
	`Device\s+(\d+):.*,\s+(gfx[0-9a-f]+)[\s:(]`,
)

func parseROCmGFXTargets(output string) map[int]string {
	gfxByIndex := make(map[int]string)

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		if matches := gfxTargetRegex.FindStringSubmatch(scanner.Text()); matches != nil {
			idx, _ := strconv.Atoi(matches[1])
			gfxByIndex[idx] = matches[2]
		}
	}

	return gfxByIndex
}

func parseGFXTarget(gfx string) (int, int) {
	gfx, ok := strings.CutPrefix(gfx, "gfx")
	if !ok || len(gfx) < 3 {
		return 0, 0
	}

	major, err := strconv.ParseInt(gfx[:len(gfx)-2], 16, 32)
	if err != nil {
		return 0, 0
	}
	minor, err := strconv.ParseInt(gfx[len(gfx)-2:], 16, 32)
	if err != nil {
		return 0, 0
	}

	return int(major), int(minor)
}

// rocblasGFXTargets scans the rocblas library directory for supported gfx targets
// by looking for TensileLibrary_lazy_gfxNNNN.dat files.
func rocblasGFXTargets(libDirs []string) map[string]bool {
	targets := make(map[string]bool)
	for _, dir := range libDirs {
		files, _ := filepath.Glob(filepath.Join(dir, "rocblas", "library", "TensileLibrary_lazy_gfx*.dat"))
		for _, f := range files {
			base := filepath.Base(f)
			if t, ok := strings.CutPrefix(base, "TensileLibrary_lazy_"); ok {
				if t, ok = strings.CutSuffix(t, ".dat"); ok {
					targets[t] = true
				}
			}
		}
	}
	return targets
}

// filterUnsupportedROCmDevices removes ROCm devices whose gfx target doesn't have
// matching rocblas kernels bundled.
func filterUnsupportedROCmDevices(devices []ml.DeviceInfo, libDirs []string) []ml.DeviceInfo {
	supported := rocblasGFXTargets(libDirs)
	if len(supported) == 0 {
		return devices
	}

	var filtered []ml.DeviceInfo
	for _, dev := range devices {
		if dev.Library != "ROCm" {
			filtered = append(filtered, dev)
			continue
		}
		gfx := dev.GFXTarget
		if gfx == "" {
			filtered = append(filtered, dev)
			continue
		}
		if supported[gfx] {
			filtered = append(filtered, dev)
		} else {
			slog.Warn("dropping ROCm device — no rocblas support for gfx target",
				"device", dev.Name, "gfx_target", gfx, "supported", supported,
				"hint", "set HSA_OVERRIDE_GFX_VERSION to map to a supported target")
		}
	}
	return filtered
}

func detectOldAMDDriverWindows() {
	if runtime.GOOS != "windows" {
		return
	}
	_, errV6 := exec.LookPath("amdhip64_6.dll")
	_, errV7 := exec.LookPath("amdhip64_7.dll")
	if errV6 == nil && errV7 != nil {
		slog.Warn("AMD driver is too old. Update your AMD driver to enable GPU inference.")
	}
}
