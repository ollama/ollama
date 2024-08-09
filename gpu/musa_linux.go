package gpu

import (
	"bufio"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

const (
	MUSADriverVersionFile    = "/sys/module/mtgpu/version"
	MUSADriverProcfsDirGlob  = "/proc/driver/musa/gpu*"
	MUSADriverDeviceNameFile = "devname"
	MUSADriverVRAMInfoFile   = "vram_info"
	MUSADriverMemoryFile     = "memory"
)

// Gather GPU information from the mtgpu driver if any supported GPUs are detected
func MUSAGetGPUInfo() []MusaGPUInfo {
	resp := []MusaGPUInfo{}
	if !MUSADetected() {
		return resp
	}

	// Opportunistic logging of driver version to aid in troubleshooting
	driverMajor, driverMinor, err := MUSADriverVersion()
	if err != nil {
		slog.Warn("ollama recommends running the https://www.mthreads.com/pes/drivers/index", "error", err)
	}

	// Determine if the user has already pre-selected which GPUs to look at, then ignore the others
	var visibleDevices []string
	mthreadsVD := envconfig.MthreadsVisibleDevices()
	musaVD := envconfig.MusaVisibleDevices()
	switch {
	case mthreadsVD != "":
		visibleDevices = strings.Split(mthreadsVD, ",")
	case musaVD != "":
		visibleDevices = strings.Split(musaVD, ",")
	}

	matches, _ := filepath.Glob(MUSADriverProcfsDirGlob)
	for _, devDir := range matches {
		slog.Debug("evaluating mtgpu node " + devDir)

		deviceNameFile := filepath.Join(devDir, MUSADriverDeviceNameFile)
		buf, err := os.ReadFile(deviceNameFile)
		if err != nil {
			slog.Debug("failed to read procfs node", "file", deviceNameFile, "error", err)
			break
		}
		deviceName := strings.TrimSpace(string(buf))
		parts := strings.Split(deviceName, ".")
		if len(parts) != 2 {
			slog.Debug("malformed", "deviceName", deviceName)
			break
		}
		gpuIndex, err := strconv.Atoi(parts[1])
		if err != nil {
			slog.Debug("failed to parse GPU index", "error", err)
			continue
		}

		// Look up the memory for the current node
		totalMemory := uint64(0)
		usedMemory := uint64(0)

		vramInfoFile := filepath.Join(devDir, MUSADriverVRAMInfoFile)
		fp, err := os.Open(vramInfoFile)
		if err != nil {
			slog.Debug("failed to open procfs node", "file", vramInfoFile, "error", err)
			continue
		}
		defer fp.Close()

		scanner := bufio.NewScanner(fp)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if strings.HasPrefix(line, "MemTotal") {
				ver := strings.Fields(line)
				if len(ver) != 3 {
					slog.Debug("malformed", "MemTotal", line)
					continue
				}
				// values here are in hex, strip off the lead 0x and parse so we can compare the numeric (decimal) values in mtgpu
				memTotal, err := strconv.ParseUint(strings.TrimPrefix(ver[2], "0x"), 16, 64)
				if err != nil {
					slog.Debug("failed to parse procfs node", "file", vramInfoFile, "error", err)
					break
				}
				totalMemory = memTotal
			}
		}

		memoryFile := filepath.Join(devDir, MUSADriverMemoryFile)
		usedMemory, err = getMusaFreeMemory(memoryFile)
		if err != nil {
			slog.Debug("failed to update used memory", "error", err)
		}

		// Map over to DRM location to find the vendor and device
		var vendor, device uint64

		drmMatches, _ := filepath.Glob(filepath.Join(DRMDeviceDirGlob, "misc"))
		for _, devDir := range drmMatches {
			if _, err := os.Stat(filepath.Join(devDir, deviceName)); err == nil {
				for _, name := range []string{DRMVendorFile, DRMDeviceFile} {
					file := filepath.Join(devDir, "../", name)
					buf, err := os.ReadFile(file)
					if err != nil {
						slog.Debug("failed to read sysfs node", "file", file, "error", err)
						continue
					}
					str := strings.TrimPrefix(strings.TrimSpace(string(buf)), "0x")
					value, err := strconv.ParseUint(str, 16, 64)
					if err != nil {
						slog.Debug("malformed", "str", str, "error", err)
					}
					if name == DRMVendorFile {
						vendor = value
					} else {
						device = value
					}
				}
				break
			}
		}

		var name string
		// TODO - PCI ID lookup
		if vendor > 0 && device > 0 {
			name = fmt.Sprintf("%04x:%04x", vendor, device)
		}

		gpuInfo := MusaGPUInfo{
			GpuInfo: GpuInfo{
				Library: "musa",
				memInfo: memInfo{
					TotalMemory: totalMemory,
					FreeMemory:  (totalMemory - usedMemory),
				},
				ID:            fmt.Sprintf("%d", gpuIndex),
				Name:          name,
				MinimumMemory: musaMinimumMemory,
				DriverMajor:   driverMajor,
				DriverMinor:   driverMinor,
			},
			index:          gpuIndex,
			memoryFilepath: memoryFile,
		}

		// If the user wants to filter to a subset of devices, filter out if we aren't a match
		if len(visibleDevices) > 0 {
			include := false
			for _, visible := range visibleDevices {
				if visible == "all" {
					include = true
					break
				}
				if visible == gpuInfo.ID {
					include = true
					break
				}
			}
			if !include {
				slog.Info("filtering out device per user request", "id", gpuInfo.ID, "visible_devices", visibleDevices)
				continue
			}
		}

		// The GPU has passed all the verification steps and is supported
		resp = append(resp, gpuInfo)
	}

	if len(resp) == 0 {
		slog.Info("no compatible mtgpu devices detected")
	}
	return resp
}

// Quick check for MUSA driver so we can skip mtgpu discovery if not present
func MUSADetected() bool {
	// Some driver versions (older?) don't have a version file, so just lookup the parent dir
	sysfsDir := filepath.Dir(MUSADriverVersionFile)
	_, err := os.Stat(sysfsDir)
	if errors.Is(err, os.ErrNotExist) {
		slog.Debug("mtgpu driver not detected " + sysfsDir)
		return false
	} else if err != nil {
		slog.Debug("error looking up musa driver", "path", sysfsDir, "error", err)
		return false
	}
	return true
}

func MUSADriverVersion() (driverMajor, driverMinor int, err error) {
	return 0, 0, fmt.Errorf("mtgpu driver version not implemented")
}

func (gpus MusaGPUInfoList) RefreshFreeMemory() error {
	if len(gpus) == 0 {
		return nil
	}
	for i := range gpus {
		usedMemory, err := getFreeMemory(gpus[i].memoryFilepath)
		if err != nil {
			return err
		}
		slog.Debug("updating musa free memory", "gpu", gpus[i].ID, "name", gpus[i].Name, "before", format.HumanBytes2(gpus[i].FreeMemory), "now", format.HumanBytes2(gpus[i].TotalMemory-usedMemory))
		gpus[i].FreeMemory = gpus[i].TotalMemory - usedMemory
	}
	return nil
}

func getMusaFreeMemory(memoryFile string) (uint64, error) {
	fp, err := os.Open(memoryFile)
	if err != nil {
		return 0, fmt.Errorf("failed to open procfs node %s %w", memoryFile, err)
	}
	defer fp.Close()

	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "MemoryUsageAllocGPUMemLMA") {
			ver := strings.Fields(line)
			if len(ver) != 2 {
				slog.Debug("malformed", "MemoryUsageAllocGPUMemLMA", line)
				continue
			}
			memUsed, err := strconv.ParseUint(ver[1], 10, 64)
			if err != nil {
				slog.Debug("failed to parse procfs node", "file", memoryFile, "error", err)
				break
			}
			return memUsed, nil
		}
	}

	return 0, fmt.Errorf("failed to find MemoryUsageAllocGPUMemLMA in %s", memoryFile)
}
