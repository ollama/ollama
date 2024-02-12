package gpu

import (
	"bufio"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// TODO - windows vs. non-windows vs darwin

// Discovery logic for AMD/ROCm GPUs

const (
	DriverVersionFile     = "/sys/module/amdgpu/version"
	GPUPropertiesFileGlob = "/sys/class/kfd/kfd/topology/nodes/*/properties"
	// TODO probably break these down per GPU to make the logic simpler
	GPUTotalMemoryFileGlob = "/sys/class/kfd/kfd/topology/nodes/*/mem_banks/*/properties" // size_in_bytes line
	GPUUsedMemoryFileGlob  = "/sys/class/kfd/kfd/topology/nodes/*/mem_banks/*/used_memory"
)

func AMDDetected() bool {
	_, err := AMDDriverVersion()
	return err == nil
}

func AMDDriverVersion() (string, error) {
	_, err := os.Stat(DriverVersionFile)
	if err != nil {
		return "", err
	}
	fp, err := os.Open(DriverVersionFile)
	if err != nil {
		return "", err
	}
	defer fp.Close()
	verString, err := io.ReadAll(fp)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(verString)), nil
}

func AMDGFXVersions() []Version {
	res := []Version{}
	matches, _ := filepath.Glob(GPUPropertiesFileGlob)
	for _, match := range matches {
		fp, err := os.Open(match)
		if err != nil {
			slog.Debug(fmt.Sprintf("failed to open sysfs node file %s: %s", match, err))
			continue
		}
		defer fp.Close()

		scanner := bufio.NewScanner(fp)
		// optionally, resize scanner's capacity for lines over 64K, see next example
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if strings.HasPrefix(line, "gfx_target_version") {
				ver := strings.Fields(line)
				if len(ver) != 2 || len(ver[1]) < 5 {
					slog.Debug("malformed " + line)
					continue
				}
				l := len(ver[1])
				patch, err1 := strconv.ParseUint(ver[1][l-2:l], 10, 32)
				minor, err2 := strconv.ParseUint(ver[1][l-4:l-2], 10, 32)
				major, err3 := strconv.ParseUint(ver[1][:l-4], 10, 32)
				if err1 != nil || err2 != nil || err3 != nil {
					slog.Debug("malformed int " + line)
					continue
				}

				res = append(res, Version{
					Major: uint(major),
					Minor: uint(minor),
					Patch: uint(patch),
				})
			}
		}
	}
	return res
}

func (v Version) ToGFXString() string {
	return fmt.Sprintf("gfx%d%d%d", v.Major, v.Minor, v.Patch)
}
