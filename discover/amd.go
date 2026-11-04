// AMD discovery needs a small amount of backend-specific handling beyond the
// generic llama-server device list. ROCm devices expose their real capability
// as gfx targets, and the shipped rocBLAS kernels define which of those
// targets are actually usable. On Linux, KFD topology and DRM sysfs attributes
// provide the integrated-vs-discrete signal needed for scheduler decisions. On
// Windows, older HIP driver installs can also leave ROCm libraries present but
// too old to support GPU inference. These helpers keep that extra validation
// and warning logic in one place.
package discover

import (
	"bufio"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
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

var pciIDRegex = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$`)

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

// HSA_OVERRIDE_GFX_VERSION changes the effective HIP/rocBLAS target even
// though KFD/sysfs still reports the physical ASIC.
func hsaOverrideGFXTarget() string {
	return rocmGFXTargetOverride(os.Getenv("HSA_OVERRIDE_GFX_VERSION"))
}

func rocmGFXTargetOverride(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "gfx") {
		if major, minor := parseGFXTarget(value); major != 0 || minor != 0 {
			return value
		}
		return ""
	}

	parts := strings.Split(value, ".")
	if len(parts) != 3 {
		return ""
	}

	var digits [3]uint64
	for i, part := range parts {
		digit, err := strconv.ParseUint(part, 10, 8)
		if err != nil || digit > 0xf {
			return ""
		}
		digits[i] = digit
	}

	return "gfx" +
		strconv.FormatUint(digits[0], 10) +
		strconv.FormatUint(digits[1], 16) +
		strconv.FormatUint(digits[2], 16)
}

func setROCmGFXTarget(device *ml.DeviceInfo, gfx string) {
	if gfx == "" || device.Library != "ROCm" {
		return
	}
	device.GFXTarget = gfx
	device.ComputeMajor, device.ComputeMinor = parseGFXTarget(gfx)
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

type rocmLinuxSysfsDevice struct {
	pciID      string
	gfxTarget  string
	integrated bool
	known      bool
}

func refineLinuxROCmDevices(devices []ml.DeviceInfo) []ml.DeviceInfo {
	if runtime.GOOS != "linux" {
		return devices
	}
	applyLinuxROCmRefinement(devices, "/sys")
	return devices
}

func applyLinuxROCmRefinement(devices []ml.DeviceInfo, sysfsRoot string) bool {
	var rocmIndexes []int
	for i, device := range devices {
		if device.Library == "ROCm" {
			rocmIndexes = append(rocmIndexes, i)
		}
	}
	if len(rocmIndexes) == 0 {
		return false
	}

	sysfsDevices, err := readROCmLinuxSysfsDevices(sysfsRoot)
	if err != nil {
		slog.Debug("linux rocm device refinement unavailable", "error", err)
		return false
	}
	if len(sysfsDevices) != len(rocmIndexes) {
		slog.Debug("linux rocm device refinement skipped: device count mismatch",
			"llama_server_count", len(rocmIndexes), "kfd_count", len(sysfsDevices))
		return false
	}

	byPCI := map[string]rocmLinuxSysfsDevice{}
	byGFX := uniqueROCmSysfsDevicesByGFX(sysfsDevices)
	for _, sysfsDevice := range sysfsDevices {
		if sysfsDevice.pciID != "" {
			byPCI[strings.ToLower(sysfsDevice.pciID)] = sysfsDevice
		}
	}

	refined := 0
	for i, rocmIndex := range rocmIndexes {
		device := &devices[rocmIndex]
		sysfsDevice, ok := matchROCmLinuxSysfsDevice(*device, i, sysfsDevices, byPCI, byGFX)
		if !ok {
			slog.Debug("linux rocm device refinement skipped: no stable match",
				"device", device.Name, "pci_id", device.PCIID, "gfx", device.GFXTarget)
			continue
		}
		applyROCmLinuxSysfsDevice(device, sysfsDevice)
		refined++
	}

	if refined == 0 {
		return false
	}

	slog.Debug("linux rocm device refinement applied", "devices", refined)
	return true
}

func uniqueROCmSysfsDevicesByGFX(sysfsDevices []rocmLinuxSysfsDevice) map[string]rocmLinuxSysfsDevice {
	byGFX := map[string]rocmLinuxSysfsDevice{}
	duplicates := map[string]bool{}
	for _, sysfsDevice := range sysfsDevices {
		if sysfsDevice.gfxTarget == "" {
			continue
		}
		if _, ok := byGFX[sysfsDevice.gfxTarget]; ok {
			duplicates[sysfsDevice.gfxTarget] = true
			continue
		}
		byGFX[sysfsDevice.gfxTarget] = sysfsDevice
	}
	for gfx := range duplicates {
		delete(byGFX, gfx)
	}
	return byGFX
}

func matchROCmLinuxSysfsDevice(device ml.DeviceInfo, index int, sysfsDevices []rocmLinuxSysfsDevice, byPCI, byGFX map[string]rocmLinuxSysfsDevice) (rocmLinuxSysfsDevice, bool) {
	// ROCm visibility envs can remap backend ordinals while sysfs stays in
	// physical KFD order, so prefer stable identity before index fallback.
	if device.PCIID != "" {
		if sysfsDevice, ok := byPCI[strings.ToLower(device.PCIID)]; ok {
			return sysfsDevice, true
		}
	}

	if device.GFXTarget != "" {
		if sysfsDevice, ok := byGFX[device.GFXTarget]; ok {
			return sysfsDevice, true
		}
	}

	if index >= len(sysfsDevices) {
		return rocmLinuxSysfsDevice{}, false
	}
	sysfsDevice := sysfsDevices[index]
	if sysfsDevice.gfxTarget != "" && device.GFXTarget != "" && sysfsDevice.gfxTarget != device.GFXTarget {
		slog.Debug("linux rocm device refinement index mismatch",
			"device", device.Name, "llama_server_gfx", device.GFXTarget, "kfd_gfx", sysfsDevice.gfxTarget)
		return rocmLinuxSysfsDevice{}, false
	}
	return sysfsDevice, true
}

func applyROCmLinuxSysfsDevice(device *ml.DeviceInfo, sysfsDevice rocmLinuxSysfsDevice) {
	if sysfsDevice.pciID != "" {
		device.PCIID = sysfsDevice.pciID
	}
	if sysfsDevice.known {
		device.Integrated = sysfsDevice.integrated
	}
}

func readROCmLinuxSysfsDevices(sysfsRoot string) ([]rocmLinuxSysfsDevice, error) {
	nodeRoot := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes")
	entries, err := os.ReadDir(nodeRoot)
	if err != nil {
		return nil, err
	}

	sort.Slice(entries, func(i, j int) bool {
		left, _ := strconv.Atoi(entries[i].Name())
		right, _ := strconv.Atoi(entries[j].Name())
		return left < right
	})

	var devices []rocmLinuxSysfsDevice
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		properties, err := readKFDNodeProperties(filepath.Join(nodeRoot, entry.Name(), "properties"))
		if err != nil || !properties.isGPU() {
			continue
		}

		device, err := readROCmDRMDevice(sysfsRoot, properties.drmRenderMinor)
		if err != nil {
			slog.Debug("linux rocm sysfs device skipped", "node", entry.Name(), "error", err)
			continue
		}
		device.gfxTarget = gfxTargetFromKFDVersion(properties.gfxTargetVersion)
		devices = append(devices, device)
	}

	return devices, nil
}

type kfdNodeProperties struct {
	vendorID         uint64
	deviceID         uint64
	drmRenderMinor   int
	gfxTargetVersion uint64
}

func (p kfdNodeProperties) isGPU() bool {
	return p.vendorID != 0 && p.deviceID != 0 && p.drmRenderMinor != 0
}

func readKFDNodeProperties(path string) (kfdNodeProperties, error) {
	file, err := os.Open(path)
	if err != nil {
		return kfdNodeProperties{}, err
	}
	defer file.Close()

	values := make(map[string]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) >= 2 {
			values[fields[0]] = fields[1]
		}
	}
	if err := scanner.Err(); err != nil {
		return kfdNodeProperties{}, err
	}

	vendorID, _ := parseSysfsUint(values["vendor_id"])
	deviceID, _ := parseSysfsUint(values["device_id"])
	renderMinor, _ := parseSysfsUint(values["drm_render_minor"])
	gfxVersion, _ := parseSysfsUint(values["gfx_target_version"])

	return kfdNodeProperties{
		vendorID:         vendorID,
		deviceID:         deviceID,
		drmRenderMinor:   int(renderMinor),
		gfxTargetVersion: gfxVersion,
	}, nil
}

func readROCmDRMDevice(sysfsRoot string, renderMinor int) (rocmLinuxSysfsDevice, error) {
	devicePath := filepath.Join(sysfsRoot, "class", "drm", "renderD"+strconv.Itoa(renderMinor), "device")
	resolvedDevicePath, err := filepath.EvalSymlinks(devicePath)
	if err != nil {
		return rocmLinuxSysfsDevice{}, err
	}

	vendor, err := readSysfsString(filepath.Join(resolvedDevicePath, "vendor"))
	if err != nil {
		return rocmLinuxSysfsDevice{}, err
	}
	if !strings.EqualFold(vendor, "0x1002") {
		return rocmLinuxSysfsDevice{}, nil
	}

	driver, err := readSysfsDriverName(filepath.Join(resolvedDevicePath, "driver"))
	if err != nil {
		return rocmLinuxSysfsDevice{}, err
	}
	if driver != "amdgpu" {
		return rocmLinuxSysfsDevice{}, nil
	}

	device := rocmLinuxSysfsDevice{pciID: pciIDFromPath(resolvedDevicePath)}
	if sysfsFileExists(filepath.Join(resolvedDevicePath, "mem_info_vram_vendor")) ||
		sysfsFileExists(filepath.Join(resolvedDevicePath, "board_info")) {
		device.known = true
		return device, nil
	}

	vramTotal, ok := readROCmLinuxMemoryInfo(resolvedDevicePath, "mem_info_vram_total")
	if !ok {
		return device, nil
	}
	gttTotal, ok := readROCmLinuxMemoryInfo(resolvedDevicePath, "mem_info_gtt_total")
	if !ok {
		return device, nil
	}

	const (
		maxIntegratedVRAM = 4 << 30
		minSharedGTT      = 8 << 30
	)
	if vramTotal > 0 && vramTotal <= maxIntegratedVRAM && gttTotal >= minSharedGTT && gttTotal >= 4*vramTotal {
		device.integrated = true
		device.known = true
	}

	return device, nil
}

func readROCmLinuxMemoryInfo(devicePath, name string) (uint64, bool) {
	value, err := readSysfsUint(filepath.Join(devicePath, name))
	return value, err == nil
}

func readSysfsString(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func readSysfsDriverName(path string) (string, error) {
	driver, readErr := readSysfsString(path)
	if readErr == nil {
		return driver, nil
	}
	driverPath, err := filepath.EvalSymlinks(path)
	if err == nil {
		return filepath.Base(driverPath), nil
	}
	return "", readErr
}

func readSysfsUint(path string) (uint64, error) {
	value, err := readSysfsString(path)
	if err != nil {
		return 0, err
	}
	return parseSysfsUint(value)
}

func parseSysfsUint(value string) (uint64, error) {
	return strconv.ParseUint(strings.TrimSpace(value), 0, 64)
}

func sysfsFileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func pciIDFromPath(path string) string {
	base := filepath.Base(path)
	if pciIDRegex.MatchString(base) {
		return base
	}
	return ""
}

func gfxTargetFromKFDVersion(version uint64) string {
	if version == 0 {
		return ""
	}
	major := version / 10000
	minor := (version / 100) % 100
	stepping := version % 100
	if minor > 0xf || stepping > 0xf {
		return ""
	}
	return "gfx" + strconv.FormatUint(major, 10) + strconv.FormatUint(minor, 16) + strconv.FormatUint(stepping, 16)
}

// filterUnsupportedROCmDevices removes ROCm devices whose gfx target doesn't have
// matching rocblas kernels bundled.
func filterUnsupportedROCmDevices(devices []ml.DeviceInfo, libDirs []string) []ml.DeviceInfo {
	supported := rocblasGFXTargets(libDirs)
	if len(supported) == 0 {
		return devices
	}

	override := hsaOverrideGFXTarget()
	var filtered []ml.DeviceInfo
	for _, dev := range devices {
		if dev.Library != "ROCm" {
			filtered = append(filtered, dev)
			continue
		}

		setROCmGFXTarget(&dev, override)
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
