//go:build windows

package llm

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	windowsROCmRuntimeDLLNames   = []string{"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"}
	windowsROCmCompanionDLLNames = []string{"amd_comgr.dll", "amd_comgr_2.dll"}
)

type windowsROCmRuntimeDLL struct {
	path         string
	source       string
	name         string
	bundledPath  string
	bundledVer   windowsFileVersion
	bundledVerOK bool
	systemPath   string
	systemVer    windowsFileVersion
	systemVerOK  bool
	systemDir    string
	bundledDir   string
}

type windowsFileVersion struct {
	ms uint32
	ls uint32
}

func WindowsROCmRuntimeDLLPath(libDirs []string) (string, error) {
	choice, err := windowsROCmRuntimeDLLChoice(libDirs)
	if err != nil {
		return "", err
	}
	return choice.path, nil
}

func adjustPlatformLibraryPaths(paths, gpuLibs []string) []string {
	paths = adjustWindowsROCmLibraryPaths(paths, gpuLibs)
	return adjustWindowsVulkanLibraryPaths(paths, gpuLibs)
}

func adjustWindowsROCmLibraryPaths(paths, gpuLibs []string) []string {
	rocmDir := firstWindowsROCmLibDir(gpuLibs)
	if rocmDir == "" {
		return paths
	}

	choice, err := windowsROCmRuntimeDLLChoice(gpuLibs)
	if err != nil {
		slog.Debug("windows ROCm runtime selection unavailable", "error", err)
		return paths
	}

	slog.Debug("selected windows ROCm runtime",
		"runtime_source", choice.source,
		"name", choice.name,
		"path", choice.path,
		"bundled", choice.bundledPath,
		"bundled_version", choice.bundledVer.String(),
		"system", choice.systemPath,
		"system_version", choice.systemVer.String(),
	)

	if choice.source != "system" || choice.systemDir == "" {
		return paths
	}

	before := choice.bundledDir
	if before == "" {
		before = rocmDir
	}

	return insertPathBefore(paths, choice.systemDir, before)
}

// AMD's Windows driver also ships ROCm runtime DLLs in System32. Within the
// same DLL name/major, use the newer driver or bundled copy; never search the
// ROCm SDK installation paths.
func windowsROCmRuntimeDLLChoice(libDirs []string) (windowsROCmRuntimeDLL, error) {
	systemDir, err := windows.GetSystemDirectory()
	if err != nil {
		return windowsROCmRuntimeDLL{}, err
	}
	systemDir = filepath.Clean(systemDir)

	for _, name := range windowsROCmRuntimeDLLNames {
		bundledPath := firstExistingFile(libDirs, name)
		if bundledPath == "" {
			continue
		}

		choice := windowsROCmRuntimeDLL{
			path:        bundledPath,
			source:      "bundled",
			name:        name,
			bundledPath: bundledPath,
			bundledDir:  filepath.Dir(bundledPath),
			systemPath:  filepath.Join(systemDir, name),
			systemDir:   systemDir,
		}
		choice.bundledVer, choice.bundledVerOK = readWindowsFileVersion(choice.bundledPath)
		if fileExists(choice.systemPath) {
			choice.systemVer, choice.systemVerOK = readWindowsFileVersion(choice.systemPath)
			if choice.systemVerOK && choice.bundledVerOK && choice.systemVer.Compare(choice.bundledVer) > 0 {
				choice.path = choice.systemPath
				choice.source = "system"
			}
		}
		if choice.source == "system" && !systemROCmCompanionDLLsCompatible(libDirs, choice.systemDir) {
			choice.path = choice.bundledPath
			choice.source = "bundled"
		}
		return choice, nil
	}

	for _, name := range windowsROCmRuntimeDLLNames {
		path := filepath.Join(systemDir, name)
		if !fileExists(path) {
			continue
		}
		choice := windowsROCmRuntimeDLL{
			path:       path,
			source:     "system",
			name:       name,
			systemPath: path,
			systemDir:  systemDir,
		}
		choice.systemVer, choice.systemVerOK = readWindowsFileVersion(path)
		return choice, nil
	}

	return windowsROCmRuntimeDLL{}, errors.New("no amdhip64 runtime DLL found")
}

func systemROCmCompanionDLLsCompatible(libDirs []string, systemDir string) bool {
	for _, name := range windowsROCmCompanionDLLNames {
		bundledPath := firstExistingFile(libDirs, name)
		if bundledPath == "" {
			continue
		}

		systemPath := filepath.Join(systemDir, name)
		if !fileExists(systemPath) {
			continue
		}

		bundledVer, bundledOK := readWindowsFileVersion(bundledPath)
		systemVer, systemOK := readWindowsFileVersion(systemPath)
		if !bundledOK || !systemOK {
			slog.Debug("keeping bundled ROCm runtime because companion DLL version is unavailable",
				"name", name, "bundled", bundledPath, "system", systemPath)
			return false
		}
		if systemVer.Compare(bundledVer) < 0 {
			slog.Debug("keeping bundled ROCm runtime because system companion DLL is older",
				"name", name,
				"bundled", bundledPath,
				"bundled_version", bundledVer.String(),
				"system", systemPath,
				"system_version", systemVer.String(),
			)
			return false
		}
	}
	return true
}

func firstWindowsROCmLibDir(libDirs []string) string {
	for _, dir := range libDirs {
		if dir == "" {
			continue
		}
		base := strings.ToLower(filepath.Base(dir))
		if strings.Contains(base, "rocm") || strings.Contains(base, "hip") {
			return filepath.Clean(dir)
		}
		if fileExists(filepath.Join(dir, "ggml-hip.dll")) || fileExists(filepath.Join(dir, "libggml-hip.dll")) {
			return filepath.Clean(dir)
		}
	}
	return ""
}

func firstExistingFile(dirs []string, name string) string {
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		path := filepath.Join(dir, name)
		if fileExists(path) {
			return filepath.Clean(path)
		}
	}
	return ""
}

func insertPathBefore(paths []string, insert, before string) []string {
	insert = filepath.Clean(insert)
	before = filepath.Clean(before)
	insertKey := strings.ToLower(insert)
	beforeKey := strings.ToLower(before)

	out := make([]string, 0, len(paths)+1)
	inserted := false
	for _, path := range paths {
		clean := filepath.Clean(path)
		key := strings.ToLower(clean)
		if key == insertKey {
			continue
		}
		if !inserted && key == beforeKey {
			out = append(out, insert)
			inserted = true
		}
		out = append(out, path)
	}
	if !inserted {
		out = append(out, insert)
	}
	return out
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func readWindowsFileVersion(path string) (windowsFileVersion, bool) {
	var zero windows.Handle
	infoSize, err := windows.GetFileVersionInfoSize(path, &zero)
	if err != nil || infoSize == 0 {
		return windowsFileVersion{}, false
	}

	versionInfo := make([]byte, infoSize)
	if err := windows.GetFileVersionInfo(path, 0, infoSize, unsafe.Pointer(&versionInfo[0])); err != nil {
		return windowsFileVersion{}, false
	}

	var fixedInfo *windows.VS_FIXEDFILEINFO
	var fixedInfoLen uint32
	if err := windows.VerQueryValue(unsafe.Pointer(&versionInfo[0]), `\`, unsafe.Pointer(&fixedInfo), &fixedInfoLen); err != nil {
		return windowsFileVersion{}, false
	}
	if fixedInfo == nil {
		return windowsFileVersion{}, false
	}
	return windowsFileVersion{ms: fixedInfo.FileVersionMS, ls: fixedInfo.FileVersionLS}, true
}

func (v windowsFileVersion) Compare(other windowsFileVersion) int {
	if v.ms < other.ms {
		return -1
	}
	if v.ms > other.ms {
		return 1
	}
	if v.ls < other.ls {
		return -1
	}
	if v.ls > other.ls {
		return 1
	}
	return 0
}

func (v windowsFileVersion) String() string {
	if v.ms == 0 && v.ls == 0 {
		return ""
	}
	return fmt.Sprintf("%d.%d.%d.%d",
		(v.ms>>16)&0xffff,
		v.ms&0xffff,
		(v.ls>>16)&0xffff,
		v.ls&0xffff,
	)
}
