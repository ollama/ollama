//go:build windows

package llm

import (
	"errors"
	"log/slog"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

const windowsVulkanRuntimeDLLName = "vulkan-1.dll"

type windowsVulkanRuntimeDLL struct {
	path         string
	source       string
	bundledPath  string
	bundledVer   windowsFileVersion
	bundledVerOK bool
	systemPath   string
	systemVer    windowsFileVersion
	systemVerOK  bool
	systemDir    string
	bundledDir   string
}

func WindowsVulkanRuntimeDLLPath(libDirs []string) (string, error) {
	choice, err := windowsVulkanRuntimeDLLChoice(libDirs)
	if err != nil {
		return "", err
	}
	return choice.path, nil
}

func adjustWindowsVulkanLibraryPaths(paths, gpuLibs []string) []string {
	vulkanDir := firstWindowsVulkanLibDir(gpuLibs)
	if vulkanDir == "" {
		return paths
	}

	choice, err := windowsVulkanRuntimeDLLChoice(gpuLibs)
	if err != nil {
		slog.Debug("windows Vulkan loader selection unavailable", "error", err)
		return paths
	}

	slog.Debug("selected windows Vulkan loader",
		"loader_source", choice.source,
		"path", choice.path,
		"bundled", choice.bundledPath,
		"bundled_version", choice.bundledVer.String(),
		"system", choice.systemPath,
		"system_version", choice.systemVer.String(),
	)

	if choice.source != "system" || choice.systemDir == "" {
		return paths
	}

	return insertPathBefore(paths, choice.systemDir, vulkanDir)
}

// Prefer the system Vulkan loader when present, but keep a bundled loader as a
// fallback for hosts without a Vulkan runtime. If both versions are available
// and the bundled loader is newer, select the bundled copy.
func windowsVulkanRuntimeDLLChoice(libDirs []string) (windowsVulkanRuntimeDLL, error) {
	systemDir, err := windows.GetSystemDirectory()
	if err != nil {
		return windowsVulkanRuntimeDLL{}, err
	}
	systemDir = filepath.Clean(systemDir)

	bundledPath := firstExistingFile(libDirs, windowsVulkanRuntimeDLLName)
	systemPath := filepath.Join(systemDir, windowsVulkanRuntimeDLLName)
	systemExists := fileExists(systemPath)

	choice := windowsVulkanRuntimeDLL{
		path:        bundledPath,
		source:      "bundled",
		bundledPath: bundledPath,
		systemPath:  systemPath,
		systemDir:   systemDir,
	}
	if bundledPath != "" {
		choice.bundledDir = filepath.Dir(bundledPath)
		choice.bundledVer, choice.bundledVerOK = readWindowsFileVersion(bundledPath)
	}
	if systemExists {
		choice.systemVer, choice.systemVerOK = readWindowsFileVersion(systemPath)
	}

	switch {
	case bundledPath != "" && systemExists:
		if choice.bundledVerOK && choice.systemVerOK && choice.bundledVer.Compare(choice.systemVer) > 0 {
			return choice, nil
		}
		choice.path = systemPath
		choice.source = "system"
		return choice, nil
	case systemExists:
		choice.path = systemPath
		choice.source = "system"
		return choice, nil
	case bundledPath != "":
		return choice, nil
	default:
		return windowsVulkanRuntimeDLL{}, errors.New("no vulkan-1.dll runtime DLL found")
	}
}

func firstWindowsVulkanLibDir(libDirs []string) string {
	for _, dir := range libDirs {
		if dir == "" {
			continue
		}
		base := strings.ToLower(filepath.Base(dir))
		if strings.Contains(base, "vulkan") {
			return filepath.Clean(dir)
		}
		if fileExists(filepath.Join(dir, "ggml-vulkan.dll")) || fileExists(filepath.Join(dir, "libggml-vulkan.dll")) {
			return filepath.Clean(dir)
		}
	}
	return ""
}
