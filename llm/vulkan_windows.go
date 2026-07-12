//go:build windows

package llm

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

const windowsVulkanRuntimeDLLName = "vulkan-1.dll"

func WindowsVulkanRuntimeDLLPath(libDirs []string) (string, error) {
	systemDir, err := windows.GetSystemDirectory()
	if err != nil {
		return "", err
	}
	return windowsVulkanRuntimeDLLPath(systemDir, os.Getenv("PATH"), libDirs, fileExists)
}

func adjustWindowsVulkanLibraryPaths(paths, gpuLibs []string) []string {
	vulkanDir := firstWindowsVulkanLibDir(gpuLibs)
	if vulkanDir == "" {
		return paths
	}

	vulkanPath, err := WindowsVulkanRuntimeDLLPath(gpuLibs)
	if err != nil {
		slog.Debug("windows Vulkan loader selection unavailable", "error", err)
		return paths
	}

	slog.Debug("selected windows Vulkan loader", "path", vulkanPath)

	return insertPathBefore(paths, filepath.Dir(vulkanPath), vulkanDir)
}

// Use the host Vulkan loader supplied by the installed Vulkan runtime or GPU
// driver. Ollama no longer packages the loader; exclude backend library
// directories from PATH probing so stale app-local copies from older installs
// cannot shadow the host runtime.
func windowsVulkanRuntimeDLLPath(
	systemDir string,
	pathEnv string,
	libDirs []string,
	exists func(string) bool,
) (string, error) {
	systemDir = filepath.Clean(systemDir)

	systemPath := filepath.Join(systemDir, windowsVulkanRuntimeDLLName)
	if exists(systemPath) {
		return systemPath, nil
	}

	if path := firstWindowsVulkanRuntimeDLLOnPath(pathEnv, libDirs, exists); path != "" {
		return path, nil
	}

	return "", errors.New("no host vulkan-1.dll runtime DLL found")
}

func firstWindowsVulkanRuntimeDLLOnPath(pathEnv string, excludedDirs []string, exists func(string) bool) string {
	for _, dir := range filepath.SplitList(pathEnv) {
		dir = strings.Trim(filepath.Clean(strings.Trim(dir, `"`)), `"`)
		if dir == "." || dir == "" || windowsDirInList(dir, excludedDirs) {
			continue
		}

		path := filepath.Join(dir, windowsVulkanRuntimeDLLName)
		if exists(path) {
			return filepath.Clean(path)
		}
	}
	return ""
}

func windowsDirInList(dir string, dirs []string) bool {
	dir = strings.ToLower(filepath.Clean(dir))
	for _, candidate := range dirs {
		candidate = strings.ToLower(filepath.Clean(candidate))
		if candidate == "" || candidate == "." {
			continue
		}
		if dir == candidate || strings.HasPrefix(dir, candidate+string(filepath.Separator)) {
			return true
		}
	}
	return false
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
