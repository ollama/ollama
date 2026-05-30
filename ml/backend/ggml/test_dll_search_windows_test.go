//go:build windows

package ggml

import (
	"os"
	"strings"
	"sync"

	"golang.org/x/sys/windows"
)

// init configures the Windows DLL search path so cgo-loaded backends can find
// their transitive deps (ROCm rocBLAS, CUDA cuBLAS, etc.) when running tests.
// Go's runtime restricts the default search to System32; without explicit
// AddDllDirectory calls, ggml-hip.dll fails to load even with the deps in
// PATH or co-located with the .dll.
//
// Set OLLAMA_TEST_DLL_DIRS to a semicolon-separated list of dirs to add.
// Example:
//
//	OLLAMA_TEST_DLL_DIRS="C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
func init() {
	dllInitOnce.Do(addTestDllDirs)
}

var dllInitOnce sync.Once

// LOAD_LIBRARY_SEARCH_* flag bits for SetDefaultDllDirectories.
const (
	loadLibrarySearchUserDirs    = 0x00000400
	loadLibrarySearchDefaultDirs = 0x00001000
)

func addTestDllDirs() {
	// Combine OLLAMA_TEST_DLL_DIRS (system runtimes like ROCm/CUDA bin) with
	// OLLAMA_LIBRARY_PATH (the lib/ollama dir holding ggml-base.dll etc.).
	// Both must be added because Go's restrictive search doesn't include the
	// loaded-DLL's own directory for transitive dep resolution.
	dirs := os.Getenv("OLLAMA_TEST_DLL_DIRS")
	if libPath := os.Getenv("OLLAMA_LIBRARY_PATH"); libPath != "" {
		if dirs != "" {
			dirs = dirs + ";" + libPath
		} else {
			dirs = libPath
		}
	}
	if dirs == "" {
		return
	}

	// Enable user-dir search so AddDllDirectory entries take effect.
	// LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = USER_DIRS | APPLICATION_DIR | SYSTEM32.
	k32 := windows.NewLazySystemDLL("kernel32.dll")
	setDefault := k32.NewProc("SetDefaultDllDirectories")
	if setDefault.Find() == nil {
		setDefault.Call(uintptr(loadLibrarySearchDefaultDirs))
	}

	for _, dir := range strings.Split(dirs, ";") {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			continue
		}
		ptr, err := windows.UTF16PtrFromString(dir)
		if err != nil {
			os.Stderr.WriteString("dll_search: utf16 err for " + dir + ": " + err.Error() + "\n")
			continue
		}
		cookie, err := windows.AddDllDirectory(ptr)
		if err != nil {
			os.Stderr.WriteString("dll_search: AddDllDirectory failed for " + dir + ": " + err.Error() + "\n")
			continue
		}
		os.Stderr.WriteString("dll_search: AddDllDirectory(" + dir + ") cookie=" + uintptrToHex(cookie) + "\n")
	}

	// Sanity: also pre-load the key transitive deps by full path so they're
	// already mapped when ggml-hip.dll's deps are resolved.
	for _, name := range []string{"amdhip64_7.dll", "libhipblaslt.dll", "rocblas.dll", "libhipblas.dll"} {
		for _, dir := range strings.Split(dirs, ";") {
			dir = strings.TrimSpace(dir)
			full := dir + `\` + name
			if _, err := os.Stat(full); err != nil {
				continue
			}
			ptr, err := windows.UTF16PtrFromString(full)
			if err != nil {
				continue
			}
			h, err := windows.LoadLibrary(full)
			if err != nil {
				os.Stderr.WriteString("dll_search: preload " + full + " failed: " + err.Error() + "\n")
			} else {
				os.Stderr.WriteString("dll_search: preloaded " + full + " handle=" + uintptrToHex(uintptr(h)) + "\n")
			}
			_ = ptr
			break
		}
	}
}

func uintptrToHex(v uintptr) string {
	const hex = "0123456789abcdef"
	if v == 0 {
		return "0x0"
	}
	var buf [18]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = hex[v&0xf]
		v >>= 4
	}
	i--
	buf[i] = 'x'
	i--
	buf[i] = '0'
	return string(buf[i:])
}
