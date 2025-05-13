package discover

import (
	"os"
	"path/filepath"
	"runtime"
)

// LibPath is a path to lookup dynamic libraries
// in development it's usually 'build/lib/ollama'
// in distribution builds it's 'lib/ollama' on Windows
// '../lib/ollama' on Linux and the executable's directory on macOS
// note: distribution builds, additional GPU-specific libraries are
// found in subdirectories of the returned path, such as
// 'cuda_v11', 'cuda_v12', 'rocm', etc.
var LibOllamaPath string = func() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	var libPath string
	switch runtime.GOOS {
	case "windows":
		libPath = filepath.Join(filepath.Dir(exe), "lib", "ollama")
	case "linux":
		libPath = filepath.Join(filepath.Dir(exe), "..", "lib", "ollama")
	case "darwin":
		libPath = filepath.Dir(exe)
	}

	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}

	paths := []string{
		libPath,

		// build paths for development
		filepath.Join(filepath.Dir(exe), "build", "lib", "ollama"),
		filepath.Join(cwd, "build", "lib", "ollama"),
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	return filepath.Dir(exe)
}()
