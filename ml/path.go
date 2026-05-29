package ml

import (
	"os"
	"path/filepath"
	"runtime"
)

type libOllamaPathSearch struct {
	executable string
	workingDir string
	goos       string
	goarch     string
}

// LibOllamaPath is the root used to find bundled llama.cpp and MLX runtime
// libraries. GPU-specific libraries live in backend subdirectories such as
// cuda_v12, rocm_v7_2, vulkan, and mlx_cuda_v13.
var LibOllamaPath = func() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}

	return findLibOllamaPath(libOllamaPathSearch{
		executable: exe,
		workingDir: cwd,
		goos:       runtime.GOOS,
		goarch:     runtime.GOARCH,
	})
}()

func findLibOllamaPath(search libOllamaPathSearch) string {
	candidates := libOllamaPathCandidates(search)
	for _, path := range candidates {
		if libOllamaPathExists(path) {
			return path
		}
	}

	if search.executable != "" {
		return filepath.Dir(search.executable)
	}
	return ""
}

func libOllamaPathCandidates(search libOllamaPathSearch) []string {
	goos := search.goos
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := search.goarch
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	seen := map[string]bool{}
	var candidates []string
	add := func(path string) {
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	if search.executable != "" {
		exeDir := filepath.Dir(search.executable)
		switch goos {
		case "darwin":
			// Local dist output and standard installs keep helpers under lib/ollama.
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		case "linux":
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
			add(filepath.Join(exeDir, "lib", "ollama"))
		case "windows":
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		default:
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		}
		addLocalLibOllamaPaths(add, exeDir, goos, goarch)
		if goos == "darwin" {
			// macOS release artifacts colocate native helpers with ollama.
			add(exeDir)
		}
	}
	addLocalLibOllamaPaths(add, search.workingDir, goos, goarch)

	return candidates
}

func addLocalLibOllamaPaths(add func(string), base, goos, goarch string) {
	if base == "" {
		return
	}
	add(filepath.Join(base, "build", "lib", "ollama"))
	add(filepath.Join(base, "dist", goos+"-"+goarch, "lib", "ollama"))
	if goos+"_"+goarch != goos+"-"+goarch {
		add(filepath.Join(base, "dist", goos+"_"+goarch, "lib", "ollama"))
	}
	if goos == "darwin" {
		add(filepath.Join(base, "dist", "darwin"))
	}
}

func libOllamaPathExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
