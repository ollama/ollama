package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/ml"
)

type llamaCppBinarySearch struct {
	libOllamaPath string
	executable    string
	workingDir    string
	goos          string
	goarch        string
}

func defaultLlamaCppBinarySearch() llamaCppBinarySearch {
	executable, _ := os.Executable()
	if executable != "" {
		if eval, err := filepath.EvalSymlinks(executable); err == nil {
			executable = eval
		}
	}

	workingDir, _ := os.Getwd()
	return llamaCppBinarySearch{
		libOllamaPath: ml.LibOllamaPath,
		executable:    executable,
		workingDir:    workingDir,
		goos:          runtime.GOOS,
		goarch:        runtime.GOARCH,
	}
}

// FindLlamaCppBinary locates a llama.cpp helper binary in installed and local
// development layouts.
func FindLlamaCppBinary(name string) (string, error) {
	path, candidates, err := findLlamaCppBinary(name, defaultLlamaCppBinarySearch())
	if err != nil {
		return "", fmt.Errorf("%s binary not found (checked: %s)", name, strings.Join(candidates, ", "))
	}
	return path, nil
}

func findLlamaCppBinary(name string, search llamaCppBinarySearch) (string, []string, error) {
	candidates := llamaCppBinaryCandidates(name, search)
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, candidates, nil
		}
	}

	return "", candidates, os.ErrNotExist
}

func llamaCppBinaryCandidates(name string, search llamaCppBinarySearch) []string {
	goos := search.goos
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := search.goarch
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	suffix := llamaCppBinaryName(name, goos)
	seen := map[string]bool{}
	var candidates []string
	add := func(dir string) {
		if dir == "" {
			return
		}
		path := filepath.Clean(filepath.Join(dir, suffix))
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	add(search.libOllamaPath)

	addPackagedLayoutDirs := func(base string) {
		if base == "" {
			return
		}
		switch goos {
		case "darwin":
			// macOS tarballs and apps colocate llama.cpp helpers with ollama.
			add(base)
			// Per-architecture local dist output keeps helpers under lib/ollama.
			add(filepath.Join(base, "lib", "ollama"))
			// Standard CMake installs put ollama in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		case "linux":
			// Linux packages install ollama in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		case "windows":
			// Windows packages keep ollama.exe at top level with lib/ as a peer.
			add(filepath.Join(base, "lib", "ollama"))
			// Standard CMake installs put ollama.exe in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		default:
			add(filepath.Join(base, "lib", "ollama"))
			add(filepath.Join(base, "..", "lib", "ollama"))
		}
	}

	addLocalLayoutDirs := func(base string) {
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

	if search.executable != "" {
		exeDir := filepath.Dir(search.executable)
		addPackagedLayoutDirs(exeDir)
		addLocalLayoutDirs(exeDir)
	}
	if search.workingDir != "" {
		addLocalLayoutDirs(search.workingDir)
	}

	addBuildOutputDirs := func(base string) {
		if base == "" {
			return
		}
		matches, _ := filepath.Glob(filepath.Join(base, "build", "llama-server-*", "bin"))
		slices.SortFunc(matches, func(a, b string) int {
			if rank := llamaCppBuildOutputRank(a) - llamaCppBuildOutputRank(b); rank != 0 {
				return rank
			}
			return strings.Compare(a, b)
		})
		for _, m := range matches {
			add(m)
		}
	}
	if search.executable != "" {
		addBuildOutputDirs(filepath.Dir(search.executable))
	}
	addBuildOutputDirs(search.workingDir)

	return candidates
}

func llamaCppBinaryName(name, goos string) string {
	if goos == "windows" && filepath.Ext(name) == "" {
		return name + ".exe"
	}
	return name
}

func llamaCppBuildOutputRank(path string) int {
	if strings.Contains(path, "llama-server-darwin") ||
		strings.Contains(path, "llama-server-cuda") ||
		strings.Contains(path, "llama-server-rocm") {
		return 0
	}
	return 1
}
