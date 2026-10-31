package mlx

import (
	"debug/elf"
	"os"
	"path/filepath"
	"strings"
)

// LibraryPathsForLoadedLibrary returns the runtime library search paths needed
// for the selected MLX dynamic library.
func LibraryPathsForLoadedLibrary(libraryPath string) []string {
	return libraryPathsForLibrary(libraryPath)
}

// CUDAToolkitDirForLoadedLibrary returns the selected MLX CUDA variant
// directory when it carries bundled CUDA headers for MLX JIT compilation.
func CUDAToolkitDirForLoadedLibrary(libraryPath string) (string, bool) {
	return cudaToolkitDirForLibrary(libraryPath)
}

func libraryPathsForLibrary(libraryPath string) []string {
	var paths []string
	seen := map[string]bool{}
	add := func(path string) {
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		if !seen[path] {
			seen[path] = true
			paths = append(paths, path)
		}
	}

	dir := filepath.Dir(libraryPath)
	add(dir)
	if strings.HasPrefix(filepath.Base(dir), "mlx_") {
		if depDir, ok := dependencyDir(dir); ok {
			add(depDir)
		}
	}

	return paths
}

// dependencyDir maps an MLX CUDA variant directory to its shared dependency
// directory. Variants like mlx_cuda_v12 automatically use cuda_v12 when
// present; non-CUDA variants do not have a sibling dependency payload.
func dependencyDir(mlxDir string) (string, bool) {
	name := filepath.Base(mlxDir)
	cudaName, ok := strings.CutPrefix(name, "mlx_cuda_")
	if !ok || cudaName == "" {
		return "", false
	}
	depName := "cuda_" + cudaName

	depDir := filepath.Join(filepath.Dir(mlxDir), depName)
	if !dirExists(depDir) {
		return "", false
	}
	return depDir, true
}

// sharedDependencies lists, dependencies first, the libraries in depDir that
// lib needs, directly or through the libraries beside it.
func sharedDependencies(lib, depDir string) []string {
	mlxDir := filepath.Dir(lib)
	var order []string
	seen := map[string]bool{}
	var visit func(path string)
	visit = func(path string) {
		for _, name := range neededLibraries(path) {
			// A soname has no directory part; anything else is not ours to load.
			if seen[name] || strings.ContainsRune(name, '/') {
				continue
			}
			seen[name] = true
			if local := filepath.Join(mlxDir, name); fileExists(local) {
				visit(local)
			} else if dep := filepath.Join(depDir, name); fileExists(dep) {
				visit(dep)
				order = append(order, dep)
			}
		}
	}
	visit(lib)
	return order
}

// neededLibraries returns an ELF file's DT_NEEDED sonames, or nil if path is
// not an ELF file.
func neededLibraries(path string) []string {
	f, err := elf.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	libs, _ := f.ImportedLibraries()
	return libs
}

func cudaToolkitDirForLibrary(libraryPath string) (string, bool) {
	return cudaToolkitDirForMLXDir(filepath.Dir(libraryPath))
}

func cudaToolkitDirForMLXDir(mlxDir string) (string, bool) {
	if !strings.HasPrefix(filepath.Base(mlxDir), "mlx_cuda_") {
		return "", false
	}
	if !dirExists(filepath.Join(mlxDir, "include")) {
		return "", false
	}
	return filepath.Clean(mlxDir), true
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}
