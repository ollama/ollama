package mlx

import (
	"os"
	"path/filepath"
	"strings"
)

// LibraryPathsForLoadedLibrary returns the runtime library search paths needed
// for the selected MLX dynamic library.
func LibraryPathsForLoadedLibrary(root, libraryPath string) []string {
	return libraryPathsForLibrary(root, libraryPath)
}

// CUDAToolkitDirForLoadedLibrary returns the selected MLX CUDA variant
// directory when it carries bundled CUDA headers for MLX JIT compilation.
func CUDAToolkitDirForLoadedLibrary(libraryPath string) (string, bool) {
	return cudaToolkitDirForLibrary(libraryPath)
}

func libraryPathsForLibrary(root, libraryPath string) []string {
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

	add(root)
	dir := filepath.Dir(libraryPath)
	add(dir)
	if strings.HasPrefix(filepath.Base(dir), "mlx_") {
		if depDir, ok := dependencyDir(dir); ok {
			add(depDir)
		}
	}

	return paths
}

// dependencyDir maps an MLX variant directory to its shared dependency
// directory. The mapping is naming-based so variants like mlx_cuda_v12
// automatically use cuda_v12 when present.
func dependencyDir(mlxDir string) (string, bool) {
	name := filepath.Base(mlxDir)
	depName, ok := strings.CutPrefix(name, "mlx_")
	if !ok || depName == "" {
		return "", false
	}

	depDir := filepath.Join(filepath.Dir(mlxDir), depName)
	if !dirExists(depDir) {
		return "", false
	}
	return depDir, true
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
