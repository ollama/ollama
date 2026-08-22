package mlx

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLibraryPathsForLoadedLibraryAddsOnlySelectedMLXDependency(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{
		"mlx_cuda_v13",
		"cuda_v13",
		"mlx_cuda_v12",
		"cuda_v12",
	} {
		if err := os.Mkdir(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	got := LibraryPathsForLoadedLibrary(root, filepath.Join(root, "mlx_cuda_v13", "mlxc.dll"))
	want := []string{
		root,
		filepath.Join(root, "mlx_cuda_v13"),
		filepath.Join(root, "cuda_v13"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("LibraryPathsForLoadedLibrary() = %#v, want %#v", got, want)
	}
}

func TestLibraryPathsForLoadedLibraryAddsStandaloneDir(t *testing.T) {
	root := t.TempDir()
	libDir := filepath.Join(root, "standalone")
	got := LibraryPathsForLoadedLibrary(root, filepath.Join(libDir, "libmlxc.so"))
	want := []string{root, libDir}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("LibraryPathsForLoadedLibrary() = %#v, want %#v", got, want)
	}
}

func TestDependencyDirRequiresExistingSibling(t *testing.T) {
	root := t.TempDir()
	mlxDir := filepath.Join(root, "mlx_cuda_v13")
	if err := os.Mkdir(mlxDir, 0o755); err != nil {
		t.Fatal(err)
	}

	if depDir, ok := dependencyDir(mlxDir); ok {
		t.Fatalf("dependencyDir() = %q, true; want missing dependency", depDir)
	}

	cudaDir := filepath.Join(root, "cuda_v13")
	if err := os.Mkdir(cudaDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if depDir, ok := dependencyDir(mlxDir); !ok || depDir != cudaDir {
		t.Fatalf("dependencyDir() = %q, %v; want %q, true", depDir, ok, cudaDir)
	}
}

func TestCUDAToolkitDirForLoadedLibraryRequiresMLXCUDAInclude(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{
		"mlx_cuda_v13",
		"mlx_test_v1",
	} {
		if err := os.Mkdir(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	libraryPath := filepath.Join(root, "mlx_cuda_v13", "mlxc.dll")

	if cudaDir, ok := CUDAToolkitDirForLoadedLibrary(libraryPath); ok {
		t.Fatalf("CUDAToolkitDirForLoadedLibrary() = %q, true; want no CUDA headers", cudaDir)
	}

	want := filepath.Join(root, "mlx_cuda_v13")
	if err := os.Mkdir(filepath.Join(want, "include"), 0o755); err != nil {
		t.Fatal(err)
	}
	if cudaDir, ok := CUDAToolkitDirForLoadedLibrary(libraryPath); !ok || cudaDir != want {
		t.Fatalf("CUDAToolkitDirForLoadedLibrary() = %q, %v; want %q, true", cudaDir, ok, want)
	}
}

func TestCUDAToolkitDirForLoadedLibraryUsesSelectedCUDAName(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{
		filepath.Join(root, "mlx_cuda_v12", "include"),
		filepath.Join(root, "mlx_cuda_v13", "include"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	want := filepath.Join(root, "mlx_cuda_v12")
	if cudaDir, ok := CUDAToolkitDirForLoadedLibrary(filepath.Join(want, "mlxc.dll")); !ok || cudaDir != want {
		t.Fatalf("CUDAToolkitDirForLoadedLibrary() = %q, %v; want %q, true", cudaDir, ok, want)
	}
}

func TestCUDAToolkitDirForMLXDirRejectsNonCUDAVariants(t *testing.T) {
	root := t.TempDir()
	otherDir := filepath.Join(root, "mlx_test_v1")
	if err := os.MkdirAll(filepath.Join(otherDir, "include"), 0o755); err != nil {
		t.Fatal(err)
	}

	if cudaDir, ok := cudaToolkitDirForMLXDir(otherDir); ok {
		t.Fatalf("cudaToolkitDirForMLXDir() = %q, true; want rejected non-CUDA variant", cudaDir)
	}
}
