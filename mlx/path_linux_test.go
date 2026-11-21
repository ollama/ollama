package mlx

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

// buildSharedLibrary compiles a one-function shared library to out with the
// given soname, linked against needs (file names found in linkDirs).
func buildSharedLibrary(t *testing.T, cc, out, soname, fn string, linkDirs, needs []string) {
	t.Helper()
	src := filepath.Join(t.TempDir(), "lib.c")
	if err := os.WriteFile(src, []byte("int "+fn+"(void) { return 1; }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	args := []string{"-shared", "-fPIC", "-o", out, src, "-Wl,--no-as-needed"}
	if soname != "" {
		args = append(args, "-Wl,-soname,"+soname)
	}
	for _, linkDir := range linkDirs {
		args = append(args, "-L"+linkDir)
	}
	for _, need := range needs {
		if filepath.Base(need) != need {
			// Linking a library without a soname by path records that path.
			args = append(args, need)
			continue
		}
		args = append(args, "-l:"+need)
	}
	cmd := exec.CommandContext(t.Context(), cc, args...)
	cmd.Dir = filepath.Dir(out)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("cc %v: %v\n%s", args, err, output)
	}
}

func TestSharedDependenciesListsBundledLibrariesDependenciesFirst(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("no C compiler to build test libraries")
	}
	root := t.TempDir()
	mlxDir := filepath.Join(root, "mlx_cuda_v13")
	depDir := filepath.Join(root, "cuda_v13")
	for _, dir := range []string{mlxDir, depDir} {
		if err := os.Mkdir(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	lib := func(dir, name string) string { return filepath.Join(dir, name) }

	// The layout that broke: the MLX library needs a versioned library beside
	// it that needs one only the dependency directory has, which needs another
	// there. Libraries nothing in that chain needs stay unloaded.
	buildSharedLibrary(t, cc, lib(depDir, "libinner.so.1"), "libinner.so.1", "inner", nil, nil)
	buildSharedLibrary(t, cc, lib(depDir, "libouter.so.1"), "libouter.so.1", "outer", []string{depDir}, []string{"libinner.so.1"})
	buildSharedLibrary(t, cc, lib(depDir, "libunused.so.1"), "libunused.so.1", "unused", nil, nil)
	buildSharedLibrary(t, cc, lib(mlxDir, "libother.so.1"), "libother.so.1", "other", []string{depDir}, []string{"libunused.so.1"})
	buildSharedLibrary(t, cc, lib(mlxDir, "libsolver.so.1.2"), "libsolver.so.1", "solver", []string{depDir}, []string{"libouter.so.1"})
	if err := os.Symlink("libsolver.so.1.2", lib(mlxDir, "libsolver.so.1")); err != nil {
		t.Fatal(err)
	}
	buildSharedLibrary(t, cc, lib(mlxDir, "libmlxc.so"), "libmlxc.so", "mlxc", []string{mlxDir}, []string{"libsolver.so.1"})

	got := sharedDependencies(lib(mlxDir, "libmlxc.so"), depDir)
	want := []string{lib(depDir, "libinner.so.1"), lib(depDir, "libouter.so.1")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sharedDependencies() = %#v, want %#v", got, want)
	}
}

func TestSharedDependenciesSkipsNamesWithDirectories(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("no C compiler to build test libraries")
	}
	root := t.TempDir()
	mlxDir := filepath.Join(root, "mlx_cuda_v13")
	depDir := filepath.Join(root, "cuda_v13")
	for _, dir := range []string{mlxDir, filepath.Join(depDir, "sub")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	// A library with no soname, linked by a relative path, is recorded as
	// needed under that path; joined to depDir it would name a file there.
	buildSharedLibrary(t, cc, filepath.Join(depDir, "sub", "libpath.so"), "", "path", nil, nil)
	if err := os.Symlink(filepath.Join(depDir, "sub"), filepath.Join(mlxDir, "sub")); err != nil {
		t.Fatal(err)
	}
	buildSharedLibrary(t, cc, filepath.Join(mlxDir, "libmlxc.so"), "libmlxc.so", "mlxc", nil, []string{"sub/libpath.so"})
	if needed := neededLibraries(filepath.Join(mlxDir, "libmlxc.so")); !slices.Contains(needed, "sub/libpath.so") {
		t.Fatalf("libmlxc.so needs %q; want it to need sub/libpath.so", needed)
	}
	if err := os.Remove(filepath.Join(mlxDir, "sub")); err != nil {
		t.Fatal(err)
	}

	if got := sharedDependencies(filepath.Join(mlxDir, "libmlxc.so"), depDir); got != nil {
		t.Fatalf("sharedDependencies() = %#v, want nil", got)
	}
}
