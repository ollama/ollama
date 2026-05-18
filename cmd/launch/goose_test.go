package launch

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestGooseIntegration(t *testing.T) {
	g := &Goose{}

	t.Run("String", func(t *testing.T) {
		if got := g.String(); got != "Goose" {
			t.Errorf("String() = %q, want %q", got, "Goose")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = g
	})
}

func TestGooseEnvVars(t *testing.T) {
	g := &Goose{}

	t.Run("with model", func(t *testing.T) {
		env := g.envVars("llama3.2")

		mustContain := []string{"GOOSE_PROVIDER=ollama", "GOOSE_MODEL=llama3.2"}
		for _, want := range mustContain {
			if !slices.Contains(env, want) {
				t.Errorf("envVars missing %q, got %v", want, env)
			}
		}

		var hasHost bool
		for _, e := range env {
			if strings.HasPrefix(e, "OLLAMA_HOST=") {
				hasHost = true
				if strings.TrimPrefix(e, "OLLAMA_HOST=") == "" {
					t.Errorf("OLLAMA_HOST is empty: %q", e)
				}
			}
		}
		if !hasHost {
			t.Errorf("envVars missing OLLAMA_HOST=, got %v", env)
		}
	})

	t.Run("without model", func(t *testing.T) {
		env := g.envVars("")
		for _, e := range env {
			if strings.HasPrefix(e, "GOOSE_MODEL=") {
				t.Errorf("envVars should not include GOOSE_MODEL when model empty, got %v", env)
			}
		}
		if !slices.Contains(env, "GOOSE_PROVIDER=ollama") {
			t.Errorf("envVars missing GOOSE_PROVIDER=ollama, got %v", env)
		}
	})
}

func TestGooseDesktopAppAvailable(t *testing.T) {
	g := &Goose{}

	t.Run("non-darwin always false", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "linux"

		prevStat := gooseStatFn
		t.Cleanup(func() { gooseStatFn = prevStat })
		gooseStatFn = func(string) (os.FileInfo, error) { return nil, nil } // pretend app exists

		if g.desktopAppAvailable() {
			t.Error("desktopAppAvailable should be false on non-darwin")
		}
	})

	t.Run("darwin with app present", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "darwin"

		prevStat := gooseStatFn
		t.Cleanup(func() { gooseStatFn = prevStat })
		gooseStatFn = func(string) (os.FileInfo, error) { return nil, nil }

		if !g.desktopAppAvailable() {
			t.Error("desktopAppAvailable should be true when stat succeeds on darwin")
		}
	})

	t.Run("darwin with app missing", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "darwin"

		prevStat := gooseStatFn
		t.Cleanup(func() { gooseStatFn = prevStat })
		gooseStatFn = func(string) (os.FileInfo, error) { return nil, &fs.PathError{Err: errors.New("not exist")} }

		if g.desktopAppAvailable() {
			t.Error("desktopAppAvailable should be false when stat returns error")
		}
	})

	t.Run("windows with app present in local app data", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "windows"

		prevRunPath := gooseRunPath
		t.Cleanup(func() { gooseRunPath = prevRunPath })
		gooseRunPath = func() string { return "" }

		prevStartID := gooseStartID
		t.Cleanup(func() { gooseStartID = prevStartID })
		gooseStartID = func() string { return "" }

		local := filepath.Join(t.TempDir(), "LocalAppData")
		t.Setenv("LOCALAPPDATA", local)
		path := filepath.Join(local, "Programs", "Goose", "Goose.exe")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0o755); err != nil {
			t.Fatal(err)
		}

		if !g.desktopAppAvailable() {
			t.Error("desktopAppAvailable should be true when Windows Goose.exe exists")
		}
	})

	t.Run("windows with start menu app", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "windows"

		prevRunPath := gooseRunPath
		t.Cleanup(func() { gooseRunPath = prevRunPath })
		gooseRunPath = func() string { return "" }

		prevStartID := gooseStartID
		t.Cleanup(func() { gooseStartID = prevStartID })
		gooseStartID = func() string { return "Block.Goose_abc123!App" }

		prevStat := gooseStatFn
		t.Cleanup(func() { gooseStatFn = prevStat })
		gooseStatFn = func(string) (os.FileInfo, error) { return nil, &fs.PathError{Err: errors.New("not exist")} }

		if !g.desktopAppAvailable() {
			t.Error("desktopAppAvailable should be true when Windows Start Menu app exists")
		}
	})
}

func TestGooseRunCLINotInstalled(t *testing.T) {
	// Force non-darwin so we hit runCLI path
	prev := gooseGOOS
	t.Cleanup(func() { gooseGOOS = prev })
	gooseGOOS = "linux"

	t.Setenv("PATH", t.TempDir()) // empty PATH

	g := &Goose{}
	err := g.Run("llama3.2", nil, nil)
	if err == nil {
		t.Fatal("expected error when goose binary not installed")
	}
	if !strings.Contains(err.Error(), "block.github.io/goose") {
		t.Errorf("expected install hint in error, got: %v", err)
	}
}

func TestGooseRunDesktopAppWindows(t *testing.T) {
	prev := gooseGOOS
	t.Cleanup(func() { gooseGOOS = prev })
	gooseGOOS = "windows"

	local := filepath.Join(t.TempDir(), "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	path := filepath.Join(local, "Programs", "Goose", "Goose.exe")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o755); err != nil {
		t.Fatal(err)
	}

	prevOpenPath := gooseOpenPath
	t.Cleanup(func() { gooseOpenPath = prevOpenPath })
	var gotPath string
	var gotEnv []string
	gooseOpenPath = func(path string, env []string) error {
		gotPath = path
		gotEnv = append([]string(nil), env...)
		return nil
	}

	if err := (&Goose{}).runDesktopApp("llama3.2"); err != nil {
		t.Fatal(err)
	}
	if gotPath != path {
		t.Fatalf("runDesktopApp opened %q, want %q", gotPath, path)
	}
	for _, want := range []string{"GOOSE_PROVIDER=ollama", "GOOSE_MODEL=llama3.2"} {
		if !slices.Contains(gotEnv, want) {
			t.Errorf("runDesktopApp env missing %q, got %v", want, gotEnv)
		}
	}
}

func TestGooseInstalled(t *testing.T) {
	g := &Goose{}

	t.Run("not installed when neither app nor cli present", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "linux"
		t.Setenv("PATH", t.TempDir())

		if g.installed() {
			t.Error("installed() should be false when neither app nor cli present")
		}
	})

	t.Run("installed when cli on PATH", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "linux"

		binDir := t.TempDir()
		writeFakeBinary(t, binDir, "goose")
		t.Setenv("PATH", binDir)

		if !g.installed() {
			t.Error("installed() should be true when goose binary on PATH")
		}
	})

	t.Run("installed when windows desktop app present", func(t *testing.T) {
		prev := gooseGOOS
		t.Cleanup(func() { gooseGOOS = prev })
		gooseGOOS = "windows"
		t.Setenv("PATH", t.TempDir())

		prevRunPath := gooseRunPath
		t.Cleanup(func() { gooseRunPath = prevRunPath })
		gooseRunPath = func() string { return "" }

		prevStartID := gooseStartID
		t.Cleanup(func() { gooseStartID = prevStartID })
		gooseStartID = func() string { return "" }

		local := filepath.Join(t.TempDir(), "LocalAppData")
		t.Setenv("LOCALAPPDATA", local)
		path := filepath.Join(local, "Programs", "Goose", "Goose.exe")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0o755); err != nil {
			t.Fatal(err)
		}

		if !g.installed() {
			t.Error("installed() should be true when Windows desktop app exists")
		}
	})
}
