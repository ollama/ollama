package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestNanoclawIntegration(t *testing.T) {
	n := &Nanoclaw{}

	t.Run("String", func(t *testing.T) {
		if got := n.String(); got != "NanoClaw" {
			t.Errorf("String() = %q, want %q", got, "NanoClaw")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = n
	})
}

func TestNanoclawRegistered(t *testing.T) {
	spec, err := LookupIntegrationSpec("nanoclaw")
	if err != nil {
		t.Fatalf("LookupIntegrationSpec(nanoclaw): %v", err)
	}
	if spec.Hidden {
		t.Error("nanoclaw should be visible (in launcher order), not hidden")
	}
	if spec.Install.CheckInstalled == nil {
		t.Error("nanoclaw should wire CheckInstalled")
	}
	if spec.Install.EnsureInstalled == nil {
		t.Error("nanoclaw should wire EnsureInstalled (install-if-missing)")
	}
	if !slices.Contains(launcherIntegrationOrder, "nanoclaw") {
		t.Error("nanoclaw should appear in launcherIntegrationOrder")
	}
}

func TestNanoclawInstalled(t *testing.T) {
	t.Run("false on a fresh home", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		if nanoclawInstalled() {
			t.Error("nanoclawInstalled() = true, want false with no checkout")
		}
	})

	t.Run("true for a valid checkout", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		writeNanoclawCheckout(t, nanoclawPackageName)
		if !nanoclawInstalled() {
			t.Error("nanoclawInstalled() = false, want true for a valid checkout")
		}
	})

	t.Run("false when package name mismatches", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		writeNanoclawCheckout(t, "some-other-package")
		if nanoclawInstalled() {
			t.Error("nanoclawInstalled() = true, want false when package.json name differs")
		}
	})
}

func TestNanoclawDisplayName(t *testing.T) {
	// Whatever the OS reports, a display name must never be empty — the script
	// uses it to name the first chat agent's folder.
	if got := nanoclawDisplayName(); strings.TrimSpace(got) == "" {
		t.Errorf("nanoclawDisplayName() = %q, want non-empty", got)
	}
}

func TestNanoclawRun(t *testing.T) {
	t.Run("execs entrypoint with contract flags on first run", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		dir := writeNanoclawCheckout(t, nanoclawPackageName)

		if err := (&Nanoclaw{}).Run("qwen3-coder:30b", nil, nil); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}

		got := readNanoclawInvocation(t, dir)
		assertFlagValue(t, got, "--model", "qwen3-coder:30b")
		assertFlagValue(t, got, "--base-url", envconfig.ConnectableHost().String())
		assertFlagValue(t, got, "--display-name", nanoclawDisplayName())
		assertFlagValue(t, got, "--agent-name", nanoclawAgentName)
	})

	t.Run("omits first-run flags once onboarded", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		dir := writeNanoclawCheckout(t, nanoclawPackageName)
		markNanoclawOnboarded(t, dir)

		if err := (&Nanoclaw{}).Run("gemma4:latest", nil, nil); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}

		got := readNanoclawInvocation(t, dir)
		assertFlagValue(t, got, "--model", "gemma4:latest")
		if slices.Contains(got, "--display-name") {
			t.Errorf("did not expect --display-name on re-launch, got %v", got)
		}
		if slices.Contains(got, "--agent-name") {
			t.Errorf("did not expect --agent-name on re-launch, got %v", got)
		}
	})

	t.Run("forwards extra args as a contiguous suffix after contract flags", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		dir := writeNanoclawCheckout(t, nanoclawPackageName)
		markNanoclawOnboarded(t, dir) // keep the suffix free of --display-name

		extra := []string{"--mode", "local"}
		if err := (&Nanoclaw{}).Run("gemma4:latest", nil, slices.Clone(extra)); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}

		got := readNanoclawInvocation(t, dir)
		if len(got) < len(extra) || !slices.Equal(got[len(got)-len(extra):], extra) {
			t.Errorf("extra args should be a contiguous suffix; got %v, want suffix %v", got, extra)
		}
	})

	t.Run("normalizes a 0.0.0.0 OLLAMA_HOST to a connectable loopback", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("OLLAMA_HOST", "0.0.0.0:11434")
		guardNanoclawAgainstInstall(t)
		dir := writeNanoclawCheckout(t, nanoclawPackageName)
		markNanoclawOnboarded(t, dir)

		if err := (&Nanoclaw{}).Run("gemma4:latest", nil, nil); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}

		got := readNanoclawInvocation(t, dir)
		assertFlagValue(t, got, "--base-url", envconfig.ConnectableHost().String())
		for i, a := range got {
			if a == "--base-url" && i+1 < len(got) && strings.Contains(got[i+1], "0.0.0.0") {
				t.Errorf("--base-url must not be an unconnectable 0.0.0.0 address, got %q", got[i+1])
			}
		}
	})
}

func TestNanoclawRunErrors(t *testing.T) {
	t.Run("errors when the entrypoint script is missing", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		dir := writeNanoclawCheckout(t, nanoclawPackageName)
		if err := os.Remove(nanoclawScriptPath(dir)); err != nil {
			t.Fatal(err)
		}

		err := (&Nanoclaw{}).Run("gemma4:latest", nil, nil)
		if err == nil || !strings.Contains(err.Error(), "entrypoint not found") {
			t.Fatalf("Run error = %v, want one mentioning the missing entrypoint", err)
		}
		if _, statErr := os.Stat(filepath.Join(dir, "invocation.txt")); statErr == nil {
			t.Error("entrypoint should not have been invoked when the script is missing")
		}
	})

	t.Run("errors when bash is unavailable", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		writeNanoclawCheckout(t, nanoclawPackageName)
		// An empty PATH hides bash from exec.LookPath.
		t.Setenv("PATH", t.TempDir())

		err := (&Nanoclaw{}).Run("gemma4:latest", nil, nil)
		if err == nil || !strings.Contains(err.Error(), "bash") {
			t.Fatalf("Run error = %v, want one mentioning bash", err)
		}
	})

	t.Run("bash-missing error includes a WSL2 hint on Windows", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		guardNanoclawAgainstInstall(t)
		setNanoclawGOOS(t, "windows")
		writeNanoclawCheckout(t, nanoclawPackageName)
		t.Setenv("PATH", t.TempDir()) // hide bash from exec.LookPath

		err := (&Nanoclaw{}).Run("gemma4:latest", nil, nil)
		if err == nil || !strings.Contains(err.Error(), "WSL2") {
			t.Fatalf("Run error = %v, want a WSL2 hint on Windows", err)
		}
	})
}

func TestNanoclawDockerInstallURL(t *testing.T) {
	cases := map[string]string{
		"darwin":  "docker.com/desktop",
		"windows": "docker.com/desktop",
		"linux":   "docker.com/engine",
	}
	for goos, want := range cases {
		t.Run(goos, func(t *testing.T) {
			setNanoclawGOOS(t, goos)
			if got := nanoclawDockerInstallURL(); !strings.Contains(got, want) {
				t.Errorf("nanoclawDockerInstallURL() for %s = %q, want one containing %q", goos, got, want)
			}
		})
	}
}

func TestNanoclawEnv(t *testing.T) {
	// Cloud provider credentials from the user's shell must not flow into the
	// entrypoint script or the containers it spawns.
	t.Setenv("ANTHROPIC_API_KEY", "secret")
	t.Setenv("OPENAI_API_KEY", "secret")
	t.Setenv("NANOCLAW_TEST_KEEP", "keep")

	env := nanoclawEnv()
	for _, e := range env {
		if strings.HasPrefix(e, "ANTHROPIC_API_KEY=") || strings.HasPrefix(e, "OPENAI_API_KEY=") {
			t.Errorf("provider credential leaked into handoff env: %q", e)
		}
	}
	if !slices.Contains(env, "NANOCLAW_TEST_KEEP=keep") {
		t.Error("non-provider environment variables should be preserved")
	}
}

func TestNanoclawEnsureInstalled(t *testing.T) {
	t.Run("refuses to install without a real terminal", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, false)
		forbidNanoclawConfirm(t) // a non-TTY refusal must never reach a prompt

		err := ensureNanoclawInstalled()
		if err == nil || !strings.Contains(err.Error(), "interactive terminal") {
			t.Fatalf("ensureNanoclawInstalled error = %v, want an interactive-terminal refusal", err)
		}
		if nanoclawInstalled() {
			t.Error("a refused install must not produce a checkout")
		}
	})

	t.Run("aggregates every missing prerequisite before prompting", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, true)
		forbidNanoclawConfirm(t)
		t.Setenv("PATH", t.TempDir()) // neither git nor docker on PATH

		err := ensureNanoclawInstalled()
		if err == nil {
			t.Fatal("ensureNanoclawInstalled error = nil, want a missing-dependencies error")
		}
		msg := err.Error()
		if !strings.Contains(msg, "required dependencies are missing") {
			t.Errorf("error should use the aggregated missing-deps wording, got %q", msg)
		}
		if !strings.Contains(msg, "git:") || !strings.Contains(msg, "Docker:") {
			t.Errorf("error should list both git and Docker in one message, got %q", msg)
		}
	})

	t.Run("reports only git when Docker is present", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, true)
		forbidNanoclawConfirm(t)
		bin := t.TempDir()
		writeFakeBinary(t, bin, "docker") // docker present, git absent
		t.Setenv("PATH", bin)

		err := ensureNanoclawInstalled()
		if err == nil || !strings.Contains(err.Error(), "git:") {
			t.Fatalf("ensureNanoclawInstalled error = %v, want a missing-git error", err)
		}
		if strings.Contains(err.Error(), "Docker:") {
			t.Errorf("should not list Docker when it is present, got %q", err.Error())
		}
	})

	t.Run("reports only Docker when git is present", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, true)
		forbidNanoclawConfirm(t)
		bin := t.TempDir()
		writeFakeBinary(t, bin, "git") // git present, docker absent
		t.Setenv("PATH", bin)

		err := ensureNanoclawInstalled()
		if err == nil || !strings.Contains(err.Error(), "Docker:") {
			t.Fatalf("ensureNanoclawInstalled error = %v, want a missing-Docker error", err)
		}
		if strings.Contains(err.Error(), "git:") {
			t.Errorf("should not list git when it is present, got %q", err.Error())
		}
	})

	t.Run("aborts when the first consent is declined", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, true)
		bin := t.TempDir()
		writeFakeBinary(t, bin, "git")
		writeFakeBinary(t, bin, "docker")
		t.Setenv("PATH", bin)
		stubNanoclawConfirm(t, false)

		err := ensureNanoclawInstalled()
		if err == nil || !strings.Contains(err.Error(), "cancelled") {
			t.Fatalf("ensureNanoclawInstalled error = %v, want a cancellation error", err)
		}
		if nanoclawInstalled() {
			t.Error("a cancelled install must not produce a checkout")
		}
	})

	t.Run("is a no-op when already installed", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, false) // would refuse if it didn't short-circuit
		forbidNanoclawConfirm(t)
		writeNanoclawCheckout(t, nanoclawPackageName)

		if err := ensureNanoclawInstalled(); err != nil {
			t.Fatalf("ensureNanoclawInstalled on an existing checkout = %v, want nil", err)
		}
	})

	t.Run("clones, validates, and replaces a stale invalid dir", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("fake git clone shim is a POSIX shell script")
		}
		setTestHome(t, t.TempDir())
		withInteractiveSession(t, true)
		bin := t.TempDir()
		writeFakeGitCloneShim(t, bin)
		writeFakeBinary(t, bin, "docker")
		// Prepend bin so the fake git/docker shadow the real ones, while the
		// shim can still find coreutils (mkdir/printf) on the inherited PATH.
		t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
		stubNanoclawConfirm(t, true)

		// A pre-existing non-empty invalid checkout must not wedge the install.
		dir, err := nanoclawDir()
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "stale.txt"), []byte("leftover"), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ensureNanoclawInstalled(); err != nil {
			t.Fatalf("ensureNanoclawInstalled = %v, want success", err)
		}
		if !nanoclawInstalled() {
			t.Error("expected a valid checkout after a successful install")
		}
		if _, statErr := os.Stat(filepath.Join(dir, "stale.txt")); statErr == nil {
			t.Error("the stale leftover should have been replaced")
		}
		if _, statErr := os.Stat(dir + ".tmp"); statErr == nil {
			t.Error("the temp clone dir should have been moved/cleaned, not left behind")
		}
	})
}

// writeNanoclawCheckout creates a minimal fake NanoClaw checkout under the test
// HOME with a recording entrypoint script, and returns the checkout dir. The
// script writes its arguments (one per line) to invocation.txt in its working
// directory so tests can assert the exact seam-contract flags launch passed.
func writeNanoclawCheckout(t *testing.T, packageName string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir: %v", err)
	}
	dir := filepath.Join(home, ".ollama", "launch", "nanoclaw")
	if err := os.MkdirAll(filepath.Join(dir, "scripts"), 0o755); err != nil {
		t.Fatal(err)
	}
	pkg := `{"name":"` + packageName + `"}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(pkg), 0o644); err != nil {
		t.Fatal(err)
	}
	script := "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > invocation.txt\n"
	if err := os.WriteFile(nanoclawScriptPath(dir), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

func markNanoclawOnboarded(t *testing.T, dir string) {
	t.Helper()
	dataDir := filepath.Join(dir, "data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "upgrade-state.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeFakeGitCloneShim installs a fake `git` that emulates
// `git clone --depth 1 <url> <dest>` by materializing a minimal valid NanoClaw
// checkout at the destination (the last argument), so the install path can be
// exercised hermetically without any network access.
func writeFakeGitCloneShim(t *testing.T, dir string) {
	t.Helper()
	script := "#!/bin/sh\n" +
		"for a in \"$@\"; do dest=\"$a\"; done\n" +
		"mkdir -p \"$dest/scripts\"\n" +
		"printf '%s' '{\"name\":\"nanoclaw\"}' > \"$dest/package.json\"\n" +
		"printf '#!/usr/bin/env bash\\n' > \"$dest/scripts/ollama-launch.sh\"\n" +
		"exit 0\n"
	if err := os.WriteFile(filepath.Join(dir, "git"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake git: %v", err)
	}
}

// guardNanoclawAgainstInstall makes Run happy-path tests defensively hermetic:
// the checkout already exists so ensureNanoclawInstalled short-circuits, but if
// a regression ever let it fall through, this fails loudly instead of cloning
// the real repository.
func guardNanoclawAgainstInstall(t *testing.T) {
	t.Helper()
	withInteractiveSession(t, true)
	forbidNanoclawConfirm(t)
}

// setNanoclawGOOS overrides the runtime.GOOS seam for the duration of a test so
// the OS-specific Docker hints and Windows WSL2 guidance can be exercised off
// the target platform.
func setNanoclawGOOS(t *testing.T, goos string) {
	t.Helper()
	old := nanoclawGOOS
	nanoclawGOOS = goos
	t.Cleanup(func() { nanoclawGOOS = old })
}

func forbidNanoclawConfirm(t *testing.T) {
	t.Helper()
	old := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect a confirmation prompt")
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = old })
}

func stubNanoclawConfirm(t *testing.T, answer bool) {
	t.Helper()
	old := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
		return answer, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = old })
}

func readNanoclawInvocation(t *testing.T, dir string) []string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "invocation.txt"))
	if err != nil {
		t.Fatalf("entrypoint was not invoked (no invocation.txt): %v", err)
	}
	var args []string
	for _, line := range strings.Split(strings.TrimRight(string(data), "\n"), "\n") {
		if line != "" {
			args = append(args, line)
		}
	}
	return args
}

func assertFlagValue(t *testing.T, args []string, flag, want string) {
	t.Helper()
	for i, a := range args {
		if a == flag {
			if i+1 >= len(args) {
				t.Fatalf("flag %q has no value in %v", flag, args)
			}
			if args[i+1] != want {
				t.Errorf("%s = %q, want %q", flag, args[i+1], want)
			}
			return
		}
	}
	t.Errorf("flag %q not found in %v", flag, args)
}
