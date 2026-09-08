package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func withTalosPlatform(t *testing.T, goos string) {
	t.Helper()
	old := talosGOOS
	talosGOOS = goos
	t.Cleanup(func() {
		talosGOOS = old
	})
}

func withTalosUserHome(t *testing.T, dir string) {
	t.Helper()
	old := talosUserHome
	talosUserHome = func() (string, error) { return dir, nil }
	t.Cleanup(func() {
		talosUserHome = old
	})
}

// clearTalosEnvVars keeps the host's Talos environment out of tests: the
// installer-supported overrides would redirect prefix and secrets resolution.
func clearTalosEnvVars(t *testing.T) {
	t.Helper()
	for _, key := range []string{"TALOS_PREFIX", "TALOS_SECRETS_ENV", "TALOS_MODEL_PROVIDER", "TALOS_MODEL", "OLLAMA_API_KEY"} {
		t.Setenv(key, "")
	}
}

func writeTalosScript(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTalosIntegration(t *testing.T) {
	talos := &Talos{}

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = talos
	})

	t.Run("implements managed single model", func(t *testing.T) {
		var _ ManagedSingleModel = talos
	})

	t.Run("implements supported integration", func(t *testing.T) {
		var _ SupportedIntegration = talos
	})
}

func TestTalosSupportedByPlatform(t *testing.T) {
	withTalosPlatform(t, "windows")
	if err := (&Talos{}).Supported(); err == nil {
		t.Fatal("expected windows to be unsupported")
	}

	withTalosPlatform(t, "darwin")
	if err := (&Talos{}).Supported(); err != nil {
		t.Fatalf("expected darwin to be supported, got %v", err)
	}
}

func TestTalosCommandPrefersPathShim(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)

	bin := writeTalosScript(t, tmpDir, "talos", "#!/bin/sh\nexit 0\n")

	argv, err := (&Talos{}).command()
	if err != nil {
		t.Fatalf("command returned error: %v", err)
	}
	if len(argv) != 1 || argv[0] != bin {
		t.Fatalf("expected PATH shim [%s], got %v", bin, argv)
	}
}

func TestTalosCommandFallsBackToInstallerVenv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withTalosUserHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)

	// Older installations have only the venv, without bin/talos.
	python := filepath.Join(tmpDir, "talos", ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(python), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(python, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	argv, err := (&Talos{}).command()
	if err != nil {
		t.Fatalf("command returned error: %v", err)
	}
	want := []string{python, "-m", "talos"}
	if strings.Join(argv, " ") != strings.Join(want, " ") {
		t.Fatalf("expected %v, got %v", want, argv)
	}
}

func TestTalosCommandHonorsPrefixOverride(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)

	prefix := filepath.Join(tmpDir, "custom-prefix")
	python := filepath.Join(prefix, ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(python), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(python, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TALOS_PREFIX", prefix)

	argv, err := (&Talos{}).command()
	if err != nil {
		t.Fatalf("command returned error: %v", err)
	}
	if argv[0] != python {
		t.Fatalf("expected venv python under TALOS_PREFIX, got %v", argv)
	}
}

func TestTalosConfigureLocalModelUsesTalosConfigSurface(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	log := filepath.Join(tmpDir, "talos-invocations.log")
	writeTalosScript(t, tmpDir, "talos", fmt.Sprintf("#!/bin/sh\nprintf '[%%s]\\n' \"$*\" >> %q\n", log))

	if err := (&Talos{}).Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	want := []string{
		"[config set TALOS_MODEL_PROVIDER ollama]",
		"[config set TALOS_MODEL gemma4]",
	}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Fatalf("expected invocations %v, got %v", want, lines)
	}
}

func TestTalosConfigureCloudModelStripsCloudTag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	log := filepath.Join(tmpDir, "talos-invocations.log")
	writeTalosScript(t, tmpDir, "talos", fmt.Sprintf("#!/bin/sh\nprintf '[%%s]\\n' \"$*\" >> %q\n", log))

	if err := (&Talos{}).Configure("kimi-k2.5:cloud"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	want := []string{
		"[config set TALOS_MODEL_PROVIDER ollama-cloud]",
		"[config set TALOS_MODEL kimi-k2.5]",
	}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Fatalf("expected invocations %v, got %v", want, lines)
	}
}

func TestTalosConfigureWrapsConfigSetFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	writeTalosScript(t, tmpDir, "talos", "#!/bin/sh\necho 'unknown key' >&2\nexit 1\n")

	err := (&Talos{}).Configure("gemma4")
	if err == nil || !strings.Contains(err.Error(), "talos config set TALOS_MODEL_PROVIDER") {
		t.Fatalf("expected wrapped config set failure, got %v", err)
	}
}

func TestTalosConfigureRejectsConflictingEnvOverride(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	t.Setenv("TALOS_MODEL", "qwen3.5")

	log := filepath.Join(tmpDir, "talos-invocations.log")
	writeTalosScript(t, tmpDir, "talos", fmt.Sprintf("#!/bin/sh\nprintf '[%%s]\\n' \"$*\" >> %q\n", log))

	err := (&Talos{}).Configure("gemma4")
	if err == nil || !strings.Contains(err.Error(), "TALOS_MODEL") || !strings.Contains(err.Error(), "overrides") {
		t.Fatalf("expected env override error, got %v", err)
	}

	// Nothing may be written when the saved config would not take effect.
	if _, statErr := os.Stat(log); !os.IsNotExist(statErr) {
		t.Fatalf("expected no talos invocations, log exists")
	}
}

func TestTalosConfigureAcceptsMatchingEnvOverride(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	t.Setenv("TALOS_MODEL_PROVIDER", "ollama")
	t.Setenv("TALOS_MODEL", "gemma4")

	log := filepath.Join(tmpDir, "talos-invocations.log")
	writeTalosScript(t, tmpDir, "talos", fmt.Sprintf("#!/bin/sh\nprintf '[%%s]\\n' \"$*\" >> %q\n", log))

	if err := (&Talos{}).Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "[config set TALOS_MODEL gemma4]") {
		t.Fatalf("expected config set calls, got %q", data)
	}
}

func TestTalosCurrentModel(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withTalosUserHome(t, tmpDir)
	clearTalosEnvVars(t)

	writeConfig := func(content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(tmpDir, "talos", "talos.env"), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(tmpDir, "talos"), 0o755); err != nil {
		t.Fatal(err)
	}

	t.Run("returns empty when nothing is configured", func(t *testing.T) {
		writeConfig("# fresh install\n")
		if got := (&Talos{}).CurrentModel(); got != "" {
			t.Fatalf("expected empty current model, got %q", got)
		}
	})

	t.Run("returns local ollama model", func(t *testing.T) {
		writeConfig("TALOS_MODEL_PROVIDER=ollama\nTALOS_MODEL=gemma4\n")
		if got := (&Talos{}).CurrentModel(); got != "gemma4" {
			t.Fatalf("expected gemma4, got %q", got)
		}
	})

	t.Run("reports cloud model with launch naming", func(t *testing.T) {
		writeConfig("TALOS_MODEL_PROVIDER=ollama-cloud\nTALOS_MODEL=kimi-k2.5\n")
		if got := (&Talos{}).CurrentModel(); got != "kimi-k2.5:cloud" {
			t.Fatalf("expected kimi-k2.5:cloud, got %q", got)
		}
	})

	t.Run("ignores models owned by another provider", func(t *testing.T) {
		writeConfig("TALOS_MODEL_PROVIDER=anthropic\nTALOS_MODEL=claude-opus\n")
		if got := (&Talos{}).CurrentModel(); got != "" {
			t.Fatalf("expected empty current model, got %q", got)
		}
	})

	t.Run("secrets file overrides the main config", func(t *testing.T) {
		writeConfig("TALOS_MODEL_PROVIDER=anthropic\nTALOS_MODEL=claude-opus\n")
		secretsPath := filepath.Join(tmpDir, "secrets.env")
		if err := os.WriteFile(secretsPath, []byte("TALOS_MODEL_PROVIDER=ollama\nTALOS_MODEL=qwen3.5\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		t.Setenv("TALOS_SECRETS_ENV", secretsPath)
		if got := (&Talos{}).CurrentModel(); got != "qwen3.5" {
			t.Fatalf("expected qwen3.5, got %q", got)
		}
	})
}

func TestTalosRunPassthroughArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	clearTalosEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	writeTalosScript(t, tmpDir, "talos", "#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/talos-invocations.log\"\n")

	if err := (&Talos{}).Run("", nil, []string{"--continue"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "talos-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "[chat --continue]" {
		t.Fatalf("expected chat session with passthrough args, got %q", got)
	}
}

func TestTalosEnsureInstalledUnixPromptsBeforeInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withTalosUserHome(t, tmpDir)
	clearTalosEnvVars(t)
	withLauncherHooks(t)
	t.Setenv("PATH", tmpDir)

	writeTalosScript(t, tmpDir, "curl", "#!/bin/sh\nexit 0\n")
	// The fake installer lays down the venv the real one would create.
	writeTalosScript(t, tmpDir, "bash", fmt.Sprintf(`#!/bin/sh
printf '%%s\n' "$*" >> %q
/bin/mkdir -p %q
/bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
/bin/chmod +x %q
exit 0
`, filepath.Join(tmpDir, "bash.log"), filepath.Join(tmpDir, "talos", ".venv", "bin"), filepath.Join(tmpDir, "talos", ".venv", "bin", "python"), filepath.Join(tmpDir, "talos", ".venv", "bin", "python")))

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != "Talos is not installed. Install now?" {
			t.Fatalf("unexpected install prompt %q", prompt)
		}
		return true, nil
	}

	if err := (&Talos{}).ensureInstalled(); err != nil {
		t.Fatalf("ensureInstalled returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "bash.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "-o pipefail -c "+talosInstallScript) {
		t.Fatalf("expected official install script invocation, got logs:\n%s", data)
	}
}

func TestTalosEnsureInstalledUnixCanBeDeclined(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withTalosUserHome(t, tmpDir)
	clearTalosEnvVars(t)
	withLauncherHooks(t)
	t.Setenv("PATH", tmpDir)

	for _, name := range []string{"curl", "bash"} {
		writeTalosScript(t, tmpDir, name, "#!/bin/sh\nexit 0\n")
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != "Talos is not installed. Install now?" {
			t.Fatalf("unexpected install prompt %q", prompt)
		}
		return false, nil
	}

	err := (&Talos{}).ensureInstalled()
	if err == nil || !strings.Contains(err.Error(), "talos installation cancelled") {
		t.Fatalf("expected install cancellation error, got %v", err)
	}
}

func TestTalosParseEnvFile(t *testing.T) {
	parsed := talosParseEnvFile([]byte(`
# a comment
TALOS_MODEL_PROVIDER=ollama
TALOS_MODEL="qwen3.5"
QUOTED='single'
BROKEN_LINE
EMPTY=
`))

	if got := parsed["TALOS_MODEL_PROVIDER"]; got != "ollama" {
		t.Fatalf("expected ollama, got %q", got)
	}
	if got := parsed["TALOS_MODEL"]; got != "qwen3.5" {
		t.Fatalf("expected quotes stripped, got %q", got)
	}
	if got := parsed["QUOTED"]; got != "single" {
		t.Fatalf("expected single quotes stripped, got %q", got)
	}
	if _, ok := parsed["BROKEN_LINE"]; ok {
		t.Fatal("expected line without = to be skipped")
	}
	if got, ok := parsed["EMPTY"]; !ok || got != "" {
		t.Fatalf("expected empty value to be kept, got %q (present=%v)", got, ok)
	}
}

func TestTalosInstallerWrapperAndConfigStayTogether(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX launcher")
	}
	tmp := t.TempDir()
	clearTalosEnvVars(t)
	withTalosUserHome(t, tmp)
	prefix := filepath.Join(tmp, "installed with spaces")
	for _, dir := range []string{"bin", ".venv/bin", "talos"} {
		if err := os.MkdirAll(filepath.Join(prefix, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	wrapper := writeTalosScript(t, filepath.Join(prefix, "bin"), "talos", "#!/bin/sh\nexit 0\n")
	writeTalosScript(t, filepath.Join(prefix, ".venv/bin"), "python", "#!/bin/sh\nexit 0\n")
	if err := os.WriteFile(filepath.Join(prefix, "talos/__main__.py"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(prefix, "talos.env"), []byte("TALOS_MODEL_PROVIDER=ollama\nTALOS_MODEL=fixture-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TALOS_SECRETS_ENV", filepath.Join(tmp, "no-secrets"))
	t.Run("prefix without PATH entry", func(t *testing.T) {
		t.Setenv("PATH", tmp)
		t.Setenv("TALOS_PREFIX", prefix)
		argv, err := (&Talos{}).command()
		if err != nil || len(argv) != 1 || argv[0] != wrapper {
			t.Fatalf("wrapper not selected: %v, %v", argv, err)
		}
	})
	t.Run("PATH symlink wins over another prefix", func(t *testing.T) {
		bin := filepath.Join(tmp, "commands")
		if err := os.Mkdir(bin, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(wrapper, filepath.Join(bin, "talos")); err != nil {
			t.Fatal(err)
		}
		t.Setenv("PATH", bin)
		t.Setenv("TALOS_PREFIX", filepath.Join(tmp, "different-installation"))
		if got := (&Talos{}).CurrentModel(); got != "fixture-model" {
			t.Fatalf("read config from wrong installation: %q", got)
		}
		paths := (&Talos{}).Paths()
		canonical, err := filepath.EvalSymlinks(filepath.Join(prefix, "talos.env"))
		if err != nil {
			t.Fatal(err)
		}
		if len(paths) != 1 || paths[0] != canonical {
			t.Fatalf("wrong config path: %v", paths)
		}
	})
}

func TestTalosLegacyVenvKeepsCallerDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX launcher")
	}
	tmp := t.TempDir()
	clearTalosEnvVars(t)
	prefix := filepath.Join(tmp, "legacy install")
	bin := filepath.Join(prefix, ".venv/bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	log := filepath.Join(tmp, "invocations")
	writeTalosScript(t, bin, "python", fmt.Sprintf("#!/bin/sh\nprintf '%%s|%%s|%%s\\n' \"$PWD\" \"$PYTHONPATH\" \"$*\" >> %q\n", log))
	caller := filepath.Join(tmp, "project")
	if err := os.Mkdir(caller, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(caller)
	t.Setenv("PATH", tmp)
	t.Setenv("TALOS_PREFIX", prefix)
	t.Setenv("PYTHONPATH", filepath.Join(tmp, "existing-python-path"))
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	if err := (&Talos{}).Configure("fixture-model"); err != nil {
		t.Fatal(err)
	}
	if err := (&Talos{}).Run("", nil, []string{"--continue"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected configure and run: %q", data)
	}
	wantPrefix := caller + "|" + prefix + string(os.PathListSeparator) + filepath.Join(tmp, "existing-python-path") + "|"
	for _, line := range lines {
		if !strings.HasPrefix(line, wantPrefix) {
			t.Fatalf("caller/import context lost: %q", line)
		}
	}
	if !strings.HasSuffix(lines[2], "-m talos chat --continue") {
		t.Fatalf("arguments changed: %q", lines[2])
	}
}

func TestTalosInstallExecutesBashAndPropagatesDownloadFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX launcher")
	}
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash unavailable")
	}
	for _, failDownload := range []bool{false, true} {
		t.Run(fmt.Sprintf("download-failure-%v", failDownload), func(t *testing.T) {
			tmp := t.TempDir()
			clearTalosEnvVars(t)
			withTalosUserHome(t, tmp)
			withLauncherHooks(t)
			if err := os.Symlink(bash, filepath.Join(tmp, "bash")); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink("/bin/sh", filepath.Join(tmp, "sh")); err != nil {
				t.Fatal(err)
			}
			prefix := filepath.Join(tmp, "prefix")
			t.Setenv("TALOS_PREFIX", prefix)
			t.Setenv("PATH", tmp)
			marker := filepath.Join(tmp, "bash-proof")
			curl := fmt.Sprintf("#!/bin/sh\n/bin/cat <<'INSTALLER'\nset -euo pipefail\ntrap ':' ERR\nvalues=(bash-only)\nprintf '%%s' \"${values[0]}\" > %q\n/bin/mkdir -p %q\nprintf '#!/bin/sh\\nexit 0\\n' > %q\n/bin/chmod +x %q\nINSTALLER\n", marker, filepath.Join(prefix, "bin"), filepath.Join(prefix, "bin/talos"), filepath.Join(prefix, "bin/talos"))
			if failDownload {
				curl += "exit 22\n"
			}
			writeTalosScript(t, tmp, "curl", curl)
			prompts := 0
			DefaultConfirmPrompt = func(_ string, _ ConfirmOptions) (bool, error) { prompts++; return true, nil }
			err := (&Talos{}).ensureInstalled()
			if failDownload {
				if err == nil || !strings.Contains(err.Error(), "failed to install talos") {
					t.Fatalf("download failure was lost: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(marker)
				if err != nil || string(data) != "bash-only" {
					t.Fatalf("Bash installer did not run: %q %v", data, err)
				}
			}
			if prompts != 1 {
				t.Fatalf("install consent requested %d times", prompts)
			}
		})
	}
}
