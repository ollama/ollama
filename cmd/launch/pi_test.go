package launch

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestPiIntegration(t *testing.T) {
	pi := &Pi{}

	t.Run("String", func(t *testing.T) {
		if got := pi.String(); got != "Pi" {
			t.Errorf("String() = %q, want %q", got, "Pi")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = pi
	})
}

func TestPiInstallSpec_UsesOfficialPackage(t *testing.T) {
	spec, err := LookupIntegrationSpec("pi")
	if err != nil {
		t.Fatalf("LookupIntegrationSpec(pi) error = %v", err)
	}

	want := []string{"npm", "install", "-g", piNpmPackage + "@latest"}
	if got := spec.Install.Command; !slices.Equal(got, want) {
		t.Fatalf("pi install command = %v, want %v", got, want)
	}
}

func TestPiRun_InstallLifecycle(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	writeScript := func(t *testing.T, path, content string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	seedPiScript := func(t *testing.T, dir string) {
		t.Helper()
		plugin := filepath.Join(dir, "ollama-plugin")
		if err := os.MkdirAll(plugin, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, data := range map[string]string{
			"package.json": `{"name":"@ollama/pi"}`,
			"index.ts":     "",
		} {
			if err := os.WriteFile(filepath.Join(plugin, name), []byte(data), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		writeScript(t, filepath.Join(dir, "pi"), fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "--version" ]; then echo 1.5.2; exit 0; fi
if [ "$1" = "list" ]; then echo %q; fi
`, filepath.Join(dir, "pi.log"), plugin))
	}
	withConfirm := func(t *testing.T, accept bool) {
		t.Helper()
		oldConfirm := DefaultConfirmPrompt
		DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return accept, nil }
		t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
	}

	t.Run("Pi missing and user accepts install", func(t *testing.T) {
		root := t.TempDir()
		setTestHome(t, root)
		t.Setenv("PATH", root)
		seedPiScript(t, root)
		piPath := filepath.Join(root, "pi")
		template, err := os.ReadFile(piPath)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(piPath); err != nil {
			t.Fatal(err)
		}
		writeScript(t, filepath.Join(root, "npm"), fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
/bin/cat > %q <<'EOS'
%s
EOS
/bin/chmod +x %q
`, filepath.Join(root, "npm.log"), piPath, template, piPath))
		withConfirm(t, true)
		if err := (&Pi{}).Run("", nil, []string{"--continue"}); err != nil {
			t.Fatal(err)
		}
		npmCalls, err := os.ReadFile(filepath.Join(root, "npm.log"))
		if err != nil || string(npmCalls) != "install -g "+piNpmPackage+"@latest\n" {
			t.Fatalf("unexpected npm calls: %s (%v)", npmCalls, err)
		}
		piCalls, err := os.ReadFile(filepath.Join(root, "pi.log"))
		if err != nil || string(piCalls) != "--version\nlist\n--continue\n" {
			t.Fatalf("unexpected Pi calls: %s (%v)", piCalls, err)
		}
	})

	t.Run("Pi missing and user declines install", func(t *testing.T) {
		root := t.TempDir()
		setTestHome(t, root)
		t.Setenv("PATH", root)
		writeScript(t, filepath.Join(root, "npm"), "#!/bin/sh\nexit 1\n")
		withConfirm(t, false)
		err := (&Pi{}).Run("", nil, nil)
		if err == nil || !strings.Contains(err.Error(), "pi installation cancelled") {
			t.Fatalf("expected cancellation, got %v", err)
		}
	})

	for _, npmAvailable := range []bool{false, true} {
		t.Run(fmt.Sprintf("installed Pi is reused with npm available=%t", npmAvailable), func(t *testing.T) {
			root := t.TempDir()
			setTestHome(t, root)
			t.Setenv("PATH", root)
			seedPiScript(t, root)
			if npmAvailable {
				writeScript(t, filepath.Join(root, "npm"), fmt.Sprintf("#!/bin/sh\necho called >> %q\nexit 1\n", filepath.Join(root, "npm.log")))
			}
			if err := (&Pi{}).Run("", nil, []string{"--continue"}); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(root, "npm.log")); !os.IsNotExist(err) {
				t.Fatalf("installed Pi should not require npm: %v", err)
			}
			calls, err := os.ReadFile(filepath.Join(root, "pi.log"))
			if err != nil || string(calls) != "--version\nlist\n--continue\n" {
				t.Fatalf("unexpected Pi calls: %s (%v)", calls, err)
			}
		})
	}
}

func TestPiRequiresSupportedVersionBeforePluginSetup(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX fake Pi executable")
	}
	for _, tc := range []struct {
		version string
		accept  bool
	}{
		{"0.67.0", false},
		{"1.5.1", false},
		{"1.5.2-rc.1", false},
		{"1.5.2", true},
		{"v1.5.2", true},
		{"1.5.10", true},
		{"unknown", false},
	} {
		t.Run(tc.version, func(t *testing.T) {
			root := t.TempDir()
			setTestHome(t, root)
			t.Setenv("PATH", root)
			log := filepath.Join(root, "pi.log")
			bin := filepath.Join(root, "pi")
			script := fmt.Sprintf("#!/bin/sh\necho \"$@\" >> %q\nif [ \"$1\" = \"--version\" ]; then echo %q; exit 0; fi\nexit 1\n", log, tc.version)
			if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
				t.Fatal(err)
			}
			if tc.accept {
				if err := checkPiVersion(bin); err != nil {
					t.Fatal(err)
				}
				return
			}
			_, err := preparePi()
			if err == nil || !strings.Contains(err.Error(), "Pi 1.5.2 or newer") || !strings.Contains(err.Error(), "npm install -g "+piNpmPackage+"@latest") {
				t.Fatalf("expected a manual-upgrade message, got %v", err)
			}
			calls, err := os.ReadFile(log)
			if err != nil || string(calls) != "--version\n" {
				t.Fatalf("incompatible Pi must stop before package setup: %s (%v)", calls, err)
			}
		})
	}
}
