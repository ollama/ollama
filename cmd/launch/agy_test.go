package launch

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAgyIntegration(t *testing.T) {
	a := &Agy{}

	t.Run("String", func(t *testing.T) {
		if got := a.String(); got != "Agy" {
			t.Errorf("String() = %q, want %q", got, "Agy")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = a
	})
}

func TestEnsureAgyInstalled(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("PATH", tmpDir)

	agyPath := filepath.Join(tmpDir, "agy")
	npmScript := fmt.Sprintf(`#!/bin/sh
printf '%%s\n' "$*" > "$HOME/npm-calls.log"
/bin/cat > %q <<'EOF'
#!/bin/sh
exit 0
EOF
/bin/chmod +x %q
exit 0
`, agyPath, agyPath)
	if err := os.WriteFile(filepath.Join(tmpDir, "npm"), []byte(npmScript), 0o755); err != nil {
		t.Fatal(err)
	}

	oldConfirmPrompt := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != "Agy is not installed. Install now?" {
			t.Fatalf("unexpected prompt: %q", prompt)
		}
		return true, nil
	}
	defer func() { DefaultConfirmPrompt = oldConfirmPrompt }()

	bin, err := ensureAgyInstalled()
	if err != nil {
		t.Fatalf("ensureAgyInstalled() error = %v", err)
	}
	if bin != "agy" {
		t.Fatalf("ensureAgyInstalled() bin = %q, want %q", bin, "agy")
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "npm-calls.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "install -g agy" {
		t.Fatalf("npm args = %q, want %q", got, "install -g agy")
	}
}

func TestAgyRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	t.Setenv("PATH", tmpDir)

	agyPath := filepath.Join(tmpDir, "agy")
	os.WriteFile(agyPath, []byte("#!/bin/sh\nexit 0"), 0o755)

	a := &Agy{}
	
	t.Run("runs with model", func(t *testing.T) {
		err := a.Run("gemma4:cloud", nil, []string{})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	})

	t.Run("runs with args", func(t *testing.T) {
		err := a.Run("gemma4:cloud", nil, []string{"--help"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	})
}
