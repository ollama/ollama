package launch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAtomCodeIntegration(t *testing.T) {
	a := &AtomCode{}
	if got := a.String(); got != "AtomCode" {
		t.Errorf("String() = %q, want %q", got, "AtomCode")
	}
	var _ Runner = a
}

func TestWriteAtomCodeConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ollama-launch.toml")
	if err := writeAtomCodeConfig(path, "qwen3:32b", []LaunchModel{{Name: "qwen3:32b", ContextLength: 32768}}); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		`default_provider = "ollama"`,
		"[providers.ollama]",
		`type = "ollama"`,
		`model = "qwen3:32b"`,
		"base_url = ",
		"context_window = 32768",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("config missing %q:\n%s", want, text)
		}
	}
}

func TestWriteAtomCodeConfigOmitsUnknownContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ollama-launch.toml")
	if err := writeAtomCodeConfig(path, "qwen3:32b", nil); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "context_window") {
		t.Errorf("context_window should be omitted when model context is unknown:\n%s", data)
	}
}
