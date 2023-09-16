package paths

import (
	"os"
	"testing"
)

func TestOllamaHomeDirWithEnvVarSet(t *testing.T) {
	err := os.Setenv(ollamaHomeEnvVar, "/haha/hihi")
	if err != nil {
		t.Fatalf("could not set env var %q: %s", ollamaHomeEnvVar, err)
	}

	got, err := OllamaHomeDir()
	if err != nil {
		t.Fatalf("error on OllamaHomeDir(): %s", err)
	}

	want := "/haha/hihi"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestOllamaHomeDirWithoutEnvVarSet(t *testing.T) {
	// unset env var
	err := os.Setenv(ollamaHomeEnvVar, "")
	if err != nil {
		t.Fatalf("could not unset %q: %s", ollamaHomeEnvVar, err)
	}

	userHomeDir, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("error on UserHomeDir(): %s", err)
	}

	got, err := OllamaHomeDir()
	if err != nil {
		t.Fatalf("error on OllamaHomeDir(): %s", err)
	}

	want := userHomeDir
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
