package cmd

import (
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func writeSkillFixture(t *testing.T) string {
	t.Helper()

	src := t.TempDir()

	skillToml := `name = "echoer"
description = "Echoes text input"
version = "0.1.0"
command = "./run.sh"
args = ["--static"]

[io]
inputs = ["text"]
outputs = ["text"]

[permissions]
required = ["filesystem.read"]
`
	if err := os.WriteFile(filepath.Join(src, "skill.toml"), []byte(skillToml), 0o644); err != nil {
		t.Fatalf("write skill.toml: %v", err)
	}

	script := "#!/bin/sh\nprintf 'run:%s %s\\n' \"$1\" \"$2\"\n"
	if err := os.WriteFile(filepath.Join(src, "run.sh"), []byte(script), 0o755); err != nil {
		t.Fatalf("write run.sh: %v", err)
	}

	return src
}

func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()

	root := NewCLI()
	root.SetContext(t.Context())
	root.SetArgs(args)

	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	defer func() {
		os.Stdout = oldStdout
	}()

	err := root.Execute()
	w.Close()
	data, _ := io.ReadAll(r)

	return string(data), err
}

func TestSkillsCommandLifecycle(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillFixture(t)

	out, err := runCLI(t, "skills", "install", src)
	if err != nil {
		t.Fatalf("skills install failed: %v", err)
	}
	if !strings.Contains(out, "Installed skill 'echoer'.") {
		t.Fatalf("skills install output = %q", out)
	}

	out, err = runCLI(t, "skills", "list")
	if err != nil {
		t.Fatalf("skills list failed: %v", err)
	}
	if !strings.Contains(out, "echoer") || !strings.Contains(out, "no") {
		t.Fatalf("skills list output = %q", out)
	}

	out, err = runCLI(t, "skills", "enable", "echoer")
	if err != nil {
		t.Fatalf("skills enable failed: %v", err)
	}
	if !strings.Contains(out, "Enabled skill 'echoer'.") {
		t.Fatalf("skills enable output = %q", out)
	}

	out, err = runCLI(t, "skills", "list")
	if err != nil {
		t.Fatalf("skills list failed: %v", err)
	}
	if !strings.Contains(out, "echoer") || !strings.Contains(out, "yes") {
		t.Fatalf("skills list output after enable = %q", out)
	}

	out, err = runCLI(t, "skills", "disable", "echoer")
	if err != nil {
		t.Fatalf("skills disable failed: %v", err)
	}
	if !strings.Contains(out, "Disabled skill 'echoer'.") {
		t.Fatalf("skills disable output = %q", out)
	}
}

func TestSkillsRunCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillFixture(t)

	if _, err := runCLI(t, "skills", "install", src); err != nil {
		t.Fatalf("skills install failed: %v", err)
	}
	if _, err := runCLI(t, "skills", "enable", "echoer"); err != nil {
		t.Fatalf("skills enable failed: %v", err)
	}

	out, err := runCLI(t, "skills", "run", "echoer", "--allow", "filesystem.read", "hello")
	if err != nil {
		t.Fatalf("skills run failed: %v", err)
	}
	if got, want := out, "run:--static hello\n"; got != want {
		t.Fatalf("skills run output = %q, want %q", got, want)
	}
}

func TestSkillsRunCommandRequiresPermission(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillFixture(t)

	if _, err := runCLI(t, "skills", "install", src); err != nil {
		t.Fatalf("skills install failed: %v", err)
	}
	if _, err := runCLI(t, "skills", "enable", "echoer"); err != nil {
		t.Fatalf("skills enable failed: %v", err)
	}

	if _, err := runCLI(t, "skills", "run", "echoer", "hello"); err == nil {
		t.Fatalf("skills run expected permission error, got nil")
	}
}
