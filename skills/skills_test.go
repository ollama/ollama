package skills

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeSkillSource(t *testing.T) string {
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

func writeSkillSourceWithScript(t *testing.T, scriptBody string) string {
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

	if err := os.WriteFile(filepath.Join(src, "run.sh"), []byte(scriptBody), 0o755); err != nil {
		t.Fatalf("write run.sh: %v", err)
	}

	return src
}

func TestInstallListEnableDisable(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	src := writeSkillSource(t)

	installed, err := Install(src)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if installed.Spec.Name != "echoer" {
		t.Fatalf("Install() name = %q, want %q", installed.Spec.Name, "echoer")
	}
	if installed.Enabled {
		t.Fatalf("Install() enabled = true, want false")
	}

	all, err := List()
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(all) != 1 {
		t.Fatalf("List() len = %d, want 1", len(all))
	}
	if all[0].Enabled {
		t.Fatalf("List()[0].Enabled = true, want false")
	}

	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	enabled, err := Enabled()
	if err != nil {
		t.Fatalf("Enabled() error = %v", err)
	}
	if len(enabled) != 1 {
		t.Fatalf("Enabled() len = %d, want 1", len(enabled))
	}
	if !enabled[0].Enabled {
		t.Fatalf("Enabled()[0].Enabled = false, want true")
	}

	if err := Disable("echoer"); err != nil {
		t.Fatalf("Disable() error = %v", err)
	}

	enabled, err = Enabled()
	if err != nil {
		t.Fatalf("Enabled() after disable error = %v", err)
	}
	if len(enabled) != 0 {
		t.Fatalf("Enabled() after disable len = %d, want 0", len(enabled))
	}
}

func TestRunRequiresEnabledSkill(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillSource(t)

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}

	err := Run(context.Background(), "echoer", []string{"hello"}, nil, &bytes.Buffer{}, &bytes.Buffer{})
	if !errors.Is(err, ErrSkillNotEnabled) {
		t.Fatalf("Run() error = %v, want ErrSkillNotEnabled", err)
	}
}

func TestRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	t.Setenv("OLLAMA_SKILL_ALLOW", "filesystem.read")
	src := writeSkillSource(t)

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	var out bytes.Buffer
	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &out, &out); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got, want := out.String(), "run:--static hello\n"; got != want {
		t.Fatalf("Run() output = %q, want %q", got, want)
	}
}

func TestRunRequiresPermissionGrant(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillSource(t)

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	err := Run(context.Background(), "echoer", []string{"hello"}, nil, &bytes.Buffer{}, &bytes.Buffer{})
	if !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("Run() error = %v, want ErrPermissionDenied", err)
	}
}

func TestLoadSpecRequiresCommand(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "skill.toml"), []byte("name = \"bad\"\n"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	_, err := LoadSpec(filepath.Join(tmp, "skill.toml"))
	if err == nil {
		t.Fatalf("LoadSpec() expected error, got nil")
	}
}

func TestLoadSpecRejectsInvalidPermission(t *testing.T) {
	tmp := t.TempDir()
	bad := `name = "bad"
command = "./run.sh"
[permissions]
required = ["bad permission"]
`
	if err := os.WriteFile(filepath.Join(tmp, "skill.toml"), []byte(bad), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	_, err := LoadSpec(filepath.Join(tmp, "skill.toml"))
	if err == nil {
		t.Fatalf("LoadSpec() expected error, got nil")
	}
}

func TestInstallReplacesExistingAndKeepsEnabledState(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	t.Setenv("OLLAMA_SKILL_ALLOW", "filesystem.read")

	first := writeSkillSourceWithScript(t, "#!/bin/sh\nprintf 'v1:%s\\n' \"$1\"\n")
	if _, err := Install(first); err != nil {
		t.Fatalf("Install(first) error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	second := writeSkillSourceWithScript(t, "#!/bin/sh\nprintf 'v2:%s\\n' \"$1\"\n")
	installed, err := Install(second)
	if err != nil {
		t.Fatalf("Install(second) error = %v", err)
	}
	if !installed.Enabled {
		t.Fatalf("Install(second) enabled = false, want true")
	}

	var out bytes.Buffer
	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &out, &out); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got, want := out.String(), "v2:--static\n"; got != want {
		t.Fatalf("Run() output = %q, want %q", got, want)
	}
}
