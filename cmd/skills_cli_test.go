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
	if !strings.Contains(out, "run:--static hello\n") || !strings.Contains(out, "Trace: skill=echoer status=ok") {
		t.Fatalf("skills run output = %q", out)
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

func TestSkillsSearchInfoAndGrantCommands(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillFixture(t)

	if _, err := runCLI(t, "skills", "install", src); err != nil {
		t.Fatalf("skills install failed: %v", err)
	}

	out, err := runCLI(t, "skills", "search", "echoes")
	if err != nil {
		t.Fatalf("skills search failed: %v", err)
	}
	if !strings.Contains(out, "echoer") {
		t.Fatalf("skills search output = %q", out)
	}

	out, err = runCLI(t, "skills", "search", "echoer", "--installed=true", "--catalog=false", "--verified")
	if err != nil {
		t.Fatalf("skills search verified local failed: %v", err)
	}
	if !strings.Contains(out, "No skills matched") {
		t.Fatalf("skills search verified local output = %q", out)
	}

	out, err = runCLI(t, "skills", "search", "web", "--installed=false", "--catalog=true", "--verified")
	if err != nil {
		t.Fatalf("skills search catalog failed: %v", err)
	}
	if !strings.Contains(out, "web-research") {
		t.Fatalf("skills search catalog output = %q", out)
	}

	out, err = runCLI(t, "skills", "info", "echoer")
	if err != nil {
		t.Fatalf("skills info failed: %v", err)
	}
	if !strings.Contains(out, "Name: echoer") || !strings.Contains(out, "Required permissions: filesystem.read") {
		t.Fatalf("skills info output = %q", out)
	}

	out, err = runCLI(t, "skills", "allow", "echoer", "filesystem.read")
	if err != nil {
		t.Fatalf("skills allow failed: %v", err)
	}
	if !strings.Contains(out, "Granted permissions for 'echoer': filesystem.read") {
		t.Fatalf("skills allow output = %q", out)
	}

	out, err = runCLI(t, "skills", "revoke", "echoer", "filesystem.read")
	if err != nil {
		t.Fatalf("skills revoke failed: %v", err)
	}
	if !strings.Contains(out, "Remaining permissions for 'echoer': -") {
		t.Fatalf("skills revoke output = %q", out)
	}
}

func TestSkillsDryRunAndLogs(t *testing.T) {
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
	if _, err := runCLI(t, "skills", "allow", "echoer", "filesystem.read"); err != nil {
		t.Fatalf("skills allow failed: %v", err)
	}

	out, err := runCLI(t, "skills", "run", "echoer", "--dry-run", "hello")
	if err != nil {
		t.Fatalf("skills run --dry-run failed: %v", err)
	}
	if !strings.Contains(out, "Skill: echoer") || strings.Contains(out, "run:--static hello") {
		t.Fatalf("skills dry-run output = %q", out)
	}

	out, err = runCLI(t, "skills", "run", "echoer", "hello")
	if err != nil {
		t.Fatalf("skills run failed: %v", err)
	}
	if !strings.Contains(out, "run:--static hello\n") || !strings.Contains(out, "Trace: skill=echoer status=ok") {
		t.Fatalf("skills run output = %q", out)
	}

	out, err = runCLI(t, "skills", "logs", "echoer", "--last", "5")
	if err != nil {
		t.Fatalf("skills logs failed: %v", err)
	}
	if !strings.Contains(out, "status=ok") {
		t.Fatalf("skills logs output = %q", out)
	}
}

func TestSkillsUpdateRollbackAndUninstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	first := writeSkillFixture(t)
	if _, err := runCLI(t, "skills", "install", first); err != nil {
		t.Fatalf("skills install(first) failed: %v", err)
	}
	if _, err := runCLI(t, "skills", "enable", "echoer"); err != nil {
		t.Fatalf("skills enable failed: %v", err)
	}
	if _, err := runCLI(t, "skills", "allow", "echoer", "filesystem.read"); err != nil {
		t.Fatalf("skills allow failed: %v", err)
	}

	second := t.TempDir()
	skillToml := `name = "echoer"
description = "Echoes text input"
version = "0.2.0"
command = "./run.sh"
args = ["--next"]

[io]
inputs = ["text"]
outputs = ["text"]

[permissions]
required = ["filesystem.read"]
`
	if err := os.WriteFile(filepath.Join(second, "skill.toml"), []byte(skillToml), 0o644); err != nil {
		t.Fatalf("write second skill.toml: %v", err)
	}
	if err := os.WriteFile(filepath.Join(second, "run.sh"), []byte("#!/bin/sh\nprintf 'new:%s %s\\n' \"$1\" \"$2\"\n"), 0o755); err != nil {
		t.Fatalf("write second run.sh: %v", err)
	}

	if _, err := runCLI(t, "skills", "update", "echoer", "--source", second); err != nil {
		t.Fatalf("skills update failed: %v", err)
	}

	out, err := runCLI(t, "skills", "run", "echoer", "hello")
	if err != nil {
		t.Fatalf("skills run(updated) failed: %v", err)
	}
	if !strings.Contains(out, "new:--next hello") {
		t.Fatalf("skills run(updated) output = %q", out)
	}

	if _, err := runCLI(t, "skills", "rollback", "echoer"); err != nil {
		t.Fatalf("skills rollback failed: %v", err)
	}
	out, err = runCLI(t, "skills", "run", "echoer", "hello")
	if err != nil {
		t.Fatalf("skills run(rollback) failed: %v", err)
	}
	if !strings.Contains(out, "run:--static hello") {
		t.Fatalf("skills run(rollback) output = %q", out)
	}

	if _, err := runCLI(t, "skills", "uninstall", "echoer"); err != nil {
		t.Fatalf("skills uninstall failed: %v", err)
	}
	out, err = runCLI(t, "skills", "list")
	if err != nil {
		t.Fatalf("skills list after uninstall failed: %v", err)
	}
	if strings.Contains(out, "echoer") {
		t.Fatalf("skills list after uninstall output = %q", out)
	}
}

func TestSkillsVerifyCommand(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillFixture(t)

	if _, err := runCLI(t, "skills", "install", src); err != nil {
		t.Fatalf("skills install failed: %v", err)
	}

	out, err := runCLI(t, "skills", "verify", "echoer")
	if err != nil {
		t.Fatalf("skills verify failed: %v", err)
	}
	if !strings.Contains(out, "Skill: echoer") || !strings.Contains(out, "Verified: no") {
		t.Fatalf("skills verify output = %q", out)
	}
}
