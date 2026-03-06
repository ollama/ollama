package skills

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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

func TestSearchAndInfoMetadata(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	src := writeSkillSource(t)
	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}

	matches, err := Search("echoes text")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("Search() len = %d, want 1", len(matches))
	}
	if !strings.Contains(matches[0].Metadata.Source, src) {
		t.Fatalf("metadata source = %q, want local source path", matches[0].Metadata.Source)
	}
}

func TestAllowRevokeAndDryRun(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillSource(t)

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}
	if _, err := Allow("echoer", []string{"filesystem.read"}); err != nil {
		t.Fatalf("Allow() error = %v", err)
	}

	var out bytes.Buffer
	if err := RunWithOptions(context.Background(), "echoer", []string{"hello"}, nil, &out, &out, RunOptions{
		DryRun: true,
	}); err != nil {
		t.Fatalf("RunWithOptions(dry-run) error = %v", err)
	}
	if !strings.Contains(out.String(), "Skill: echoer") {
		t.Fatalf("dry-run output = %q", out.String())
	}

	remaining, err := Revoke("echoer", []string{"filesystem.read"})
	if err != nil {
		t.Fatalf("Revoke() error = %v", err)
	}
	if len(remaining) != 0 {
		t.Fatalf("Revoke() remaining = %v, want empty", remaining)
	}
}

func TestUpdateRollbackUninstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	t.Setenv("OLLAMA_SKILL_ALLOW", "filesystem.read")

	first := writeSkillSourceWithScript(t, "#!/bin/sh\nprintf 'one:%s\\n' \"$1\"\n")
	installed, err := Install(first)
	if err != nil {
		t.Fatalf("Install(first) error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	second := writeSkillSourceWithScript(t, "#!/bin/sh\nprintf 'two:%s\\n' \"$1\"\n")
	updated, err := Update("echoer", second, "")
	if err != nil {
		t.Fatalf("Update() error = %v", err)
	}
	if updated.Metadata.Source == installed.Metadata.Source {
		t.Fatalf("Update() metadata source did not change")
	}

	rolledBack, err := Rollback("echoer")
	if err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	var out bytes.Buffer
	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &out, &out); err != nil {
		t.Fatalf("Run() after rollback error = %v", err)
	}
	if !strings.Contains(out.String(), "one:") {
		t.Fatalf("Run() output after rollback = %q", out.String())
	}
	if rolledBack.Spec.Name != "echoer" {
		t.Fatalf("Rollback() name = %q, want echoer", rolledBack.Spec.Name)
	}

	if err := Uninstall("echoer"); err != nil {
		t.Fatalf("Uninstall() error = %v", err)
	}
	if _, err := Get("echoer"); !errors.Is(err, ErrSkillNotFound) {
		t.Fatalf("Get() after uninstall error = %v, want ErrSkillNotFound", err)
	}
}

func TestInstallGitSourceRequiresPinnedRef(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	if _, err := Install("owner/repo"); err == nil {
		t.Fatalf("Install() expected pinned-ref error, got nil")
	}
}

func TestInstallFromGitPinnedRef(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is required for this test")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	repo := writeSkillSourceWithScript(t, "#!/bin/sh\nprintf 'git:%s\\n' \"$1\"\n")

	if _, err := exec.Command("git", "-C", repo, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v", err)
	}
	if _, err := exec.Command("git", "-C", repo, "config", "user.email", "skill@test").CombinedOutput(); err != nil {
		t.Fatalf("git config user.email failed: %v", err)
	}
	if _, err := exec.Command("git", "-C", repo, "config", "user.name", "Skill Test").CombinedOutput(); err != nil {
		t.Fatalf("git config user.name failed: %v", err)
	}
	if _, err := exec.Command("git", "-C", repo, "add", ".").CombinedOutput(); err != nil {
		t.Fatalf("git add failed: %v", err)
	}
	if _, err := exec.Command("git", "-C", repo, "commit", "-m", "initial").CombinedOutput(); err != nil {
		t.Fatalf("git commit failed: %v", err)
	}
	revOut, err := exec.Command("git", "-C", repo, "rev-parse", "HEAD").Output()
	if err != nil {
		t.Fatalf("git rev-parse failed: %v", err)
	}
	rev := strings.TrimSpace(string(revOut))

	installed, err := Install(repo + "@" + rev)
	if err != nil {
		t.Fatalf("Install(git) error = %v", err)
	}
	if installed.Metadata.Commit == "" {
		t.Fatalf("Install(git) commit metadata is empty")
	}
	if installed.Metadata.Ref != rev {
		t.Fatalf("Install(git) ref = %q, want %q", installed.Metadata.Ref, rev)
	}
}

func TestRunWritesLogs(t *testing.T) {
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

	lines, err := ReadLogs("echoer", 0)
	if err != nil {
		t.Fatalf("ReadLogs() error = %v", err)
	}
	if len(lines) == 0 {
		t.Fatalf("ReadLogs() returned no lines")
	}
}

func TestSearchCatalogFilters(t *testing.T) {
	results, err := SearchCatalog(CatalogFilter{
		Query:        "web",
		Tags:         []string{"research"},
		Permissions:  []string{"network.fetch"},
		VerifiedOnly: true,
	})
	if err != nil {
		t.Fatalf("SearchCatalog() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("SearchCatalog() returned no results")
	}
	if results[0].Name != "web-research" {
		t.Fatalf("SearchCatalog() first result = %q, want web-research", results[0].Name)
	}
}

func TestAllowSessionGrant(t *testing.T) {
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
	if _, err := AllowSession("echoer", []string{"filesystem.read"}); err != nil {
		t.Fatalf("AllowSession() error = %v", err)
	}

	var out bytes.Buffer
	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &out, &out); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
}

func TestPermissionErrorHasMissingPermissions(t *testing.T) {
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
	missing := MissingPermissions(err)
	if len(missing) != 1 || missing[0] != "filesystem.read" {
		t.Fatalf("MissingPermissions() = %v, want [filesystem.read]", missing)
	}
}

func TestSandboxPolicyBlocksPermissionMismatch(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := t.TempDir()

	skillToml := `name = "net-skill"
command = "./run.sh"

[permissions]
required = ["network.fetch"]

[sandbox]
allow_network = false
`
	if err := os.WriteFile(filepath.Join(src, "skill.toml"), []byte(skillToml), 0o644); err != nil {
		t.Fatalf("write skill.toml: %v", err)
	}
	if err := os.WriteFile(filepath.Join(src, "run.sh"), []byte("#!/bin/sh\necho ok\n"), 0o755); err != nil {
		t.Fatalf("write run.sh: %v", err)
	}

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("net-skill"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}
	if _, err := Allow("net-skill", []string{"network.fetch"}); err != nil {
		t.Fatalf("Allow() error = %v", err)
	}

	err := Run(context.Background(), "net-skill", nil, nil, &bytes.Buffer{}, &bytes.Buffer{})
	if !errors.Is(err, ErrSandboxDenied) {
		t.Fatalf("Run() error = %v, want ErrSandboxDenied", err)
	}
}

func TestSandboxTimeoutAndOutputLimit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	src := t.TempDir()
	skillToml := `name = "slow-skill"
command = "./run.sh"

[permissions]
required = []

[sandbox]
timeout_seconds = 1
max_output_bytes = 16
`
	if err := os.WriteFile(filepath.Join(src, "skill.toml"), []byte(skillToml), 0o644); err != nil {
		t.Fatalf("write skill.toml: %v", err)
	}
	script := "#!/bin/sh\nprintf 'abcdefghijklmnopqrstuvwxyz\\n'; sleep 2\n"
	if err := os.WriteFile(filepath.Join(src, "run.sh"), []byte(script), 0o755); err != nil {
		t.Fatalf("write run.sh: %v", err)
	}

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("slow-skill"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}

	err := Run(context.Background(), "slow-skill", nil, nil, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil {
		t.Fatalf("Run() expected sandbox failure, got nil")
	}
	if !errors.Is(err, ErrSandboxDenied) &&
		!errors.Is(err, context.DeadlineExceeded) &&
		!strings.Contains(strings.ToLower(err.Error()), "killed") {
		t.Fatalf("Run() error = %v, want sandbox-denied or timeout", err)
	}
}

func appendProvenance(t *testing.T, dir string, sha, signature, publicKey string) {
	t.Helper()

	path := filepath.Join(dir, "skill.toml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read skill.toml: %v", err)
	}

	block := "\n[provenance]\n"
	if sha != "" {
		block += fmt.Sprintf("sha256 = %q\n", sha)
	}
	if signature != "" {
		block += fmt.Sprintf("signature = %q\n", signature)
	}
	if publicKey != "" {
		block += fmt.Sprintf("public_key = %q\n", publicKey)
	}

	if err := os.WriteFile(path, append(data, []byte(block)...), 0o644); err != nil {
		t.Fatalf("write skill.toml with provenance: %v", err)
	}
}

func TestInstallAndVerifySignedProvenance(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	t.Setenv("OLLAMA_SKILLS", t.TempDir())
	src := writeSkillSource(t)

	digest, err := computeSkillDigest(src)
	if err != nil {
		t.Fatalf("computeSkillDigest() error = %v", err)
	}
	digestBytes, err := hex.DecodeString(digest)
	if err != nil {
		t.Fatalf("hex decode digest: %v", err)
	}

	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	signature := ed25519.Sign(privateKey, digestBytes)
	appendProvenance(t, src, digest, base64.StdEncoding.EncodeToString(signature), base64.StdEncoding.EncodeToString(publicKey))

	installed, err := Install(src)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if !installed.Metadata.Verified {
		t.Fatalf("Install() metadata verified = false, want true")
	}
	if !installed.Metadata.Signed {
		t.Fatalf("Install() metadata signed = false, want true")
	}
	if installed.Metadata.Digest == "" {
		t.Fatalf("Install() metadata digest is empty")
	}

	verified, err := Verify("echoer")
	if err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
	if !verified.Verified || !verified.Signed {
		t.Fatalf("Verify() = %+v, want verified+signed", verified)
	}
}

func TestInstallRejectsProvenanceMismatch(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	src := writeSkillSource(t)
	appendProvenance(t, src, strings.Repeat("0", 64), "", "")

	_, err := Install(src)
	if !errors.Is(err, ErrProvenanceVerification) {
		t.Fatalf("Install() error = %v, want ErrProvenanceVerification", err)
	}
}

func TestPolicyRequireSignature(t *testing.T) {
	root := t.TempDir()
	t.Setenv("OLLAMA_SKILLS", root)

	if err := os.WriteFile(filepath.Join(root, policyFile), []byte(`{"require_signature":true}`), 0o644); err != nil {
		t.Fatalf("write policy: %v", err)
	}

	src := writeSkillSource(t)
	_, err := Install(src)
	if !errors.Is(err, ErrPolicyDenied) {
		t.Fatalf("Install() error = %v, want ErrPolicyDenied", err)
	}
}

func TestPolicyChangeBlocksPreviouslyInstalledUnverifiedSkill(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script execution not supported in this test on windows")
	}

	root := t.TempDir()
	t.Setenv("OLLAMA_SKILLS", root)
	src := writeSkillSource(t)

	if _, err := Install(src); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if err := Enable("echoer"); err != nil {
		t.Fatalf("Enable() error = %v", err)
	}
	if _, err := Allow("echoer", []string{"filesystem.read"}); err != nil {
		t.Fatalf("Allow() error = %v", err)
	}

	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("Run() before policy change error = %v", err)
	}

	if err := os.WriteFile(filepath.Join(root, policyFile), []byte(`{"require_sha256":true}`), 0o644); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	err := Run(context.Background(), "echoer", []string{"hello"}, nil, &bytes.Buffer{}, &bytes.Buffer{})
	if !errors.Is(err, ErrPolicyDenied) {
		t.Fatalf("Run() after policy change error = %v, want ErrPolicyDenied", err)
	}
}

func TestAuditLogsIncludeSkillEvents(t *testing.T) {
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
	if _, err := Allow("echoer", []string{"filesystem.read"}); err != nil {
		t.Fatalf("Allow() error = %v", err)
	}

	if err := Run(context.Background(), "echoer", []string{"hello"}, nil, &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if _, err := Revoke("echoer", []string{"filesystem.read"}); err != nil {
		t.Fatalf("Revoke() error = %v", err)
	}

	lines, err := ReadAuditLogs(0)
	if err != nil {
		t.Fatalf("ReadAuditLogs() error = %v", err)
	}
	joined := strings.Join(lines, "\n")
	for _, expected := range []string{`"action":"install"`, `"action":"allow"`, `"action":"run"`, `"action":"revoke"`} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("audit logs missing %s in %q", expected, joined)
		}
	}
}

func TestAuditLogRotation(t *testing.T) {
	t.Setenv("OLLAMA_SKILLS", t.TempDir())

	oldMaxBytes := auditMaxLogBytes
	oldMaxBackups := auditMaxLogBackups
	auditMaxLogBytes = 64
	auditMaxLogBackups = 1
	defer func() {
		auditMaxLogBytes = oldMaxBytes
		auditMaxLogBackups = oldMaxBackups
	}()

	path, err := auditPath()
	if err != nil {
		t.Fatalf("auditPath() error = %v", err)
	}

	if err := os.WriteFile(path, bytes.Repeat([]byte("x"), 128), 0o644); err != nil {
		t.Fatalf("write oversized audit.log: %v", err)
	}
	if err := appendAuditEvent(auditEvent{Action: "run", Success: true}); err != nil {
		t.Fatalf("appendAuditEvent() error = %v", err)
	}

	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("ReadDir() error = %v", err)
	}
	backupCount := 0
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), auditFile+".") {
			backupCount++
		}
	}
	if backupCount != 1 {
		t.Fatalf("backupCount = %d, want 1", backupCount)
	}

	if err := os.WriteFile(path, bytes.Repeat([]byte("y"), 128), 0o644); err != nil {
		t.Fatalf("write oversized audit.log second time: %v", err)
	}
	if err := appendAuditEvent(auditEvent{Action: "verify", Success: true}); err != nil {
		t.Fatalf("appendAuditEvent() second error = %v", err)
	}

	entries, err = os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("ReadDir() second error = %v", err)
	}
	backupCount = 0
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), auditFile+".") {
			backupCount++
		}
	}
	if backupCount != 1 {
		t.Fatalf("backupCount after prune = %d, want 1", backupCount)
	}
}
