package skills

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestLoadCatalogReadsSkillMetadata(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, filepath.Join(dir, "go-code"), "go-code", "Write idiomatic Go code.")

	catalog, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(catalog.Skills) != 1 {
		t.Fatalf("skills = %d, want 1: %#v", len(catalog.Skills), catalog)
	}
	if got := catalog.Skills[0].Name; got != "go-code" {
		t.Fatalf("skill name = %q", got)
	}
	if prompt := catalog.SystemPrompt(true); !strings.Contains(prompt, "go-code: Write idiomatic Go code.") || !strings.Contains(prompt, "call the skill tool") {
		t.Fatalf("system prompt missing skill metadata: %q", prompt)
	}
}

func TestLoadCatalogSkipsInvalidSkills(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, filepath.Join(dir, "bad"), "Bad_Name", "bad")

	catalog, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(catalog.Skills) != 0 {
		t.Fatalf("skills = %#v, want none", catalog.Skills)
	}
	if len(catalog.Warnings) == 0 {
		t.Fatal("expected invalid skill warning")
	}
}

func TestImportToDirCopiesCanonicalSkill(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	src := filepath.Join(home, ".claude", "skills", "go-code")
	writeSkill(t, src, "go-code", "Write idiomatic Go code.")
	if err := os.WriteFile(filepath.Join(src, "notes.md"), []byte("notes"), 0o644); err != nil {
		t.Fatal(err)
	}

	dest := filepath.Join(home, ".ollama", "skills")
	results, err := ImportToDir("claude", dest, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Skipped {
		t.Fatalf("results = %#v", results)
	}
	if _, err := os.Stat(filepath.Join(dest, "go-code", "notes.md")); err != nil {
		t.Fatal(err)
	}
}

func TestReadMetadataRejectsSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions vary on Windows")
	}
	dir := t.TempDir()
	real := filepath.Join(dir, "real.md")
	if err := os.WriteFile(real, []byte("---\nname: go-code\ndescription: Write idiomatic Go code.\n---\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, SkillFile)
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadMetadata(link); err == nil || !strings.Contains(err.Error(), "must not be a symlink") {
		t.Fatalf("ReadMetadata error = %v, want symlink rejection", err)
	}
}

func TestImportToDirReportsSymlinkedSkillDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions vary on Windows")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	srcRoot := filepath.Join(home, ".claude", "skills")
	real := filepath.Join(home, "elsewhere", "go-code")
	writeSkill(t, real, "go-code", "Write idiomatic Go code.")
	if err := os.MkdirAll(srcRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(srcRoot, "go-code")); err != nil {
		t.Fatal(err)
	}

	results, err := ImportToDir("claude", filepath.Join(home, ".ollama", "skills"), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || !results[0].Skipped {
		t.Fatalf("results = %#v, want one skipped symlink directory", results)
	}
	if !strings.Contains(results[0].Error, "symlinked skill directories") {
		t.Fatalf("error = %q, want symlink directory warning", results[0].Error)
	}
}

func TestImportToDirReportsSkippedSymlinkEntries(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions vary on Windows")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	src := filepath.Join(home, ".claude", "skills", "go-code")
	writeSkill(t, src, "go-code", "Write idiomatic Go code.")
	target := filepath.Join(home, "outside.md")
	if err := os.WriteFile(target, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(src, "outside.md")); err != nil {
		t.Fatal(err)
	}

	results, err := ImportToDir("claude", filepath.Join(home, ".ollama", "skills"), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Skipped {
		t.Fatalf("results = %#v, want one imported skill", results)
	}
	if !strings.Contains(results[0].Error, "skipped symlinks: outside.md") {
		t.Fatalf("error = %q, want skipped symlink warning", results[0].Error)
	}
	if _, err := os.Lstat(filepath.Join(home, ".ollama", "skills", "go-code", "outside.md")); !os.IsNotExist(err) {
		t.Fatalf("copied symlink err = %v, want missing symlink", err)
	}
}

func writeSkill(t *testing.T, dir, name, description string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: " + name + "\ndescription: " + description + "\n---\n\n# " + name + "\n\nUse this skill.\n"
	if err := os.WriteFile(filepath.Join(dir, SkillFile), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
