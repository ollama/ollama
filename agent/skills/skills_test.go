package skills

import (
	"os"
	"path/filepath"
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
