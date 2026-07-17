package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeCatalogSkill(t *testing.T, dir, name, content string) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(content, "---") {
		content = "---\nname: " + name + "\ndescription: Test skill.\n---\n" + content
	}
	if err := os.WriteFile(filepath.Join(path, skillFilename), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDiscoverAndLoadSkills(t *testing.T) {
	dir := t.TempDir()
	writeCatalogSkill(t, dir, "release-notes", "---\nname: release-notes\ndescription: Draft concise release notes.\nmetadata:\n  author: Ollama\n  labels:\n    - release\n    - docs\n---\n# Release notes\n\nUse short bullets.")
	catalog, err := DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	list := catalog.List()
	if len(list) != 1 || list[0].Name != "release-notes" || list[0].Description != "Draft concise release notes." {
		t.Fatalf("skills = %#v", list)
	}
	skill, err := catalog.Load("release-notes")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(skill.Content(), `<skill name="release-notes">`) || !strings.Contains(skill.Content(), "Use short bullets.") {
		t.Fatalf("skill content = %q", skill.Content())
	}
	if context := catalog.SystemContext(); !strings.Contains(context, "release-notes: Draft concise release notes.") || !strings.Contains(context, "normal approval rules") {
		t.Fatalf("system context = %q", context)
	}
}

func TestDiscoverSkillsSkipsMalformedEntries(t *testing.T) {
	dir := t.TempDir()
	writeCatalogSkill(t, dir, "valid", "do the useful thing")
	writeCatalogSkill(t, dir, "mismatched", "---\nname: whatever\ndescription: wrong name\n---\nbody")
	// Genuinely malformed front matter (a line without a key:value pair) is still rejected.
	writeCatalogSkill(t, dir, "broken", "---\nname: broken\ndescription\n---\nnope")
	writeCatalogSkill(t, dir, "missing-name", "---\ndescription: missing name\n---\nbody")
	writeCatalogSkill(t, dir, "missing-description", "---\nname: missing-description\n---\nbody")
	writeCatalogSkill(t, dir, "bad-name", "---\nname: bad_name\ndescription: invalid name\n---\nbody")
	writeCatalogSkill(t, dir, "under_score", "---\nname: under_score\ndescription: invalid directory\n---\nbody")
	if err := os.MkdirAll(filepath.Join(dir, "no-front-matter"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "no-front-matter", skillFilename), []byte("body"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(catalog.List()), 1; got != want {
		t.Fatalf("valid skills = %d, want %d", got, want)
	}
	if got, want := len(catalog.Diagnostics()), 7; got != want {
		t.Fatalf("diagnostics = %d, want %d: %#v", got, want, catalog.Diagnostics())
	}
	if _, err := catalog.Load("broken"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("load broken error = %v", err)
	}
	if _, err := catalog.Load("../valid"); err == nil || !strings.Contains(err.Error(), "invalid skill name") {
		t.Fatalf("unsafe name error = %v", err)
	}
}

func TestDiscoverSkillsFollowsSymlinks(t *testing.T) {
	dir := t.TempDir()
	target := t.TempDir()
	writeCatalogSkill(t, target, "shared", "---\nname: shared\ndescription: From a linked repo.\n---\nshared instructions")
	if err := os.Symlink(filepath.Join(target, "shared"), filepath.Join(dir, "shared")); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	catalog, err := DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	list := catalog.List()
	if len(list) != 1 || list[0].Name != "shared" || list[0].Description != "From a linked repo." {
		t.Fatalf("symlinked skills = %#v", list)
	}
	if !strings.Contains(list[0].Content(), "shared instructions") {
		t.Fatalf("symlinked skill content = %q", list[0].Content())
	}
}

func TestLoadDefaultSkillsContinuesAfterBadRoot(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	project := t.TempDir()
	writeCatalogSkill(t, filepath.Join(project, ".ollama", "skills"), "release-notes", "project instructions")

	badRoot := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(badRoot, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(SkillsDirEnv, badRoot)

	catalog, err := LoadDefaultSkills(project)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := catalog.Load("release-notes"); err != nil {
		t.Fatalf("valid skill was hidden by bad root: %v", err)
	}
	if _, err := catalog.Load(bundledSkillCreatorName); err != nil {
		t.Fatalf("bundled skill was hidden by bad root: %v", err)
	}
	var foundDiagnostic bool
	for _, diagnostic := range catalog.Diagnostics() {
		if strings.Contains(diagnostic.Error(), badRoot) {
			foundDiagnostic = true
			break
		}
	}
	if !foundDiagnostic {
		t.Fatalf("diagnostics = %#v, want bad root %q", catalog.Diagnostics(), badRoot)
	}
}

func TestLoadDefaultSkillsInstallsBundledSkillCreator(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(SkillsDirEnv, dir)

	catalog, err := LoadDefaultSkills("")
	if err != nil {
		t.Fatal(err)
	}
	skill, err := catalog.Load(bundledSkillCreatorName)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, bundledSkillCreatorName, skillFilename)
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != bundledSkillCreatorContent {
		t.Fatalf("installed skill = %q, want bundled contents", contents)
	}
	if skill.Path != path {
		t.Fatalf("skill path = %q, want %q", skill.Path, path)
	}
	if !strings.Contains(skill.Content(), "Skill directory: "+filepath.Dir(path)) {
		t.Fatalf("skill content does not identify its directory: %q", skill.Content())
	}
}

func TestLoadDefaultSkillsUpdatesExistingSkillCreator(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(SkillsDirEnv, dir)
	writeCatalogSkill(t, dir, bundledSkillCreatorName, "custom instructions")

	if _, err := LoadDefaultSkills(""); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(filepath.Join(dir, bundledSkillCreatorName, skillFilename))
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != bundledSkillCreatorContent {
		t.Fatalf("installed skill = %q, want bundled contents", contents)
	}
}

func TestSkillsDirUsesOverrideAndXDG(t *testing.T) {
	base := t.TempDir()

	override := filepath.Join(base, "skills-override")
	t.Setenv(SkillsDirEnv, override)
	got, err := SkillsDir()
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(override)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("SkillsDir override = %q, want %q", got, want)
	}

	t.Setenv(SkillsDirEnv, "")
	xdg := filepath.Join(base, "xdg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	if got, err := SkillsDir(); err != nil || got != filepath.Join(xdg, "ollama", "skills") {
		t.Fatalf("SkillsDir xdg = %q, want %q, %v", got, filepath.Join(xdg, "ollama", "skills"), err)
	}

	t.Setenv("XDG_CONFIG_HOME", "")
	home := filepath.Join(base, "home")
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if got, err := SkillsDir(); err != nil || got != filepath.Join(home, ".ollama", "skills") {
		t.Fatalf("SkillsDir default = %q, want %q, %v", got, filepath.Join(home, ".ollama", "skills"), err)
	}
}

func TestLoadDefaultSkillsPrecedenceAndCollisions(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home) // Windows: os.UserHomeDir uses %USERPROFILE%

	userOllama := t.TempDir()
	t.Setenv(SkillsDirEnv, userOllama)

	userAgents := filepath.Join(home, ".agents", "skills")
	project := t.TempDir()
	projectAgents := filepath.Join(project, ".agents", "skills")
	projectOllama := filepath.Join(project, ".ollama", "skills")

	// release-notes exists in all four roots; project ollama must win.
	writeCatalogSkill(t, userAgents, "release-notes", "from user agents")
	writeCatalogSkill(t, userOllama, "release-notes", "from user ollama")
	writeCatalogSkill(t, projectOllama, "release-notes", "from project ollama")
	// code-review exists in both project roots; project ollama beats project agents.
	writeCatalogSkill(t, projectAgents, "code-review", "from project agents")
	writeCatalogSkill(t, projectOllama, "code-review", "from project ollama")
	// unique appears only in user ollama (via env override).
	writeCatalogSkill(t, userOllama, "unique", "only here")

	catalog, err := LoadDefaultSkills(project)
	if err != nil {
		t.Fatal(err)
	}
	rn, err := catalog.Load("release-notes")
	if err != nil || !strings.Contains(rn.Instructions, "from project ollama") || !strings.Contains(rn.Path, ".ollama") {
		t.Fatalf("release-notes = %#v, want project ollama to win", rn)
	}
	cr, err := catalog.Load("code-review")
	if err != nil || !strings.Contains(cr.Instructions, "from project ollama") {
		t.Fatalf("code-review = %#v, want project ollama to win over project agents", cr)
	}
	if _, err := catalog.Load("unique"); err != nil {
		t.Fatalf("unique should load from user ollama: %v", err)
	}
	// Collisions are resolved silently by precedence — no diagnostics.
	for _, d := range catalog.Diagnostics() {
		if strings.Contains(d.Error(), "shadows") {
			t.Fatalf("unexpected shadow diagnostic: %v", d)
		}
	}
}

func TestSkillContentListsDirectoryAndResources(t *testing.T) {
	root := t.TempDir()
	skillDir := filepath.Join(root, "pdf-processing")
	if err := os.MkdirAll(filepath.Join(skillDir, "scripts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(skillDir, "references"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\nname: pdf-processing\ndescription: Handle PDFs.\n---\nHandle PDFs."), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "scripts", "extract.py"), []byte("#!/usr/bin/env python3"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "references", "ref.md"), []byte("ref"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := DiscoverSkills(root)
	if err != nil {
		t.Fatal(err)
	}
	skill, err := catalog.Load("pdf-processing")
	if err != nil {
		t.Fatal(err)
	}
	content := skill.Content()
	if !strings.Contains(content, "Skill directory:") || !strings.Contains(content, skillDir) {
		t.Fatalf("content missing skill directory: %q", content)
	}
	if !strings.Contains(content, "<file>scripts/extract.py</file>") || !strings.Contains(content, "<file>references/ref.md</file>") {
		t.Fatalf("content missing resource listing: %q", content)
	}
}
