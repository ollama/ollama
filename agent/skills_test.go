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

func writeImportFixtureSkill(t *testing.T, dir string) {
	t.Helper()
	contents, err := os.ReadFile(filepath.Join("testdata", "import", "release-notes", skillFilename))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "release-notes", skillFilename)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, contents, 0o644); err != nil {
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

func TestSkillCatalogExcludeNames(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"release-notes", "system", "exit"} {
		writeCatalogSkill(t, dir, name, "instructions")
	}
	catalog, err := DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := strings.Join(catalog.ExcludeNames([]string{"/system", "EXIT"}), ","), "exit,system"; got != want {
		t.Fatalf("excluded skills = %q, want %q", got, want)
	}
	if _, err := catalog.Load("system"); err == nil {
		t.Fatal("excluded system skill should not load")
	}
	if _, err := catalog.Load("exit"); err == nil {
		t.Fatal("excluded exit skill should not load")
	}
	if _, err := catalog.Load("release-notes"); err != nil {
		t.Fatalf("non-conflicting skill should remain available: %v", err)
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

func TestImportSkillsCopiesFixtureAndIsIdempotent(t *testing.T) {
	source := t.TempDir()
	destination := t.TempDir()
	writeImportFixtureSkill(t, source)
	writeCatalogSkill(t, source, "broken", "---\nname: another-skill\ndescription: Deliberately invalid.\n---\nIgnore this.")
	if err := os.MkdirAll(filepath.Join(source, "release-notes", "references"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "release-notes", "references", "style.txt"), []byte("Keep it short.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(source, "release-notes", "scripts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "release-notes", "scripts", "prepare.sh"), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "ignored.md"), []byte("Ignored root file.\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := importSkillsFromDir("codex", source, destination)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(result.Imported, ","), "release-notes"; got != want {
		t.Fatalf("imported = %q, want %q", got, want)
	}
	catalog, err := DiscoverSkills(destination)
	if err != nil {
		t.Fatal(err)
	}
	skill, err := catalog.Load("release-notes")
	if err != nil || skill.Description != "Draft concise release notes." {
		t.Fatalf("imported skill = %#v, %v", skill, err)
	}
	if got := len(result.Failures); got != 1 || result.Failures[0].Name != "broken" {
		t.Fatalf("failures = %#v, want broken fixture failure", result.Failures)
	}
	for _, file := range []string{skillFilename, filepath.Join("references", "style.txt"), filepath.Join("scripts", "prepare.sh")} {
		if _, err := os.Stat(filepath.Join(destination, "release-notes", file)); err != nil {
			t.Fatalf("imported fixture file %q: %v", file, err)
		}
	}

	result, err = importSkillsFromDir("codex", source, destination)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(result.Existing, ","), "release-notes"; got != want {
		t.Fatalf("existing = %q, want %q", got, want)
	}
	if len(result.Imported) != 0 {
		t.Fatalf("repeated import copied skills: %#v", result.Imported)
	}
}

func TestImportSkillsLeavesConflictsAndUnsafeSourcesUntouched(t *testing.T) {
	source := t.TempDir()
	destination := t.TempDir()
	writeCatalogSkill(t, source, "release-notes", "source instructions")
	writeCatalogSkill(t, destination, "release-notes", "existing instructions")
	writeCatalogSkill(t, source, "nested-link", "safe manifest")
	if err := os.Symlink(filepath.Join(source, "release-notes", skillFilename), filepath.Join(source, "nested-link", "reference")); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	if err := os.Symlink(filepath.Join(source, "release-notes"), filepath.Join(source, "linked-skill")); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	result, err := importSkillsFromDir("codex", source, destination)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Imported) != 0 || len(result.Existing) != 0 {
		t.Fatalf("unexpected successful import: %#v", result)
	}
	if got, err := os.ReadFile(filepath.Join(destination, "release-notes", skillFilename)); err != nil || !strings.Contains(string(got), "existing instructions") {
		t.Fatalf("conflicting destination changed: %q, %v", got, err)
	}
	failed := make(map[string]bool)
	for _, failure := range result.Failures {
		failed[failure.Name] = true
	}
	for _, name := range []string{"release-notes", "nested-link", "linked-skill"} {
		if !failed[name] {
			t.Fatalf("missing failure for %q: %#v", name, result.Failures)
		}
	}
}

func TestImportSkillsRejectsSymlinkedRoot(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(t.TempDir(), "codex-skills")
	if err := os.Symlink(root, source); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	result, err := importSkillsFromDir("codex", source, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "symlinks are not supported") {
		t.Fatalf("symlinked root error = %v", err)
	}
	if len(result.Imported) != 0 || len(result.Existing) != 0 || len(result.Failures) != 0 {
		t.Fatalf("symlinked root result = %#v", result)
	}
}

func TestImportSkillsMissingRootAndConfiguredRoots(t *testing.T) {
	result, err := importSkillsFromDir("codex", filepath.Join(t.TempDir(), "missing"), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Imported) != 0 || len(result.Existing) != 0 || len(result.Failures) != 0 {
		t.Fatalf("missing root result = %#v", result)
	}

	destination := t.TempDir()
	rootBase := t.TempDir()
	roots := map[string]string{
		"codex":  filepath.Join(rootBase, "codex"),
		"claude": filepath.Join(rootBase, "claude"),
		"pi":     filepath.Join(rootBase, "pi"),
	}
	for _, test := range []struct {
		source string
		root   string
		name   string
	}{
		{source: "codex", root: roots["codex"], name: "from-codex"},
		{source: "claude", root: roots["claude"], name: "from-claude"},
		{source: "pi", root: roots["pi"], name: "from-pi"},
	} {
		t.Run(test.source, func(t *testing.T) {
			writeCatalogSkill(t, test.root, test.name, "from "+test.source)
			result, err = importSkillsFromRoots(test.source, roots, destination)
			if err != nil {
				t.Fatal(err)
			}
			if result.SourceDir != test.root {
				t.Fatalf("source dir = %q, want %q", result.SourceDir, test.root)
			}
			if _, err := os.Stat(filepath.Join(destination, test.name, skillFilename)); err != nil {
				t.Fatalf("conventional source was not imported: %v", err)
			}
		})
	}
	if _, err := importSkillsFromRoots("unknown", roots, destination); err == nil || !strings.Contains(err.Error(), "unknown skill source") {
		t.Fatalf("unknown source error = %v", err)
	}
}

func TestConventionalSkillImportRoots(t *testing.T) {
	home := t.TempDir()
	roots := conventionalSkillImportRoots(home)
	for source, want := range map[string]string{
		"codex":  filepath.Join(home, ".codex", "skills"),
		"claude": filepath.Join(home, ".claude", "skills"),
		"pi":     filepath.Join(home, ".pi", "agent", "skills"),
	} {
		if got := roots[source]; got != want {
			t.Fatalf("%s root = %q, want %q", source, got, want)
		}
	}
}

func TestImportSkillsRejectsUnreadableManifest(t *testing.T) {
	source := t.TempDir()
	writeCatalogSkill(t, source, "private", "do not read")
	manifest := filepath.Join(source, "private", skillFilename)
	if err := os.Chmod(manifest, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(manifest, 0o644) })
	if _, err := os.ReadFile(manifest); err == nil {
		t.Skip("test user can read a mode-000 file")
	}
	result, err := importSkillsFromDir("codex", source, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Failures) != 1 || result.Failures[0].Name != "private" {
		t.Fatalf("failures = %#v", result.Failures)
	}
}
