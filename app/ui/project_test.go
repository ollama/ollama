//go:build windows || darwin

package ui

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/ui/responses"
)

func writeFile(t *testing.T, root, rel, content string) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScanProjectFiles(t *testing.T) {
	root := t.TempDir()

	writeFile(t, root, ".gitignore", "*.log\nsecret/\n/anchored.txt\n!keep.log\n")
	writeFile(t, root, "main.go", "package main")
	writeFile(t, root, "src/app.ts", "export {}")
	writeFile(t, root, "src/.gitignore", "generated/\n")
	writeFile(t, root, "src/generated/api.ts", "ignored")
	writeFile(t, root, "debug.log", "ignored")
	writeFile(t, root, "keep.log", "kept by negation")
	writeFile(t, root, "secret/token.txt", "ignored")
	writeFile(t, root, "anchored.txt", "ignored")
	writeFile(t, root, "sub/anchored.txt", "not ignored: pattern is anchored")
	writeFile(t, root, "node_modules/pkg/index.js", "ignored")
	writeFile(t, root, ".git/config", "ignored")

	files, truncated := scanProjectFiles(root)
	if truncated {
		t.Fatal("expected listing not to be truncated")
	}

	var paths []string
	for _, f := range files {
		if !f.IsDir {
			paths = append(paths, f.Path)
		}
	}

	want := []string{"keep.log", "main.go", "src/app.ts", "sub/anchored.txt"}
	// .gitignore files themselves are listed too
	for _, p := range paths {
		if strings.HasSuffix(p, ".gitignore") {
			continue
		}
		if !slices.Contains(want, p) {
			t.Errorf("unexpected file in listing: %s", p)
		}
	}
	for _, w := range want {
		if !slices.Contains(paths, w) {
			t.Errorf("missing file in listing: %s", w)
		}
	}
}

func TestResolveFileRefs(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "src/app.ts", "hello")
	writeFile(t, root, "big.txt", strings.Repeat("x", maxMentionFileBytes+100))

	p := &projectState{Root: root}

	refs := p.resolveFileRefs([]string{
		"src/app.ts",
		"src/app.ts", // duplicate is skipped
		"missing.txt",
		"../outside.txt",
		"/etc/passwd",
		"big.txt",
	}, map[string]bool{})

	if len(refs) != 2 {
		t.Fatalf("expected 2 resolved refs, got %d", len(refs))
	}
	if refs[0].Filename != "src/app.ts" || string(refs[0].Data) != "hello" {
		t.Errorf("unexpected first ref: %+v", refs[0])
	}
	if refs[1].Filename != "big.txt" {
		t.Fatalf("unexpected second ref: %s", refs[1].Filename)
	}
	if !strings.HasSuffix(string(refs[1].Data), "[truncated]") {
		t.Error("expected oversized file to be truncated")
	}

	// already-attached filenames are skipped
	refs = p.resolveFileRefs([]string{"src/app.ts"}, map[string]bool{"src/app.ts": true})
	if len(refs) != 0 {
		t.Errorf("expected already-attached ref to be skipped, got %d", len(refs))
	}
}

func TestParseFrontmatter(t *testing.T) {
	name, desc, ok := parseFrontmatter("---\nname: my-skill\ndescription: \"Does things\"\n---\nbody")
	if !ok || name != "my-skill" || desc != "Does things" {
		t.Errorf("got name=%q desc=%q ok=%v", name, desc, ok)
	}

	if _, _, ok := parseFrontmatter("no frontmatter here"); ok {
		t.Error("expected no frontmatter")
	}
}

func TestProjectEndpoints(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "main.go", "package main")
	writeFile(t, root, "AGENTS.md", "Be nice.")

	testStore := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	defer testStore.Close()

	server := &Server{Store: testStore, Restart: func() {}}

	// no project open initially
	rr := httptest.NewRecorder()
	if err := server.getProject(rr, httptest.NewRequest("GET", "/api/v1/project", nil)); err != nil {
		t.Fatal(err)
	}
	var project responses.ProjectResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &project); err != nil {
		t.Fatal(err)
	}
	if project.Root != "" {
		t.Fatalf("expected no project, got %q", project.Root)
	}

	// open the project
	body, _ := json.Marshal(map[string]string{"path": root})
	rr = httptest.NewRecorder()
	if err := server.openProject(rr, httptest.NewRequest("POST", "/api/v1/project/open", bytes.NewReader(body))); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &project); err != nil {
		t.Fatal(err)
	}
	if project.Root != root || !project.HasAgentsMd {
		t.Fatalf("unexpected project response: %+v", project)
	}
	if !slices.Contains(project.Recent, root) {
		t.Errorf("expected %q in recents, got %v", root, project.Recent)
	}

	// project dir persisted for reopening on restart
	if dir, err := testStore.ProjectDir(); err != nil || dir != root {
		t.Errorf("persisted project dir = %q, %v; want %q", dir, err, root)
	}

	// file listing includes the scanned file
	rr = httptest.NewRecorder()
	if err := server.getProjectFiles(rr, httptest.NewRequest("GET", "/api/v1/project/files", nil)); err != nil {
		t.Fatal(err)
	}
	var files responses.ProjectFilesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &files); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, f := range files.Files {
		if f.Path == "main.go" && !f.IsDir {
			found = true
		}
	}
	if !found {
		t.Errorf("main.go missing from listing: %+v", files.Files)
	}

	// refresh picks up new files
	writeFile(t, root, "new.txt", "hi")
	rr = httptest.NewRecorder()
	if err := server.getProjectFiles(rr, httptest.NewRequest("GET", "/api/v1/project/files?refresh=1", nil)); err != nil {
		t.Fatal(err)
	}
	files = responses.ProjectFilesResponse{}
	if err := json.Unmarshal(rr.Body.Bytes(), &files); err != nil {
		t.Fatal(err)
	}
	found = false
	for _, f := range files.Files {
		if f.Path == "new.txt" {
			found = true
		}
	}
	if !found {
		t.Error("new.txt missing after refresh")
	}

	// close the project
	rr = httptest.NewRecorder()
	if err := server.closeProject(rr, httptest.NewRequest("POST", "/api/v1/project/close", nil)); err != nil {
		t.Fatal(err)
	}
	if p := server.activeProject(); p != nil {
		t.Errorf("expected no active project after close, got %q", p.Root)
	}
	if dir, _ := testStore.ProjectDir(); dir != "" {
		t.Errorf("expected cleared project dir, got %q", dir)
	}
	// recents survive closing
	if recents, _ := testStore.RecentProjects(); !slices.Contains(recents, root) {
		t.Errorf("expected %q to remain in recents, got %v", root, recents)
	}
}

func TestProjectSystemPrompt(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "AGENTS.md", "Always use tabs.")
	writeFile(t, root, ".agents/skills/deploy/SKILL.md", "---\nname: deploy\ndescription: Deploys the app\n---\n")

	p, err := loadProject(root)
	if err != nil {
		t.Fatal(err)
	}

	prompt := p.systemPrompt()
	for _, want := range []string{"Always use tabs.", "deploy", "Deploys the app"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("system prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestGetProjectFile(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "main.go", "package main\n")
	writeFile(t, root, "big.txt", strings.Repeat("a", maxViewFileBytes+100))
	writeFile(t, root, "bin.dat", "head\x00\x01binary")
	writeFile(t, root, "img.png", "\x89PNG\r\n\x1a\n fake")

	server := &Server{Restart: func() {}}
	server.project = &projectState{Root: root}
	server.projectLoaded = true

	get := func(t *testing.T, path string) responses.ProjectFileResponse {
		t.Helper()
		rr := httptest.NewRecorder()
		req := httptest.NewRequest("GET", "/api/v1/project/file?path="+url.QueryEscape(path), nil)
		if err := server.getProjectFile(rr, req); err != nil {
			t.Fatalf("getProjectFile(%q): %v", path, err)
		}
		var resp responses.ProjectFileResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		return resp
	}

	if resp := get(t, "main.go"); resp.Content != "package main\n" || resp.Binary || resp.Truncated {
		t.Errorf("unexpected response for main.go: %+v", resp)
	}

	resp := get(t, "big.txt")
	if !resp.Truncated || len(resp.Content) != maxViewFileBytes {
		t.Errorf("big.txt: truncated=%v len=%d, want truncated with %d bytes", resp.Truncated, len(resp.Content), maxViewFileBytes)
	}

	if resp := get(t, "bin.dat"); !resp.Binary || resp.Content != "" {
		t.Errorf("unexpected response for bin.dat: %+v", resp)
	}

	resp = get(t, "img.png")
	if resp.MimeType != "image/png" || !resp.Binary {
		t.Errorf("unexpected response for img.png: %+v", resp)
	}
	if data, err := base64.StdEncoding.DecodeString(resp.Content); err != nil || !strings.HasPrefix(string(data), "\x89PNG") {
		t.Errorf("img.png content not base64 png: %v", err)
	}

	// paths outside the project and directories are rejected
	for _, bad := range []string{"../escape.txt", "/etc/passwd", "", "."} {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest("GET", "/api/v1/project/file?path="+url.QueryEscape(bad), nil)
		if err := server.getProjectFile(rr, req); err == nil {
			t.Errorf("expected error for path %q", bad)
		}
	}
}
