//go:build windows || darwin

package ui

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/ui/responses"
)

const (
	// maxProjectFiles caps how many entries the project scanner returns
	maxProjectFiles = 20000

	// maxAgentsFileBytes caps how much of AGENTS.md is injected as context
	maxAgentsFileBytes = 32 * 1024

	// maxMentionFileBytes caps the content injected for a single @-mentioned file
	maxMentionFileBytes = 64 * 1024

	// maxMentionTotalBytes caps the combined content injected for all
	// @-mentioned files in a single message (~32k tokens at ~4 bytes/token)
	maxMentionTotalBytes = 128 * 1024

	// maxViewFileBytes caps how much of a file the viewer returns; longer
	// files are truncated and flagged as such
	maxViewFileBytes = 512 * 1024

	// maxViewImageBytes caps images returned inline (base64) to the viewer
	maxViewImageBytes = 8 * 1024 * 1024

	// maxRecentProjects caps the persisted recent projects list
	maxRecentProjects = 8
)

// defaultIgnoredDirs are directory names skipped at any depth regardless of
// .gitignore contents, since they are large and rarely useful as chat context
var defaultIgnoredDirs = map[string]bool{
	".git":         true,
	"node_modules": true,
	"build":        true,
	"dist":         true,
	"out":          true,
	"target":       true,
	"vendor":       true,
	".next":        true,
	".nuxt":        true,
	".venv":        true,
	"venv":         true,
	"__pycache__":  true,
	".cache":       true,
	"coverage":     true,
	"DerivedData":  true,
	".gradle":      true,
}

type projectState struct {
	Root      string
	Files     []responses.ProjectFile
	Truncated bool
	AgentsMD  string
	Skills    []responses.ProjectSkill
}

// activeProject returns the currently open project, lazily restoring the
// last opened project from the store on first use. Returns nil when no
// project is open.
func (s *Server) activeProject() *projectState {
	s.projectMu.Lock()
	defer s.projectMu.Unlock()

	if !s.projectLoaded {
		s.projectLoaded = true
		if s.Store != nil {
			if dir, err := s.Store.ProjectDir(); err == nil && dir != "" {
				if p, err := loadProject(dir); err == nil {
					s.project = p
				} else {
					s.log().Warn("failed to restore last project", "dir", dir, "error", err)
				}
			}
		}
	}

	return s.project
}

// loadProject scans root and builds the project state, including AGENTS.md
// and .agents/skills metadata
func loadProject(root string) (*projectState, error) {
	root = filepath.Clean(root)
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("stat project dir: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", root)
	}

	p := &projectState{Root: root}
	p.Files, p.Truncated = scanProjectFiles(root)

	if data, err := os.ReadFile(filepath.Join(root, "AGENTS.md")); err == nil {
		if len(data) > maxAgentsFileBytes {
			data = data[:maxAgentsFileBytes]
		}
		p.AgentsMD = string(data)
	}

	p.Skills = scanProjectSkills(root)

	return p, nil
}

// Name returns the base name of the project folder
func (p *projectState) Name() string {
	return filepath.Base(p.Root)
}

// ignoreRule is a single parsed .gitignore pattern
type ignoreRule struct {
	segs     []string
	negate   bool
	dirOnly  bool
	anchored bool
	base     string // slash-separated dir (relative to root) containing the .gitignore
}

func parseGitignore(base string, data []byte) []ignoreRule {
	var rules []ignoreRule
	for line := range strings.SplitSeq(string(data), "\n") {
		line = strings.TrimRight(line, "\r")
		// unescape trailing spaces is not supported; plain trim is close enough
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		rule := ignoreRule{base: base}
		if strings.HasPrefix(line, "!") {
			rule.negate = true
			line = line[1:]
		}
		line = strings.TrimPrefix(line, "\\")
		if strings.HasSuffix(line, "/") {
			rule.dirOnly = true
			line = strings.TrimSuffix(line, "/")
		}
		if strings.HasPrefix(line, "/") {
			rule.anchored = true
			line = strings.TrimPrefix(line, "/")
		} else if strings.Contains(line, "/") {
			// patterns with a slash anywhere are anchored to the .gitignore dir
			rule.anchored = true
		}
		if line == "" {
			continue
		}
		rule.segs = strings.Split(line, "/")
		rules = append(rules, rule)
	}
	return rules
}

// matches reports whether rel (slash-separated, relative to root) matches
// the rule. rel must not be "."
func (r ignoreRule) matches(rel string, isDir bool) bool {
	if r.dirOnly && !isDir {
		return false
	}

	// scope rel to the directory containing the .gitignore
	if r.base != "" {
		prefix := r.base + "/"
		if !strings.HasPrefix(rel, prefix) {
			return false
		}
		rel = strings.TrimPrefix(rel, prefix)
	}

	pathSegs := strings.Split(rel, "/")
	if !r.anchored {
		// unanchored single-segment patterns match the basename at any depth
		ok, err := path.Match(r.segs[0], pathSegs[len(pathSegs)-1])
		return err == nil && ok
	}

	return matchSegs(r.segs, pathSegs)
}

// matchSegs matches gitignore pattern segments (with ** support) against
// path segments
func matchSegs(patSegs, pathSegs []string) bool {
	if len(patSegs) == 0 {
		return len(pathSegs) == 0
	}
	if patSegs[0] == "**" {
		for i := 0; i <= len(pathSegs); i++ {
			if matchSegs(patSegs[1:], pathSegs[i:]) {
				return true
			}
		}
		return false
	}
	if len(pathSegs) == 0 {
		return false
	}
	if ok, err := path.Match(patSegs[0], pathSegs[0]); err != nil || !ok {
		return false
	}
	return matchSegs(patSegs[1:], pathSegs[1:])
}

func ignored(rules []ignoreRule, rel string, isDir bool) bool {
	result := false
	for _, r := range rules {
		if r.matches(rel, isDir) {
			result = !r.negate
		}
	}
	return result
}

// scanProjectFiles walks root and returns a flat listing of files and
// directories, honoring .gitignore files and skipping well-known heavy
// directories. The returned bool reports whether the listing was truncated.
func scanProjectFiles(root string) ([]responses.ProjectFile, bool) {
	var files []responses.ProjectFile
	var rules []ignoreRule
	truncated := false

	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // skip unreadable entries
		}

		rel, err := filepath.Rel(root, p)
		if err != nil {
			return nil //nolint:nilerr
		}
		rel = filepath.ToSlash(rel)

		if rel == "." {
			// load the root .gitignore before visiting children
			if data, err := os.ReadFile(filepath.Join(p, ".gitignore")); err == nil {
				rules = append(rules, parseGitignore("", data)...)
			}
			return nil
		}

		name := d.Name()

		if d.IsDir() {
			if defaultIgnoredDirs[name] || ignored(rules, rel, true) {
				return fs.SkipDir
			}
			if data, err := os.ReadFile(filepath.Join(p, ".gitignore")); err == nil {
				rules = append(rules, parseGitignore(rel, data)...)
			}
			files = append(files, responses.ProjectFile{Path: rel, IsDir: true})
		} else {
			if name == ".DS_Store" || d.Type()&fs.ModeSymlink != 0 || !d.Type().IsRegular() {
				return nil
			}
			if ignored(rules, rel, false) {
				return nil
			}
			var size int64
			if info, err := d.Info(); err == nil {
				size = info.Size()
			}
			files = append(files, responses.ProjectFile{Path: rel, Size: size})
		}

		if len(files) >= maxProjectFiles {
			truncated = true
			return fs.SkipAll
		}
		return nil
	})

	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, truncated
}

// scanProjectSkills reads skill metadata from .agents/skills in the project
// root. Both <skills>/<name>/SKILL.md and <skills>/<name>.md layouts are
// supported.
func scanProjectSkills(root string) []responses.ProjectSkill {
	skillsDir := filepath.Join(root, ".agents", "skills")
	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		return nil
	}

	var skills []responses.ProjectSkill
	for _, entry := range entries {
		var skillFile, name string
		if entry.IsDir() {
			skillFile = filepath.Join(skillsDir, entry.Name(), "SKILL.md")
			name = entry.Name()
		} else if strings.HasSuffix(entry.Name(), ".md") {
			skillFile = filepath.Join(skillsDir, entry.Name())
			name = strings.TrimSuffix(entry.Name(), ".md")
		} else {
			continue
		}

		data, err := os.ReadFile(skillFile)
		if err != nil {
			continue
		}

		skill := responses.ProjectSkill{Name: name}
		if fmName, fmDesc, ok := parseFrontmatter(string(data)); ok {
			if fmName != "" {
				skill.Name = fmName
			}
			skill.Description = fmDesc
		}
		skills = append(skills, skill)
	}
	return skills
}

// parseFrontmatter extracts name and description from a YAML frontmatter
// block delimited by "---" lines
func parseFrontmatter(content string) (name, description string, ok bool) {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return "", "", false
	}
	for _, line := range lines[1:] {
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			return name, description, true
		}
		if v, found := strings.CutPrefix(trimmed, "name:"); found {
			name = strings.Trim(strings.TrimSpace(v), `"'`)
		} else if v, found := strings.CutPrefix(trimmed, "description:"); found {
			description = strings.Trim(strings.TrimSpace(v), `"'`)
		}
	}
	return "", "", false
}

// systemPrompt builds the system message injected into chats while the
// project is open
func (p *projectState) systemPrompt() string {
	var b strings.Builder
	fmt.Fprintf(&b, "The user has the project folder %q open (full path: %s). File contents shared in messages come from this project.", p.Name(), p.Root)

	if p.AgentsMD != "" {
		b.WriteString("\n\nThe project provides the following instructions (AGENTS.md):\n\n")
		b.WriteString(p.AgentsMD)
	}

	if len(p.Skills) > 0 {
		b.WriteString("\n\nThe project defines the following skills in .agents/skills:\n")
		for _, skill := range p.Skills {
			if skill.Description != "" {
				fmt.Fprintf(&b, "- %s: %s\n", skill.Name, skill.Description)
			} else {
				fmt.Fprintf(&b, "- %s\n", skill.Name)
			}
		}
	}

	return b.String()
}

// resolvePath maps a project-relative reference to its slash-separated
// relative form and absolute path on disk. It reports false for absolute
// paths and for anything that would escape the project root.
func (p *projectState) resolvePath(ref string) (rel, full string, ok bool) {
	cleaned := filepath.Clean(filepath.FromSlash(ref))
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) || filepath.IsAbs(cleaned) {
		return "", "", false
	}
	return filepath.ToSlash(cleaned), filepath.Join(p.Root, cleaned), true
}

// resolveFileRefs reads the contents of @-mentioned project files, skipping
// paths outside the project root and filenames already present in skip.
// Contents are truncated to per-file and total budgets.
func (p *projectState) resolveFileRefs(refs []string, skip map[string]bool) []store.File {
	var files []store.File
	total := 0

	for _, ref := range refs {
		if skip[ref] {
			continue
		}
		skip[ref] = true

		rel, full, ok := p.resolvePath(ref)
		if !ok {
			continue
		}

		info, err := os.Stat(full)
		if err != nil || info.IsDir() {
			continue
		}

		if total >= maxMentionTotalBytes {
			break
		}

		data, err := os.ReadFile(full)
		if err != nil {
			continue
		}

		budget := min(maxMentionFileBytes, maxMentionTotalBytes-total)
		if len(data) > budget {
			data = append(data[:budget], []byte("\n... [truncated]")...)
		}
		total += len(data)

		files = append(files, store.File{
			Filename: rel,
			Data:     data,
		})
	}

	return files
}

func (s *Server) recentProjects() []string {
	if s.Store == nil {
		return nil
	}
	recents, err := s.Store.RecentProjects()
	if err != nil {
		s.log().Warn("failed to load recent projects", "error", err)
		return nil
	}
	return recents
}

func (s *Server) addRecentProject(dir string) {
	if s.Store == nil {
		return
	}
	recents := s.recentProjects()
	recents = slices.DeleteFunc(recents, func(r string) bool { return r == dir })
	recents = append([]string{dir}, recents...)
	if len(recents) > maxRecentProjects {
		recents = recents[:maxRecentProjects]
	}
	if err := s.Store.SetRecentProjects(recents); err != nil {
		s.log().Warn("failed to save recent projects", "error", err)
	}
}

func (s *Server) projectResponse(p *projectState) responses.ProjectResponse {
	resp := responses.ProjectResponse{Recent: s.recentProjects()}
	if resp.Recent == nil {
		resp.Recent = []string{}
	}
	if p != nil {
		resp.Root = p.Root
		resp.Name = p.Name()
		resp.HasAgentsMd = p.AgentsMD != ""
		resp.Skills = p.Skills
	}
	if resp.Skills == nil {
		resp.Skills = []responses.ProjectSkill{}
	}
	return resp
}

func (s *Server) getProject(w http.ResponseWriter, r *http.Request) error {
	p := s.activeProject()
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(s.projectResponse(p))
}

func (s *Server) openProject(w http.ResponseWriter, r *http.Request) error {
	var req struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}
	if req.Path == "" {
		return fmt.Errorf("path is required")
	}

	p, err := loadProject(req.Path)
	if err != nil {
		return err
	}

	s.projectMu.Lock()
	s.project = p
	s.projectLoaded = true
	s.projectMu.Unlock()

	if s.Store != nil {
		if err := s.Store.SetProjectDir(p.Root); err != nil {
			s.log().Warn("failed to persist project dir", "error", err)
		}
		s.addRecentProject(p.Root)
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(s.projectResponse(p))
}

func (s *Server) closeProject(w http.ResponseWriter, r *http.Request) error {
	s.projectMu.Lock()
	s.project = nil
	s.projectLoaded = true
	s.projectMu.Unlock()

	if s.Store != nil {
		if err := s.Store.SetProjectDir(""); err != nil {
			s.log().Warn("failed to clear project dir", "error", err)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(s.projectResponse(nil))
}

func (s *Server) getProjectFiles(w http.ResponseWriter, r *http.Request) error {
	p := s.activeProject()
	if p == nil {
		return fmt.Errorf("no project is open")
	}

	if r.URL.Query().Get("refresh") == "1" {
		refreshed, err := loadProject(p.Root)
		if err != nil {
			return err
		}
		s.projectMu.Lock()
		// only replace if the project didn't change while rescanning
		if s.project != nil && s.project.Root == refreshed.Root {
			s.project = refreshed
			p = refreshed
		}
		s.projectMu.Unlock()
	}

	files := p.Files
	if files == nil {
		files = []responses.ProjectFile{}
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(responses.ProjectFilesResponse{
		Files:     files,
		Truncated: p.Truncated,
	})
}

// viewImageTypes maps image extensions the viewer renders inline to their
// MIME type
var viewImageTypes = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
	".svg":  "image/svg+xml",
	".ico":  "image/x-icon",
}

// trimPartialRune drops the incomplete UTF-8 sequence a byte-size cut may
// have left at the end of data
func trimPartialRune(data []byte) []byte {
	for range 3 {
		if len(data) == 0 {
			break
		}
		if r, size := utf8.DecodeLastRune(data); r != utf8.RuneError || size > 1 {
			break
		}
		data = data[:len(data)-1]
	}
	return data
}

// isBinary reports whether data looks like something other than UTF-8 text
func isBinary(data []byte) bool {
	return bytes.IndexByte(data, 0) >= 0 || !utf8.Valid(trimPartialRune(data))
}

// getProjectFile returns the contents of a single file of the active project
// so the UI can preview it. Text is returned as-is (truncated past
// maxViewFileBytes), images as base64, and other binaries without content.
func (s *Server) getProjectFile(w http.ResponseWriter, r *http.Request) error {
	p := s.activeProject()
	if p == nil {
		return fmt.Errorf("no project is open")
	}

	ref := r.URL.Query().Get("path")
	if ref == "" {
		return fmt.Errorf("path is required")
	}

	rel, full, ok := p.resolvePath(ref)
	if !ok {
		return fmt.Errorf("invalid path %q", ref)
	}

	info, err := os.Stat(full)
	if err != nil {
		return fmt.Errorf("read %s: %w", rel, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s is a directory", rel)
	}

	resp := responses.ProjectFileResponse{Path: rel, Size: info.Size()}

	if mime, isImage := viewImageTypes[strings.ToLower(filepath.Ext(full))]; isImage {
		resp.Binary = true
		if info.Size() > maxViewImageBytes {
			w.Header().Set("Content-Type", "application/json")
			return json.NewEncoder(w).Encode(resp)
		}
		data, err := os.ReadFile(full)
		if err != nil {
			return fmt.Errorf("read %s: %w", rel, err)
		}
		resp.MimeType = mime
		resp.Content = base64.StdEncoding.EncodeToString(data)
		w.Header().Set("Content-Type", "application/json")
		return json.NewEncoder(w).Encode(resp)
	}

	f, err := os.Open(full)
	if err != nil {
		return fmt.Errorf("read %s: %w", rel, err)
	}
	defer f.Close()

	// read one byte past the cap to tell a full file from a truncated one
	data, err := io.ReadAll(io.LimitReader(f, maxViewFileBytes+1))
	if err != nil {
		return fmt.Errorf("read %s: %w", rel, err)
	}
	if len(data) > maxViewFileBytes {
		data = data[:maxViewFileBytes]
		resp.Truncated = true
	}

	if isBinary(data) {
		resp.Binary = true
	} else {
		if resp.Truncated {
			data = trimPartialRune(data)
		}
		resp.Content = string(data)
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}
