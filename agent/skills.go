package agent

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	// SkillsDirEnv overrides the user-level Ollama-owned skills directory. The
	// cross-client .agents/skills/ convention and project-level .ollama/skills/
	// are also scanned (see LoadDefaultSkills); on a name collision, Ollama-owned
	// directories take precedence over .agents/skills/, and project-level takes
	// precedence over user-level.
	SkillsDirEnv  = "OLLAMA_SKILLS"
	skillFilename = "SKILL.md"
	maxSkillBytes = 1 << 20

	bundledSkillCreatorName    = "skill-creator"
	bundledSkillCreatorContent = `---
name: skill-creator
description: Create or improve reusable skills. Use when the user wants a reusable skill, asks how to author SKILL.md, or needs help installing a skill.
---

# Create a skill

Create a focused, reusable instruction package. Treat a skill as guidance for the model, not as a way to gain new permissions or bypass safety controls.

## Choose the location

Create user skills beside this one. The skill directory shown in the loaded skill context is this skill's location; its parent is the user skill root. This bundled skill normally lives at ~/.ollama/skills/skill-creator, so new user skills normally go at ~/.ollama/skills/<skill-name>/SKILL.md.

Use a project-local skill directory only when the user asks to keep the skill with that project. Do not overwrite an existing skill without the user's approval. New and changed skills are discovered when the agent starts, so tell the user to begin a new agent session afterward.

## Follow the required shape

Use the directory name as the skill name. Use lowercase letters, numbers, and single hyphens only. Keep the name short and no longer than 64 characters.

Every skill needs a SKILL.md with YAML frontmatter followed by Markdown instructions:

~~~md
---
name: release-notes
description: Draft concise release notes from completed changes. Use when the user asks for a changelog, release notes, or GitHub release copy.
---

# Draft release notes

Write the workflow here.
~~~

Require a non-empty description that says both what the skill does and when to use it. Keep the body procedural and concise. Put detailed schemas, long examples, and variant-specific guidance in references/ only when the skill needs them.

Use scripts/ for repeatable or fragile operations that benefit from deterministic execution. Use assets/ for files that belong in generated output. Do not add README files, changelogs, or setup notes that do not help the model perform the task.

## Create safely

1. Identify the repeated task, expected inputs, and useful output.
2. Choose the smallest name and description that reliably trigger the skill.
3. Create the folder and SKILL.md; add resources only when they remove real repeated work.
4. Re-read the completed file and verify its frontmatter, directory-name match, and relative resource paths.
5. Tell the user where it was created and that a new agent session will discover it.

Skills provide instructions only. They do not grant filesystem, network, shell, or approval privileges, and they do not make a tool available. Use only the tools that are actually available, follow their normal approval rules, and ask before actions that need user authorization.
`
)

var skillName = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

// SkillsDir returns the canonical runtime-owned skill directory.
func SkillsDir() (string, error) {
	if path := strings.TrimSpace(os.Getenv(SkillsDirEnv)); path != "" {
		return filepath.Abs(path)
	}
	if xdg := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME")); xdg != "" {
		return filepath.Join(xdg, "ollama", "skills"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "skills"), nil
}

// Skill is a validated, loadable instruction set. It never grants tool
// permissions; it is supplied to the model as ordinary tool-result content.
type Skill struct {
	Name         string
	Description  string
	Instructions string
	Path         string
}

func (s Skill) Content() string {
	var b strings.Builder
	fmt.Fprintf(&b, "<skill name=%q>\n%s\n", s.Name, strings.TrimSpace(s.Instructions))
	if s.Path != "" {
		dir := filepath.Dir(s.Path)
		fmt.Fprintf(&b, "Skill directory: %s\n", dir)
		b.WriteString("Relative paths in this skill are relative to the skill directory.\n")
	}
	if resources := s.resources(); len(resources) > 0 {
		b.WriteString("<skill_resources>\n")
		for _, r := range resources {
			fmt.Fprintf(&b, "  <file>%s</file>\n", r)
		}
		b.WriteString("</skill_resources>\n")
	}
	b.WriteString("</skill>")
	return b.String()
}

// resources lists bundled files one level deep under scripts/, references/,
// and assets/ without reading them, so the model can load them on demand.
func (s Skill) resources() []string {
	if s.Path == "" {
		return nil
	}
	dir := filepath.Dir(s.Path)
	var resources []string
	for _, sub := range []string{"scripts", "references", "assets"} {
		entries, err := os.ReadDir(filepath.Join(dir, sub))
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			resources = append(resources, sub+"/"+e.Name())
		}
	}
	sort.Strings(resources)
	return resources
}

// SkillCatalog contains valid skills and diagnostics for ignored invalid
// entries, so one malformed skill cannot hide the rest.
type SkillCatalog struct {
	dir         string
	skills      map[string]Skill
	diagnostics []error
}

func DiscoverSkills(dir string) (*SkillCatalog, error) {
	dir, err := filepath.Abs(strings.TrimSpace(dir))
	if err != nil {
		return nil, err
	}
	catalog := &SkillCatalog{dir: dir, skills: make(map[string]Skill)}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return catalog, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read skills directory: %w", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		// Follow symlinks so users can point at shared skill repositories.
		// The link name (not the target) is the canonical skill name.
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			catalog.diagnostics = append(catalog.diagnostics, fmt.Errorf("skill %q: %w", name, err))
			continue
		}
		if !info.IsDir() {
			continue
		}
		if !skillName.MatchString(name) {
			catalog.diagnostics = append(catalog.diagnostics, fmt.Errorf("invalid skill directory %q", name))
			continue
		}
		skill, err := parseSkill(filepath.Join(dir, name, skillFilename), name)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			catalog.diagnostics = append(catalog.diagnostics, err)
			continue
		}
		catalog.skills[skill.Name] = skill
	}
	return catalog, nil
}

// LoadDefaultSkills discovers skills from the spec's scopes, merged with
// deterministic precedence. Roots are scanned lowest-precedence first so later
// roots override earlier ones on name collisions (recording a diagnostic):
//
//  1. ~/.agents/skills/              (user, cross-client)
//  2. user Ollama skills dir          (user, Ollama-owned; SkillsDir)
//  3. <project>/.agents/skills/      (project, cross-client)
//  4. <project>/.ollama/skills/      (project, Ollama-owned)
//
// Project-level overrides user-level, and within a scope Ollama-owned
// directories override .agents/skills/. projectDir is the agent's working
// directory at startup (discovery is a session-start snapshot per the spec).
func LoadDefaultSkills(projectDir string) (*SkillCatalog, error) {
	roots, err := defaultSkillRoots(projectDir)
	if err != nil {
		return nil, err
	}
	catalog := &SkillCatalog{skills: make(map[string]Skill)}
	bundled, err := bundledSkillCreator()
	if err != nil {
		return nil, err
	}
	catalog.skills[bundled.Name] = bundled
	if err := installBundledSkillCreator(); err != nil {
		catalog.diagnostics = append(catalog.diagnostics, err)
	}
	for _, root := range roots {
		sub, err := DiscoverSkills(root.path)
		if err != nil {
			catalog.diagnostics = append(catalog.diagnostics, fmt.Errorf("discover skills in %s: %w", root.path, err))
			continue
		}
		catalog.diagnostics = append(catalog.diagnostics, sub.diagnostics...)
		for _, skill := range sub.skills {
			// Name collisions across roots are expected precedence resolution,
			// not errors: later (higher-precedence) roots legitimately override
			// earlier ones. The skill is still loaded; no diagnostic needed.
			catalog.skills[skill.Name] = skill
		}
	}
	return catalog, nil
}

func bundledSkillCreator() (Skill, error) {
	skill, err := parseSkillContent("", bundledSkillCreatorName, bundledSkillCreatorContent)
	if err != nil {
		return Skill{}, fmt.Errorf("load bundled %s skill: %w", bundledSkillCreatorName, err)
	}
	return skill, nil
}

func installBundledSkillCreator() error {
	dir, err := SkillsDir()
	if err != nil {
		return fmt.Errorf("resolve bundled skill directory: %w", err)
	}
	path := filepath.Join(dir, bundledSkillCreatorName, skillFilename)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create bundled skill directory: %w", err)
	}
	contents, err := os.ReadFile(path)
	if err == nil && string(contents) == bundledSkillCreatorContent {
		return nil
	}
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("read bundled skill: %w", err)
	}
	if err := os.WriteFile(path, []byte(bundledSkillCreatorContent), 0o644); err != nil {
		return fmt.Errorf("write bundled skill: %w", err)
	}
	return nil
}

type skillRoot struct {
	path string
}

// SkillImportResult describes one import attempt. Failed skills do not prevent
// other valid skills in the same source root from being imported.
type SkillImportResult struct {
	Source      string
	SourceDir   string
	Destination string
	Imported    []string
	Existing    []string
	Failures    []SkillImportFailure
}

// SkillImportFailure identifies a source skill that was deliberately skipped.
// The destination is never changed for a failed skill.
type SkillImportFailure struct {
	Name string
	Err  error
}

// ImportSkills imports skills from a conventional coding-agent source into the
// canonical Ollama skills directory. Supported sources are codex, claude, and
// pi. Existing skills are left untouched: an identical directory is reported
// as existing, and a differing one is reported as a conflict.
func ImportSkills(source string) (SkillImportResult, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return SkillImportResult{}, fmt.Errorf("resolve home directory: %w", err)
	}

	destination, err := SkillsDir()
	if err != nil {
		return SkillImportResult{}, fmt.Errorf("resolve Ollama skills directory: %w", err)
	}
	return importSkillsFromRoots(source, conventionalSkillImportRoots(home), destination)
}

func conventionalSkillImportRoots(home string) map[string]string {
	return map[string]string{
		"codex":  filepath.Join(home, ".codex", "skills"),
		"claude": filepath.Join(home, ".claude", "skills"),
		"pi":     filepath.Join(home, ".pi", "agent", "skills"),
	}
}

func importSkillsFromRoots(source string, roots map[string]string, destination string) (SkillImportResult, error) {
	source = strings.ToLower(strings.TrimSpace(source))
	sourceDir, ok := roots[source]
	if !ok {
		return SkillImportResult{}, fmt.Errorf("unknown skill source %q", source)
	}
	return importSkillsFromDir(source, sourceDir, destination)
}

func importSkillsFromDir(source, sourceDir, destination string) (SkillImportResult, error) {
	result := SkillImportResult{Source: source, SourceDir: sourceDir, Destination: destination}
	info, err := os.Lstat(sourceDir)
	if errors.Is(err, fs.ErrNotExist) {
		return result, nil
	}
	if err != nil {
		return result, fmt.Errorf("inspect %s skills directory: %w", source, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return result, fmt.Errorf("inspect %s skills directory: symlinks are not supported", source)
	}
	if !info.IsDir() {
		return result, fmt.Errorf("inspect %s skills directory: not a directory", source)
	}

	entries, err := os.ReadDir(sourceDir)
	if err != nil {
		return result, fmt.Errorf("read %s skills directory: %w", source, err)
	}
	for _, entry := range entries {
		name := entry.Name()
		path := filepath.Join(sourceDir, name)
		if entry.Type()&os.ModeSymlink != 0 {
			result.Failures = append(result.Failures, SkillImportFailure{Name: name, Err: errors.New("symlinked skill directories are not supported")})
			continue
		}
		info, err := entry.Info()
		if err != nil {
			result.Failures = append(result.Failures, SkillImportFailure{Name: name, Err: fmt.Errorf("inspect source: %w", err)})
			continue
		}
		if !info.IsDir() {
			continue
		}
		if !skillName.MatchString(name) {
			result.Failures = append(result.Failures, SkillImportFailure{Name: name, Err: errors.New("invalid skill directory name")})
			continue
		}
		if err := validateImportSkill(path, name); err != nil {
			result.Failures = append(result.Failures, SkillImportFailure{Name: name, Err: err})
			continue
		}

		state, err := importSkillDirectory(path, filepath.Join(destination, name))
		if err != nil {
			result.Failures = append(result.Failures, SkillImportFailure{Name: name, Err: err})
			continue
		}
		if state == skillImportExisting {
			result.Existing = append(result.Existing, name)
		} else {
			result.Imported = append(result.Imported, name)
		}
	}
	return result, nil
}

func validateImportSkill(dir, name string) error {
	manifest := filepath.Join(dir, skillFilename)
	info, err := os.Lstat(manifest)
	if err != nil {
		return fmt.Errorf("inspect %s: %w", skillFilename, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return fmt.Errorf("%s must be a regular, non-symlinked file", skillFilename)
	}
	if _, err := parseSkill(manifest, name); err != nil {
		return err
	}
	return walkImportTree(dir, func(path string, entry fs.DirEntry, info fs.FileInfo) error {
		if info.IsDir() || path == dir {
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("only regular files may be imported: %s", path)
		}
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}
		return file.Close()
	})
}

func walkImportTree(root string, visit func(string, fs.DirEntry, fs.FileInfo) error) error {
	return filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe skill path %q", path)
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlinks may not be imported: %s", path)
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		return visit(path, entry, info)
	})
}

type skillImportState int

const (
	skillImportCopied skillImportState = iota
	skillImportExisting
)

func importSkillDirectory(source, destination string) (skillImportState, error) {
	if info, err := os.Lstat(destination); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return 0, errors.New("destination exists but is not a regular directory")
		}
		same, err := sameImportTree(source, destination)
		if err != nil {
			return 0, fmt.Errorf("inspect existing destination: %w", err)
		}
		if same {
			return skillImportExisting, nil
		}
		return 0, errors.New("destination skill already exists with different contents")
	} else if !errors.Is(err, fs.ErrNotExist) {
		return 0, fmt.Errorf("inspect destination: %w", err)
	}

	if err := ensureImportDestination(filepath.Dir(destination)); err != nil {
		return 0, err
	}
	stage, err := os.MkdirTemp(filepath.Dir(destination), "."+filepath.Base(destination)+".import-")
	if err != nil {
		return 0, fmt.Errorf("create import staging directory: %w", err)
	}
	defer os.RemoveAll(stage)
	if err := copyImportTree(source, stage); err != nil {
		return 0, err
	}
	if _, err := os.Lstat(destination); err == nil {
		return 0, errors.New("destination skill was created during import")
	} else if !errors.Is(err, fs.ErrNotExist) {
		return 0, fmt.Errorf("inspect destination before install: %w", err)
	}
	if err := os.Rename(stage, destination); err != nil {
		return 0, fmt.Errorf("install imported skill: %w", err)
	}
	return skillImportCopied, nil
}

func ensureImportDestination(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create Ollama skills directory: %w", err)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return fmt.Errorf("inspect Ollama skills directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("Ollama skills directory must be a regular, non-symlinked directory")
	}
	return nil
}

func copyImportTree(source, destination string) error {
	return walkImportTree(source, func(path string, entry fs.DirEntry, info fs.FileInfo) error {
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := destination
		if rel != "." {
			target = filepath.Join(destination, rel)
		}
		if info.IsDir() {
			if rel == "." {
				return nil
			}
			return os.Mkdir(target, info.Mode().Perm())
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("only regular files may be imported: %s", path)
		}
		return copyImportFile(path, target, info.Mode().Perm())
	})
}

func copyImportFile(source, destination string, mode fs.FileMode) error {
	in, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("read %s: %w", source, err)
	}
	defer in.Close()
	out, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return fmt.Errorf("create %s: %w", destination, err)
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return fmt.Errorf("copy %s: %w", source, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("write %s: %w", destination, closeErr)
	}
	return nil
}

func sameImportTree(source, destination string) (bool, error) {
	seen := make(map[string]struct{})
	same := true
	err := walkImportTree(source, func(path string, entry fs.DirEntry, info fs.FileInfo) error {
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		seen[rel] = struct{}{}
		other := destination
		if rel != "." {
			other = filepath.Join(destination, rel)
		}
		otherInfo, err := os.Lstat(other)
		if errors.Is(err, fs.ErrNotExist) {
			same = false
			return nil
		}
		if err != nil {
			return err
		}
		if otherInfo.Mode()&os.ModeSymlink != 0 || otherInfo.IsDir() != info.IsDir() || (!info.IsDir() && !otherInfo.Mode().IsRegular()) {
			same = false
			return nil
		}
		if info.Mode().IsRegular() {
			equal, err := sameImportFile(path, other)
			if err != nil {
				return err
			}
			if !equal {
				same = false
			}
		}
		return nil
	})
	if err != nil || !same {
		return same, err
	}
	err = walkImportTree(destination, func(path string, entry fs.DirEntry, info fs.FileInfo) error {
		rel, err := filepath.Rel(destination, path)
		if err != nil {
			return err
		}
		if _, ok := seen[rel]; !ok {
			same = false
		}
		return nil
	})
	return same, err
}

func sameImportFile(first, second string) (bool, error) {
	a, err := os.Open(first)
	if err != nil {
		return false, err
	}
	defer a.Close()
	b, err := os.Open(second)
	if err != nil {
		return false, err
	}
	defer b.Close()

	left := make([]byte, 32*1024)
	right := make([]byte, len(left))
	for {
		n, errA := a.Read(left)
		m, errB := b.Read(right)
		if n != m || !bytes.Equal(left[:n], right[:m]) {
			return false, nil
		}
		if errA == io.EOF && errB == io.EOF {
			return true, nil
		}
		if errA != nil && errA != io.EOF {
			return false, errA
		}
		if errB != nil && errB != io.EOF {
			return false, errB
		}
		if errA == io.EOF || errB == io.EOF {
			return false, nil
		}
	}
}

// defaultSkillRoots returns skill directories ordered lowest- to
// highest-precedence. Non-existent directories are scanned harmlessly
// (DiscoverSkills skips them).
func defaultSkillRoots(projectDir string) ([]skillRoot, error) {
	var roots []skillRoot

	if home, err := os.UserHomeDir(); err == nil && home != "" {
		roots = append(roots, skillRoot{path: filepath.Join(home, ".agents", "skills")})
	}

	userOllama, err := SkillsDir()
	if err != nil {
		return nil, err
	}
	roots = append(roots, skillRoot{path: userOllama})

	projectDir = strings.TrimSpace(projectDir)
	if projectDir != "" {
		if abs, err := filepath.Abs(projectDir); err == nil {
			roots = append(roots,
				skillRoot{path: filepath.Join(abs, ".agents", "skills")},
				skillRoot{path: filepath.Join(abs, ".ollama", "skills")},
			)
		}
	}
	return roots, nil
}

func (c *SkillCatalog) Dir() string {
	if c == nil {
		return ""
	}
	return c.dir
}

func (c *SkillCatalog) List() []Skill {
	if c == nil {
		return nil
	}
	list := make([]Skill, 0, len(c.skills))
	for _, skill := range c.skills {
		list = append(list, skill)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Name < list[j].Name })
	return list
}

func (c *SkillCatalog) Diagnostics() []error {
	if c == nil {
		return nil
	}
	return append([]error(nil), c.diagnostics...)
}

// ExcludeNames removes skills whose names are reserved by a caller. It returns
// the excluded names in sorted order.
func (c *SkillCatalog) ExcludeNames(names []string) []string {
	if c == nil {
		return nil
	}
	reserved := make(map[string]struct{}, len(names))
	for _, name := range names {
		name = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(name)), "/")
		if name != "" {
			reserved[name] = struct{}{}
		}
	}
	var excluded []string
	for name := range c.skills {
		if _, ok := reserved[name]; !ok {
			continue
		}
		delete(c.skills, name)
		excluded = append(excluded, name)
	}
	sort.Strings(excluded)
	return excluded
}

func (c *SkillCatalog) Load(name string) (Skill, error) {
	name = strings.TrimSpace(name)
	if !skillName.MatchString(name) {
		return Skill{}, fmt.Errorf("invalid skill name %q", name)
	}
	if c == nil {
		return Skill{}, errors.New("skills are unavailable")
	}
	skill, ok := c.skills[name]
	if !ok {
		return Skill{}, fmt.Errorf("skill %q not found in %s", name, c.dir)
	}
	return skill, nil
}

// SystemContext advertises the catalog without expanding full instructions in
// every request. The skill call is the explicit loading boundary.
func (c *SkillCatalog) SystemContext() string {
	list := c.List()
	if len(list) == 0 {
		return ""
	}
	lines := []string{"<available_skills>"}
	for _, skill := range list {
		description := skill.Description
		if description == "" {
			description = "No description provided."
		}
		lines = append(lines, fmt.Sprintf("- %s: %s", skill.Name, description))
	}
	lines = append(lines, "</available_skills>", "Load a matching skill with the skill tool before following its instructions. Skills only provide instructions; use ordinary tools for filesystem or network access, with their normal approval rules.")
	return strings.Join(lines, "\n")
}

func parseSkill(path, directoryName string) (Skill, error) {
	// Stat (not Lstat) so a symlinked SKILL.md resolves to its target file.
	info, err := os.Stat(path)
	if err != nil {
		return Skill{}, err
	}
	if !info.Mode().IsRegular() {
		return Skill{}, fmt.Errorf("skill %q: %s is not a regular file", directoryName, skillFilename)
	}
	if info.Size() > maxSkillBytes {
		return Skill{}, fmt.Errorf("skill %q: %s exceeds %d bytes", directoryName, skillFilename, maxSkillBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, fmt.Errorf("read skill %q: %w", directoryName, err)
	}
	return parseSkillContent(path, directoryName, string(data))
}

func parseSkillContent(path, directoryName, input string) (Skill, error) {
	instructions := strings.TrimSpace(input)
	if instructions == "" {
		return Skill{}, fmt.Errorf("skill %q: %s is empty", directoryName, skillFilename)
	}
	if !strings.HasPrefix(instructions, "---\n") && !strings.HasPrefix(instructions, "---\r\n") {
		return Skill{}, fmt.Errorf("skill %q: missing YAML front matter", directoryName)
	}
	metadata, body, err := skillFrontMatter(instructions)
	if err != nil {
		return Skill{}, fmt.Errorf("skill %q: %w", directoryName, err)
	}
	if metadata.Name == "" {
		return Skill{}, fmt.Errorf("skill %q: front matter requires name", directoryName)
	}
	if metadata.Description == "" {
		return Skill{}, fmt.Errorf("skill %q: front matter requires description", directoryName)
	}
	if !skillName.MatchString(metadata.Name) {
		return Skill{}, fmt.Errorf("skill %q: invalid front matter name %q", directoryName, metadata.Name)
	}
	if metadata.Name != directoryName {
		return Skill{}, fmt.Errorf("skill %q: front matter name %q must match directory name", directoryName, metadata.Name)
	}
	skill := Skill{Name: metadata.Name, Description: metadata.Description, Path: path}
	instructions = body
	if strings.TrimSpace(instructions) == "" {
		return Skill{}, fmt.Errorf("skill %q: instructions are empty", directoryName)
	}
	skill.Instructions = strings.TrimSpace(instructions)
	return skill, nil
}

type skillFrontMatterMetadata struct {
	Name        string         `yaml:"name"`
	Description string         `yaml:"description"`
	Metadata    map[string]any `yaml:"metadata"`
}

func skillFrontMatter(input string) (skillFrontMatterMetadata, string, error) {
	input = strings.ReplaceAll(input, "\r\n", "\n")
	lines := strings.Split(input, "\n")
	if len(lines) < 3 || lines[0] != "---" {
		return skillFrontMatterMetadata{}, "", errors.New("invalid front matter")
	}
	for i := 1; i < len(lines); i++ {
		if lines[i] == "---" {
			var metadata skillFrontMatterMetadata
			if err := yaml.Unmarshal([]byte(strings.Join(lines[1:i], "\n")), &metadata); err != nil {
				return skillFrontMatterMetadata{}, "", fmt.Errorf("parse YAML front matter: %w", err)
			}
			metadata.Name = strings.TrimSpace(metadata.Name)
			metadata.Description = strings.TrimSpace(metadata.Description)
			return metadata, strings.Join(lines[i+1:], "\n"), nil
		}
	}
	return skillFrontMatterMetadata{}, "", errors.New("front matter is not closed")
}
