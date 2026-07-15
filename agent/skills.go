package agent

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	// SkillsDirEnv overrides the runtime-owned skills directory. Ollama never
	// searches Codex or Claude skill directories.
	SkillsDirEnv  = "OLLAMA_SKILLS"
	skillFilename = "SKILL.md"
	maxSkillBytes = 1 << 20
)

var skillName = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)

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
	return fmt.Sprintf("<skill name=%q>\n%s\n</skill>", s.Name, strings.TrimSpace(s.Instructions))
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

func LoadDefaultSkills() (*SkillCatalog, error) {
	dir, err := SkillsDir()
	if err != nil {
		return nil, err
	}
	return DiscoverSkills(dir)
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
	instructions := strings.TrimSpace(string(data))
	if instructions == "" {
		return Skill{}, fmt.Errorf("skill %q: %s is empty", directoryName, skillFilename)
	}
	skill := Skill{Name: directoryName, Path: path}
	if strings.HasPrefix(instructions, "---\n") || strings.HasPrefix(instructions, "---\r\n") {
		metadata, body, err := skillFrontMatter(instructions)
		if err != nil {
			return Skill{}, fmt.Errorf("skill %q: %w", directoryName, err)
		}
		// The directory name is the canonical skill name; a front matter "name"
		// field is accepted for compatibility but is not required to match.
		skill.Description = metadata["description"]
		instructions = body
	}
	if strings.TrimSpace(instructions) == "" {
		return Skill{}, fmt.Errorf("skill %q: instructions are empty", directoryName)
	}
	skill.Instructions = strings.TrimSpace(instructions)
	return skill, nil
}

func skillFrontMatter(input string) (map[string]string, string, error) {
	input = strings.ReplaceAll(input, "\r\n", "\n")
	lines := strings.Split(input, "\n")
	if len(lines) < 3 || lines[0] != "---" {
		return nil, "", errors.New("invalid front matter")
	}
	metadata := make(map[string]string)
	for i := 1; i < len(lines); i++ {
		if lines[i] == "---" {
			return metadata, strings.Join(lines[i+1:], "\n"), nil
		}
		key, value, ok := strings.Cut(lines[i], ":")
		if !ok || strings.TrimSpace(key) == "" || strings.TrimSpace(value) == "" {
			return nil, "", fmt.Errorf("invalid front matter line %q", lines[i])
		}
		if _, exists := metadata[key]; exists {
			return nil, "", fmt.Errorf("duplicate front matter field %q", key)
		}
		metadata[key] = strings.Trim(strings.TrimSpace(value), `"`)
	}
	return nil, "", errors.New("front matter is not closed")
}
