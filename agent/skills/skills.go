package skills

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	SkillFile = "SKILL.md"
)

var validName = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

type Skill struct {
	Name        string
	Description string
	Dir         string
	File        string
}

type Catalog struct {
	Dir      string
	Skills   []Skill
	Warnings []string
}

type frontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

func DefaultDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".ollama", "skills"), nil
}

func LoadDefault() (*Catalog, error) {
	dir, err := DefaultDir()
	if err != nil {
		return nil, err
	}
	return Load(dir)
}

func Load(dir string) (*Catalog, error) {
	catalog := &Catalog{Dir: dir}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return catalog, nil
		}
		return nil, fmt.Errorf("read skills directory: %w", err)
	}

	seen := make(map[string]string)
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}

		skillDir := filepath.Join(dir, entry.Name())
		skill, err := ReadMetadata(filepath.Join(skillDir, SkillFile))
		if err != nil {
			catalog.Warnings = append(catalog.Warnings, fmt.Sprintf("%s: %v", skillDir, err))
			continue
		}
		skill.Dir = skillDir
		skill.File = filepath.Join(skillDir, SkillFile)
		if previous, ok := seen[skill.Name]; ok {
			catalog.Warnings = append(catalog.Warnings, fmt.Sprintf("%s: duplicate skill name %q already loaded from %s", skillDir, skill.Name, previous))
			continue
		}
		seen[skill.Name] = skillDir
		catalog.Skills = append(catalog.Skills, skill)
	}

	slices.SortFunc(catalog.Skills, func(a, b Skill) int {
		return strings.Compare(a.Name, b.Name)
	})
	return catalog, nil
}

func ReadMetadata(path string) (Skill, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, err
	}

	meta, _, err := parseSkillFile(data)
	if err != nil {
		return Skill{}, err
	}
	if err := validateMetadata(meta); err != nil {
		return Skill{}, err
	}
	return Skill{Name: meta.Name, Description: meta.Description}, nil
}

func (c *Catalog) Empty() bool {
	return c == nil || len(c.Skills) == 0
}

func (c *Catalog) Find(name string) (Skill, bool) {
	if c == nil {
		return Skill{}, false
	}
	name = NormalizeName(name)
	for _, skill := range c.Skills {
		if skill.Name == name {
			return skill, true
		}
	}
	return Skill{}, false
}

func (c *Catalog) SummaryMarkdown() string {
	if c.Empty() {
		return "No skills are installed."
	}
	var b strings.Builder
	b.WriteString("Installed skills:\n\n")
	for _, skill := range c.Skills {
		b.WriteString("- **")
		b.WriteString(skill.Name)
		b.WriteString("**: ")
		b.WriteString(skill.Description)
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

func (c *Catalog) SystemPrompt(toolAvailable bool) string {
	if c.Empty() {
		return ""
	}

	var b strings.Builder
	b.WriteString("Agent skills are available. Skills are reusable instruction packages stored under ")
	b.WriteString(c.Dir)
	b.WriteString(".\n")
	b.WriteString("Use a skill when its description matches the user's task. Load only metadata up front; load full instructions only when needed.\n")
	if toolAvailable {
		b.WriteString("To load a skill, call the skill tool with the skill name. After loading SKILL.md, follow it. Resolve relative references from the returned skill directory.\n")
	} else {
		b.WriteString("This model cannot call tools in this session. Follow any skill instructions that are explicitly provided by the user or system.\n")
	}
	b.WriteString("\nAvailable skills:\n")
	for _, skill := range c.Skills {
		b.WriteString("- ")
		b.WriteString(skill.Name)
		b.WriteString(": ")
		b.WriteString(skill.Description)
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

func (s Skill) Read() (string, error) {
	if s.File == "" {
		return "", fmt.Errorf("skill %q has no %s path", s.Name, SkillFile)
	}
	data, err := os.ReadFile(s.File)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func NormalizeName(name string) string {
	name = strings.TrimSpace(name)
	name = strings.TrimPrefix(name, "/")
	return strings.ToLower(name)
}

func parseSkillFile(data []byte) (frontmatter, string, error) {
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	if !strings.HasPrefix(text, "---\n") {
		return frontmatter{}, "", fmt.Errorf("%s must start with YAML frontmatter", SkillFile)
	}

	rest := text[len("---\n"):]
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return frontmatter{}, "", fmt.Errorf("%s frontmatter is not closed", SkillFile)
	}

	var meta frontmatter
	if err := yaml.Unmarshal([]byte(rest[:end]), &meta); err != nil {
		return frontmatter{}, "", fmt.Errorf("parse frontmatter: %w", err)
	}

	body := rest[end+len("\n---"):]
	body = strings.TrimPrefix(body, "\n")
	return meta, body, nil
}

func validateMetadata(meta frontmatter) error {
	if !validName.MatchString(meta.Name) {
		return fmt.Errorf("invalid skill name %q", meta.Name)
	}
	if strings.TrimSpace(meta.Description) == "" {
		return fmt.Errorf("skill %q has empty description", meta.Name)
	}
	if len([]rune(meta.Description)) > 1024 {
		return fmt.Errorf("skill %q description exceeds 1024 characters", meta.Name)
	}
	return nil
}

func copyDir(src, dst string, force bool) error {
	if force {
		if err := os.RemoveAll(dst); err != nil {
			return err
		}
	}
	if _, err := os.Stat(dst); err == nil {
		return fs.ErrExist
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}

	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)

		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}

		info, err := d.Info()
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}
