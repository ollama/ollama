package skills

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type ImportResult struct {
	Source  string
	Skill   Skill
	From    string
	To      string
	Skipped bool
	Error   string
}

type skillDirCandidate struct {
	Dir     string
	Skipped bool
	Error   string
}

func Import(source string, force bool) ([]ImportResult, error) {
	dest, err := DefaultDir()
	if err != nil {
		return nil, err
	}
	return ImportToDir(source, dest, force)
}

func ImportToDir(source, dest string, force bool) ([]ImportResult, error) {
	roots, err := SourceDirs(source)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return nil, fmt.Errorf("create skills directory: %w", err)
	}

	var results []ImportResult
	for _, root := range roots {
		candidates, err := skillDirs(root)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			results = append(results, ImportResult{Source: source, From: root, Skipped: true, Error: err.Error()})
			continue
		}
		for _, candidate := range candidates {
			result := ImportResult{Source: source, From: candidate.Dir}
			if candidate.Skipped {
				result.Skipped = true
				result.Error = candidate.Error
				results = append(results, result)
				continue
			}

			skill, err := ReadMetadata(filepath.Join(candidate.Dir, SkillFile))
			if err != nil {
				result.Skipped = true
				result.Error = err.Error()
				results = append(results, result)
				continue
			}

			result.Skill = skill
			result.To = filepath.Join(dest, skill.Name)
			copyResult, err := copyDir(candidate.Dir, result.To, force)
			if errors.Is(err, os.ErrExist) {
				result.Skipped = true
				result.Error = "already exists"
			} else if err != nil {
				result.Skipped = true
				result.Error = err.Error()
			} else if len(copyResult.Skipped) > 0 {
				result.Error = "skipped symlinks: " + strings.Join(copyResult.Skipped, ", ")
			}
			results = append(results, result)
		}
	}

	slices.SortFunc(results, func(a, b ImportResult) int {
		return strings.Compare(a.Skill.Name+a.From, b.Skill.Name+b.From)
	})
	return results, nil
}

func SourceDirs(source string) ([]string, error) {
	source = strings.ToLower(strings.TrimSpace(source))
	if source == "" {
		source = "all"
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home directory: %w", err)
	}

	dirs := map[string][]string{
		"claude": {filepath.Join(home, ".claude", "skills")},
		"codex":  {filepath.Join(home, ".codex", "skills")},
		"pi":     {filepath.Join(home, ".pi", "skills"), filepath.Join(home, ".agents", "skills")},
		"agents": {filepath.Join(home, ".agents", "skills")},
	}
	if source == "all" {
		var all []string
		for _, name := range []string{"claude", "codex", "pi"} {
			all = append(all, dirs[name]...)
		}
		return uniqueStrings(all), nil
	}
	if roots, ok := dirs[source]; ok {
		return roots, nil
	}
	return nil, fmt.Errorf("unknown skill source %q (use claude, codex, pi, agents, or all)", source)
}

func skillDirs(root string) ([]skillDirCandidate, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var dirs []skillDirCandidate
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 {
			dirs = append(dirs, skillDirCandidate{
				Dir:     dir,
				Skipped: true,
				Error:   "symlinked skill directories are not supported",
			})
			continue
		}
		if !entry.IsDir() {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, SkillFile)); err == nil {
			dirs = append(dirs, skillDirCandidate{Dir: dir})
		}
	}
	slices.SortFunc(dirs, func(a, b skillDirCandidate) int {
		return strings.Compare(a.Dir, b.Dir)
	})
	return dirs, nil
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, value := range values {
		if seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}
