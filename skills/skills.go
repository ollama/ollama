package skills

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

const (
	envSkillsDir      = "OLLAMA_SKILLS"
	envSkillAllow     = "OLLAMA_SKILL_ALLOW"
	skillManifestFile = "skill.toml"
	installedDirName  = "installed"
	enabledStateFile  = "enabled.json"
)

var (
	// ErrSkillNotFound indicates a requested skill has not been installed.
	ErrSkillNotFound = errors.New("skill not found")
	// ErrSkillNotEnabled indicates a requested skill is installed but disabled.
	ErrSkillNotEnabled = errors.New("skill not enabled")
	// ErrPermissionDenied indicates required permissions were not granted.
	ErrPermissionDenied = errors.New("permission denied")

	validSkillName  = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	validPermission = regexp.MustCompile(`^[a-z][a-z0-9_.-]*$`)
)

// Spec describes the skill.toml MVP schema.
type Spec struct {
	Name        string         `toml:"name"`
	Description string         `toml:"description,omitempty"`
	Version     string         `toml:"version,omitempty"`
	Command     string         `toml:"command"`
	Args        []string       `toml:"args,omitempty"`
	IO          IOSpec         `toml:"io"`
	Permissions PermissionSpec `toml:"permissions"`
}

type IOSpec struct {
	Inputs  []string `toml:"inputs,omitempty"`
	Outputs []string `toml:"outputs,omitempty"`
}

type PermissionSpec struct {
	Required []string `toml:"required,omitempty"`
}

type Skill struct {
	Spec    Spec
	Dir     string
	Enabled bool
}

// RunOptions controls how a skill is executed.
type RunOptions struct {
	// GrantedPermissions are permissions allowed for this execution.
	// If empty, OLLAMA_SKILL_ALLOW is used.
	GrantedPermissions []string
}

func RootDir() (string, error) {
	if v := strings.TrimSpace(os.Getenv(envSkillsDir)); v != "" {
		return filepath.Clean(v), nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", "skills"), nil
}

func installedDir() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, installedDirName), nil
}

func enabledPath() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, enabledStateFile), nil
}

func ensureLayout() error {
	root, err := RootDir()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}

	installed, err := installedDir()
	if err != nil {
		return err
	}

	return os.MkdirAll(installed, 0o755)
}

func (s *Spec) normalize(fallbackName string) error {
	s.Name = strings.TrimSpace(s.Name)
	if s.Name == "" {
		s.Name = strings.TrimSpace(fallbackName)
	}
	if s.Name == "" {
		return errors.New("skill name is required")
	}
	if !validSkillName.MatchString(s.Name) {
		return fmt.Errorf("invalid skill name %q: only letters, numbers, '.', '_' and '-' are allowed", s.Name)
	}

	s.Description = strings.TrimSpace(s.Description)
	s.Version = strings.TrimSpace(s.Version)
	s.Command = strings.TrimSpace(s.Command)
	if s.Command == "" {
		return errors.New("skill command is required")
	}

	s.IO.Inputs = normalizeStringList(s.IO.Inputs)
	s.IO.Outputs = normalizeStringList(s.IO.Outputs)

	requiredPerms, err := normalizePermissions(s.Permissions.Required)
	if err != nil {
		return err
	}
	s.Permissions.Required = requiredPerms

	return nil
}

func LoadSpec(path string) (Spec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Spec{}, err
	}

	var spec Spec
	if err := toml.Unmarshal(data, &spec); err != nil {
		return Spec{}, fmt.Errorf("parse %s: %w", path, err)
	}

	if err := spec.normalize(filepath.Base(filepath.Dir(path))); err != nil {
		return Spec{}, err
	}

	return spec, nil
}

func Install(source string) (Skill, error) {
	src := strings.TrimSpace(source)
	if src == "" {
		return Skill{}, errors.New("source path is required")
	}

	absSource, err := filepath.Abs(src)
	if err != nil {
		return Skill{}, err
	}

	stat, err := os.Stat(absSource)
	if err != nil {
		if isGitLikeSource(src) {
			return Skill{}, errors.New("skill install currently supports local directories only")
		}
		return Skill{}, err
	}
	if !stat.IsDir() {
		return Skill{}, fmt.Errorf("source %q is not a directory", source)
	}

	specPath := filepath.Join(absSource, skillManifestFile)
	spec, err := LoadSpec(specPath)
	if err != nil {
		return Skill{}, err
	}

	if err := ensureLayout(); err != nil {
		return Skill{}, err
	}

	root, err := installedDir()
	if err != nil {
		return Skill{}, err
	}

	tmpDir, err := os.MkdirTemp(root, spec.Name+"-tmp-")
	if err != nil {
		return Skill{}, err
	}
	defer os.RemoveAll(tmpDir)

	if err := copyDir(absSource, tmpDir); err != nil {
		return Skill{}, err
	}

	copiedSpecPath := filepath.Join(tmpDir, skillManifestFile)
	copiedSpec, err := LoadSpec(copiedSpecPath)
	if err != nil {
		return Skill{}, err
	}

	dest := filepath.Join(root, copiedSpec.Name)
	backup := ""
	if _, err := os.Stat(dest); err == nil {
		backup = filepath.Join(root, copiedSpec.Name+".backup")
		for i := 0; ; i++ {
			candidate := backup
			if i > 0 {
				candidate = fmt.Sprintf("%s-%d", backup, i)
			}

			if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
				backup = candidate
				break
			}
		}

		if err := os.Rename(dest, backup); err != nil {
			return Skill{}, fmt.Errorf("prepare install swap: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return Skill{}, err
	}

	if err := os.Rename(tmpDir, dest); err != nil {
		if backup != "" {
			_ = os.Rename(backup, dest)
		}
		return Skill{}, fmt.Errorf("finalize install swap: %w", err)
	}

	if backup != "" {
		if err := os.RemoveAll(backup); err != nil {
			return Skill{}, fmt.Errorf("cleanup previous install backup: %w", err)
		}
	}

	state, err := readEnabledState()
	if err != nil {
		return Skill{}, err
	}

	return Skill{
		Spec:    copiedSpec,
		Dir:     dest,
		Enabled: state[copiedSpec.Name],
	}, nil
}

func List() ([]Skill, error) {
	if err := ensureLayout(); err != nil {
		return nil, err
	}

	root, err := installedDir()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	state, err := readEnabledState()
	if err != nil {
		return nil, err
	}

	out := make([]Skill, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		dir := filepath.Join(root, entry.Name())
		spec, err := LoadSpec(filepath.Join(dir, skillManifestFile))
		if err != nil {
			return nil, err
		}

		out = append(out, Skill{
			Spec:    spec,
			Dir:     dir,
			Enabled: state[spec.Name],
		})
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Spec.Name < out[j].Spec.Name
	})

	return out, nil
}

func Enabled() ([]Skill, error) {
	all, err := List()
	if err != nil {
		return nil, err
	}

	enabled := all[:0]
	for _, skill := range all {
		if skill.Enabled {
			enabled = append(enabled, skill)
		}
	}

	return slices.Clip(enabled), nil
}

func Enable(name string) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	state, err := readEnabledState()
	if err != nil {
		return err
	}

	state[skill.Spec.Name] = true
	return writeEnabledState(state)
}

func Disable(name string) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	state, err := readEnabledState()
	if err != nil {
		return err
	}

	delete(state, skill.Spec.Name)
	return writeEnabledState(state)
}

func Get(name string) (Skill, error) {
	n := strings.TrimSpace(name)
	if n == "" {
		return Skill{}, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}

	all, err := List()
	if err != nil {
		return Skill{}, err
	}

	for _, skill := range all {
		if skill.Spec.Name == n {
			return skill, nil
		}
	}

	return Skill{}, fmt.Errorf("%w: %s", ErrSkillNotFound, n)
}

func Run(ctx context.Context, name string, args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	return RunWithOptions(ctx, name, args, stdin, stdout, stderr, RunOptions{})
}

func RunWithOptions(ctx context.Context, name string, args []string, stdin io.Reader, stdout, stderr io.Writer, opts RunOptions) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	if !skill.Enabled {
		return fmt.Errorf("%w: %s", ErrSkillNotEnabled, skill.Spec.Name)
	}

	grantedPerms, err := grantedPermissions(opts.GrantedPermissions)
	if err != nil {
		return err
	}
	if err := ensurePermissionsGranted(skill.Spec.Permissions.Required, grantedPerms); err != nil {
		return err
	}

	command := resolveCommand(skill.Dir, skill.Spec.Command)
	cmd := exec.CommandContext(ctx, command, append(skill.Spec.Args, args...)...)
	cmd.Dir = skill.Dir
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	return cmd.Run()
}

func resolveCommand(skillDir, command string) string {
	if command == "" || filepath.IsAbs(command) {
		return command
	}

	if strings.Contains(command, string(filepath.Separator)) ||
		strings.HasPrefix(command, ".") {
		return filepath.Join(skillDir, command)
	}

	return command
}

func isGitLikeSource(source string) bool {
	s := strings.ToLower(strings.TrimSpace(source))
	return strings.HasPrefix(s, "http://") ||
		strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "git@") ||
		strings.HasSuffix(s, ".git")
}

func normalizeStringList(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		v := strings.TrimSpace(value)
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizePermissions(values []string) ([]string, error) {
	return normalizePermissionsInternal(values, false)
}

func normalizeGrantedPermissions(values []string) ([]string, error) {
	return normalizePermissionsInternal(values, true)
}

func normalizePermissionsInternal(values []string, allowWildcard bool) ([]string, error) {
	values = normalizeStringList(values)
	for _, permission := range values {
		if allowWildcard && permission == "*" {
			continue
		}
		if !validPermission.MatchString(permission) {
			return nil, fmt.Errorf("invalid permission %q", permission)
		}
	}
	return values, nil
}

func grantedPermissions(overrides []string) ([]string, error) {
	base := normalizeStringList(overrides)
	if len(base) == 0 {
		base = splitCommaList(os.Getenv(envSkillAllow))
	}
	return normalizeGrantedPermissions(base)
}

func splitCommaList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}

	parts := strings.Split(raw, ",")
	return normalizeStringList(parts)
}

func ensurePermissionsGranted(required, granted []string) error {
	if len(required) == 0 {
		return nil
	}

	grantedSet := map[string]bool{}
	hasWildcard := false
	for _, permission := range granted {
		grantedSet[permission] = true
		if permission == "*" {
			hasWildcard = true
		}
	}

	if hasWildcard {
		return nil
	}

	var missing []string
	for _, permission := range required {
		if !grantedSet[permission] {
			missing = append(missing, permission)
		}
	}

	if len(missing) == 0 {
		return nil
	}

	return fmt.Errorf("%w: missing required permissions: %s (grant with --allow or %s)", ErrPermissionDenied, strings.Join(missing, ", "), envSkillAllow)
}

func readEnabledState() (map[string]bool, error) {
	if err := ensureLayout(); err != nil {
		return nil, err
	}

	path, err := enabledPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]bool{}, nil
	}
	if err != nil {
		return nil, err
	}

	state := map[string]bool{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}

	return state, nil
}

func writeEnabledState(state map[string]bool) error {
	if err := ensureLayout(); err != nil {
		return err
	}

	path, err := enabledPath()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	tmpFile, err := os.CreateTemp(filepath.Dir(path), "enabled-*.json")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpPath, path)
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}

		if d.IsDir() {
			// Avoid copying Git metadata into installed skills.
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}

		if !d.Type().IsRegular() {
			return nil
		}

		srcInfo, err := d.Info()
		if err != nil {
			return err
		}

		return copyFile(path, filepath.Join(dst, rel), srcInfo.Mode())
	})
}

func copyFile(src, dst string, mode fs.FileMode) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode.Perm())
	if err != nil {
		return err
	}
	defer dstFile.Close()

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		return err
	}

	return nil
}
