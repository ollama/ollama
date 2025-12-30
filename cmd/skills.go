package cmd

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server"
)

const (
	skillFileName       = "SKILL.md"
	maxSkillDescription = 1024
	maxSkillNameLength  = 64
)

var skillNamePattern = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

type skillMetadata struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

type skillDefinition struct {
	Name        string
	Description string
	Content     string // Full SKILL.md content (without frontmatter)
	Dir         string
	SkillPath   string
}

type skillCatalog struct {
	Skills []skillDefinition
	byName map[string]skillDefinition
}

func loadSkills(paths []string) (*skillCatalog, error) {
	if len(paths) == 0 {
		return nil, nil
	}

	var skills []skillDefinition
	byName := make(map[string]skillDefinition)
	for _, root := range paths {
		info, err := os.Stat(root)
		if err != nil {
			return nil, fmt.Errorf("skills directory %q: %w", root, err)
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("skills path %q is not a directory", root)
		}

		err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				return nil
			}
			if entry.Name() != skillFileName {
				return nil
			}

			skillDir := filepath.Dir(path)
			skill, err := parseSkillFile(path, skillDir)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: skipping skill at %s: %v\n", path, err)
				return nil
			}

			if _, exists := byName[skill.Name]; exists {
				fmt.Fprintf(os.Stderr, "Warning: duplicate skill name %q at %s\n", skill.Name, path)
				return nil
			}

			byName[skill.Name] = skill
			skills = append(skills, skill)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}

	if len(skills) == 0 {
		return nil, nil
	}

	sort.Slice(skills, func(i, j int) bool {
		return skills[i].Name < skills[j].Name
	})

	return &skillCatalog{Skills: skills, byName: byName}, nil
}

// loadSkillsFromRefs loads skills from a list of SkillRef objects.
// Skills can be referenced by:
//   - Digest: loaded from the extracted skill cache (for bundled/pulled skills)
//   - Name (local path): loaded from the filesystem (for development)
func loadSkillsFromRefs(refs []api.SkillRef) (*skillCatalog, error) {
	if len(refs) == 0 {
		return nil, nil
	}

	var skills []skillDefinition
	byName := make(map[string]skillDefinition)

	for _, ref := range refs {
		var skillDir string

		if ref.Digest != "" {
			// Load from extracted skill cache
			path, err := server.GetSkillsPath(ref.Digest)
			if err != nil {
				return nil, fmt.Errorf("getting skill path for %s: %w", ref.Digest, err)
			}

			// Check if skill is already extracted
			skillMdPath := filepath.Join(path, skillFileName)
			if _, err := os.Stat(skillMdPath); os.IsNotExist(err) {
				// Try to extract the skill blob
				path, err = server.ExtractSkillBlob(ref.Digest)
				if err != nil {
					return nil, fmt.Errorf("extracting skill %s: %w", ref.Digest, err)
				}
			}

			skillDir = path
		} else if ref.Name != "" {
			// Check if this is a local path or a registry reference
			if !server.IsLocalSkillPath(ref.Name) {
				// Registry reference without a digest - skill needs to be pulled first
				// This happens when an agent references a skill that hasn't been bundled
				return nil, fmt.Errorf("skill %q is a registry reference but has no digest - the agent may need to be recreated or the skill pulled separately", ref.Name)
			}

			// Local path - resolve it
			skillPath := ref.Name
			if strings.HasPrefix(skillPath, "~") {
				home, err := os.UserHomeDir()
				if err != nil {
					return nil, fmt.Errorf("expanding home directory: %w", err)
				}
				skillPath = filepath.Join(home, skillPath[1:])
			}

			absPath, err := filepath.Abs(skillPath)
			if err != nil {
				return nil, fmt.Errorf("resolving skill path %q: %w", ref.Name, err)
			}

			// Check if this is a directory containing skills or a single skill
			info, err := os.Stat(absPath)
			if err != nil {
				return nil, fmt.Errorf("skill path %q: %w", ref.Name, err)
			}

			if info.IsDir() {
				// Check if it's a skill directory (has SKILL.md) or a parent of skill directories
				skillMdPath := filepath.Join(absPath, skillFileName)
				if _, err := os.Stat(skillMdPath); err == nil {
					// Direct skill directory
					skillDir = absPath
				} else {
					// Parent directory - walk to find skill subdirectories
					err := filepath.WalkDir(absPath, func(path string, entry fs.DirEntry, walkErr error) error {
						if walkErr != nil {
							return walkErr
						}
						if entry.IsDir() {
							return nil
						}
						if entry.Name() != skillFileName {
							return nil
						}

						skillSubDir := filepath.Dir(path)
						skill, err := parseSkillFile(path, skillSubDir)
						if err != nil {
							fmt.Fprintf(os.Stderr, "Warning: skipping skill at %s: %v\n", path, err)
							return nil
						}

						if _, exists := byName[skill.Name]; exists {
							fmt.Fprintf(os.Stderr, "Warning: duplicate skill name %q at %s\n", skill.Name, path)
							return nil
						}

						byName[skill.Name] = skill
						skills = append(skills, skill)
						return nil
					})
					if err != nil {
						return nil, err
					}
					continue
				}
			} else {
				return nil, fmt.Errorf("skill path %q is not a directory", ref.Name)
			}
		} else {
			// Both empty - skip
			continue
		}

		// Parse the skill from skillDir if set
		if skillDir != "" {
			skillMdPath := filepath.Join(skillDir, skillFileName)
			skill, err := parseSkillFile(skillMdPath, skillDir)
			if err != nil {
				return nil, fmt.Errorf("parsing skill at %s: %w", skillDir, err)
			}

			if _, exists := byName[skill.Name]; exists {
				fmt.Fprintf(os.Stderr, "Warning: duplicate skill name %q\n", skill.Name)
				continue
			}

			byName[skill.Name] = skill
			skills = append(skills, skill)
		}
	}

	if len(skills) == 0 {
		return nil, nil
	}

	sort.Slice(skills, func(i, j int) bool {
		return skills[i].Name < skills[j].Name
	})

	return &skillCatalog{Skills: skills, byName: byName}, nil
}

func parseSkillFile(path, skillDir string) (skillDefinition, error) {
	rawContent, err := os.ReadFile(path)
	if err != nil {
		return skillDefinition{}, err
	}

	frontmatter, bodyContent, err := extractFrontmatterAndContent(string(rawContent))
	if err != nil {
		return skillDefinition{}, err
	}

	var meta skillMetadata
	if err := yaml.Unmarshal([]byte(frontmatter), &meta); err != nil {
		return skillDefinition{}, fmt.Errorf("invalid frontmatter: %w", err)
	}

	if err := validateSkillMetadata(meta, skillDir); err != nil {
		return skillDefinition{}, err
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return skillDefinition{}, err
	}
	absDir, err := filepath.Abs(skillDir)
	if err != nil {
		return skillDefinition{}, err
	}

	return skillDefinition{
		Name:        meta.Name,
		Description: meta.Description,
		Content:     bodyContent,
		Dir:         absDir,
		SkillPath:   absPath,
	}, nil
}

func extractFrontmatterAndContent(content string) (frontmatter string, body string, err error) {
	scanner := bufio.NewScanner(strings.NewReader(content))
	if !scanner.Scan() {
		return "", "", errors.New("empty SKILL.md")
	}
	if strings.TrimSpace(scanner.Text()) != "---" {
		return "", "", errors.New("missing YAML frontmatter")
	}

	var fmLines []string
	foundEnd := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "---" {
			foundEnd = true
			break
		}
		fmLines = append(fmLines, line)
	}
	if !foundEnd {
		return "", "", errors.New("frontmatter not terminated")
	}

	// Collect remaining content as body
	var bodyLines []string
	for scanner.Scan() {
		bodyLines = append(bodyLines, scanner.Text())
	}

	return strings.Join(fmLines, "\n"), strings.TrimSpace(strings.Join(bodyLines, "\n")), nil
}

func validateSkillMetadata(meta skillMetadata, skillDir string) error {
	name := strings.TrimSpace(meta.Name)
	description := strings.TrimSpace(meta.Description)

	switch {
	case name == "":
		return errors.New("missing skill name")
	case len(name) > maxSkillNameLength:
		return fmt.Errorf("skill name exceeds %d characters", maxSkillNameLength)
	case !skillNamePattern.MatchString(name):
		return fmt.Errorf("invalid skill name %q", name)
	}

	if description == "" {
		return errors.New("missing skill description")
	}
	if len(description) > maxSkillDescription {
		return fmt.Errorf("skill description exceeds %d characters", maxSkillDescription)
	}

	// Skip directory name check for digest-based paths (extracted from blobs)
	dirName := filepath.Base(skillDir)
	if !strings.HasPrefix(dirName, "sha256-") && dirName != name {
		return fmt.Errorf("skill directory %q does not match name %q", dirName, name)
	}

	return nil
}

func (c *skillCatalog) SystemPrompt() string {
	if c == nil || len(c.Skills) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("# Skills\n\n")
	b.WriteString("You have the following skills loaded. Each skill provides instructions and may include executable scripts.\n\n")
	b.WriteString("## Available Tools\n\n")
	b.WriteString("- `run_skill_script`: Execute a script bundled with a skill. Use this when the skill instructions tell you to run a script.\n")
	b.WriteString("- `read_skill_file`: Read additional files from a skill directory.\n\n")

	for _, skill := range c.Skills {
		fmt.Fprintf(&b, "## Skill: %s\n\n", skill.Name)
		fmt.Fprintf(&b, "%s\n\n", skill.Content)
		b.WriteString("---\n\n")
	}

	return b.String()
}

func (c *skillCatalog) Tools() api.Tools {
	if c == nil || len(c.Skills) == 0 {
		return nil
	}

	return api.Tools{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "run_skill_script",
				Description: "Execute a script or command within a skill's directory. Use this to run Python scripts, shell scripts, or other executables bundled with a skill.",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"skill", "command"},
					Properties: map[string]api.ToolProperty{
						"skill": {
							Type:        api.PropertyType{"string"},
							Description: "The name of the skill containing the script",
						},
						"command": {
							Type:        api.PropertyType{"string"},
							Description: "The command to execute (e.g., 'python scripts/calculate.py 25 4' or './scripts/run.sh')",
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "read_skill_file",
				Description: "Read a file from a skill's directory. Use this to read additional documentation, reference files, or data files bundled with a skill.",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"skill", "path"},
					Properties: map[string]api.ToolProperty{
						"skill": {
							Type:        api.PropertyType{"string"},
							Description: "The name of the skill containing the file",
						},
						"path": {
							Type:        api.PropertyType{"string"},
							Description: "The relative path to the file within the skill directory",
						},
					},
				},
			},
		},
	}
}

func (c *skillCatalog) RunToolCall(call api.ToolCall) (api.Message, bool, error) {
	switch call.Function.Name {
	case "read_skill_file":
		skillName, err := requireStringArg(call.Function.Arguments, "skill")
		if err != nil {
			return toolMessage(call, err.Error()), true, nil
		}
		relPath, err := requireStringArg(call.Function.Arguments, "path")
		if err != nil {
			return toolMessage(call, err.Error()), true, nil
		}
		skill, ok := c.byName[skillName]
		if !ok {
			return toolMessage(call, fmt.Sprintf("unknown skill %q", skillName)), true, nil
		}
		content, err := readSkillFile(skill.Dir, relPath)
		if err != nil {
			return toolMessage(call, err.Error()), true, nil
		}
		return toolMessage(call, content), true, nil

	case "run_skill_script":
		skillName, err := requireStringArg(call.Function.Arguments, "skill")
		if err != nil {
			return toolMessage(call, err.Error()), true, nil
		}
		command, err := requireStringArg(call.Function.Arguments, "command")
		if err != nil {
			return toolMessage(call, err.Error()), true, nil
		}
		skill, ok := c.byName[skillName]
		if !ok {
			return toolMessage(call, fmt.Sprintf("unknown skill %q", skillName)), true, nil
		}
		output, err := runSkillScript(skill.Dir, command)
		if err != nil {
			return toolMessage(call, fmt.Sprintf("error: %v\noutput: %s", err, output)), true, nil
		}
		return toolMessage(call, output), true, nil

	default:
		return api.Message{}, false, nil
	}
}

// runSkillScript executes a shell command within a skill's directory.
//
// SECURITY LIMITATIONS (TODO):
//   - No sandboxing: commands run with full user permissions
//   - No path validation: model can run any command, not just scripts in skill dir
//   - Shell injection risk: sh -c is used, malicious input could be crafted
//   - No executable allowlist: any program can be called (curl, rm, etc.)
//   - No environment isolation: scripts inherit full environment variables
//
// POTENTIAL IMPROVEMENTS:
//   - Restrict commands to only reference files within skill directory
//   - Allowlist specific executables (python3, node, bash)
//   - Use sandboxing (Docker, nsjail, seccomp)
//   - Require explicit script registration in SKILL.md frontmatter
//   - Add per-skill configurable timeouts
func runSkillScript(skillDir, command string) (string, error) {
	// Validate the skill directory exists
	absSkillDir, err := filepath.Abs(skillDir)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(absSkillDir); err != nil {
		return "", fmt.Errorf("skill directory not found: %w", err)
	}

	// Create command with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = absSkillDir

	// Inject the current working directory (where ollama run was called from)
	// as an environment variable so scripts can reference files in that directory
	workingDir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}
	cmd.Env = append(os.Environ(), "OLLAMA_WORKING_DIR="+workingDir)

	// Capture both stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()

	// Combine output
	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += stderr.String()
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return output, fmt.Errorf("command timed out after 30 seconds")
		}
		return output, err
	}

	return output, nil
}

func readSkillFile(skillDir, relPath string) (string, error) {
	relPath = filepath.Clean(strings.TrimSpace(relPath))
	if relPath == "" {
		return "", errors.New("path is required")
	}
	if filepath.IsAbs(relPath) {
		return "", errors.New("path must be relative to the skill directory")
	}

	target := filepath.Join(skillDir, relPath)
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", err
	}
	absSkillDir, err := filepath.Abs(skillDir)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(absSkillDir, absTarget)
	if err != nil {
		return "", err
	}
	if strings.HasPrefix(rel, "..") {
		return "", errors.New("path escapes the skill directory")
	}

	content, err := os.ReadFile(absTarget)
	if err != nil {
		return "", fmt.Errorf("failed to read %q: %w", relPath, err)
	}

	return string(content), nil
}

func requireStringArg(args api.ToolCallFunctionArguments, name string) (string, error) {
	value, ok := args[name]
	if !ok {
		return "", fmt.Errorf("missing required argument %q", name)
	}
	str, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("argument %q must be a string", name)
	}
	if strings.TrimSpace(str) == "" {
		return "", fmt.Errorf("argument %q cannot be empty", name)
	}
	return str, nil
}

func toolMessage(call api.ToolCall, content string) api.Message {
	msg := api.Message{
		Role:     "tool",
		Content:  content,
		ToolName: call.Function.Name,
	}
	if call.ID != "" {
		msg.ToolCallID = call.ID
	}
	return msg
}
