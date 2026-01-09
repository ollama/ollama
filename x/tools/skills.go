// Package tools provides built-in tool implementations for the agent loop.
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// SkillSpec defines the specification for a custom skill.
// Skills can be loaded from JSON files and registered with the tool registry.
type SkillSpec struct {
	// Name is the unique identifier for the skill
	Name string `json:"name"`
	// Description is a human-readable description shown to the LLM
	Description string `json:"description"`
	// Parameters defines the input schema for the skill
	Parameters []SkillParameter `json:"parameters,omitempty"`
	// Executor defines how the skill is executed
	Executor SkillExecutor `json:"executor"`
}

// SkillParameter defines a single parameter for a skill.
type SkillParameter struct {
	// Name is the parameter name
	Name string `json:"name"`
	// Type is the JSON schema type (string, number, boolean, array, object)
	Type string `json:"type"`
	// Description explains what this parameter is for
	Description string `json:"description"`
	// Required indicates if this parameter must be provided
	Required bool `json:"required"`
}

// SkillExecutor defines how to execute a skill.
type SkillExecutor struct {
	// Type is the executor type: "script", "http", or "builtin"
	Type string `json:"type"`
	// Command is the command to run for "script" type
	// Arguments are passed as JSON via stdin, result is read from stdout
	Command string `json:"command,omitempty"`
	// Args are additional arguments appended to the command
	Args []string `json:"args,omitempty"`
	// Timeout is the maximum execution time in seconds (default: 60)
	Timeout int `json:"timeout,omitempty"`
	// URL is the endpoint for "http" type executors
	URL string `json:"url,omitempty"`
	// Method is the HTTP method (default: POST)
	Method string `json:"method,omitempty"`
}

// SkillsFile represents a file containing skill definitions.
type SkillsFile struct {
	// Version is the spec version (currently "1")
	Version string `json:"version"`
	// Skills is the list of skill definitions
	Skills []SkillSpec `json:"skills"`
}

// SkillTool wraps a SkillSpec to implement the Tool interface.
type SkillTool struct {
	spec SkillSpec
}

// NewSkillTool creates a Tool from a SkillSpec.
func NewSkillTool(spec SkillSpec) *SkillTool {
	return &SkillTool{spec: spec}
}

// Name returns the skill name.
func (s *SkillTool) Name() string {
	return s.spec.Name
}

// Description returns the skill description.
func (s *SkillTool) Description() string {
	return s.spec.Description
}

// Schema returns the tool's parameter schema for the LLM.
func (s *SkillTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	var required []string

	for _, param := range s.spec.Parameters {
		props.Set(param.Name, api.ToolProperty{
			Type:        api.PropertyType{param.Type},
			Description: param.Description,
		})
		if param.Required {
			required = append(required, param.Name)
		}
	}

	return api.ToolFunction{
		Name:        s.spec.Name,
		Description: s.spec.Description,
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   required,
		},
	}
}

// Execute runs the skill with the given arguments.
func (s *SkillTool) Execute(args map[string]any) (string, error) {
	switch s.spec.Executor.Type {
	case "script":
		return s.executeScript(args)
	case "http":
		return s.executeHTTP(args)
	default:
		return "", fmt.Errorf("unknown executor type: %s", s.spec.Executor.Type)
	}
}

// executeScript runs a script-based skill.
func (s *SkillTool) executeScript(args map[string]any) (string, error) {
	if s.spec.Executor.Command == "" {
		return "", fmt.Errorf("script executor requires command")
	}

	timeout := time.Duration(s.spec.Executor.Timeout) * time.Second
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Build command
	cmdArgs := append([]string{}, s.spec.Executor.Args...)
	cmd := exec.CommandContext(ctx, s.spec.Executor.Command, cmdArgs...)

	// Pass arguments as JSON via stdin
	inputJSON, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("marshaling arguments: %w", err)
	}
	cmd.Stdin = bytes.NewReader(inputJSON)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()

	// Build output
	var sb strings.Builder
	if stdout.Len() > 0 {
		output := stdout.String()
		if len(output) > maxOutputSize {
			output = output[:maxOutputSize] + "\n... (output truncated)"
		}
		sb.WriteString(output)
	}

	if stderr.Len() > 0 {
		stderrOutput := stderr.String()
		if len(stderrOutput) > maxOutputSize {
			stderrOutput = stderrOutput[:maxOutputSize] + "\n... (stderr truncated)"
		}
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("stderr:\n")
		sb.WriteString(stderrOutput)
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return sb.String() + fmt.Sprintf("\n\nError: command timed out after %d seconds", s.spec.Executor.Timeout), nil
		}
		if exitErr, ok := err.(*exec.ExitError); ok {
			return sb.String() + fmt.Sprintf("\n\nExit code: %d", exitErr.ExitCode()), nil
		}
		return sb.String(), fmt.Errorf("executing skill: %w", err)
	}

	if sb.Len() == 0 {
		return "(no output)", nil
	}

	return sb.String(), nil
}

// executeHTTP runs an HTTP-based skill.
func (s *SkillTool) executeHTTP(args map[string]any) (string, error) {
	// HTTP executor is a placeholder for future implementation
	return "", fmt.Errorf("http executor not yet implemented")
}

// LoadSkillsFile loads skill definitions from a JSON file.
func LoadSkillsFile(path string) (*SkillsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading skills file: %w", err)
	}

	var file SkillsFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parsing skills file: %w", err)
	}

	if file.Version == "" {
		file.Version = "1"
	}

	return &file, nil
}

// RegisterSkillsFromFile loads skills from a file and registers them with the registry.
func RegisterSkillsFromFile(registry *Registry, path string) error {
	file, err := LoadSkillsFile(path)
	if err != nil {
		return err
	}

	for _, spec := range file.Skills {
		if err := validateSkillSpec(spec); err != nil {
			return fmt.Errorf("invalid skill %q: %w", spec.Name, err)
		}
		registry.Register(NewSkillTool(spec))
	}

	return nil
}

// FindSkillsFiles searches for skill definition files in standard locations.
// It looks for:
// - ./ollama-skills.json (current directory)
// - ~/.ollama/skills.json (user config)
// - ~/.config/ollama/skills.json (XDG config)
func FindSkillsFiles() []string {
	var files []string

	// Current directory
	if _, err := os.Stat("ollama-skills.json"); err == nil {
		files = append(files, "ollama-skills.json")
	}

	// Home directory
	home, err := os.UserHomeDir()
	if err == nil {
		paths := []string{
			filepath.Join(home, ".ollama", "skills.json"),
			filepath.Join(home, ".config", "ollama", "skills.json"),
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				files = append(files, p)
			}
		}
	}

	return files
}

// LoadAllSkills loads skills from all discovered skill files into the registry.
func LoadAllSkills(registry *Registry) ([]string, error) {
	files := FindSkillsFiles()
	var loaded []string

	for _, path := range files {
		if err := RegisterSkillsFromFile(registry, path); err != nil {
			return loaded, fmt.Errorf("loading %s: %w", path, err)
		}
		loaded = append(loaded, path)
	}

	return loaded, nil
}

// validateSkillSpec validates a skill specification.
func validateSkillSpec(spec SkillSpec) error {
	if spec.Name == "" {
		return fmt.Errorf("name is required")
	}
	if spec.Description == "" {
		return fmt.Errorf("description is required")
	}
	if spec.Executor.Type == "" {
		return fmt.Errorf("executor.type is required")
	}

	switch spec.Executor.Type {
	case "script":
		if spec.Executor.Command == "" {
			return fmt.Errorf("executor.command is required for script type")
		}
	case "http":
		if spec.Executor.URL == "" {
			return fmt.Errorf("executor.url is required for http type")
		}
	default:
		return fmt.Errorf("unknown executor type: %s", spec.Executor.Type)
	}

	for _, param := range spec.Parameters {
		if param.Name == "" {
			return fmt.Errorf("parameter name is required")
		}
		if param.Type == "" {
			return fmt.Errorf("parameter type is required for %s", param.Name)
		}
	}

	return nil
}
