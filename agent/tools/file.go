package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	maxReadBytes   = 200000
	maxListEntries = 300
)

type Read struct{}

func NewRead() *Read {
	return &Read{}
}

func (r *Read) Name() string {
	return "read"
}

func (r *Read) Description() string {
	return "Read a text file from the current working directory."
}

func (r *Read) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Path to the file to read, relative to the working directory.",
	})
	return api.ToolFunction{
		Name:        r.Name(),
		Description: r.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"path"},
		},
	}
}

func (r *Read) Execute(ctx context.Context, toolCtx agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || strings.TrimSpace(path) == "" {
		return agent.ToolResult{}, fmt.Errorf("path parameter is required")
	}

	resolved, err := resolvePath(toolCtx.WorkingDir, path)
	if err != nil {
		return agent.ToolResult{}, err
	}

	info, err := os.Stat(resolved)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if info.IsDir() {
		return agent.ToolResult{}, fmt.Errorf("%s is a directory", path)
	}
	if info.Size() > maxReadBytes {
		return agent.ToolResult{}, fmt.Errorf("%s is too large to read (%d bytes)", path, info.Size())
	}

	select {
	case <-ctx.Done():
		return agent.ToolResult{}, ctx.Err()
	default:
	}

	content, err := os.ReadFile(resolved)
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: string(content)}, nil
}

type List struct{}

func NewList() *List {
	return &List{}
}

func (l *List) Name() string {
	return "list"
}

func (l *List) Description() string {
	return "List files and directories in the current working directory."
}

func (l *List) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Directory path to list, relative to the working directory. Defaults to the working directory.",
	})
	return api.ToolFunction{
		Name:        l.Name(),
		Description: l.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
		},
	}
}

func (l *List) Execute(ctx context.Context, toolCtx agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	path := "."
	if raw, ok := args["path"].(string); ok && strings.TrimSpace(raw) != "" {
		path = raw
	}

	resolved, err := resolvePath(toolCtx.WorkingDir, path)
	if err != nil {
		return agent.ToolResult{}, err
	}

	entries, err := os.ReadDir(resolved)
	if err != nil {
		return agent.ToolResult{}, err
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir()
		}
		return entries[i].Name() < entries[j].Name()
	})

	var sb strings.Builder
	count := len(entries)
	limit := count
	if limit > maxListEntries {
		limit = maxListEntries
	}
	for i := 0; i < limit; i++ {
		select {
		case <-ctx.Done():
			return agent.ToolResult{}, ctx.Err()
		default:
		}

		name := entries[i].Name()
		if entries[i].IsDir() {
			name += string(os.PathSeparator)
		}
		sb.WriteString(name)
		sb.WriteByte('\n')
	}
	if count > limit {
		sb.WriteString(fmt.Sprintf("... (%d entries omitted)\n", count-limit))
	}
	if sb.Len() == 0 {
		return agent.ToolResult{Content: "(empty directory)"}, nil
	}
	return agent.ToolResult{Content: sb.String()}, nil
}

type Edit struct{}

func NewEdit() *Edit {
	return &Edit{}
}

func (e *Edit) Name() string {
	return "edit"
}

func (e *Edit) Description() string {
	return "Edit a text file in the current working directory by replacing exact text."
}

func (e *Edit) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Path to the file to edit, relative to the working directory.",
	})
	props.Set("old_text", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Exact text to replace.",
	})
	props.Set("new_text", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Replacement text.",
	})
	props.Set("replace_all", api.ToolProperty{
		Type:        api.PropertyType{"boolean"},
		Description: "Replace every occurrence. Defaults to false and requires old_text to match exactly once.",
	})
	return api.ToolFunction{
		Name:        e.Name(),
		Description: e.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"path", "old_text", "new_text"},
		},
	}
}

func (e *Edit) RequiresApproval(map[string]any) bool {
	return true
}

func (e *Edit) Execute(ctx context.Context, toolCtx agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || strings.TrimSpace(path) == "" {
		return agent.ToolResult{}, fmt.Errorf("path parameter is required")
	}

	oldText, ok := args["old_text"].(string)
	if !ok || oldText == "" {
		return agent.ToolResult{}, fmt.Errorf("old_text parameter is required")
	}

	newText, ok := args["new_text"].(string)
	if !ok {
		return agent.ToolResult{}, fmt.Errorf("new_text parameter is required")
	}

	replaceAll, _ := args["replace_all"].(bool)

	resolved, err := resolvePath(toolCtx.WorkingDir, path)
	if err != nil {
		return agent.ToolResult{}, err
	}

	info, err := os.Stat(resolved)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if info.IsDir() {
		return agent.ToolResult{}, fmt.Errorf("%s is a directory", path)
	}
	if info.Size() > maxReadBytes {
		return agent.ToolResult{}, fmt.Errorf("%s is too large to edit (%d bytes)", path, info.Size())
	}

	select {
	case <-ctx.Done():
		return agent.ToolResult{}, ctx.Err()
	default:
	}

	contentBytes, err := os.ReadFile(resolved)
	if err != nil {
		return agent.ToolResult{}, err
	}
	content := string(contentBytes)
	matches := strings.Count(content, oldText)
	if matches == 0 {
		return agent.ToolResult{}, fmt.Errorf("old_text was not found in %s", path)
	}
	if matches > 1 && !replaceAll {
		return agent.ToolResult{}, fmt.Errorf("old_text matched %d times in %s; set replace_all to true to replace every match", matches, path)
	}

	var updated string
	if replaceAll {
		updated = strings.ReplaceAll(content, oldText, newText)
	} else {
		updated = strings.Replace(content, oldText, newText, 1)
	}
	if len(updated) > maxReadBytes {
		return agent.ToolResult{}, fmt.Errorf("edited content is too large (%d bytes)", len(updated))
	}

	if err := os.WriteFile(resolved, []byte(updated), info.Mode().Perm()); err != nil {
		return agent.ToolResult{}, err
	}

	return agent.ToolResult{Content: fmt.Sprintf("Updated %s (%d replacement%s).", path, matches, plural(matches))}, nil
}

func resolvePath(workingDir, path string) (string, error) {
	base := workingDir
	if base == "" {
		var err error
		base, err = os.Getwd()
		if err != nil {
			return "", err
		}
	}

	if filepath.IsAbs(path) {
		return "", fmt.Errorf("absolute paths are not allowed")
	}

	baseAbs, err := canonicalPath(base)
	if err != nil {
		return "", err
	}
	resolved := filepath.Clean(filepath.Join(baseAbs, path))
	resolvedForCheck := resolved
	if canonical, err := canonicalPath(resolved); err == nil {
		resolvedForCheck = canonical
	}
	rel, err := filepath.Rel(baseAbs, resolvedForCheck)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes working directory")
	}

	return resolved, nil
}

func canonicalPath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err == nil {
		return resolved, nil
	}
	return abs, nil
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
