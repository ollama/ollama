package tools

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	maxReadBytes = 200000
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
	props.Set("start", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "Optional 1-based line to start reading from.",
	})
	props.Set("end", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "Optional 1-based inclusive line to stop reading at.",
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

	file, info, err := openRegularFile(toolCtx.WorkingDir, path)
	if err != nil {
		return agent.ToolResult{}, err
	}
	defer file.Close()

	selection, err := readSelectionFromArgs(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if !selection.enabled && info.Size() > maxReadBytes {
		return agent.ToolResult{}, fmt.Errorf("%s is too large to read (%d bytes)", path, info.Size())
	}

	select {
	case <-ctx.Done():
		return agent.ToolResult{}, ctx.Err()
	default:
	}

	var content string
	if selection.enabled {
		content, err = readLineSelection(file, selection)
	} else {
		var contentBytes []byte
		contentBytes, err = io.ReadAll(file)
		content = string(contentBytes)
	}
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: content}, nil
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

	if err := rejectFinalSymlink(toolCtx.WorkingDir, path); err != nil {
		return agent.ToolResult{}, err
	}

	file, info, err := openRegularFile(toolCtx.WorkingDir, path)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if info.Size() > maxReadBytes {
		file.Close()
		return agent.ToolResult{}, fmt.Errorf("%s is too large to edit (%d bytes)", path, info.Size())
	}

	select {
	case <-ctx.Done():
		file.Close()
		return agent.ToolResult{}, ctx.Err()
	default:
	}

	contentBytes, err := io.ReadAll(file)
	if closeErr := file.Close(); err == nil && closeErr != nil {
		err = closeErr
	}
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

	if err := writeFileAtomic(toolCtx.WorkingDir, path, []byte(updated), info.Mode().Perm()); err != nil {
		return agent.ToolResult{}, err
	}

	return agent.ToolResult{Content: fmt.Sprintf("Updated %s (%d replacement%s).", path, matches, plural(matches))}, nil
}

func cleanRelativePath(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("path parameter is required")
	}
	if filepath.IsAbs(path) {
		return "", fmt.Errorf("absolute paths are not allowed")
	}
	cleaned := filepath.Clean(path)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes working directory")
	}
	return cleaned, nil
}

func openRegularFile(workingDir, path string) (*os.File, os.FileInfo, error) {
	rel, err := cleanRelativePath(path)
	if err != nil {
		return nil, nil, err
	}
	root, err := openWorkingRoot(workingDir)
	if err != nil {
		return nil, nil, err
	}
	defer root.Close()

	file, err := root.Open(rel)
	if err != nil {
		return nil, nil, rootPathError(err)
	}
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, err
	}
	if info.IsDir() {
		file.Close()
		return nil, nil, fmt.Errorf("%s is a directory", path)
	}
	return file, info, nil
}

func writeFileAtomic(workingDir, path string, data []byte, perm os.FileMode) error {
	rel, err := cleanRelativePath(path)
	if err != nil {
		return err
	}
	root, err := openWorkingRoot(workingDir)
	if err != nil {
		return err
	}
	defer root.Close()
	if err := rejectRootFinalSymlink(root, rel, path); err != nil {
		return err
	}

	parent, name := filepath.Split(rel)
	tmpBase := fmt.Sprintf(".%s.ollama-tmp-%d", name, os.Getpid())
	for i := 0; ; i++ {
		candidateName := tmpBase
		if i > 0 {
			candidateName = fmt.Sprintf("%s-%d", tmpBase, i)
		}
		candidate := filepath.Join(parent, candidateName)
		file, err := root.OpenFile(candidate, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
		if os.IsExist(err) {
			continue
		}
		if err != nil {
			return rootPathError(err)
		}
		writeErr := writeAllAndSync(file, data)
		closeErr := file.Close()
		if writeErr != nil || closeErr != nil {
			_ = root.Remove(candidate)
			if writeErr != nil {
				return writeErr
			}
			return closeErr
		}
		if err := root.Rename(candidate, rel); err != nil {
			_ = root.Remove(candidate)
			return rootPathError(err)
		}
		return nil
	}
}

func rejectFinalSymlink(workingDir, path string) error {
	rel, err := cleanRelativePath(path)
	if err != nil {
		return err
	}
	root, err := openWorkingRoot(workingDir)
	if err != nil {
		return err
	}
	defer root.Close()
	return rejectRootFinalSymlink(root, rel, path)
}

func rejectRootFinalSymlink(root *os.Root, rel, path string) error {
	info, err := root.Lstat(rel)
	if err != nil {
		return rootPathError(err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%s is a symlink; edit the target file directly", path)
	}
	return nil
}

func rootPathError(err error) error {
	if err != nil && strings.Contains(err.Error(), "path escapes") {
		return fmt.Errorf("path escapes working directory")
	}
	return err
}

func openWorkingRoot(workingDir string) (*os.Root, error) {
	base, err := workingDirAbs(workingDir)
	if err != nil {
		return nil, err
	}
	return os.OpenRoot(base)
}

func writeAllAndSync(file *os.File, data []byte) error {
	if _, err := file.Write(data); err != nil {
		return err
	}
	return file.Sync()
}

func workingDirAbs(workingDir string) (string, error) {
	base := workingDir
	if base == "" {
		var err error
		base, err = os.Getwd()
		if err != nil {
			return "", err
		}
	}
	return canonicalPath(base)
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

type readSelection struct {
	enabled bool
	start   int
	end     int
}

func readSelectionFromArgs(args map[string]any) (readSelection, error) {
	selection := readSelection{start: 1}

	if start, ok, err := intReadArg(args, "start"); err != nil {
		return readSelection{}, err
	} else if ok {
		selection.enabled = true
		selection.start = start
	}
	if end, ok, err := intReadArg(args, "end"); err != nil {
		return readSelection{}, err
	} else if ok {
		selection.enabled = true
		selection.end = end
	}

	if !selection.enabled {
		return selection, nil
	}
	if selection.start < 1 {
		return readSelection{}, fmt.Errorf("start must be greater than 0")
	}
	if selection.end > 0 && selection.end < selection.start {
		return readSelection{}, fmt.Errorf("end must be greater than or equal to start")
	}
	return selection, nil
}

func readLineSelection(file *os.File, selection readSelection) (string, error) {
	reader := bufio.NewReader(file)
	var b strings.Builder
	for lineNo := 1; ; lineNo++ {
		line, err := reader.ReadString('\n')
		if lineNo >= selection.start && (selection.end == 0 || lineNo <= selection.end) {
			if b.Len()+len(line) > maxReadBytes {
				return "", fmt.Errorf("selected content is too large (%d byte limit)", maxReadBytes)
			}
			b.WriteString(line)
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		if selection.end > 0 && lineNo >= selection.end {
			break
		}
	}
	return b.String(), nil
}

func intReadArg(args map[string]any, key string) (int, bool, error) {
	value, ok := args[key]
	if !ok {
		return 0, false, nil
	}
	switch v := value.(type) {
	case int:
		return v, true, nil
	case int64:
		return int(v), true, nil
	case float64:
		if v != float64(int(v)) {
			return 0, true, fmt.Errorf("%s must be a whole number", key)
		}
		return int(v), true, nil
	case string:
		v = strings.TrimSpace(v)
		if v == "" {
			return 0, false, nil
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return 0, true, fmt.Errorf("%s must be a whole number", key)
		}
		return n, true, nil
	default:
		return 0, true, fmt.Errorf("%s must be a whole number", key)
	}
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
