package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	maxReadLines = 400
	maxReadBytes = 48 << 10
	maxSearchHit = 50
	maxSearchOut = 24 << 10
)

type finalReviewTool struct {
	corpus *reviewCorpus

	mu        sync.Mutex
	review    review
	submitted bool
}

func (t *finalReviewTool) Name() string { return "final_review" }
func (t *finalReviewTool) Description() string {
	return "Submit the completed review exactly once. Call this only after inspecting the needed diff and context."
}

func (t *finalReviewTool) Schema() api.ToolFunction {
	findings := api.NewToolPropertiesMap()
	findings.Set("category", api.ToolProperty{Type: api.PropertyType{"string"}, Enum: []any{"ux", "efficiency", "go", "security", "correctness"}})
	findings.Set("severity", api.ToolProperty{Type: api.PropertyType{"string"}, Enum: []any{"low", "medium", "high", "critical"}})
	findings.Set("confidence", api.ToolProperty{Type: api.PropertyType{"number"}})
	findings.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	findings.Set("line", api.ToolProperty{Type: api.PropertyType{"integer"}})
	findings.Set("title", api.ToolProperty{Type: api.PropertyType{"string"}})
	findings.Set("impact", api.ToolProperty{Type: api.PropertyType{"string"}})
	findings.Set("recommendation", api.ToolProperty{Type: api.PropertyType{"string"}})
	findings.Set("test", api.ToolProperty{Type: api.PropertyType{"string"}})
	findings.Set("evidence", api.ToolProperty{Type: api.PropertyType{"string"}})

	props := api.NewToolPropertiesMap()
	props.Set("summary", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("findings", api.ToolProperty{
		Type: api.PropertyType{"array"},
		Items: api.ToolProperty{
			Type:       api.PropertyType{"object"},
			Properties: findings,
			Required:   []string{"category", "severity", "confidence", "path", "line", "title", "impact", "recommendation", "test", "evidence"},
		},
	})
	return api.ToolFunction{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"summary", "findings"},
		},
	}
}

func (t *finalReviewTool) Execute(_ context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	receipt, err := t.corpus.toolOutput("")
	if err != nil {
		return agent.ToolResult{}, err
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("%s final_review rejected: encode final review: %w", receipt, err)
	}
	result, err := parseReview(string(encoded), t.corpus)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("%s final_review rejected: %w", receipt, err)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.submitted {
		return agent.ToolResult{}, fmt.Errorf("%s final_review rejected: final review was already submitted", receipt)
	}
	t.review = result
	t.submitted = true
	return agent.ToolResult{Content: receipt + "\n\nFinal review recorded. Do not call more tools; finish now."}, nil
}

func (t *finalReviewTool) result() (review, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.review, t.submitted
}

type reviewDiffTool struct{ corpus *reviewCorpus }

func (t *reviewDiffTool) Name() string { return "review_diff" }
func (t *reviewDiffTool) Description() string {
	return "Read the pull request change manifest or a patch for one changed file."
}

func (t *reviewDiffTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "Optional changed path."})
	return api.ToolFunction{Name: t.Name(), Description: t.Description(), Parameters: api.ToolFunctionParameters{Type: "object", Properties: props}}
}

func (t *reviewDiffTool) Execute(_ context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	requested, _ := args["path"].(string)
	if requested == "" {
		content := "Changed files:\n" + strings.Join(sortedPaths(t.corpus), "\n")
		content, err := t.corpus.toolOutput(content)
		if err != nil {
			return agent.ToolResult{}, err
		}
		return agent.ToolResult{Content: content}, nil
	}
	file, ok := t.corpus.files[requested]
	if !ok {
		return agent.ToolResult{}, fmt.Errorf("%s is not an eligible changed file", requested)
	}
	content, err := t.corpus.toolOutput(file.Patch)
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: content}, nil
}

type readFileTool struct{ corpus *reviewCorpus }

func (t *readFileTool) Name() string { return "read_file" }
func (t *readFileTool) Description() string {
	return "Read a bounded range from a tracked text file at the pull request head or base."
}

func (t *readFileTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("ref", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "head or base; defaults to head."})
	props.Set("start", api.ToolProperty{Type: api.PropertyType{"integer"}})
	props.Set("end", api.ToolProperty{Type: api.PropertyType{"integer"}})
	return api.ToolFunction{Name: t.Name(), Description: t.Description(), Parameters: api.ToolFunctionParameters{Type: "object", Properties: props, Required: []string{"path"}}}
}

func (t *readFileTool) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	path, _ := args["path"].(string)
	if !cleanPath(path) {
		return agent.ToolResult{}, fmt.Errorf("path must be a clean repository-relative path")
	}
	ref, _ := args["ref"].(string)
	sha := t.corpus.head
	if ref == "base" {
		sha = t.corpus.base
	} else if ref != "" && ref != "head" {
		return agent.ToolResult{}, fmt.Errorf("ref must be head or base")
	}
	if err := readableRegularTextFile(ctx, sha, path); err != nil {
		return agent.ToolResult{}, err
	}
	output, err := gitOutput(ctx, "show", "--no-textconv", sha+":"+path)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if strings.IndexByte(output, 0) >= 0 {
		return agent.ToolResult{}, fmt.Errorf("path is a binary file")
	}
	content, err := t.corpus.toolOutput(truncate(selectLines(output, args), maxReadBytes))
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: content}, nil
}

func readableRegularTextFile(ctx context.Context, sha, path string) error {
	tree, err := gitOutput(ctx, "ls-tree", sha, "--", path)
	if err != nil {
		return err
	}
	fields := strings.Fields(tree)
	if len(fields) < 3 || (fields[0] != "100644" && fields[0] != "100755") || fields[1] != "blob" {
		return fmt.Errorf("path is not a regular tracked file")
	}
	return nil
}

type searchTextTool struct{ corpus *reviewCorpus }

func (t *searchTextTool) Name() string { return "search_text" }
func (t *searchTextTool) Description() string {
	return "Search tracked text at the pull request head or base using a literal query."
}

func (t *searchTextTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("ref", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "head or base; defaults to head."})
	return api.ToolFunction{Name: t.Name(), Description: t.Description(), Parameters: api.ToolFunctionParameters{Type: "object", Properties: props, Required: []string{"query"}}}
}

func (t *searchTextTool) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	query, _ := args["query"].(string)
	if strings.TrimSpace(query) == "" || len(query) > 200 {
		return agent.ToolResult{}, fmt.Errorf("query must be between 1 and 200 bytes")
	}
	ref, _ := args["ref"].(string)
	sha := t.corpus.head
	if ref == "base" {
		sha = t.corpus.base
	} else if ref != "" && ref != "head" {
		return agent.ToolResult{}, fmt.Errorf("ref must be head or base")
	}
	output, err := gitOutput(ctx, "grep", "-n", "-I", "-F", "--max-count="+strconv.Itoa(maxSearchHit), "-e", query, sha, "--")
	if err != nil {
		if strings.Contains(err.Error(), "exit status 1") {
			content, err := t.corpus.toolOutput("No matches.")
			if err != nil {
				return agent.ToolResult{}, err
			}
			return agent.ToolResult{Content: content}, nil
		}
		return agent.ToolResult{}, err
	}
	content, err := t.corpus.toolOutput(truncate(searchResults(output), maxSearchOut))
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: content}, nil
}

func searchResults(output string) string {
	lines := strings.Split(strings.TrimSuffix(output, "\n"), "\n")
	if len(lines) > maxSearchHit {
		lines = lines[:maxSearchHit]
	}
	return strings.Join(lines, "\n")
}

func gitOutput(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Env = append(cmd.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", args[0], err)
	}
	return string(output), nil
}

func selectLines(content string, args map[string]any) string {
	start := intArg(args["start"], 1)
	end := intArg(args["end"], start+maxReadLines-1)
	if start < 1 || end < start || end-start >= maxReadLines {
		return "Invalid requested line range."
	}
	lines := strings.Split(content, "\n")
	if start > len(lines) {
		return ""
	}
	if end > len(lines) {
		end = len(lines)
	}
	return strings.Join(lines[start-1:end], "\n")
}

func intArg(value any, fallback int) int {
	if value, ok := value.(float64); ok {
		return int(value)
	}
	return fallback
}

func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "\n[truncated]"
}
