package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
)

type pullRequestFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
	Changes   int    `json:"changes"`
	Patch     string `json:"patch"`
}

type reviewCorpus struct {
	base   string
	head   string
	files  map[string]pullRequestFile
	lines  map[string]map[int]string
	budget toolBudget
}

type toolBudget struct {
	mu    sync.Mutex
	calls int
}

const maxToolCalls = 100

func newReviewCorpus(base, head string, files []pullRequestFile) *reviewCorpus {
	c := &reviewCorpus{base: base, head: head, files: make(map[string]pullRequestFile), lines: make(map[string]map[int]string)}
	changedLines := 0
	patchBytes := 0
	for _, file := range files {
		if !eligibleFile(file) || len(c.files) >= 25 || changedLines+file.Changes > 1000 || patchBytes+len(file.Patch) > 120<<10 {
			continue
		}
		c.files[file.Filename] = file
		c.lines[file.Filename] = changedRightLines(file.Patch)
		changedLines += file.Changes
		patchBytes += len(file.Patch)
	}
	return c
}

func (c *reviewCorpus) hasEligibleFiles() bool { return len(c.files) > 0 }

func (c *reviewCorpus) toolOutput(content string) (string, error) {
	c.budget.mu.Lock()
	defer c.budget.mu.Unlock()
	if c.budget.calls >= maxToolCalls {
		return "", fmt.Errorf("review tool call budget reached")
	}
	c.budget.calls++
	return toolCallReceipt(c.budget.calls) + "\n\n" + content, nil
}

func toolCallReceipt(calls int) string {
	return fmt.Sprintf("[CI review tool calls: %d/%d used; %d remaining. Call final_review when ready.]", calls, maxToolCalls, maxToolCalls-calls)
}

func eligibleFile(file pullRequestFile) bool {
	if file.Filename == "" || file.Patch == "" || file.Changes == 0 || file.Changes > 1000 {
		return false
	}
	name := strings.ToLower(file.Filename)
	if strings.HasPrefix(name, "docs/") || strings.HasSuffix(name, ".md") || strings.HasPrefix(name, "vendor/") || strings.Contains(name, "/testdata/") || strings.HasSuffix(name, ".lock") {
		return false
	}
	return !strings.Contains(name, "generated") && !strings.HasSuffix(name, ".png") && !strings.HasSuffix(name, ".jpg") && !strings.HasSuffix(name, ".pdf")
}

var hunkHeader = regexp.MustCompile(`^@@ -(?:\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

func changedRightLines(patch string) map[int]string {
	lines := make(map[int]string)
	lineNo := 0
	for _, line := range strings.Split(patch, "\n") {
		if match := hunkHeader.FindStringSubmatch(line); len(match) == 2 {
			fmt.Sscanf(match[1], "%d", &lineNo)
			continue
		}
		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			lines[lineNo] = strings.TrimPrefix(line, "+")
			lineNo++
			continue
		}
		if !strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "\\") {
			lineNo++
		}
	}
	return lines
}

type review struct {
	Summary  string    `json:"summary"`
	Findings []finding `json:"findings"`
}

type finding struct {
	Category       string  `json:"category"`
	Severity       string  `json:"severity"`
	Confidence     float64 `json:"confidence"`
	Path           string  `json:"path"`
	Line           int     `json:"line"`
	Title          string  `json:"title"`
	Impact         string  `json:"impact"`
	Recommendation string  `json:"recommendation"`
	Test           string  `json:"test"`
	Evidence       string  `json:"evidence"`
}

const maxSummaryBytes = 1200

func parseReview(content string, corpus *reviewCorpus) (review, error) {
	var out review
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &out); err != nil {
		return review{}, err
	}
	out.Summary = strings.TrimSpace(out.Summary)
	if out.Summary == "" || len(out.Summary) > maxSummaryBytes || containsSensitiveText(out.Summary) {
		return review{}, fmt.Errorf("review summary is missing, oversized, or sensitive")
	}
	supplied := len(out.Findings)
	if supplied > maxFindings {
		out.Findings = out.Findings[:maxFindings]
	}
	seen := make(map[string]bool)
	valid := make([]finding, 0, len(out.Findings))
	invalid := make([]string, 0, len(out.Findings))
	for index, finding := range out.Findings {
		if err := validateFinding(finding, corpus); err != nil {
			invalid = append(invalid, fmt.Sprintf("finding %d: %v", index+1, err))
			continue
		}
		key := findingKey(finding)
		if seen[key] {
			invalid = append(invalid, fmt.Sprintf("finding %d: duplicate", index+1))
			continue
		}
		seen[key] = true
		valid = append(valid, finding)
	}
	out.Findings = valid
	if supplied > 0 && len(valid) == 0 {
		return review{}, fmt.Errorf("no findings passed validation (%s); submit final_review again with only findings at confidence >= 0.85 that quote an added changed line, or use findings: []", strings.Join(invalid, "; "))
	}
	return out, nil
}

func validateFinding(f finding, corpus *reviewCorpus) error {
	if f.Confidence < .85 || !oneOf(f.Category, "ux", "efficiency", "go", "security", "correctness") || !oneOf(f.Severity, "low", "medium", "high", "critical") {
		return fmt.Errorf("invalid category, severity, or confidence")
	}
	if !cleanPath(f.Path) || f.Line < 1 || !boundedText(f.Title, 160) || !boundedText(f.Impact, 500) || !boundedText(f.Recommendation, 500) || !boundedText(f.Test, 300) || !boundedText(f.Evidence, 300) {
		return fmt.Errorf("finding fields are incomplete")
	}
	changed, ok := corpus.lines[f.Path]
	if !ok || !strings.Contains(changed[f.Line], strings.TrimSpace(f.Evidence)) {
		return fmt.Errorf("finding is not grounded in a changed line")
	}
	if containsSensitiveText(f.Title) || containsSensitiveText(f.Evidence) || containsSensitiveText(f.Impact) || containsSensitiveText(f.Recommendation) || containsSensitiveText(f.Test) {
		return fmt.Errorf("finding contains sensitive text")
	}
	return nil
}

func boundedText(value string, limit int) bool {
	return strings.TrimSpace(value) != "" && len(value) <= limit
}

func oneOf(value string, values ...string) bool {
	for _, allowed := range values {
		if value == allowed {
			return true
		}
	}
	return false
}

func cleanPath(value string) bool {
	return value != "" && !strings.ContainsAny(value, "\\\x00:") && !strings.HasPrefix(value, "-") && !strings.HasPrefix(value, ".git/") && !strings.HasPrefix(value, "/") && path.Clean(value) == value && value != "." && !strings.HasPrefix(value, "../")
}

func containsSensitiveText(value string) bool {
	lower := strings.ToLower(value)
	return strings.Contains(lower, "authorization: bearer") || strings.Contains(lower, "ollama_api_key") || strings.Contains(lower, "private key")
}

func findingKey(f finding) string {
	sum := sha256.Sum256([]byte(strings.Join([]string{f.Path, fmt.Sprint(f.Line), f.Category, strings.ToLower(f.Title)}, "\x00")))
	return fmt.Sprintf("%x", sum[:])
}

func score(r review) int {
	value := 5
	for _, finding := range r.Findings {
		switch finding.Severity {
		case "critical":
			value = min(value, 1)
		case "high":
			value = min(value, 2)
		case "medium":
			value = min(value, 3)
		case "low":
			value = min(value, 4)
		}
	}
	return value
}

func summary(r review, head string) string {
	return fmt.Sprintf("Reviewed `%s` with %s: **%d/5** risk score and %d validated finding(s).", shortSHA(head), reviewModelDisplay, score(r), len(r.Findings))
}

func scoreComment(r review, head string) string {
	return fmt.Sprintf("<!-- ollama-ai-review -->\n## Ollama AI review: %d/5\n\n%s\n\nReviewed head `%s` across UX, efficiency, Go cleanliness, security, and correctness. This score covers only validated findings in the changed code; it does not replace human review.", score(r), r.Summary, head)
}

func dryRunDraft(r review, head string) map[string]any {
	comments := make([]map[string]any, 0, len(r.Findings))
	for _, finding := range r.Findings {
		comments = append(comments, map[string]any{
			"path": finding.Path,
			"line": finding.Line,
			"side": "RIGHT",
			"body": findingBody(finding),
		})
	}
	return map[string]any{
		"head":            head,
		"score":           score(r),
		"summary":         summary(r, head),
		"sticky_comment":  scoreComment(r, head),
		"inline_comments": comments,
	}
}

func shortSHA(sha string) string {
	if len(sha) > 7 {
		return sha[:7]
	}
	return sha
}

func sortedPaths(c *reviewCorpus) []string {
	paths := make([]string, 0, len(c.files))
	for p := range c.files {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}
