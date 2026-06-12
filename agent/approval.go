package agent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"mvdan.cc/sh/v3/syntax"
)

type ApprovalDecision string

const (
	ApprovalAllowOnce    ApprovalDecision = "allow_once"
	ApprovalAllowSession ApprovalDecision = "allow_session"
	ApprovalDeny         ApprovalDecision = "deny"
)

type ApprovalRisk string

const (
	ApprovalRiskLow    ApprovalRisk = "low"
	ApprovalRiskMedium ApprovalRisk = "medium"
	ApprovalRiskHigh   ApprovalRisk = "high"
)

type ApprovalRequest struct {
	ToolCallID string
	ToolName   string
	Args       map[string]any
	WorkingDir string
	Summary    string
	Risk       ApprovalRisk
	Reasons    []string
}

type ApprovalResult struct {
	Decision ApprovalDecision
	Reason   string
}

type ApprovalHandler interface {
	RequiresApproval(context.Context, Tool, ApprovalRequest) bool
	Approve(context.Context, ApprovalRequest) (ApprovalResult, error)
}

type ApprovalPrompter interface {
	PromptApproval(context.Context, ApprovalRequest) (ApprovalResult, error)
}

type ApprovalPolicy interface {
	EvaluateApproval(context.Context, ApprovalRequest) ApprovalEvaluation
}

type ApprovalEvaluation struct {
	Decision      ApprovalDecision
	RequirePrompt bool
	Risk          ApprovalRisk
	Summary       string
	Reasons       []string
	SessionKey    string
}

type AutoAllowApproval struct{}

func (AutoAllowApproval) RequiresApproval(context.Context, Tool, ApprovalRequest) bool {
	return false
}

func (AutoAllowApproval) Approve(context.Context, ApprovalRequest) (ApprovalResult, error) {
	return ApprovalResult{Decision: ApprovalAllowOnce}, nil
}

type ApprovalManagerOptions struct {
	Policy      ApprovalPolicy
	Prompter    ApprovalPrompter
	AutoApprove bool
}

type ApprovalManager struct {
	policy      ApprovalPolicy
	prompter    ApprovalPrompter
	autoApprove bool

	mu             *sync.Mutex
	sessionAllowed map[string]struct{}
}

func NewApprovalManager(opts ApprovalManagerOptions) *ApprovalManager {
	policy := opts.Policy
	if policy == nil {
		policy = DefaultApprovalPolicy{}
	}
	return &ApprovalManager{
		policy:         policy,
		prompter:       opts.Prompter,
		autoApprove:    opts.AutoApprove,
		mu:             &sync.Mutex{},
		sessionAllowed: make(map[string]struct{}),
	}
}

func (m *ApprovalManager) WithPrompter(prompter ApprovalPrompter) *ApprovalManager {
	if m == nil {
		return NewApprovalManager(ApprovalManagerOptions{Prompter: prompter})
	}
	return &ApprovalManager{
		policy:         m.policy,
		prompter:       prompter,
		autoApprove:    m.autoApprove,
		mu:             m.mu,
		sessionAllowed: m.sessionAllowed,
	}
}

func (m *ApprovalManager) AutoApproveEnabled() bool {
	return m != nil && m.autoApprove
}

func (m *ApprovalManager) RequiresApproval(ctx context.Context, tool Tool, req ApprovalRequest) bool {
	if m == nil || m.autoApprove {
		return false
	}
	evaluation := m.evaluate(ctx, req)
	if ToolRequiresApproval(tool, req.Args) && evaluation.Decision != ApprovalDeny {
		evaluation.RequirePrompt = true
		if evaluation.Summary == "" {
			evaluation.Summary = fmt.Sprintf("%s wants to run", toolApprovalDisplayName(req.ToolName))
		}
		if len(evaluation.Reasons) == 0 {
			evaluation.Reasons = []string{"tool requires approval"}
		}
	}
	if evaluation.Decision == ApprovalDeny {
		return true
	}
	if evaluation.RequirePrompt {
		return !m.sessionAllowedFor(evaluation.SessionKey)
	}
	return false
}

func (m *ApprovalManager) Approve(ctx context.Context, req ApprovalRequest) (ApprovalResult, error) {
	if m == nil || m.autoApprove {
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}

	evaluation := m.evaluate(ctx, req)
	req = approvalRequestWithEvaluation(req, evaluation)

	if evaluation.Decision == ApprovalDeny {
		reason := strings.Join(evaluation.Reasons, "; ")
		if reason == "" {
			reason = "Tool execution denied."
		}
		return ApprovalResult{Decision: ApprovalDeny, Reason: reason}, nil
	}

	if !evaluation.RequirePrompt || m.sessionAllowedFor(evaluation.SessionKey) {
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}

	if m.prompter == nil {
		return ApprovalResult{
			Decision: ApprovalDeny,
			Reason:   "Tool execution requires approval, but no approval prompter is available. Re-run with --auto-approve-tools to allow tool execution.",
		}, nil
	}

	result, err := m.prompter.PromptApproval(ctx, req)
	if err != nil {
		return ApprovalResult{}, err
	}
	if result.Decision == "" {
		result.Decision = ApprovalDeny
	}
	if result.Decision == ApprovalAllowSession {
		m.allowSession(evaluation.SessionKey)
	}
	return result, nil
}

func (m *ApprovalManager) evaluate(ctx context.Context, req ApprovalRequest) ApprovalEvaluation {
	if m == nil || m.policy == nil {
		return DefaultApprovalPolicy{}.EvaluateApproval(ctx, req)
	}
	evaluation := m.policy.EvaluateApproval(ctx, req)
	if evaluation.Risk == "" {
		evaluation.Risk = ApprovalRiskLow
	}
	if evaluation.SessionKey == "" {
		evaluation.SessionKey = approvalSessionKey(req)
	}
	return evaluation
}

func (m *ApprovalManager) sessionAllowedFor(key string) bool {
	if m == nil || key == "" {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.sessionAllowed[key]
	return ok
}

func (m *ApprovalManager) allowSession(key string) {
	if m == nil || key == "" {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessionAllowed[key] = struct{}{}
}

func approvalRequestWithEvaluation(req ApprovalRequest, evaluation ApprovalEvaluation) ApprovalRequest {
	req.Summary = evaluation.Summary
	req.Risk = evaluation.Risk
	req.Reasons = slices.Clone(evaluation.Reasons)
	return req
}

func approvalSessionKey(req ApprovalRequest) string {
	switch req.ToolName {
	case "bash":
		if command, ok := stringApprovalArg(req.Args, "command"); ok {
			return "bash:" + command
		}
	case "edit":
		if path, ok := stringApprovalArg(req.Args, "path"); ok {
			return "edit:" + path
		}
	case "web_search":
		if query, ok := stringApprovalArg(req.Args, "query"); ok {
			return "web_search:" + query
		}
	case "web_fetch":
		if targetURL, ok := stringApprovalArg(req.Args, "url"); ok {
			return "web_fetch:" + targetURL
		}
	}
	return req.ToolName + ":" + stableApprovalArgs(req.Args)
}

type DefaultApprovalPolicy struct{}

func (DefaultApprovalPolicy) EvaluateApproval(_ context.Context, req ApprovalRequest) ApprovalEvaluation {
	switch req.ToolName {
	case "read":
		if path, ok := stringApprovalArg(req.Args, "path"); ok {
			if reason := approvalPathEscapeReason(req.WorkingDir, path); reason != "" {
				return denyApproval(req.ToolName, ApprovalRiskHigh, reason)
			}
		}
		return ApprovalEvaluation{Decision: ApprovalAllowOnce, Risk: ApprovalRiskLow, Summary: fmt.Sprintf("%s can run without approval", toolApprovalDisplayName(req.ToolName))}
	case "web_search", "web_fetch":
		return ApprovalEvaluation{Decision: ApprovalAllowOnce, Risk: ApprovalRiskLow, Summary: fmt.Sprintf("%s can run without approval", toolApprovalDisplayName(req.ToolName))}
	case "edit":
		return evaluateEditApproval(req)
	case "bash":
		return evaluateBashApproval(req)
	default:
		return ApprovalEvaluation{
			RequirePrompt: true,
			Risk:          ApprovalRiskMedium,
			Summary:       fmt.Sprintf("%s wants to run", toolApprovalDisplayName(req.ToolName)),
			Reasons:       []string{"unknown tool effects"},
			SessionKey:    approvalSessionKey(req),
		}
	}
}

func evaluateEditApproval(req ApprovalRequest) ApprovalEvaluation {
	path, ok := stringApprovalArg(req.Args, "path")
	if !ok {
		return denyApproval("edit", ApprovalRiskHigh, "missing path argument")
	}
	if reason := approvalPathEscapeReason(req.WorkingDir, path); reason != "" {
		return denyApproval("edit", ApprovalRiskHigh, reason)
	}

	reasons := []string{"writes to a file"}
	if replaceAll, _ := req.Args["replace_all"].(bool); replaceAll {
		reasons = append(reasons, "may replace multiple matches")
	}
	return ApprovalEvaluation{
		RequirePrompt: true,
		Risk:          ApprovalRiskMedium,
		Summary:       fmt.Sprintf("Edit wants to modify %s", path),
		Reasons:       reasons,
		SessionKey:    "edit:" + path,
	}
}

func evaluateBashApproval(req ApprovalRequest) ApprovalEvaluation {
	command, ok := stringApprovalArg(req.Args, "command")
	if !ok || strings.TrimSpace(command) == "" {
		return denyApproval("bash", ApprovalRiskHigh, "missing command argument")
	}

	risk, reasons := classifyBashCommand(command)
	if len(reasons) == 0 {
		reasons = append(reasons, "runs shell commands")
	}
	return ApprovalEvaluation{
		RequirePrompt: true,
		Risk:          risk,
		Summary:       "Bash wants to run a command",
		Reasons:       reasons,
		SessionKey:    "bash:" + command,
	}
}

func denyApproval(toolName string, risk ApprovalRisk, reason string) ApprovalEvaluation {
	return ApprovalEvaluation{
		Decision:   ApprovalDeny,
		Risk:       risk,
		Summary:    fmt.Sprintf("%s cannot run", toolApprovalDisplayName(toolName)),
		Reasons:    []string{reason},
		SessionKey: approvalSessionKey(ApprovalRequest{ToolName: toolName}),
	}
}

// Bash approval is static analysis of model-generated commands, not a sandbox.
// Globs, aliases, environment, and runtime shell state are intentionally out of scope.
func classifyBashCommand(command string) (ApprovalRisk, []string) {
	parser := syntax.NewParser(syntax.Variant(syntax.LangBash))
	file, err := parser.Parse(strings.NewReader(command), "")
	if err != nil {
		return ApprovalRiskHigh, []string{"could not parse shell command"}
	}

	classifier := bashClassifier{risk: ApprovalRiskMedium}
	if len(file.Stmts) > 1 {
		classifier.addReason("runs multiple shell statements")
	}

	syntax.Walk(file, func(node syntax.Node) bool {
		switch n := node.(type) {
		case *syntax.BinaryCmd:
			classifier.addReason("uses shell control operator " + n.Op.String())
			classifier.high = true
		case *syntax.Stmt:
			if n.Semicolon.IsValid() {
				classifier.addReason("uses shell statement separator")
				classifier.high = true
			}
			if n.Background {
				classifier.addReason("runs a command in the background")
				classifier.high = true
			}
			if len(n.Redirs) > 0 {
				for _, redir := range n.Redirs {
					classifier.addRedirect(redir)
				}
			}
		case *syntax.CmdSubst:
			classifier.addReason("uses command substitution")
			classifier.high = true
		case *syntax.ProcSubst:
			classifier.addReason("uses process substitution")
			classifier.high = true
		case *syntax.Subshell:
			classifier.addReason("uses a subshell")
			classifier.high = true
		case *syntax.FuncDecl:
			classifier.addReason("defines shell functions")
			classifier.high = true
		case *syntax.CallExpr:
			classifier.addCall(n)
		}
		return true
	})

	if classifier.high {
		classifier.risk = ApprovalRiskHigh
	}
	return classifier.risk, classifier.reasons
}

type bashClassifier struct {
	risk    ApprovalRisk
	high    bool
	reasons []string
}

func (c *bashClassifier) addReason(reason string) {
	if reason == "" || slices.Contains(c.reasons, reason) {
		return
	}
	c.reasons = append(c.reasons, reason)
}

func (c *bashClassifier) addRedirect(redir *syntax.Redirect) {
	if redir == nil {
		return
	}
	op := redir.Op.String()
	if strings.Contains(op, ">") {
		c.addReason("writes or redirects files")
		c.high = true
		return
	}
	c.addReason("uses shell redirection")
}

func (c *bashClassifier) addCall(call *syntax.CallExpr) {
	if call == nil || len(call.Args) == 0 {
		return
	}
	if wordHasCommandNameExpansion(call.Args[0]) {
		c.addReason("uses dynamic command name")
		c.high = true
	}
	args := literalWords(call.Args)
	if len(args) == 0 {
		return
	}
	name := args[0]
	switch name {
	case "cd":
		c.addReason("changes directory")
		c.high = true
	case "eval":
		c.addReason("evaluates shell code")
		c.high = true
	case "source", ".":
		c.addReason("sources shell code")
		c.high = true
	case "exec":
		c.addReason("replaces the shell process")
		c.high = true
	case "sudo":
		c.addReason("runs with elevated privileges")
		c.high = true
	case "rm":
		if hasAnyFlag(args[1:], "r", "R", "recursive") || hasAnyFlag(args[1:], "f", "force") {
			c.addReason("removes files destructively")
			c.high = true
		}
	case "git":
		if isGitResetHard(args) {
			c.addReason("runs destructive git reset")
			c.high = true
		}
		if isGitCleanDestructive(args) {
			c.addReason("runs destructive git clean")
			c.high = true
		}
	case "chmod", "chown":
		if hasAnyFlag(args[1:], "R", "recursive") {
			c.addReason("changes permissions or ownership recursively")
			c.high = true
		}
	case "dd", "diskutil":
		c.addReason("can write directly to disks")
		c.high = true
	case "mkfs", "newfs":
		c.addReason("formats filesystems")
		c.high = true
	case "curl", "wget":
		c.addReason("downloads remote content")
	case "sh", "bash", "zsh":
		if len(args) > 1 {
			c.addReason("runs a shell interpreter")
			c.high = true
		}
	}
}

func wordHasCommandNameExpansion(word *syntax.Word) bool {
	if word == nil {
		return false
	}
	for _, part := range word.Parts {
		switch part.(type) {
		case *syntax.Lit:
			continue
		default:
			return true
		}
	}
	return false
}

func literalWords(words []*syntax.Word) []string {
	args := make([]string, 0, len(words))
	for _, word := range words {
		if word == nil {
			continue
		}
		args = append(args, word.Lit())
	}
	return args
}

func hasAnyFlag(args []string, flags ...string) bool {
	for _, arg := range args {
		if arg == "--" {
			return false
		}
		if !strings.HasPrefix(arg, "-") {
			continue
		}
		for _, flag := range flags {
			short := len(flag) == 1
			if short && strings.HasPrefix(arg, "-") && !strings.HasPrefix(arg, "--") && strings.Contains(arg[1:], flag) {
				return true
			}
			if arg == "--"+flag {
				return true
			}
		}
	}
	return false
}

func isGitResetHard(args []string) bool {
	return len(args) >= 3 && args[1] == "reset" && slices.Contains(args[2:], "--hard")
}

func isGitCleanDestructive(args []string) bool {
	if len(args) < 2 || args[1] != "clean" {
		return false
	}
	return hasAnyFlag(args[2:], "f", "force") && (hasAnyFlag(args[2:], "d") || hasAnyFlag(args[2:], "x", "X"))
}

func approvalPathEscapeReason(workingDir, path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return "absolute paths are not allowed"
	}

	base := workingDir
	if base == "" {
		var err error
		base, err = os.Getwd()
		if err != nil {
			return "could not determine working directory"
		}
	}

	base, err := canonicalApprovalPath(base)
	if err != nil {
		return "could not resolve working directory"
	}
	resolved := filepath.Clean(filepath.Join(base, path))
	resolvedForCheck := resolved
	if canonical, err := canonicalApprovalPath(resolved); err == nil {
		resolvedForCheck = canonical
	}
	rel, err := filepath.Rel(base, resolvedForCheck)
	if err != nil {
		return "could not resolve path"
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "path escapes working directory"
	}
	return ""
}

func canonicalApprovalPath(path string) (string, error) {
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

func stringApprovalArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	return value, ok && strings.TrimSpace(value) != ""
}

func stableApprovalArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	var b bytes.Buffer
	for _, key := range keys {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%v", key, args[key])
	}
	return b.String()
}

func toolApprovalDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
	case "read":
		return "Read"
	case "edit":
		return "Edit"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}
