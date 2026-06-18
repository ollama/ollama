package agent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
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
			Reason:   "Tool execution requires approval, but no approval prompter is available. Re-run with --auto-approve-tools or --yolo to allow tool execution.",
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
	classifier := bashClassifier{risk: ApprovalRiskMedium}

	tokens, scannerReasons, scannerHigh := scanBashTokens(command)
	for _, reason := range scannerReasons {
		classifier.addReason(reason)
	}
	if scannerHigh {
		classifier.high = true
	}

	if bashFunctionDeclPattern.MatchString(command) {
		classifier.addReason("defines shell functions")
		classifier.high = true
	}
	if bashSubshellPattern.MatchString(command) {
		classifier.addReason("uses a subshell")
		classifier.high = true
	}

	for _, token := range tokens {
		if token.kind != bashTokenOperator {
			continue
		}
		switch token.value {
		case "&&", "||", "|":
			classifier.addReason("uses shell control operator " + token.value)
			classifier.high = true
		case ";", "\n":
			classifier.addReason("uses shell statement separator")
			classifier.high = true
		case "&":
			classifier.addReason("runs a command in the background")
			classifier.high = true
		case "$(", "`":
			classifier.addReason("uses command substitution")
			classifier.high = true
		case "<(", ">(":
			classifier.addReason("uses process substitution")
			classifier.high = true
		case ">", ">>", ">|", ">&", "&>":
			classifier.addReason("writes or redirects files")
			classifier.high = true
		case "<", "<<", "<&":
			classifier.addReason("uses shell redirection")
		}
	}

	for _, args := range bashCommandCalls(tokens) {
		classifier.addCall(args)
	}

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

func (c *bashClassifier) addCall(args []string) {
	args = shellCommandArgs(args)
	if len(args) == 0 {
		return
	}
	if isDynamicCommandName(args[0]) {
		c.addReason("uses dynamic command name")
		c.high = true
	}
	name := shellCommandBase(args[0])
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
	case "find":
		c.addFindReasons(args)
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

func (c *bashClassifier) addFindReasons(args []string) {
	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "-delete":
			c.addReason("deletes files via find")
			c.high = true
		case "-exec", "-execdir":
			c.addReason("executes commands via find")
			c.high = true
			end := i + 1
			for end < len(args) && args[end] != ";" && args[end] != `\;` && args[end] != "+" {
				end++
			}
			if end > i+1 {
				c.addCall(args[i+1 : end])
			}
			i = end
		}
	}
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
	subcommand := gitSubcommandArgs(args)
	return len(subcommand) >= 2 && subcommand[0] == "reset" && slices.Contains(subcommand[1:], "--hard")
}

func isGitCleanDestructive(args []string) bool {
	subcommand := gitSubcommandArgs(args)
	if len(subcommand) < 1 || subcommand[0] != "clean" {
		return false
	}
	return hasAnyFlag(subcommand[1:], "f", "force") && (hasAnyFlag(subcommand[1:], "d") || hasAnyFlag(subcommand[1:], "x", "X"))
}

func gitSubcommandArgs(args []string) []string {
	for i := 1; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			return args[i+1:]
		}
		switch {
		case arg == "-C", arg == "-c", arg == "--git-dir", arg == "--work-tree":
			i++
			continue
		case strings.HasPrefix(arg, "-C"), strings.HasPrefix(arg, "-c"):
			continue
		case strings.HasPrefix(arg, "--git-dir="), strings.HasPrefix(arg, "--work-tree="):
			continue
		case strings.HasPrefix(arg, "-"):
			continue
		default:
			return args[i:]
		}
	}
	return nil
}

type bashTokenKind int

const (
	bashTokenWord bashTokenKind = iota
	bashTokenOperator
)

type bashToken struct {
	kind  bashTokenKind
	value string
}

var (
	bashFunctionDeclPattern = regexp.MustCompile(`(?m)(^|[;&|[:space:]])(?:function[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*(?:\(\)[[:space:]]*)?\{`)
	bashSubshellPattern     = regexp.MustCompile(`(?m)(^|[;&|[:space:]])\(`)
)

func scanBashTokens(command string) ([]bashToken, []string, bool) {
	var tokens []bashToken
	var reasons []string
	var word strings.Builder
	var quote byte
	escaped := false
	high := false

	flushWord := func() {
		if word.Len() == 0 {
			return
		}
		tokens = append(tokens, bashToken{kind: bashTokenWord, value: word.String()})
		word.Reset()
	}
	addOperator := func(op string) {
		tokens = append(tokens, bashToken{kind: bashTokenOperator, value: op})
	}

	for i := 0; i < len(command); i++ {
		ch := command[i]
		if escaped {
			word.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && quote != '\'' {
			escaped = true
			continue
		}

		if quote == '\'' {
			if ch == '\'' {
				quote = 0
			} else {
				word.WriteByte(ch)
			}
			continue
		}
		if quote == '"' {
			switch {
			case ch == '"':
				quote = 0
			case ch == '`':
				addOperator("`")
				word.WriteByte(ch)
			case ch == '$' && i+1 < len(command) && command[i+1] == '(':
				addOperator("$(")
				word.WriteString("$(")
				i++
			default:
				word.WriteByte(ch)
			}
			continue
		}

		if ch == '\'' || ch == '"' {
			quote = ch
			continue
		}
		if ch == '`' {
			addOperator("`")
			word.WriteByte(ch)
			continue
		}
		if ch == '$' && i+1 < len(command) && command[i+1] == '(' {
			addOperator("$(")
			word.WriteString("$(")
			i++
			continue
		}
		if (ch == '<' || ch == '>') && i+1 < len(command) && command[i+1] == '(' {
			flushWord()
			addOperator(string([]byte{ch, '('}))
			i++
			continue
		}
		if ch == '\n' {
			flushWord()
			addOperator("\n")
			continue
		}
		if ch == ' ' || ch == '\t' || ch == '\r' {
			flushWord()
			continue
		}

		switch ch {
		case '&':
			flushWord()
			switch {
			case i+1 < len(command) && command[i+1] == '&':
				addOperator("&&")
				i++
			case i+1 < len(command) && command[i+1] == '>':
				addOperator("&>")
				i++
			default:
				addOperator("&")
			}
		case '|':
			flushWord()
			if i+1 < len(command) && command[i+1] == '|' {
				addOperator("||")
				i++
			} else {
				addOperator("|")
			}
		case ';':
			flushWord()
			addOperator(";")
		case '<':
			flushWord()
			if i+1 < len(command) && command[i+1] == '<' {
				addOperator("<<")
				i++
			} else if i+1 < len(command) && command[i+1] == '&' {
				addOperator("<&")
				i++
			} else {
				addOperator("<")
			}
		case '>':
			flushWord()
			if i+1 < len(command) && command[i+1] == '>' {
				addOperator(">>")
				i++
			} else if i+1 < len(command) && command[i+1] == '|' {
				addOperator(">|")
				i++
			} else if i+1 < len(command) && command[i+1] == '&' {
				addOperator(">&")
				i++
			} else {
				addOperator(">")
			}
		default:
			word.WriteByte(ch)
		}
	}
	if escaped || quote != 0 {
		reasons = append(reasons, "could not parse shell command")
		high = true
	}
	flushWord()
	return tokens, reasons, high
}

func bashCommandCalls(tokens []bashToken) [][]string {
	var calls [][]string
	var current []string
	flush := func() {
		if len(current) == 0 {
			return
		}
		calls = append(calls, current)
		current = nil
	}
	for _, token := range tokens {
		if token.kind == bashTokenWord {
			current = append(current, token.value)
			continue
		}
		if isBashCommandBoundary(token.value) {
			flush()
		}
	}
	flush()
	return calls
}

func isBashCommandBoundary(op string) bool {
	switch op {
	case "&&", "||", "|", ";", "&", "\n":
		return true
	default:
		return false
	}
}

func shellCommandArgs(args []string) []string {
	for len(args) > 0 && isShellAssignment(args[0]) {
		args = args[1:]
	}
	return args
}

func isShellAssignment(word string) bool {
	name, _, ok := strings.Cut(word, "=")
	if !ok || name == "" {
		return false
	}
	for i := range len(name) {
		ch := name[i]
		if i == 0 {
			if !((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_') {
				return false
			}
			continue
		}
		if !((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_') {
			return false
		}
	}
	return true
}

func isDynamicCommandName(name string) bool {
	return strings.HasPrefix(name, "$") || strings.HasPrefix(name, "`") || strings.Contains(name, "$(") || strings.Contains(name, "`")
}

func shellCommandBase(name string) string {
	name = strings.TrimSpace(name)
	if i := strings.LastIndexAny(name, `/\`); i >= 0 && i+1 < len(name) {
		name = name[i+1:]
	}
	return name
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
