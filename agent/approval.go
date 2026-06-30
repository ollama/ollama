package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"unicode"
)

type ApprovalDecision string

const (
	ApprovalAllowOnce ApprovalDecision = "allow_once"
	ApprovalAllowAll  ApprovalDecision = "allow_all"
	ApprovalDeny      ApprovalDecision = "deny"
)

type ApprovalRisk string

const (
	ApprovalRiskLow    ApprovalRisk = "low"
	ApprovalRiskMedium ApprovalRisk = "medium"
	ApprovalRiskHigh   ApprovalRisk = "high"
)

type ApprovalRequest struct {
	ToolCallID           string
	ToolName             string
	Args                 map[string]any
	WorkingDir           string
	ToolApprovalRequired bool
	Summary              string
	Risk                 ApprovalRisk
	Reasons              []string
}

type ApprovalResult struct {
	Decision ApprovalDecision
	Reason   string
}

type ToolAuthorizer interface {
	AuthorizeTool(context.Context, ToolAuthorizationRequest) (ApprovalResult, error)
}

type ToolAuthorizationRequest struct {
	ToolCallID string
	Tool       Tool
	ToolName   string
	Args       map[string]any
	WorkingDir string
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
}

type AutoAllowApproval struct{}

func (AutoAllowApproval) AuthorizeTool(context.Context, ToolAuthorizationRequest) (ApprovalResult, error) {
	return ApprovalResult{Decision: ApprovalAllowOnce}, nil
}

type ApprovalManagerOptions struct {
	Policy   ApprovalPolicy
	Prompter ApprovalPrompter
}

type ApprovalManager struct {
	policy   ApprovalPolicy
	prompter ApprovalPrompter
	allowAll bool
}

func NewApprovalManager(opts ApprovalManagerOptions) *ApprovalManager {
	policy := opts.Policy
	if policy == nil {
		policy = DefaultApprovalPolicy{}
	}
	return &ApprovalManager{
		policy:   policy,
		prompter: opts.Prompter,
	}
}

func (m *ApprovalManager) AuthorizeTool(ctx context.Context, req ToolAuthorizationRequest) (ApprovalResult, error) {
	if m == nil {
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}

	approvalReq := approvalRequestFromToolAuthorization(req)
	evaluation := applyToolApprovalRequirement(approvalReq, m.evaluate(ctx, approvalReq))
	approvalReq = approvalRequestWithEvaluation(approvalReq, evaluation)

	if evaluation.Decision == ApprovalDeny {
		reason := strings.Join(evaluation.Reasons, "; ")
		if reason == "" {
			reason = "Tool execution denied."
		}
		return ApprovalResult{Decision: ApprovalDeny, Reason: reason}, nil
	}

	if !evaluation.RequirePrompt || m.allowAll {
		return ApprovalResult{Decision: ApprovalAllowOnce}, nil
	}

	if m.prompter == nil {
		return ApprovalResult{
			Decision: ApprovalDeny,
			Reason:   "Tool execution requires approval, but no approval prompter is available.",
		}, nil
	}

	result, err := m.prompter.PromptApproval(ctx, approvalReq)
	if err != nil {
		return ApprovalResult{}, err
	}
	if result.Decision == "" {
		result.Decision = ApprovalDeny
	}
	if result.Decision == ApprovalAllowAll {
		m.allowAll = true
	}
	return result, nil
}

func approvalRequestFromToolAuthorization(req ToolAuthorizationRequest) ApprovalRequest {
	toolName := req.ToolName
	if toolName == "" && req.Tool != nil {
		toolName = req.Tool.Name()
	}
	return ApprovalRequest{
		ToolCallID:           req.ToolCallID,
		ToolName:             toolName,
		Args:                 req.Args,
		WorkingDir:           req.WorkingDir,
		ToolApprovalRequired: ToolRequiresApproval(req.Tool, req.Args),
	}
}

func (m *ApprovalManager) evaluate(ctx context.Context, req ApprovalRequest) ApprovalEvaluation {
	if m == nil || m.policy == nil {
		return DefaultApprovalPolicy{}.EvaluateApproval(ctx, req)
	}
	evaluation := m.policy.EvaluateApproval(ctx, req)
	if evaluation.Risk == "" {
		evaluation.Risk = ApprovalRiskLow
	}
	return evaluation
}

func applyToolApprovalRequirement(req ApprovalRequest, evaluation ApprovalEvaluation) ApprovalEvaluation {
	if !req.ToolApprovalRequired || evaluation.Decision == ApprovalDeny {
		return evaluation
	}
	evaluation.RequirePrompt = true
	if evaluation.Summary == "" {
		evaluation.Summary = fmt.Sprintf("%s wants to run", toolDisplayName(req.ToolName))
	}
	if len(evaluation.Reasons) == 0 {
		evaluation.Reasons = []string{"tool requires approval"}
	}
	return evaluation
}

func approvalRequestWithEvaluation(req ApprovalRequest, evaluation ApprovalEvaluation) ApprovalRequest {
	req.Summary = evaluation.Summary
	req.Risk = evaluation.Risk
	req.Reasons = slices.Clone(evaluation.Reasons)
	return req
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
		return ApprovalEvaluation{Decision: ApprovalAllowOnce, Risk: ApprovalRiskLow, Summary: fmt.Sprintf("%s can run without approval", toolDisplayName(req.ToolName))}
	case "web_search", "web_fetch":
		return evaluateWebApproval(req)
	case "edit":
		return evaluateEditApproval(req)
	case "bash", "powershell":
		return evaluateShellApproval(req)
	default:
		return ApprovalEvaluation{
			RequirePrompt: true,
			Risk:          ApprovalRiskMedium,
			Summary:       fmt.Sprintf("%s wants to run", toolDisplayName(req.ToolName)),
			Reasons:       []string{"unknown tool effects"},
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
		Summary:       fmt.Sprintf("Edit wants to modify %s", sanitizeApprovalDisplay(path)),
		Reasons:       reasons,
	}
}

func evaluateWebApproval(req ApprovalRequest) ApprovalEvaluation {
	switch req.ToolName {
	case "web_search":
		query, ok := stringApprovalArg(req.Args, "query")
		if !ok {
			return denyApproval("web_search", ApprovalRiskHigh, "missing query argument")
		}
		return ApprovalEvaluation{
			RequirePrompt: true,
			Risk:          ApprovalRiskMedium,
			Summary:       fmt.Sprintf("Web Search wants to search for %q", sanitizeApprovalDisplay(query)),
			Reasons:       []string{"searches the web"},
		}
	case "web_fetch":
		targetURL, ok := stringApprovalArg(req.Args, "url")
		if !ok {
			return denyApproval("web_fetch", ApprovalRiskHigh, "missing url argument")
		}
		return ApprovalEvaluation{
			RequirePrompt: true,
			Risk:          ApprovalRiskMedium,
			Summary:       fmt.Sprintf("Web Fetch wants to fetch %s", sanitizeApprovalDisplay(targetURL)),
			Reasons:       []string{"fetches web content"},
		}
	}
	return ApprovalEvaluation{
		RequirePrompt: true,
		Risk:          ApprovalRiskMedium,
		Summary:       fmt.Sprintf("%s wants to run", toolDisplayName(req.ToolName)),
		Reasons:       []string{"accesses the web"},
	}
}

func sanitizeApprovalDisplay(value string) string {
	value = approvalANSIEscapePattern.ReplaceAllString(value, "")
	value = strings.Map(func(r rune) rune {
		switch r {
		case '\n', '\r', '\t':
			return ' '
		}
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, value)
	value = strings.Join(strings.Fields(value), " ")
	if value == "" {
		return "(empty)"
	}
	return value
}

func denyApproval(toolName string, risk ApprovalRisk, reason string) ApprovalEvaluation {
	return ApprovalEvaluation{
		Decision: ApprovalDeny,
		Risk:     risk,
		Summary:  fmt.Sprintf("%s cannot run", toolDisplayName(toolName)),
		Reasons:  []string{reason},
	}
}

func toolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
	case "powershell":
		return "PowerShell"
	case "read":
		return "Read"
	case "list":
		return "List"
	case "edit":
		return "Edit"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}

var approvalANSIEscapePattern = regexp.MustCompile(`\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))`)

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
