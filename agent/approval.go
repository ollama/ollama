package agent

import "context"

type ApprovalDecision string

const (
	ApprovalAllowCurrent ApprovalDecision = "allow_current"
	ApprovalAllowAll     ApprovalDecision = "allow_all"
	ApprovalDeny         ApprovalDecision = "deny"
)

type ApprovalRequest struct {
	WorkingDir string
	Calls      []ApprovalToolCall
}

type ApprovalToolCall struct {
	ToolCallID string
	ToolName   string
	Args       map[string]any
}

type ApprovalResult struct {
	Decision ApprovalDecision
	Reason   string
}

type ToolAuthorizer interface {
	AuthorizeTools(context.Context, ApprovalRequest) (ApprovalResult, error)
}

type ApprovalPrompter interface {
	PromptApproval(context.Context, ApprovalRequest) (ApprovalResult, error)
}

type AutoAllowApproval struct{}

func (AutoAllowApproval) AuthorizeTools(context.Context, ApprovalRequest) (ApprovalResult, error) {
	return ApprovalResult{Decision: ApprovalAllowCurrent}, nil
}

type ApprovalManagerOptions struct {
	Prompter ApprovalPrompter
}

type ApprovalManager struct {
	prompter ApprovalPrompter
	allowAll bool
}

func NewApprovalManager(opts ApprovalManagerOptions) *ApprovalManager {
	return &ApprovalManager{prompter: opts.Prompter}
}

func (m *ApprovalManager) AuthorizeTools(ctx context.Context, req ApprovalRequest) (ApprovalResult, error) {
	if m == nil || m.allowAll || len(req.Calls) == 0 {
		return ApprovalResult{Decision: ApprovalAllowCurrent}, nil
	}
	if m.prompter == nil {
		return ApprovalResult{
			Decision: ApprovalDeny,
			Reason:   "Tool execution requires approval, but no approval prompter is available.",
		}, nil
	}

	result, err := m.prompter.PromptApproval(ctx, req)
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
