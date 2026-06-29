package agent

type ToolMode int

const (
	ToolModeReview ToolMode = iota
	ToolModeFullAccess
	ToolModeDisabled
)

type RunPolicy struct {
	ToolMode       ToolMode
	ApprovalPolicy ApprovalPolicy
	// MaxToolRounds limits consecutive model/tool cycles.
	// Zero uses the default guard; negative disables the guard for tests or
	// special callers.
	MaxToolRounds int
}

func (p RunPolicy) UsesTools() bool {
	switch p.ToolMode {
	case ToolModeReview, ToolModeFullAccess:
		return true
	default:
		return false
	}
}

func (p RunPolicy) Tools(registry *Registry) *Registry {
	if !p.UsesTools() {
		return nil
	}
	return registry
}

func (p RunPolicy) Authorizer(prompter ApprovalPrompter) ToolAuthorizer {
	if p.ToolMode == ToolModeFullAccess {
		return AutoAllowApproval{}
	}
	policy := p.ApprovalPolicy
	if policy == nil {
		policy = DefaultApprovalPolicy{}
	}
	return NewApprovalManager(ApprovalManagerOptions{
		Policy:   policy,
		Prompter: prompter,
	})
}
