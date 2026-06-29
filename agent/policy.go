package agent

import "sync"

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

func (p RunPolicy) ReviewAuthorizer(prompter ApprovalPrompter) ToolAuthorizer {
	policy := p.ApprovalPolicy
	if policy == nil {
		policy = DefaultApprovalPolicy{}
	}
	return NewApprovalManager(ApprovalManagerOptions{
		Policy:   policy,
		Prompter: prompter,
	})
}

type RunPolicyState struct {
	mu     sync.Mutex
	policy RunPolicy
}

func NewRunPolicyState(policy RunPolicy) *RunPolicyState {
	return &RunPolicyState{policy: policy}
}

func (s *RunPolicyState) Policy() RunPolicy {
	if s == nil {
		return RunPolicy{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.policy
}

func (s *RunPolicyState) ToolMode() ToolMode {
	return s.Policy().ToolMode
}

func (s *RunPolicyState) SetToolMode(mode ToolMode) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.policy.ToolMode = mode
}
