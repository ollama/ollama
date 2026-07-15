package agent

import (
	"context"
	"strings"
	"sync"
)

type ApprovalRequest struct {
	WorkingDir string
	Calls      []ApprovalToolCall
}

func (r *ApprovalRequest) AddToolCall(id, name string, args map[string]any) {
	r.Calls = append(r.Calls, ApprovalToolCall{
		ToolCallID:    id,
		ToolName:      name,
		Args:          args,
		ApprovalScope: toolApprovalScope(name, args),
	})
}

type ApprovalToolCall struct {
	ToolCallID    string
	ToolName      string
	Args          map[string]any
	ApprovalScope string
}

type Approval struct {
	Allow       bool
	AllowAll    bool
	AllowScopes []string
	Reason      string
}

type ApprovalPrompter interface {
	PromptApproval(context.Context, ApprovalRequest) (Approval, error)
}

type ApprovalState struct {
	mu       sync.RWMutex
	allowAll bool
	scopes   map[string]bool
}

func (s *ApprovalState) Set(allowAll bool, scopes map[string]bool) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.allowAll = allowAll
	s.scopes = cloneApprovalScopes(scopes)
}

// GrantAll grants blanket approval for all future tool calls.
func (s *ApprovalState) GrantAll() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.allowAll = true
}

// AllGranted reports whether blanket approval has been granted.
func (s *ApprovalState) AllGranted() bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.allowAll
}

func (s *ApprovalState) Allows(scope string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.allowAll || s.scopes[scope]
}

// Apply merges an approval's scopes and allow-all flag into the state. It
// returns true if the approval grants permission (allow-all or at least one
// scope). It does not mutate the approval; the caller sets Allow based on the
// returned value.
func (s *ApprovalState) Apply(result *Approval) bool {
	if s == nil || result == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	granted := false
	if result.AllowAll {
		s.allowAll = true
		granted = true
	}
	if len(result.AllowScopes) > 0 {
		granted = true
		s.grantScopesLocked(result.AllowScopes)
	}
	return granted
}

// GrantScopes merges the given scopes into the state.
func (s *ApprovalState) GrantScopes(scopes []string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.grantScopesLocked(scopes)
}

// grantScopesLocked adds trimmed, non-empty scopes to the state. Caller must
// hold s.mu.
func (s *ApprovalState) grantScopesLocked(scopes []string) {
	if s.scopes == nil {
		s.scopes = make(map[string]bool, len(scopes))
	}
	for _, scope := range scopes {
		scope = strings.TrimSpace(scope)
		if scope != "" {
			s.scopes[scope] = true
		}
	}
}

func cloneApprovalScopes(src map[string]bool) map[string]bool {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]bool, len(src))
	for scope, allowed := range src {
		if allowed {
			dst[scope] = true
		}
	}
	return dst
}

func (s *Session) needsApproval(tool Tool, name string, args map[string]any) bool {
	return ToolRequiresApproval(tool, args) && !s.allows(toolApprovalScope(name, args))
}

// allows reports whether scope is permitted by the session's accumulated approval state.
func (s *Session) allows(scope string) bool {
	if s == nil || s.ApprovalState == nil {
		return false
	}
	return s.ApprovalState.Allows(scope)
}

// applyApproval merges an approval result into the session's state and marks
// the result as allowed when scopes or allow-all were granted.
func (s *Session) applyApproval(result *Approval) {
	if s == nil || result == nil {
		return
	}
	if s.ApprovalState == nil {
		s.ApprovalState = &ApprovalState{}
	}
	if s.ApprovalState.Apply(result) {
		result.Allow = true
	}
}

func (s *Session) authorizeToolCalls(ctx context.Context, req ApprovalRequest) (Approval, error) {
	if s == nil || len(req.Calls) == 0 || (s.ApprovalState != nil && s.ApprovalState.AllGranted()) {
		return Approval{Allow: true}, nil
	}
	if s.ApprovalPrompter == nil {
		return Approval{
			Reason: "Tool execution requires approval, but no approval prompter is available.",
		}, nil
	}

	result, err := s.ApprovalPrompter.PromptApproval(ctx, req)
	if err != nil {
		return Approval{}, err
	}
	s.applyApproval(&result)
	return result, nil
}

// toolApprovalScope returns the approval scope key for a tool invocation.
//
// For shell tools (bash/powershell) the scope is "<tool>\x00<command>": the
// exact, trimmed command byte string. "Always allow this command" therefore
// matches ONLY that precise string — any whitespace, quoting, or casing
// variant, or any command that is a superset of the approved one, will
// re-prompt. The NUL separator is safe because a shell command string cannot
// contain a literal NUL. For all other tools the scope is the tool name.
func toolApprovalScope(toolName string, args map[string]any) string {
	toolName = strings.TrimSpace(toolName)
	if isShellApprovalTool(toolName) {
		if command, ok := stringArg(args, "command"); ok {
			command = strings.TrimSpace(command)
			if command != "" {
				return toolName + "\x00" + command
			}
		}
	}
	return toolName
}

func isShellApprovalTool(name string) bool {
	return name == "bash" || name == "powershell"
}

func stringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	return value, ok
}
