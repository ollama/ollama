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

func (s *ApprovalState) SetAllowAll(allowAll bool) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.allowAll = allowAll
}

func (s *ApprovalState) AllowAll() bool {
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

func (s *ApprovalState) Apply(result *Approval) {
	if s == nil || result == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if result.AllowAll {
		result.Allow = true
		s.allowAll = true
	}
	if len(result.AllowScopes) > 0 {
		result.Allow = true
		if s.scopes == nil {
			s.scopes = make(map[string]bool, len(result.AllowScopes))
		}
		for _, scope := range result.AllowScopes {
			scope = strings.TrimSpace(scope)
			if scope != "" {
				s.scopes[scope] = true
			}
		}
	}
}

func (s *ApprovalState) AllowScopes(scopes []string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
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
// the result as allowed when scopes or allow-all were granted. It mutates
// result.Allow so the caller can branch on the effective decision.
func (s *Session) applyApproval(result *Approval) {
	if s == nil || result == nil {
		return
	}
	if s.ApprovalState == nil {
		s.ApprovalState = &ApprovalState{}
	}
	s.ApprovalState.Apply(result)
}

func (s *Session) authorizeToolCalls(ctx context.Context, req ApprovalRequest) (Approval, error) {
	if s == nil || len(req.Calls) == 0 || (s.ApprovalState != nil && s.ApprovalState.AllowAll()) {
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
