package agent

import (
	"context"
	"strings"

	"github.com/google/uuid"

	"github.com/ollama/ollama/api"
)

// activateSkill loads opts.SkillName from the catalog and injects a synthetic
// assistant tool call plus tool result before the first model request, so the
// transcript looks like a real skill tool invocation. It emits the same
// tool_call_detected -> tool_started -> tool_finished lifecycle the model path
// uses, and returns the messages to prepend. A blank SkillName is a no-op.
func (s *Session) activateSkill(ctx context.Context, runID string, opts RunOptions) ([]api.Message, error) {
	name := strings.TrimSpace(opts.SkillName)
	if name == "" {
		return nil, nil
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	skill, err := s.Skills.Load(name)
	if err != nil {
		return nil, err
	}
	args := api.NewToolCallFunctionArguments()
	args.Set("name", skill.Name)
	call := api.ToolCall{
		ID:       "call_skill_" + uuid.NewString(),
		Function: api.ToolCallFunction{Name: "skill", Arguments: args},
	}
	result := api.Message{
		Role:       "tool",
		ToolName:   "skill",
		ToolCallID: call.ID,
		Content:    skill.Content(),
	}
	if err := s.emitToolCallDetected(runID, opts, []api.ToolCall{call}); err != nil {
		return nil, err
	}
	if err := s.emitToolStarted(runID, opts, call.ID, "skill", s.currentWorkingDir(), args.ToMap()); err != nil {
		return nil, err
	}
	if err := s.emitToolFinished(ctx, runID, opts, ToolStatusDone, call.ID, "skill", s.currentWorkingDir(), args.ToMap(), result.Content, ""); err != nil {
		return nil, err
	}
	return []api.Message{
		{Role: "assistant", ToolCalls: []api.ToolCall{call}},
		result,
	}, nil
}
