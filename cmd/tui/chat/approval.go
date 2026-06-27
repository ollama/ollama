package chat

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
)

type chatApprovalChoice struct {
	label    string
	key      string
	decision coreagent.ApprovalDecision
	reason   string
}

var chatApprovalChoices = []chatApprovalChoice{
	{label: "Approve once", key: "1", decision: coreagent.ApprovalAllowOnce},
	{label: "Approve session", key: "2", decision: coreagent.ApprovalAllowSession},
	{label: "Deny", key: "3", decision: coreagent.ApprovalDeny, reason: "Tool execution denied."},
}

type chatApprovalPrompt struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.ApprovalResult
	cursor  int
}

func (m chatModel) approvalHandlerForRun(events chan<- tea.Msg) coreagent.ApprovalHandler {
	return chatPolicyApprovalHandler{
		policy: m.policyState,
		review: approvalHandlerForRun(m.reviewApproval, events),
	}
}

func approvalHandlerForRun(handler coreagent.ApprovalHandler, events chan<- tea.Msg) coreagent.ApprovalHandler {
	prompter := chatApprovalPrompter{ch: events}
	if manager, ok := handler.(*coreagent.ApprovalManager); ok {
		return manager.WithPrompter(prompter)
	}
	if handler != nil {
		return handler
	}
	return coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{Prompter: prompter})
}

func chatReviewApprovalHandler(handler coreagent.ApprovalHandler, policy coreagent.RunPolicy) coreagent.ApprovalHandler {
	if handler != nil {
		return handler
	}
	return policy.ReviewApprovalHandler(nil)
}

func (m *chatModel) openApprovalPrompt(msg chatApprovalPromptMsg) {
	m.approvalPrompt = &chatApprovalPrompt{request: msg.request, reply: msg.reply}
	m.status = "approval required"
	m.thinking = false
	m.thinkingTokens = 0
	m.upsertApprovalToolEntry(msg.request)
}

func (m *chatModel) togglePermissionMode() (tea.Model, tea.Cmd) {
	m.ensureRunPolicy()
	nextMode := coreagent.ToolModeFullAccess
	if m.currentPolicy().ToolMode == coreagent.ToolModeFullAccess {
		nextMode = coreagent.ToolModeReview
	}
	m.policyState.SetToolMode(nextMode)
	m.opts.Policy = m.currentPolicy()
	if nextMode == coreagent.ToolModeFullAccess {
		m.permissionNotice = "full access enabled"
		m.status = "full access enabled"
		if m.approvalPrompt != nil {
			updated, cmd := m.resolveApprovalPrompt(coreagent.ApprovalAllowOnce, "")
			if model, ok := updated.(chatModel); ok {
				model.permissionNotice = "full access enabled"
				model.status = "full access enabled"
				return model, cmd
			}
			return updated, cmd
		}
		return *m, nil
	}
	m.permissionNotice = "review mode enabled"
	m.status = "review mode enabled"
	return *m, nil
}

func (m *chatModel) ensureRunPolicy() {
	if m.policyState == nil {
		m.policyState = coreagent.NewRunPolicyState(m.opts.Policy)
	}
	if m.reviewApproval == nil {
		m.reviewApproval = chatReviewApprovalHandler(m.opts.Approval, m.currentPolicy())
	}
}

func (m chatModel) currentPolicy() coreagent.RunPolicy {
	if m.policyState != nil {
		return m.policyState.Policy()
	}
	return m.opts.Policy
}

func (m chatModel) autoApproveTools() bool {
	return m.currentPolicy().ToolMode == coreagent.ToolModeFullAccess
}

func (m *chatModel) upsertApprovalToolEntry(request coreagent.ApprovalRequest) {
	idx := m.findToolEntry(request.ToolCallID)
	if idx < 0 {
		m.groupCompletedToolHistory()
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
		idx = len(m.entries) - 1
	}
	m.entries[idx].detail = request.ToolName
	m.entries[idx].label = toolInvocationLabel(request.ToolName, request.Args)
	m.entries[idx].status = "approval"
	m.entries[idx].toolID = request.ToolCallID
	m.entries[idx].args = request.Args
	m.entries[idx].startedAt = time.Now()
	m.applyToolOutputModeTo(idx)
	m.markEntryDirty(idx)
}

func (m chatModel) updateApprovalPrompt(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyLeft, tea.KeyUp:
		m.moveApprovalChoice(-1)
	case tea.KeyRight, tea.KeyDown, tea.KeyTab:
		m.moveApprovalChoice(1)
	case tea.KeyRunes:
		switch string(msg.Runes) {
		case "1", "2", "3":
			choice := chatApprovalChoices[int(msg.Runes[0]-'1')]
			return m.resolveApprovalPrompt(choice.decision, choice.reason)
		}
	case tea.KeyEnter:
		choice := chatApprovalChoices[clamp(m.approvalPrompt.cursor, 0, len(chatApprovalChoices)-1)]
		return m.resolveApprovalPrompt(choice.decision, choice.reason)
	case tea.KeyEsc, tea.KeyCtrlC:
		return m.resolveApprovalPrompt(coreagent.ApprovalDeny, "Tool execution denied.")
	}
	return m, nil
}

func (m *chatModel) moveApprovalChoice(delta int) {
	if m.approvalPrompt == nil {
		return
	}
	m.approvalPrompt.cursor = (m.approvalPrompt.cursor + delta) % len(chatApprovalChoices)
	if m.approvalPrompt.cursor < 0 {
		m.approvalPrompt.cursor += len(chatApprovalChoices)
	}
	m.markApprovalPromptEntryDirty()
}

func (m *chatModel) markApprovalPromptEntryDirty() {
	if m.approvalPrompt == nil {
		return
	}
	if idx := m.findToolEntry(m.approvalPrompt.request.ToolCallID); idx >= 0 {
		m.markEntryDirty(idx)
	}
}

func (m chatModel) resolveApprovalPrompt(decision coreagent.ApprovalDecision, reason string) (tea.Model, tea.Cmd) {
	if m.approvalPrompt == nil {
		return m, nil
	}
	prompt := m.approvalPrompt
	m.approvalPrompt = nil
	m.status = "running"
	if decision == coreagent.ApprovalDeny {
		m.status = "denied"
	}
	if idx := m.findToolEntry(prompt.request.ToolCallID); idx >= 0 && m.entries[idx].status == "approval" {
		if decision == coreagent.ApprovalDeny {
			m.entries[idx].status = "error"
			m.entries[idx].err = reason
			if m.entries[idx].err == "" {
				m.entries[idx].err = "Tool execution denied."
			}
		} else {
			m.entries[idx].status = "queued"
		}
		m.markEntryDirty(idx)
	}
	prompt.reply <- coreagent.ApprovalResult{Decision: decision, Reason: reason}
	return m, waitForChatMsg(m.events)
}

func (m chatModel) renderApprovalPromptLines(width int) []string {
	prompt := m.approvalPrompt
	if prompt == nil {
		return nil
	}
	if width <= 0 {
		width = 80
	}
	bodyWidth := max(20, width-2)

	request := prompt.request
	var lines []string
	detail := approvalRequestDetail(request, bodyWidth)
	if detail == "" && request.Summary != "" {
		lines = append(lines, wrapChatText(request.Summary, width)...)
	} else if detail == "" {
		lines = append(lines, wrapChatText(fmt.Sprintf("%s wants to run", toolDisplayName(request.ToolName)), width)...)
	}
	if detail != "" {
		lines = append(lines, indentLines(splitRenderedBody(detail), "  ")...)
	}

	lines = append(lines, "")
	lines = append(lines, indentLines(renderApprovalChoices(prompt.cursor, bodyWidth), "  ")...)
	return lines
}

func approvalRequestDetail(request coreagent.ApprovalRequest, width int) string {
	if coreagent.IsShellToolName(request.ToolName) {
		command, ok := rawStringArg(request.Args, "command")
		if !ok {
			return ""
		}
		return strings.Join(wrapChatText(shellPromptPrefix(request.ToolName)+command, width), "\n")
	}
	switch request.ToolName {
	case "edit":
		path, ok := rawStringArg(request.Args, "path")
		if !ok {
			return ""
		}
		var lines []string
		lines = append(lines, "path: "+path)
		if oldText, ok := rawStringArg(request.Args, "old_text"); ok {
			lines = append(lines, fmt.Sprintf("old_text: %d chars", len([]rune(oldText))))
		}
		if newText, ok := rawStringArg(request.Args, "new_text"); ok {
			lines = append(lines, fmt.Sprintf("new_text: %d chars", len([]rune(newText))))
		}
		return chatMetaStyle.Render(strings.Join(lines, "\n"))
	default:
		if len(request.Args) == 0 {
			return ""
		}
		return strings.Join(renderToolCallArgs(request.Args, width), "\n")
	}
}

func renderApprovalChoices(cursor int, width int) []string {
	var lines []string
	for i, choice := range chatApprovalChoices {
		label := choice.key + ". " + choice.label
		wrapped := wrapChatText(label, max(20, width-2))
		if i == clamp(cursor, 0, len(chatApprovalChoices)-1) {
			for j, line := range wrapped {
				if j == 0 {
					lines = append(lines, chatResumeSelectedStyle.Render("› "+line))
				} else {
					lines = append(lines, chatResumeSelectedStyle.Render("  "+line))
				}
			}
		} else {
			for j, line := range wrapped {
				if j == 0 {
					lines = append(lines, chatResumeTextStyle.Render("  "+line))
				} else {
					lines = append(lines, chatResumeTextStyle.Render("  "+line))
				}
			}
		}
	}
	return lines
}

type chatApprovalPrompter struct {
	ch chan<- tea.Msg
}

type chatPolicyApprovalHandler struct {
	policy *coreagent.RunPolicyState
	review coreagent.ApprovalHandler
}

func (h chatPolicyApprovalHandler) RequiresApproval(ctx context.Context, tool coreagent.Tool, req coreagent.ApprovalRequest) bool {
	if h.policy != nil && h.policy.ToolMode() == coreagent.ToolModeFullAccess {
		return false
	}
	if h.review != nil {
		return h.review.RequiresApproval(ctx, tool, req)
	}
	return req.ToolApprovalRequired || coreagent.ToolRequiresApproval(tool, req.Args)
}

func (h chatPolicyApprovalHandler) Approve(ctx context.Context, req coreagent.ApprovalRequest) (coreagent.ApprovalResult, error) {
	if h.policy != nil && h.policy.ToolMode() == coreagent.ToolModeFullAccess {
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalAllowOnce}, nil
	}
	if h.review != nil {
		return h.review.Approve(ctx, req)
	}
	return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool execution requires approval."}, nil
}

func (p chatApprovalPrompter) PromptApproval(ctx context.Context, request coreagent.ApprovalRequest) (coreagent.ApprovalResult, error) {
	reply := make(chan coreagent.ApprovalResult, 1)
	select {
	case p.ch <- chatApprovalPromptMsg{request: request, reply: reply}:
	case <-ctx.Done():
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool approval canceled."}, nil
	}

	select {
	case result := <-reply:
		return result, nil
	case <-ctx.Done():
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool approval canceled."}, nil
	}
}
