package tui

import (
	"context"
	"fmt"
	"strings"
	"sync"
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
	{label: "Approve once", key: "o", decision: coreagent.ApprovalAllowOnce},
	{label: "Approve session", key: "s", decision: coreagent.ApprovalAllowSession},
	{label: "Deny", key: "d", decision: coreagent.ApprovalDeny, reason: "Tool execution denied."},
}

type chatApprovalPrompt struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.ApprovalResult
	cursor  int
}

type chatPermissionMode struct {
	mu          sync.Mutex
	autoApprove bool
}

func newChatPermissionMode(autoApprove bool) *chatPermissionMode {
	return &chatPermissionMode{autoApprove: autoApprove}
}

func (m *chatPermissionMode) AutoApprove() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.autoApprove
}

func (m *chatPermissionMode) SetAutoApprove(autoApprove bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.autoApprove = autoApprove
}

func (m chatModel) approvalHandlerForRun(events chan<- tea.Msg) coreagent.ApprovalHandler {
	return chatPermissionApprovalHandler{
		mode:   m.permissionMode,
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

func chatReviewApprovalHandler(handler coreagent.ApprovalHandler) coreagent.ApprovalHandler {
	if handler == nil || approvalHandlerAutoApproves(handler) {
		return coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{})
	}
	return handler
}

func approvalHandlerAutoApproves(handler coreagent.ApprovalHandler) bool {
	switch h := handler.(type) {
	case coreagent.AutoAllowApproval:
		return true
	case *coreagent.AutoAllowApproval:
		return h != nil
	case interface{ AutoApproveEnabled() bool }:
		return h.AutoApproveEnabled()
	default:
		return false
	}
}

func (m *chatModel) openApprovalPrompt(msg chatApprovalPromptMsg) {
	m.approvalPrompt = &chatApprovalPrompt{request: msg.request, reply: msg.reply}
	m.status = "approval required"
	m.thinking = false
	m.thinkingTokens = 0
	m.upsertApprovalToolEntry(msg.request)
}

func (m *chatModel) togglePermissionMode() (tea.Model, tea.Cmd) {
	m.ensurePermissionMode()
	autoApprove := !m.permissionMode.AutoApprove()
	m.permissionMode.SetAutoApprove(autoApprove)
	m.opts.AutoApproveTools = autoApprove
	if autoApprove {
		m.status = "full access enabled"
		if m.approvalPrompt != nil {
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowOnce, "")
		}
		return *m, nil
	}
	m.status = "review mode enabled"
	return *m, nil
}

func (m *chatModel) ensurePermissionMode() {
	if m.permissionMode == nil {
		m.permissionMode = newChatPermissionMode(m.opts.AutoApproveTools || approvalHandlerAutoApproves(m.opts.Approval))
	}
	if m.reviewApproval == nil {
		m.reviewApproval = chatReviewApprovalHandler(m.opts.Approval)
	}
}

func (m chatModel) autoApproveTools() bool {
	if m.permissionMode != nil {
		return m.permissionMode.AutoApprove()
	}
	return m.opts.AutoApproveTools || approvalHandlerAutoApproves(m.opts.Approval)
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
	case tea.KeyCtrlO:
		m.toggleAllToolOutputs()
	case tea.KeyRunes:
		switch strings.ToLower(string(msg.Runes)) {
		case "o":
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowOnce, "")
		case "s":
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowSession, "")
		case "d":
			return m.resolveApprovalPrompt(coreagent.ApprovalDeny, "Tool execution denied.")
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
	if request.Summary != "" {
		lines = append(lines, wrapChatText(request.Summary, width)...)
	} else {
		lines = append(lines, wrapChatText(fmt.Sprintf("%s wants to run", toolDisplayName(request.ToolName)), width)...)
	}
	if detail := approvalRequestDetail(request, bodyWidth); detail != "" {
		lines = append(lines, indentLines(splitRenderedBody(detail), "  ")...)
	}

	if len(request.Reasons) > 0 {
		lines = append(lines, "  "+chatMetaStyle.Render(strings.Join(request.Reasons, " • ")))
	}
	if strings.TrimSpace(request.WorkingDir) != "" {
		lines = append(lines, "  "+chatMetaStyle.Render("cwd: "+request.WorkingDir))
	}

	lines = append(lines, "")
	lines = append(lines, indentLines(renderApprovalChoices(prompt.cursor, request, bodyWidth), "  ")...)
	lines = append(lines, "  "+chatMetaStyle.Render("enter select • o once • s session • d deny • esc deny"))
	return lines
}

func approvalSessionScope(request coreagent.ApprovalRequest) string {
	switch request.ToolName {
	case "bash":
		return "same command in this chat"
	case "edit":
		if path, ok := stringArg(request.Args, "path"); ok {
			return "edits to " + path + " in this chat"
		}
		return "matching edit calls in this chat"
	case "web_search":
		return "same search in this chat"
	case "web_fetch":
		return "same URL in this chat"
	default:
		return "matching tool arguments in this chat"
	}
}

func approvalChoiceScope(request coreagent.ApprovalRequest, decision coreagent.ApprovalDecision) string {
	if decision == coreagent.ApprovalAllowSession {
		return approvalSessionScope(request)
	}
	return ""
}

func approvalRequestDetail(request coreagent.ApprovalRequest, width int) string {
	switch request.ToolName {
	case "bash":
		command, ok := stringArg(request.Args, "command")
		if !ok {
			return ""
		}
		return strings.Join(wrapChatText("$ "+command, width), "\n")
	case "edit":
		path, ok := stringArg(request.Args, "path")
		if !ok {
			return ""
		}
		var lines []string
		lines = append(lines, "path: "+path)
		if oldText, ok := stringArg(request.Args, "old_text"); ok {
			lines = append(lines, fmt.Sprintf("old_text: %d chars", len([]rune(oldText))))
		}
		if newText, ok := stringArg(request.Args, "new_text"); ok {
			lines = append(lines, fmt.Sprintf("new_text: %d chars", len([]rune(newText))))
		}
		return chatMetaStyle.Render(strings.Join(lines, "\n"))
	default:
		if len(request.Args) == 0 {
			return ""
		}
		return chatMetaStyle.Render(formatToolArgs(request.Args))
	}
}

func renderApprovalChoices(cursor int, request coreagent.ApprovalRequest, width int) []string {
	var lines []string
	for i, choice := range chatApprovalChoices {
		label := choice.label
		if choice.key != "" {
			label = choice.label + " (" + choice.key + ")"
		}
		if scope := approvalChoiceScope(request, choice.decision); scope != "" {
			label += " - " + scope
		}
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

type chatPermissionApprovalHandler struct {
	mode   *chatPermissionMode
	review coreagent.ApprovalHandler
}

func (h chatPermissionApprovalHandler) RequiresApproval(ctx context.Context, tool coreagent.Tool, req coreagent.ApprovalRequest) bool {
	if h.mode != nil && h.mode.AutoApprove() {
		return false
	}
	if h.review != nil {
		return h.review.RequiresApproval(ctx, tool, req)
	}
	return coreagent.ToolRequiresApproval(tool, req.Args)
}

func (h chatPermissionApprovalHandler) Approve(ctx context.Context, req coreagent.ApprovalRequest) (coreagent.ApprovalResult, error) {
	if h.mode != nil && h.mode.AutoApprove() {
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
