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
	allow    bool
	allowAll bool
	reason   string
}

var chatApprovalChoices = []chatApprovalChoice{
	{label: "Approve once", key: "1", allow: true},
	{label: "Approve all", key: "2", allow: true, allowAll: true},
	{label: "Deny", key: "3", reason: "Tool execution denied."},
}

type chatApprovalPrompt struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.Approval
	cursor  int
}

func (m chatModel) approvalPrompterForRun(events chan<- tea.Msg) coreagent.ApprovalPrompter {
	if m.allowAllTools {
		return nil
	}
	if m.opts.ApprovalPrompter != nil {
		return m.opts.ApprovalPrompter
	}
	return chatApprovalPrompter{ch: events}
}

func (m *chatModel) openApprovalPrompt(msg chatApprovalPromptMsg) {
	m.approvalPrompt = &chatApprovalPrompt{request: msg.request, reply: msg.reply}
	m.status = "approval required"
	m.thinking = false
	m.thinkingTokens = 0
	m.upsertApprovalToolEntries(msg.request)
}

func (m *chatModel) togglePermissionMode() (tea.Model, tea.Cmd) {
	m.allowAllTools = !m.allowAllTools
	m.opts.AllowAllTools = m.allowAllTools
	if m.allowAllTools {
		m.permissionNotice = "full access enabled"
		m.status = "full access enabled"
		if m.approvalPrompt != nil {
			updated, cmd := m.resolveApprovalPrompt(chatApprovalChoice{allow: true, allowAll: true})
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

func (m chatModel) autoApproveTools() bool {
	return m.allowAllTools
}

func (m *chatModel) upsertApprovalToolEntries(request coreagent.ApprovalRequest) {
	for _, call := range request.Calls {
		idx := m.findToolEntry(call.ToolCallID)
		if idx < 0 {
			m.groupCompletedToolHistory()
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].detail = call.ToolName
		m.entries[idx].label = toolInvocationLabel(call.ToolName, call.Args)
		m.entries[idx].status = "approval"
		m.entries[idx].toolID = call.ToolCallID
		m.entries[idx].args = call.Args
		m.entries[idx].startedAt = time.Now()
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
	}
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
			return m.resolveApprovalPrompt(choice)
		}
	case tea.KeyEnter:
		choice := chatApprovalChoices[clamp(m.approvalPrompt.cursor, 0, len(chatApprovalChoices)-1)]
		return m.resolveApprovalPrompt(choice)
	case tea.KeyEsc, tea.KeyCtrlC:
		return m.resolveApprovalPrompt(chatApprovalChoice{reason: "Tool execution denied."})
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
	for _, call := range m.approvalPrompt.request.Calls {
		if idx := m.findToolEntry(call.ToolCallID); idx >= 0 {
			m.markEntryDirty(idx)
		}
	}
}

func (m chatModel) resolveApprovalPrompt(choice chatApprovalChoice) (tea.Model, tea.Cmd) {
	if m.approvalPrompt == nil {
		return m, nil
	}
	prompt := m.approvalPrompt
	m.approvalPrompt = nil
	m.status = "running"
	if !choice.allow {
		m.status = "denied"
	}
	if choice.allowAll {
		m.allowAllTools = true
		m.opts.AllowAllTools = true
	}
	for _, call := range prompt.request.Calls {
		if idx := m.findToolEntry(call.ToolCallID); idx >= 0 && m.entries[idx].status == "approval" {
			if !choice.allow {
				m.entries[idx].status = "error"
				m.entries[idx].err = choice.reason
				if m.entries[idx].err == "" {
					m.entries[idx].err = "Tool execution denied."
				}
			} else {
				m.entries[idx].status = "queued"
			}
			m.markEntryDirty(idx)
		}
	}
	prompt.reply <- coreagent.Approval{Allow: choice.allow, AllowAll: choice.allowAll, Reason: choice.reason}
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

	var lines []string
	detail := approvalRequestDetail(prompt.request, bodyWidth)
	if detail == "" {
		label := "Tool request"
		if len(prompt.request.Calls) == 1 {
			label = toolDisplayName(prompt.request.Calls[0].ToolName)
		}
		lines = append(lines, wrapChatText(fmt.Sprintf("%s wants to run", label), width)...)
	} else {
		lines = append(lines, indentLines(splitRenderedBody(detail), "  ")...)
	}

	lines = append(lines, "")
	lines = append(lines, indentLines(renderApprovalChoices(prompt.cursor, bodyWidth), "  ")...)
	return lines
}

func approvalRequestDetail(request coreagent.ApprovalRequest, width int) string {
	if len(request.Calls) == 0 {
		return ""
	}
	if len(request.Calls) == 1 {
		return approvalToolCallDetail(request.Calls[0], width)
	}
	lines := make([]string, 0, len(request.Calls))
	for _, call := range request.Calls {
		lines = append(lines, toolInvocationLabel(call.ToolName, call.Args))
	}
	return chatMetaStyle.Render(strings.Join(lines, "\n"))
}

func approvalToolCallDetail(call coreagent.ApprovalToolCall, width int) string {
	if isShellToolName(call.ToolName) {
		command, ok := rawStringArg(call.Args, "command")
		if !ok {
			return ""
		}
		return strings.Join(wrapChatText(shellPromptPrefix(call.ToolName)+command, width), "\n")
	}
	switch call.ToolName {
	case "edit":
		path, ok := rawStringArg(call.Args, "path")
		if !ok {
			return ""
		}
		var lines []string
		lines = append(lines, "path: "+path)
		if oldText, ok := rawStringArg(call.Args, "old_text"); ok {
			lines = append(lines, fmt.Sprintf("old_text: %d chars", len([]rune(oldText))))
		}
		if newText, ok := rawStringArg(call.Args, "new_text"); ok {
			lines = append(lines, fmt.Sprintf("new_text: %d chars", len([]rune(newText))))
		}
		return chatMetaStyle.Render(strings.Join(lines, "\n"))
	default:
		if len(call.Args) == 0 {
			return ""
		}
		return strings.Join(renderToolCallArgs(call.Args, width), "\n")
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
					lines = append(lines, chatPickerSelectedStyle.Render("> "+line))
				} else {
					lines = append(lines, chatPickerSelectedStyle.Render("  "+line))
				}
			}
		} else {
			for _, line := range wrapped {
				lines = append(lines, chatPickerTextStyle.Render("  "+line))
			}
		}
	}
	return lines
}

type chatApprovalPrompter struct {
	ch chan<- tea.Msg
}

func (p chatApprovalPrompter) PromptApproval(ctx context.Context, request coreagent.ApprovalRequest) (coreagent.Approval, error) {
	reply := make(chan coreagent.Approval, 1)
	select {
	case p.ch <- chatApprovalPromptMsg{request: request, reply: reply}:
	case <-ctx.Done():
		return coreagent.Approval{Reason: "Tool approval canceled."}, nil
	}

	select {
	case result := <-reply:
		return result, nil
	case <-ctx.Done():
		return coreagent.Approval{Reason: "Tool approval canceled."}, nil
	}
}
