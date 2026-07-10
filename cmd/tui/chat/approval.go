package chat

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
)

type chatApprovalChoice struct {
	label      string
	key        string
	allow      bool
	allowTools bool
	allowAll   bool
	reason     string
}

var chatApprovalChoices = []chatApprovalChoice{
	{label: "Approve once", key: "1", allow: true},
	{label: "Always allow tool", key: "2", allow: true, allowTools: true},
	{label: "Deny", key: "3", reason: "Tool execution denied."},
}

type chatApprovalPrompt struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.Approval
	cursor  int
}

func (m chatModel) approvalPrompterForRun(controller *chatApprovalController) coreagent.ApprovalPrompter {
	if m.opts.ApprovalPrompter != nil {
		return m.opts.ApprovalPrompter
	}
	return controller
}

func (m *chatModel) ensureApprovalState() *coreagent.ApprovalState {
	if m.approvalState == nil {
		m.approvalState = &coreagent.ApprovalState{}
		m.approvalState.Set(m.defaultAllowAll, nil)
	}
	return m.approvalState
}

func (m *chatModel) resetApprovalState() {
	m.approvalState = &coreagent.ApprovalState{}
	m.approvalState.Set(m.defaultAllowAll, nil)
}

func (m chatModel) allowAllToolsEnabled() bool {
	if m.approvalState == nil {
		return m.defaultAllowAll
	}
	return m.approvalState.AllowAll()
}

func (m *chatModel) setAllowAllTools(allowAll bool) {
	m.ensureApprovalState().SetAllowAll(allowAll)
	m.opts.AllowAllTools = allowAll
}

func (m *chatModel) openApprovalPrompt(msg chatApprovalPromptMsg) {
	m.approvalPrompt = &chatApprovalPrompt{request: msg.request, reply: msg.reply}
	m.status = "approval required"
	m.thinking = false
	m.thinkingTokens = 0
	m.upsertApprovalToolEntries(msg.request)
}

func (m *chatModel) togglePermissionMode() (tea.Model, tea.Cmd) {
	m.setAllowAllTools(!m.allowAllToolsEnabled())
	if m.allowAllToolsEnabled() {
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
	printedLines := m.flowPrintedLines
	var printedTranscript []string
	if printedLines > 0 {
		printedTranscript = slices.Clone(m.transcriptLines(m.viewWidth()))
	}
	prompt := m.approvalPrompt
	m.approvalPrompt = nil
	m.status = "running"
	if !choice.allow {
		m.status = "denied"
	}
	if choice.allowAll {
		m.setAllowAllTools(true)
	}
	allowScopes := approvalScopes(prompt.request)
	if choice.allowTools {
		m.ensureApprovalState().AllowScopes(allowScopes)
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
	result := coreagent.Approval{Allow: choice.allow, AllowAll: choice.allowAll, Reason: choice.reason}
	if choice.allowTools {
		result.AllowScopes = allowScopes
	}
	prompt.reply <- result
	return m.withFlowTranscriptRefreshAfter(printedTranscript, printedLines, waitForChatMsg(m.events))
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
	if len(prompt.request.Calls) <= 1 {
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
	}

	lines = append(lines, indentLines(renderApprovalChoices(prompt.request, prompt.cursor, bodyWidth), "  ")...)
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

func renderApprovalChoices(request coreagent.ApprovalRequest, cursor int, width int) []string {
	var lines []string
	for i, choice := range chatApprovalChoices {
		label := choice.key + ". " + approvalChoiceLabel(choice, request)
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

func approvalChoiceLabel(choice chatApprovalChoice, request coreagent.ApprovalRequest) string {
	if !choice.allowTools {
		return choice.label
	}
	scopes := approvalScopes(request)
	if len(scopes) == 1 {
		call := approvalCallForScope(request, scopes[0])
		if isShellToolName(call.ToolName) {
			if command, ok := rawStringArg(call.Args, "command"); ok && strings.TrimSpace(command) != "" {
				return "Always allow this command"
			}
		}
		return "Always allow " + toolDisplayName(call.ToolName)
	}
	return "Always allow these requests"
}

func approvalScopes(request coreagent.ApprovalRequest) []string {
	seen := make(map[string]bool, len(request.Calls))
	var scopes []string
	for _, call := range request.Calls {
		scope := approvalScope(call)
		if scope == "" || seen[scope] {
			continue
		}
		seen[scope] = true
		scopes = append(scopes, scope)
	}
	return scopes
}

func approvalCallForScope(request coreagent.ApprovalRequest, scope string) coreagent.ApprovalToolCall {
	for _, call := range request.Calls {
		if approvalScope(call) == scope {
			return call
		}
	}
	return coreagent.ApprovalToolCall{}
}

func approvalScope(call coreagent.ApprovalToolCall) string {
	if scope := strings.TrimSpace(call.ApprovalScope); scope != "" {
		return scope
	}
	return strings.TrimSpace(call.ToolName)
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

type chatApprovalController struct {
	ch    chan<- tea.Msg
	state *coreagent.ApprovalState
}

func newChatApprovalController(ch chan<- tea.Msg, state *coreagent.ApprovalState) *chatApprovalController {
	return &chatApprovalController{
		ch:    ch,
		state: state,
	}
}

func (c *chatApprovalController) PromptApproval(ctx context.Context, request coreagent.ApprovalRequest) (coreagent.Approval, error) {
	if result, ok := c.preapproved(request); ok {
		return result, nil
	}
	return chatApprovalPrompter{ch: c.ch}.PromptApproval(ctx, request)
}

func (c *chatApprovalController) preapproved(request coreagent.ApprovalRequest) (coreagent.Approval, bool) {
	if c == nil {
		return coreagent.Approval{}, false
	}
	if c.state.AllowAll() {
		return coreagent.Approval{Allow: true, AllowAll: true}, true
	}
	scopes := approvalScopes(request)
	if len(scopes) == 0 {
		return coreagent.Approval{}, false
	}
	for _, scope := range scopes {
		if !c.state.Allows(scope) {
			return coreagent.Approval{}, false
		}
	}
	return coreagent.Approval{Allow: true, AllowScopes: scopes}, true
}
