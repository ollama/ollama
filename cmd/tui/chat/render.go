package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type chatEntry struct {
	role       string
	content    string
	label      string
	detail     string
	status     string
	err        string
	toolID     string
	args       map[string]any
	expanded   bool
	startedAt  time.Time
	finishedAt time.Time
	tools      []chatEntry
	metrics    *api.Metrics

	version     int
	renderKey   chatEntryRenderKey
	renderLines []string
}

type chatEntryRenderKey struct {
	width   int
	version int
	hash    string
}

func (m chatModel) findToolEntry(toolID string) int {
	if toolID == "" {
		return -1
	}
	for i := len(m.entries) - 1; i >= 0; i-- {
		if m.entries[i].role == "tool" && m.entries[i].toolID == toolID {
			return i
		}
	}
	return -1
}

func (m chatModel) findActiveToolEntry(toolID string) int {
	idx := m.findToolEntry(toolID)
	if idx < 0 || !isToolActiveStatus(m.entries[idx].status) {
		return -1
	}
	return idx
}

func (m chatModel) toolStartedAt(toolID string) time.Time {
	idx := m.findToolEntry(toolID)
	if idx < 0 {
		return time.Time{}
	}
	return m.entries[idx].startedAt
}

func entryHasExpandableOutput(entry chatEntry) bool {
	return (entry.role == "tool" && (len(entry.args) > 0 || strings.TrimSpace(entry.content) != "")) ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func entryHasToolOutputMode(entry chatEntry) bool {
	return (entry.role == "tool" && (isToolActiveStatus(entry.status) || isToolResultStatus(entry.status) || entry.content != "")) ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func (m *chatModel) applyToolOutputMode() {
	if !m.toolOutputMode {
		return
	}
	for i := range m.entries {
		m.applyToolOutputModeTo(i)
	}
}

func (m *chatModel) applyToolOutputModeTo(index int) {
	if !m.toolOutputMode || index < 0 || index >= len(m.entries) {
		return
	}
	if !entryHasToolOutputMode(m.entries[index]) {
		return
	}
	if m.entries[index].expanded == m.toolOutputOpen {
		return
	}
	m.entries[index].expanded = m.toolOutputOpen
	m.markEntryDirty(index)
}

func (m *chatModel) groupCompletedToolHistory() {
	m.entries = groupCompletedToolEntries(m.entries)
	m.applyToolOutputMode()
}

func (m *chatModel) ensureAssistantEntry() int {
	if len(m.entries) > 0 && m.entries[len(m.entries)-1].role == "assistant" {
		return len(m.entries) - 1
	}
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "assistant"}))
	return len(m.entries) - 1
}

func (m chatModel) renderTranscript(width int) string {
	var b strings.Builder
	for index, entry := range m.entries {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		prefix, body := m.renderEntry(entry)
		prefixWidth := lipgloss.Width(prefix)
		continuation := ""
		if prefixWidth > 0 {
			continuation = strings.Repeat(" ", prefixWidth)
		}
		for i, line := range m.renderEntryLinesCached(index, entry, body, width-prefixWidth) {
			if i == 0 {
				b.WriteString(prefix)
				b.WriteString(line)
			} else {
				b.WriteString(continuation)
				b.WriteString(line)
			}
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func (m chatModel) renderEntryLinesCached(index int, entry chatEntry, body string, width int) []string {
	key := entryRenderKey(entry, body, width)
	if index >= 0 && index < len(m.entries) {
		cached := m.entries[index]
		if cached.renderKey == key && cached.renderLines != nil {
			return cached.renderLines
		}
	}

	lines := m.renderEntryLines(entry, body, width)
	if index >= 0 && index < len(m.entries) {
		m.entries[index].renderKey = key
		m.entries[index].renderLines = slices.Clone(lines)
	}
	return lines
}

func (m chatModel) transcriptLines(width int) []string {
	transcript := m.renderTranscript(width)
	transcript = strings.TrimRight(transcript, "\n")
	if transcript == "" {
		return nil
	}
	return strings.Split(transcript, "\n")
}

func (m chatModel) visibleTranscriptLinesForLines(lines []string, available int) []string {
	if available <= 0 {
		return nil
	}
	start := 0
	if len(lines) > available {
		start = m.visibleTranscriptStartLineForLines(len(lines), available)
		lines = lines[start : start+available]
	}
	lines = m.applyTranscriptSelection(lines, start)
	return lines
}

func (m chatModel) visibleTranscriptStartLine(width, available int) int {
	return m.visibleTranscriptStartLineForLines(len(m.transcriptLines(width)), available)
}

func (m chatModel) visibleTranscriptStartLineForLines(total, available int) int {
	if total <= available || available <= 0 {
		return 0
	}
	maxScroll := total - available
	scroll := clamp(m.scroll, 0, maxScroll)
	return maxScroll - scroll
}

func (m chatModel) viewWidth() int {
	if m.width > 0 {
		return m.width
	}
	return 80
}

func (m chatModel) transcriptHeight() int {
	width := m.width
	if width <= 0 {
		width = 80
	}
	height := m.height
	if height <= 0 {
		height = 24
	}
	lineCount := len(m.transcriptLines(width))
	baseHeaderHeight := 2
	baseMaxHeight := height - baseHeaderHeight
	baseBottomLines := m.bottomLines(width, baseMaxHeight)
	baseHeight := max(0, baseMaxHeight-len(baseBottomLines)-transcriptInputGap(baseMaxHeight, len(baseBottomLines), lineCount))
	if lineCount <= baseHeight {
		return lineCount
	}
	statusHeaderHeight := 3
	statusMaxHeight := height - statusHeaderHeight
	statusBottomLines := m.bottomLines(width, statusMaxHeight)
	return max(0, statusMaxHeight-len(statusBottomLines)-transcriptInputGap(statusMaxHeight, len(statusBottomLines), lineCount))
}

func (m chatModel) transcriptLayout() (top, height int) {
	width := m.width
	if width <= 0 {
		width = 80
	}
	viewHeight := m.height
	if viewHeight <= 0 {
		viewHeight = 24
	}
	headerHeight := len(m.headerLines())
	bottomLines := m.bottomLines(width, viewHeight-headerHeight)
	transcriptLineCount := len(m.transcriptLines(width))
	return headerHeight, max(0, viewHeight-headerHeight-len(bottomLines)-transcriptInputGap(viewHeight-headerHeight, len(bottomLines), transcriptLineCount))
}

func (m chatModel) maxScroll() int {
	width := m.width
	if width <= 0 {
		width = 80
	}
	return max(0, len(m.transcriptLines(width))-m.transcriptHeight())
}

func (m chatModel) bottomLines(width, maxHeight int) []string {
	var lines []string
	if m.modelPicker != nil {
		lines = append(lines, m.renderInlineModelPicker(width)...)
	} else {
		lines = append(lines, m.completionLines(width)...)
	}
	lines = append(lines, m.queuedLines(width)...)

	actionStatusLines := m.renderActionStatusLines(width)
	approvalLines := m.renderApprovalPromptLines(width)
	if len(actionStatusLines) == 0 {
		actionStatusLines = []string{""}
	}
	if maxHeight > 0 {
		maxApprovalLines := max(0, maxHeight-len(lines)-len(actionStatusLines)-3)
		if len(approvalLines) > maxApprovalLines {
			approvalLines = approvalLines[:maxApprovalLines]
		}
		maxActionStatusLines := max(1, maxHeight-len(lines)-len(approvalLines)-3)
		if len(actionStatusLines) > maxActionStatusLines {
			actionStatusLines = actionStatusLines[:maxActionStatusLines]
		}
	}

	lines = append(lines, actionStatusLines...)
	lines = append(lines, approvalLines...)
	modelLines := m.renderModelStatusLines(width)
	fixedLines := len(lines) + len(modelLines) + 2
	inputBodyLines := maxInputBoxBodyLines
	if maxHeight > 0 {
		inputBodyLines = min(inputBodyLines, max(1, maxHeight-fixedLines))
	}
	inputCursor := m.normalizedInputCursor()
	if m.approvalPrompt != nil || m.cloudAuthPrompt != nil {
		inputCursor = -1
	}
	lines = append(lines, renderInputBoxLines(string(m.input), inputCursor, width, inputBodyLines, m.emptyInputPlaceholder())...)
	lines = append(lines, modelLines...)
	return lines
}

func (m chatModel) renderModelStatusLines(width int) []string {
	if m.modelPicker != nil {
		return nil
	}
	var parts []string
	if model := strings.TrimSpace(m.opts.Model); model != "" {
		parts = append(parts, model)
	}
	if contextStatus := m.contextStatus(); contextStatus != "" {
		parts = append(parts, contextStatus)
	}
	if notice := m.permissionModeNotice(); notice != "" {
		parts = append(parts, notice)
	}
	if len(parts) == 0 {
		return nil
	}
	lines := wrapChatText(strings.Join(parts, "  "), width)
	for i := range lines {
		lines[i] = renderFooterPlainLine(lines[i])
	}
	return lines
}

func (m chatModel) renderActionStatusLines(width int) []string {
	if activity := m.activityLine(); activity != "" {
		return []string{chatMetaStyle.Render(activity)}
	}
	if notificationLines := m.renderNotificationLines(width); len(notificationLines) > 0 {
		return notificationLines
	}
	return nil
}

func transcriptInputGap(maxHeight, bottomLineCount, _ int) int {
	const desiredGap = 1
	if maxHeight <= 0 {
		return desiredGap
	}
	available := maxHeight - bottomLineCount
	if available <= 1 {
		return 0
	}
	return min(desiredGap, available-1)
}

func (m *chatModel) scrollBy(lines int) {
	if lines == 0 {
		return
	}
	m.scroll = clamp(m.scroll+lines, 0, m.maxScroll())
}

var chatANSISequencePattern = regexp.MustCompile(`\x1b\[[0-9;:]*[A-Za-z]`)

func stripChatANSI(s string) string {
	return chatANSISequencePattern.ReplaceAllString(s, "")
}

func (m chatModel) normalizedSelectionRange() (chatSelectionPoint, chatSelectionPoint, bool) {
	return normalizedSelectionRangeFor(m.selection)
}

func normalizedSelectionRangeFor(selection chatSelection) (chatSelectionPoint, chatSelectionPoint, bool) {
	if !selection.active {
		return chatSelectionPoint{}, chatSelectionPoint{}, false
	}
	start, end := selection.anchor, selection.cursor
	if start.line > end.line || (start.line == end.line && start.col > end.col) {
		start, end = end, start
	}
	if start.line == end.line && start.col == end.col {
		return chatSelectionPoint{}, chatSelectionPoint{}, false
	}
	return start, end, true
}

func (m chatModel) applyTranscriptSelection(lines []string, offset int) []string {
	start, end, ok := m.normalizedSelectionRange()
	if !ok || len(lines) == 0 {
		return lines
	}
	out := slices.Clone(lines)
	for i, line := range out {
		lineIndex := offset + i
		if lineIndex < start.line || lineIndex > end.line {
			continue
		}
		text := stripChatANSI(line)
		startCol, endCol := 0, len([]rune(text))
		if lineIndex == start.line {
			startCol = displayColumnToRuneIndex(text, start.col)
		}
		if lineIndex == end.line {
			endCol = displayColumnToRuneIndex(text, end.col)
		}
		if startCol > endCol {
			startCol, endCol = endCol, startCol
		}
		out[i] = renderSelectedTranscriptLine(text, startCol, endCol)
	}
	return out
}

func renderSelectedTranscriptLine(line string, startCol, endCol int) string {
	runes := []rune(line)
	startCol = clamp(startCol, 0, len(runes))
	endCol = clamp(endCol, 0, len(runes))
	if startCol == endCol {
		return line
	}
	return string(runes[:startCol]) + chatSelectionStyle.Render(string(runes[startCol:endCol])) + string(runes[endCol:])
}

func displayColumnToRuneIndex(line string, col int) int {
	if col <= 0 {
		return 0
	}
	width := 0
	for i, r := range []rune(line) {
		next := width + runewidth.RuneWidth(r)
		if col < next {
			return i
		}
		width = next
	}
	return len([]rune(line))
}

func (m chatModel) selectedTranscriptText(width int) string {
	start, end, ok := m.normalizedSelectionRange()
	if !ok {
		return ""
	}
	lines := m.transcriptLines(width)
	if len(lines) == 0 {
		return ""
	}
	start.line = clamp(start.line, 0, len(lines)-1)
	end.line = clamp(end.line, 0, len(lines)-1)
	var selected []string
	for lineIndex := start.line; lineIndex <= end.line; lineIndex++ {
		text := stripChatANSI(lines[lineIndex])
		runes := []rune(text)
		startCol, endCol := 0, len(runes)
		if lineIndex == start.line {
			startCol = displayColumnToRuneIndex(text, start.col)
		}
		if lineIndex == end.line {
			endCol = displayColumnToRuneIndex(text, end.col)
		}
		if startCol > endCol {
			startCol, endCol = endCol, startCol
		}
		selected = append(selected, string(runes[startCol:endCol]))
	}
	return strings.TrimRight(strings.Join(selected, "\n"), "\n")
}

func (m chatModel) renderEntry(entry chatEntry) (string, string) {
	switch entry.role {
	case "user":
		return "", entry.content
	case "assistant":
		return chatAssistantStyle.Render("•") + " ", entry.content
	case "slash":
		return chatMetaStyle.Render("•") + " ", entry.content
	case "compaction_summary":
		prefix := toolStatusStyle(entry.status).Render("•") + " "
		return prefix, compactionSummaryStatusLine(entry)
	case "tool":
		prefix := toolStatusStyle(entry.status).Render("•") + " "
		return prefix, toolStatusLine(entry)
	case "tool_group":
		prefix := toolGroupPrefixStyle(entry).Render("•") + " "
		return prefix, toolGroupStatusLine(entry)
	case "error":
		return chatErrorStyle.Render("err ") + " ", entry.content
	case "system":
		return "", entry.content
	default:
		return "", entry.content
	}
}

func (m chatModel) renderEntryLines(entry chatEntry, body string, width int) []string {
	if width < 20 {
		width = 20
	}
	switch entry.role {
	case "assistant":
		lines := splitRenderedBody(renderMarkdownForView(body, width))
		lines = append(lines, renderMetricsLines(entry.metrics, width)...)
		return lines
	case "system", "slash":
		return splitRenderedBody(renderMarkdownForView(body, width))
	case "user":
		return renderUserMessageLines(body, width)
	case "compaction_summary":
		return renderCompactionSummaryLines(entry, width)
	case "tool":
		if entryHasExpandableOutput(entry) {
			return renderToolResultLines(entry, width)
		}
		if isToolResultStatus(entry.status) {
			return renderToolResultLines(entry, width)
		}
		return wrapChatText(body, width)
	case "tool_group":
		return renderToolGroupLines(entry, width)
	default:
		return wrapChatText(body, width)
	}
}

func renderUserMessageLines(content string, width int) []string {
	return renderPromptRow(chatPromptPrefix+content, width)
}

func renderMetricsLines(metrics *api.Metrics, width int) []string {
	summary := metricsSummaryLines(metrics)
	if len(summary) == 0 {
		return nil
	}
	var lines []string
	for _, line := range summary {
		for _, wrapped := range wrapChatText(line, width) {
			lines = append(lines, chatMetaStyle.Render(wrapped))
		}
	}
	return lines
}

func metricsSummaryLines(metrics *api.Metrics) []string {
	if metrics == nil || metricsEmpty(*metrics) {
		return nil
	}
	var lines []string
	if metrics.TotalDuration > 0 {
		lines = append(lines, fmt.Sprintf("total duration:       %v", metrics.TotalDuration))
	}
	if metrics.LoadDuration > 0 {
		lines = append(lines, fmt.Sprintf("load duration:        %v", metrics.LoadDuration))
	}
	if metrics.PromptEvalCount > 0 {
		lines = append(lines, fmt.Sprintf("prompt eval count:    %d token(s)", metrics.PromptEvalCount))
	}
	if metrics.PromptEvalDuration > 0 {
		lines = append(lines, fmt.Sprintf("prompt eval duration: %s", metrics.PromptEvalDuration))
		lines = append(lines, fmt.Sprintf("prompt eval rate:     %.2f tokens/s", float64(metrics.PromptEvalCount)/metrics.PromptEvalDuration.Seconds()))
	}
	if metrics.EvalCount > 0 {
		lines = append(lines, fmt.Sprintf("eval count:           %d token(s)", metrics.EvalCount))
	}
	if metrics.EvalDuration > 0 {
		lines = append(lines, fmt.Sprintf("eval duration:        %s", metrics.EvalDuration))
		lines = append(lines, fmt.Sprintf("eval rate:            %.2f tokens/s", float64(metrics.EvalCount)/metrics.EvalDuration.Seconds()))
	}
	return lines
}

func metricsEmpty(metrics api.Metrics) bool {
	return metrics.TotalDuration <= 0 &&
		metrics.LoadDuration <= 0 &&
		metrics.PromptEvalCount <= 0 &&
		metrics.PromptEvalDuration <= 0 &&
		metrics.EvalCount <= 0 &&
		metrics.EvalDuration <= 0
}

func renderHistoryMessages(messages []api.Message, width int) []string {
	if width < 20 {
		width = 20
	}
	var lines []string
	for _, msg := range messages {
		role := strings.TrimSpace(msg.Role)
		if role == "" {
			role = "message"
		}
		lines = append(lines, renderHistoryStyledLine("", role, width, historyRoleStyle(role))...)

		var meta []string
		if msg.ToolName != "" {
			meta = append(meta, "tool: `"+msg.ToolName+"`")
		}
		if msg.ToolCallID != "" {
			meta = append(meta, "tool call: `"+msg.ToolCallID+"`")
		}
		if len(meta) > 0 {
			lines = append(lines, "  "+renderHistoryMetaLine(strings.Join(meta, " · ")))
		}

		if strings.TrimSpace(msg.Thinking) != "" {
			lines = append(lines, renderHistoryFieldLines("  ", "thinking", msg.Thinking, width)...)
		}
		if len(msg.ToolCalls) > 0 {
			lines = append(lines, "  "+chatHistoryLabelStyle.Render("tool calls:"))
			for _, call := range msg.ToolCalls {
				lines = append(lines, renderHistoryToolCallLines(call, width)...)
			}
		}
		if strings.TrimSpace(msg.Content) != "" {
			content := msg.Content
			if msg.Role == "tool" {
				content = stripInternalToolTruncationMarkers(content)
			}
			lines = append(lines, renderHistoryFieldLines("  ", "content", content, width)...)
		}
		if strings.TrimSpace(msg.Thinking) == "" && len(msg.ToolCalls) == 0 && strings.TrimSpace(msg.Content) == "" {
			lines = append(lines, "  "+chatHistoryTextStyle.Render("_empty_"))
		}
		lines = append(lines, "")
	}
	return stringsTrimTrailingEmptyLines(lines)
}

func renderHistoryFieldLines(indent, label, content string, width int) []string {
	content = strings.TrimRight(content, "\n")
	if content == "" {
		return nil
	}
	if !strings.Contains(content, "\n") {
		return renderHistoryLabelValue(indent, label, content, width)
	}
	lines := []string{indent + chatHistoryLabelStyle.Render(label+":")}
	for _, line := range strings.Split(content, "\n") {
		lines = append(lines, renderHistoryCodeLine(indent, line, width)...)
	}
	return lines
}

func renderHistoryToolCallLines(call api.ToolCall, width int) []string {
	name := call.Function.Name
	if name == "" {
		name = "tool"
	}
	line := "    - "
	if call.ID != "" {
		line += "`" + call.ID + "` "
	}
	line += toolDisplayName(name)
	lines := renderHistoryStyledLine("", line, width, chatHistoryTextStyle)

	args := call.Function.Arguments.ToMap()
	if len(args) == 0 {
		return lines
	}
	lines = append(lines, "      "+chatHistoryLabelStyle.Render("args:"))
	data, err := json.MarshalIndent(args, "", "  ")
	if err != nil {
		lines = append(lines, renderHistoryCodeLine("      ", fmt.Sprint(args), width)...)
		return lines
	}
	for _, line := range strings.Split(string(data), "\n") {
		lines = append(lines, renderHistoryCodeLine("      ", line, width)...)
	}
	return lines
}

func stringsTrimTrailingEmptyLines(lines []string) []string {
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func renderHistoryLabelValue(indent, label, value string, width int) []string {
	labelText := label + ":"
	value = strings.TrimSpace(value)
	if value == "" {
		return []string{indent + chatHistoryLabelStyle.Render(labelText)}
	}

	prefixWidth := lipgloss.Width(indent) + lipgloss.Width(labelText) + 1
	wrapped := wrapChatText(value, max(10, width-prefixWidth))
	lines := make([]string, 0, len(wrapped))
	for i, line := range wrapped {
		if i == 0 {
			lines = append(lines, indent+chatHistoryLabelStyle.Render(labelText)+" "+renderHistoryInline(line, chatHistoryTextStyle))
			continue
		}
		lines = append(lines, indent+strings.Repeat(" ", lipgloss.Width(labelText)+1)+renderHistoryInline(line, chatHistoryTextStyle))
	}
	return lines
}

func renderHistoryStyledLine(indent, text string, width int, style lipgloss.Style) []string {
	wrapped := wrapChatText(text, max(10, width-lipgloss.Width(indent)))
	for i, line := range wrapped {
		wrapped[i] = indent + renderHistoryInline(line, style)
	}
	return wrapped
}

func renderHistoryCodeLine(indent, code string, width int) []string {
	codeIndent := indent + "  "
	wrapped := wrapChatText(code, max(10, width-lipgloss.Width(codeIndent)))
	for i, line := range wrapped {
		wrapped[i] = codeIndent + chatHistoryCodeStyle.Render(line)
	}
	return wrapped
}

func renderHistoryMetaLine(text string) string {
	parts := strings.Split(text, " · ")
	for i, part := range parts {
		if label, value, ok := historyLabelValue(part); ok {
			parts[i] = chatHistoryLabelStyle.Render(label+":") + " " + renderHistoryInline(value, chatHistoryTextStyle)
			continue
		}
		parts[i] = renderHistoryInline(part, chatHistoryTextStyle)
	}
	return strings.Join(parts, chatHistoryLabelStyle.Render(" · "))
}

func renderHistoryInline(text string, style lipgloss.Style) string {
	text = strings.ReplaceAll(text, "**", "")
	var b strings.Builder
	for {
		before, rest, ok := strings.Cut(text, "`")
		b.WriteString(style.Render(before))
		if !ok {
			break
		}
		code, after, ok := strings.Cut(rest, "`")
		if !ok {
			b.WriteString(style.Render("`" + rest))
			break
		}
		b.WriteString(chatHistoryCodeStyle.Render(code))
		text = after
	}
	return b.String()
}

func historyRoleStyle(role string) lipgloss.Style {
	switch role {
	case "system":
		return chatHistorySystemRoleStyle
	case "user":
		return chatHistoryUserRoleStyle
	case "assistant":
		return chatHistoryAssistantRoleStyle
	case "tool":
		return chatHistoryToolRoleStyle
	default:
		return chatHistoryTitleStyle
	}
}

func historyLabelValue(text string) (string, string, bool) {
	label, value, ok := strings.Cut(text, ":")
	if !ok {
		return "", "", false
	}
	label = strings.TrimSpace(label)
	if !historyLabel(label) {
		return "", "", false
	}
	return label, strings.TrimSpace(value), true
}

func historyLabel(label string) bool {
	switch label {
	case "args", "content", "thinking", "tool", "tool call", "tool calls":
		return true
	default:
		return false
	}
}

func (m chatModel) queuedLines(width int) []string {
	if len(m.queued) == 0 {
		return nil
	}
	limit := 2
	if width < 40 {
		limit = 1
	}
	lines := make([]string, 0, min(len(m.queued), limit)+1)
	for i, queued := range m.queued {
		if i >= limit {
			lines = append(lines, chatMetaStyle.Render(fmt.Sprintf("queued +%d more", len(m.queued)-i)))
			break
		}
		label := fmt.Sprintf("queued %d: %s", i+1, queued)
		lines = append(lines, chatMetaStyle.Render(truncateRunes(label, max(20, width))))
	}
	return lines
}

func renderToolResultLines(entry chatEntry, width int) []string {
	lines := wrapChatText(toolStatusLine(entry), width)
	if !entry.expanded {
		return lines
	}

	detailLines := renderToolCallDetailLines(entry, width)
	if len(detailLines) > 0 {
		lines = append(lines, "")
		lines = append(lines, detailLines...)
	}
	if strings.TrimSpace(entry.content) != "" {
		lines = append(lines, "")
		lines = append(lines, renderToolOutputLines(entry, entry.content, width)...)
	}
	return lines
}

func renderToolGroupLines(entry chatEntry, width int) []string {
	lines := wrapChatText(toolGroupStatusLine(entry), width)
	if !entry.expanded {
		return lines
	}

	for i, tool := range entry.tools {
		if i > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, "  "+toolGroupChildStatusLine(tool))
		if detailLines := renderToolCallDetailLines(tool, width-4); len(detailLines) > 0 {
			lines = append(lines, indentLines(detailLines, "    ")...)
		}
		if strings.TrimSpace(tool.content) == "" {
			continue
		}
		lines = append(lines, indentLines(renderToolOutputLines(tool, tool.content, width-4), "    ")...)
	}
	return lines
}

func renderCompactionSummaryLines(entry chatEntry, width int) []string {
	lines := wrapChatText(compactionSummaryStatusLine(entry), width)
	if !entry.expanded || strings.TrimSpace(entry.content) == "" {
		return lines
	}
	lines = append(lines, "")
	lines = append(lines, indentLines(splitRenderedBody(renderMarkdownForView(entry.content, width-2)), "  ")...)
	return lines
}

func compactionSummaryStatusLine(entry chatEntry) string {
	segment := toolStatusStyle(entry.status).Render(toolStatusLabel(entry))
	if segment == "" {
		return "Compacted summary"
	}
	return fmt.Sprintf("Compacted summary %s", segment)
}

func toolGroupChildStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	segment := renderToolStatusSegment(entry)
	if segment == "" {
		return boldToolInvocationName(label)
	}
	return fmt.Sprintf("%s %s", boldToolInvocationName(label), segment)
}

func boldToolInvocationName(label string) string {
	name, rest, ok := strings.Cut(label, "(")
	if !ok || name == "" {
		return chatHeaderStyle.Render(label)
	}
	return chatHeaderStyle.Render(name) + "(" + rest
}

func renderToolOutputLines(entry chatEntry, output string, width int) []string {
	output = stripInternalToolTruncationMarkers(output)
	if looksLikeUnifiedDiff(output) {
		return splitRenderedBody(renderDiffForView(output, width))
	}
	if toolOutputUsesMarkdown(entry.detail) {
		return splitRenderedBody(renderMarkdownForView(output, width))
	}
	return wrapChatText(output, width)
}

func stripInternalToolTruncationMarkers(output string) string {
	lines := strings.Split(output, "\n")
	filtered := lines[:0]
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "[tool output truncated: ") && strings.HasSuffix(trimmed, "]") {
			continue
		}
		filtered = append(filtered, line)
	}
	return strings.TrimSpace(strings.Join(filtered, "\n"))
}

func looksLikeUnifiedDiff(output string) bool {
	lines := strings.Split(output, "\n")
	hasOldFile := false
	hasNewFile := false
	hasHunk := false
	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "diff --git "):
			return true
		case strings.HasPrefix(line, "--- "):
			hasOldFile = true
		case strings.HasPrefix(line, "+++ "):
			hasNewFile = true
		case strings.HasPrefix(line, "@@ "):
			hasHunk = true
		}
	}
	return hasHunk || (hasOldFile && hasNewFile)
}

func renderDiffForView(diff string, width int) string {
	if width < 20 {
		width = 20
	}
	lines := strings.Split(strings.TrimRight(diff, "\n"), "\n")
	rendered := make([]string, 0, len(lines))
	for _, line := range lines {
		style := diffLineStyle(line)
		for _, wrapped := range wrapChatText(line, width) {
			rendered = append(rendered, style.Render(wrapped))
		}
	}
	return strings.Join(rendered, "\n")
}

func diffLineStyle(line string) lipgloss.Style {
	switch {
	case strings.HasPrefix(line, "diff --git "),
		strings.HasPrefix(line, "--- "),
		strings.HasPrefix(line, "+++ "):
		return chatDiffFileStyle
	case strings.HasPrefix(line, "@@ "):
		return chatDiffHunkStyle
	case strings.HasPrefix(line, "+"):
		return chatDiffAddStyle
	case strings.HasPrefix(line, "-"):
		return chatDiffDeleteStyle
	case strings.HasPrefix(line, "index "),
		strings.HasPrefix(line, "new file "),
		strings.HasPrefix(line, "deleted file "),
		strings.HasPrefix(line, "similarity index "),
		strings.HasPrefix(line, "rename from "),
		strings.HasPrefix(line, "rename to "),
		strings.HasPrefix(line, "\\ "):
		return chatDiffMetaStyle
	default:
		return chatToolStyle
	}
}

func isToolActiveStatus(status string) bool {
	return status == "queued" || status == "running" || status == "approval"
}

func isToolResultStatus(status string) bool {
	return status == "done" || status == "error"
}

// renderToolStatusSegment returns the styled status text for a tool entry,
// or "" when the entry has no status word to show (e.g. a completed or failed
// tool, where the colored dot alone conveys the result).
func renderToolStatusSegment(entry chatEntry) string {
	s := toolStatusLabel(entry)
	if s == "" {
		return ""
	}
	return toolStatusStyle(entry.status).Render(s)
}

func toolStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	segment := renderToolStatusSegment(entry)
	if segment == "" {
		return label
	}
	return fmt.Sprintf("%s %s", label, segment)
}

func toolGroupStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = fmt.Sprintf("Tool calls (%d)", len(entry.tools))
	}

	// Color the label with the group's outcome color (green/red/amber/yellow);
	// the small dot prefix already carries the same color, so the label
	// reinforces which outcome the group had without adding a status word.
	styledLabel := toolGroupPrefixStyle(entry).Render(label)
	segment := renderToolStatusSegment(entry)
	if segment == "" {
		return styledLabel
	}
	return fmt.Sprintf("%s %s", styledLabel, segment)
}

func toolGroupPrefixStyle(entry chatEntry) lipgloss.Style {
	succeeded, failed := toolGroupResultCounts(entry.tools)
	switch {
	case succeeded > 0 && failed > 0:
		return chatToolMixedStyle
	case succeeded > 0:
		return chatToolDoneStyle
	default:
		return toolStatusStyle(entry.status)
	}
}

func toolGroupResultCounts(tools []chatEntry) (succeeded int, failed int) {
	for _, tool := range tools {
		if tool.err != "" || tool.status == "error" {
			failed++
			continue
		}
		if tool.status == "done" {
			succeeded++
		}
	}
	return succeeded, failed
}

func toolStatusLabel(entry chatEntry) string {
	if entry.status == "approval" {
		return "needs approval"
	}
	if entry.status == "running" || entry.status == "queued" {
		return "in progress"
	}
	return ""
}

func toolStatusStyle(status string) lipgloss.Style {
	switch status {
	case "done":
		return chatToolDoneStyle
	case "error":
		return chatErrorStyle
	default:
		return chatToolRunningStyle
	}
}

func toolInvocationLabel(name string, args map[string]any) string {
	displayName := toolDisplayName(name)
	for _, key := range []string{"query", "url", "command", "path", "name"} {
		if value, ok := rawStringArg(args, key); ok {
			if isShellToolName(name) && key == "command" {
				value = truncateRunes(value, 100)
			}
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(value))
		}
	}
	if len(args) == 0 {
		return displayName
	}
	return fmt.Sprintf("%s(%s)", displayName, formatDisplayArgs(args))
}

func toolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
	case "powershell":
		return "PowerShell"
	case "read":
		return "Read"
	case "list":
		return "List"
	case "edit":
		return "Edit"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}

func toolOutputUsesMarkdown(name string) bool {
	switch name {
	case "read", "skill", "web_search", "web_fetch":
		return true
	default:
		return false
	}
}

func renderToolCallDetailLines(entry chatEntry, width int) []string {
	if len(entry.args) == 0 {
		return nil
	}
	if isShellToolName(entry.detail) {
		if command, ok := rawStringArg(entry.args, "command"); ok {
			return renderToolCallArgLine(shellPromptPrefix(entry.detail)+command, width)
		}
	}
	switch entry.detail {
	case "web_search":
		if query, ok := rawStringArg(entry.args, "query"); ok {
			return renderToolCallArgLine("query: "+query, width)
		}
	case "web_fetch":
		if targetURL, ok := rawStringArg(entry.args, "url"); ok {
			return renderToolCallArgLine("url: "+targetURL, width)
		}
	}
	return renderToolCallArgs(entry.args, width)
}

func isShellToolName(name string) bool {
	return name == "bash" || name == "powershell"
}

func formatDisplayArgs(args map[string]any) string {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		value := truncateRunes(fmt.Sprintf("%v", args[key]), 100)
		parts = append(parts, fmt.Sprintf("%s=%s", key, strconv.Quote(value)))
	}
	return strings.Join(parts, ", ")
}

func shellPromptPrefix(toolName string) string {
	if toolName == "powershell" {
		return "PS> "
	}
	return "$ "
}

func renderToolCallArgs(args map[string]any, width int) []string {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var lines []string
	for _, key := range keys {
		value := toolArgDisplayValue(args[key])
		if strings.Contains(value, "\n") {
			lines = append(lines, chatMetaStyle.Render(key+":"))
			lines = append(lines, indentLines(renderToolCallArgLine(value, max(20, width-2)), "  ")...)
			continue
		}
		lines = append(lines, renderToolCallArgLine(key+": "+value, width)...)
	}
	return lines
}

func renderToolCallArgLine(line string, width int) []string {
	wrapped := wrapChatText(line, width)
	for i := range wrapped {
		wrapped[i] = chatMetaStyle.Render(wrapped[i])
	}
	return wrapped
}

func toolArgDisplayValue(value any) string {
	if value == nil {
		return "null"
	}
	if text, ok := value.(string); ok {
		return text
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err == nil {
		return string(data)
	}
	return fmt.Sprint(value)
}

func rawStringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return value, true
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func (m chatModel) notificationLine() string {
	status := strings.TrimSpace(m.status)
	if status == "" || status == "ready" {
		return ""
	}
	if m.running || m.compacting || m.approvalPrompt != nil {
		return ""
	}
	switch status {
	case "queued", "running", "compacting", "approval required", "full access enabled", "review mode enabled":
		return ""
	default:
		return status
	}
}

func (m chatModel) renderNotificationLines(width int) []string {
	line := m.notificationLine()
	if line == "" {
		return nil
	}
	lines := wrapChatText(line, width)
	for i, wrapped := range lines {
		lines[i] = chatNotificationStyle.Render(wrapped)
	}
	return lines
}

func renderFooterPlainLine(line string) string {
	const fullAccess = "full access"
	if !strings.Contains(line, fullAccess) {
		return chatMetaStyle.Render(line)
	}

	var b strings.Builder
	for {
		before, after, ok := strings.Cut(line, fullAccess)
		if before != "" {
			b.WriteString(chatMetaStyle.Render(before))
		}
		if !ok {
			break
		}
		b.WriteString(chatFullAccessStyle.Render(fullAccess))
		line = after
	}
	return b.String()
}

func (m chatModel) permissionModeNotice() string {
	if notice := strings.TrimSpace(m.permissionNotice); notice != "" {
		return notice
	}
	switch strings.TrimSpace(m.status) {
	case "full access enabled", "review mode enabled":
		return strings.TrimSpace(m.status)
	}
	if m.allowAllTools && m.notificationLine() == "" {
		return "full access enabled"
	}
	return ""
}

func (m *chatModel) refreshContextWindowTokens(modelName string) {
	if m == nil || m.opts.ContextWindowTokensForModel == nil {
		return
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return
	}
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	tokens := m.opts.ContextWindowTokensForModel(ctx, modelName, m.opts.ContextWindowTokens)
	m.updateContextWindowTokens(tokens)
}

func (m *chatModel) updateContextWindowTokens(tokens int) {
	if tokens <= 0 || tokens == m.opts.ContextWindowTokens {
		return
	}
	m.opts.ContextWindowTokens = tokens
	if compactor, ok := m.opts.Compactor.(*coreagent.SimpleCompactor); ok && compactor != nil {
		compactor.Options.ContextWindowTokens = tokens
	}
}

func (m chatModel) responseModelName(response *api.ChatResponse) string {
	if response != nil {
		if strings.TrimSpace(response.Model) != "" {
			return response.Model
		}
		if strings.TrimSpace(response.RemoteModel) != "" {
			return response.RemoteModel
		}
	}
	return m.opts.Model
}

func (m chatModel) currentWorkingDir() string {
	if strings.TrimSpace(m.workingDir) != "" {
		return m.workingDir
	}
	return m.opts.WorkingDir
}

func (m chatModel) activityLine() string {
	if m.approvalPrompt != nil {
		return ""
	}
	if m.preloadingModel != "" && !m.running && !m.compacting {
		return ""
	}
	if !m.running && !m.compacting && m.preloadingModel == "" && m.approvalPrompt == nil {
		return ""
	}
	label := m.activityLabel()
	if label == "" {
		if !m.waitingForModel() || m.spinner < waitingSpinnerTicks {
			return ""
		}
		if m.preloadingModel != "" {
			return statusWithSpinner(m.spinnerFrame(), "Working")
		}
		return statusWithSpinner(m.spinnerFrame(), "Working")
	}
	return statusWithSpinner(m.spinnerFrame(), label)
}

func (m chatModel) activityLabel() string {
	if m.status == "canceling" {
		return "canceling"
	}
	if m.compacting {
		if m.compactingTokens > 0 {
			return "Compacting " + formatTokenCount(m.compactingTokens)
		}
		return "Compacting"
	}
	if m.thinking {
		if m.thinkingTokens > 0 {
			return "Thinking " + formatTokenCount(m.thinkingTokens)
		}
		return "Thinking"
	}
	start := m.currentTurnEntryStart()
	for i := len(m.entries) - 1; i >= start; i-- {
		entry := m.entries[i]
		switch entry.role {
		case "tool":
			if isToolActiveStatus(entry.status) {
				return ""
			}
		case "assistant":
			if entry.content != "" {
				return ""
			}
		}
	}
	return ""
}

func (m chatModel) waitingForModel() bool {
	if m.preloadingModel != "" && !m.compacting && m.approvalPrompt == nil && m.status != "canceling" {
		return true
	}
	if !m.running || m.compacting || m.approvalPrompt != nil || m.thinking || m.status == "canceling" {
		return false
	}
	if m.awaitingModel {
		return true
	}
	start := m.currentTurnEntryStart()
	for i := len(m.entries) - 1; i >= start; i-- {
		entry := m.entries[i]
		switch entry.role {
		case "tool":
			if isToolActiveStatus(entry.status) {
				return false
			}
		case "assistant":
			return entry.content == ""
		}
	}
	return true
}

func (m chatModel) currentTurnEntryStart() int {
	for i := len(m.entries) - 1; i >= 0; i-- {
		if m.entries[i].role == "user" {
			return i + 1
		}
	}
	return 0
}

func (m *chatModel) applyResponseMetrics(response *api.ChatResponse) {
	if response == nil {
		return
	}
	tokens := 0
	if response.PromptEvalCount > 0 {
		tokens = response.PromptEvalCount
	}
	if response.EvalCount > 0 {
		tokens += response.EvalCount
	}
	if tokens <= 0 {
		return
	}
	if m.running {
		if tokens > m.contextTokens {
			m.contextTokens = tokens
		}
		return
	}
	if tokens > 0 {
		m.contextTokens = tokens
		m.contextEstimate = false
	}
}

func (m chatModel) estimatePromptTokens(messages []api.Message, systemPrompt string) int {
	if strings.TrimSpace(systemPrompt) == "" {
		systemPrompt = m.systemPrompt("")
	}
	var tools api.Tools
	if m.opts.Tools != nil {
		tools = m.opts.Tools.Tools()
	}
	return estimatePromptTokenCount(systemPrompt, messages, tools, m.opts.Format)
}

func estimatePromptTokenCount(systemPrompt string, messages []api.Message, tools api.Tools, format string) int {
	total := approximateTokenCount(systemPrompt)
	for _, msg := range messages {
		total += approximateTokenCount(msg.Role)
		total += approximateTokenCount(msg.Content)
		total += approximateTokenCount(msg.Thinking)
		total += approximateTokenCount(msg.ToolName)
		total += approximateTokenCount(msg.ToolCallID)
		for _, call := range msg.ToolCalls {
			total += approximateTokenCount(call.Function.Name)
			total += approximateTokenCount(call.Function.Arguments.String())
		}
	}
	total += approximateTokenCount(tools.String())
	total += approximateTokenCount(format)
	return total
}

func approximateTokenCount(text string) int {
	n := len([]rune(text))
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}

func formatTokenCount(count int) string {
	if count == 1 {
		return "1 token"
	}
	return fmt.Sprintf("%d tokens", count)
}

func (m chatModel) contextStatus() string {
	window := coreagent.ResolveContextWindowTokens(m.opts.Options, m.opts.ContextWindowTokens)
	if window <= 0 {
		return ""
	}
	used := max(m.contextTokens, 0)
	percent := 0
	if window > 0 {
		percent = (used*100 + window/2) / window
	}

	prefix := ""
	if m.contextEstimate {
		prefix = "~"
	}

	threshold := coreagent.ResolveCompactionThreshold(m.opts.CompactionThreshold)
	compactAt := int(float64(window)*threshold + 0.999999)
	if compactAt <= 0 || compactAt > window {
		compactAt = window
	}

	if used >= compactAt {
		return fmt.Sprintf("ctx %s%s/%s (%d%%)", prefix, formatContextTokenCount(used), formatContextTokenCount(window), percent)
	}

	noticeDistance := int(float64(window)*0.1 + 0.999999)
	if noticeDistance < 1 {
		noticeDistance = 1
	}
	if compactAt-used <= noticeDistance {
		return fmt.Sprintf("ctx %s%s/%s (%d%%)", prefix, formatContextTokenCount(used), formatContextTokenCount(window), percent)
	}

	if percent > 60 {
		return fmt.Sprintf("ctx %s%s/%s (%d%%)", prefix, formatContextTokenCount(used), formatContextTokenCount(window), percent)
	}

	return ""
}

func formatContextTokenCount(value int) string {
	sign := ""
	if value < 0 {
		sign = "-"
		value = -value
	}
	if value >= 950_000 {
		return fmt.Sprintf("%s%dM", sign, int(float64(value)/1_000_000+0.5))
	}
	if value >= 10_240 {
		return fmt.Sprintf("%s%dk", sign, int(float64(value)/1024+0.5))
	}
	return sign + formatInteger(value)
}

func formatInteger(value int) string {
	sign := ""
	if value < 0 {
		sign = "-"
		value = -value
	}
	s := strconv.Itoa(value)
	if len(s) <= 3 {
		return sign + s
	}
	var b strings.Builder
	b.WriteString(sign)
	first := len(s) % 3
	if first == 0 {
		first = 3
	}
	b.WriteString(s[:first])
	for i := first; i < len(s); i += 3 {
		b.WriteByte(',')
		b.WriteString(s[i : i+3])
	}
	return b.String()
}

func (m chatModel) spinnerFrame() string {
	if len(chatSpinnerFrames) == 0 {
		return ""
	}
	return chatSpinnerFrames[m.spinner%len(chatSpinnerFrames)]
}

func statusWithSpinner(frame, label string) string {
	label = strings.TrimSpace(label)
	if label == "" {
		return strings.TrimSpace(frame)
	}
	return label + frame
}

func renderFullFrame(content string, width, height int) string {
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}
	rendered := lipgloss.NewStyle().MaxWidth(width).Render(content)
	lines := strings.Split(strings.TrimRight(rendered, "\n"), "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func renderFrameLines(lines []string, width, height int) string {
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	out := make([]string, 0, height)
	for _, line := range lines {
		out = append(out, padRenderedLine(clipRenderedLine(line, width), width))
	}
	for len(out) < height {
		out = append(out, strings.Repeat(" ", width))
	}
	return strings.Join(out, "\n")
}

func truncateRenderedLine(line string, width int) string {
	if width <= 0 || lipgloss.Width(line) <= width {
		return line
	}
	return lipgloss.NewStyle().MaxWidth(width).Render(line)
}

func clipRenderedLine(line string, width int) string {
	if width <= 0 {
		return ""
	}
	line, _, _ = strings.Cut(line, "\n")
	if lipgloss.Width(line) <= width {
		return line
	}
	clipped := lipgloss.NewStyle().MaxWidth(width).Render(line)
	clipped, _, _ = strings.Cut(clipped, "\n")
	return clipped
}

func clamp(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func newChatEntry(entry chatEntry) chatEntry {
	if entry.version <= 0 {
		entry.version = 1
	}
	return entry
}

func newSlashEntry(content string) chatEntry {
	return newChatEntry(chatEntry{role: "slash", content: content})
}

func (m *chatModel) markEntryDirty(index int) {
	if index < 0 || index >= len(m.entries) {
		return
	}
	entry := &m.entries[index]
	entry.version++
	if entry.version <= 0 {
		entry.version = 1
	}
	entry.renderKey = chatEntryRenderKey{}
	entry.renderLines = nil
}

func entryRenderKey(entry chatEntry, body string, width int) chatEntryRenderKey {
	if entry.version > 0 {
		return chatEntryRenderKey{width: width, version: entry.version}
	}
	return chatEntryRenderKey{width: width, hash: fallbackEntryRenderHash(entry, body)}
}

func fallbackEntryRenderHash(entry chatEntry, body string) string {
	var b strings.Builder
	writeEntryRenderHash(&b, entry)
	b.WriteString("\x00body\x00")
	b.WriteString(body)
	return b.String()
}

func writeEntryRenderHash(b *strings.Builder, entry chatEntry) {
	for _, value := range []string{
		entry.role,
		entry.content,
		entry.label,
		entry.detail,
		entry.status,
		entry.err,
		entry.toolID,
		strconv.FormatBool(entry.expanded),
		strconv.FormatInt(entry.startedAt.UnixNano(), 10),
		strconv.FormatInt(entry.finishedAt.UnixNano(), 10),
		strings.Join(metricsSummaryLines(entry.metrics), "\n"),
	} {
		b.WriteString(value)
		b.WriteByte(0)
	}
	for _, tool := range entry.tools {
		writeEntryRenderHash(b, tool)
	}
}

func entriesFromMessages(messages []api.Message) []chatEntry {
	entries := make([]chatEntry, 0, len(messages))
	toolCalls := make(map[string]api.ToolCall)
	for _, msg := range messages {
		switch msg.Role {
		case "user", "system":
			if summary, ok := compactionSummaryContent(msg); ok {
				entries = append(entries, newChatEntry(chatEntry{
					role:    "compaction_summary",
					content: summary,
					status:  "done",
				}))
				continue
			}
			entries = append(entries, newChatEntry(chatEntry{role: msg.Role, content: msg.Content}))
		case "assistant":
			for _, call := range msg.ToolCalls {
				if call.ID != "" {
					toolCalls[call.ID] = call
				}
			}
			if strings.TrimSpace(msg.Content) != "" {
				entries = append(entries, newChatEntry(chatEntry{role: "assistant", content: msg.Content}))
			}
		case "tool":
			if summary, ok := compactionSummaryContent(msg); ok {
				entries = append(entries, newChatEntry(chatEntry{
					role:    "compaction_summary",
					content: summary,
					status:  "done",
				}))
				continue
			}
			toolName := msg.ToolName
			var args map[string]any
			if call, ok := toolCalls[msg.ToolCallID]; ok {
				if toolName == "" {
					toolName = call.Function.Name
				}
				args = call.Function.Arguments.ToMap()
			}
			entries = append(entries, newChatEntry(chatEntry{
				role:    "tool",
				content: msg.Content,
				label:   toolInvocationLabel(toolName, args),
				detail:  toolName,
				status:  "done",
				toolID:  msg.ToolCallID,
				args:    args,
			}))
		}
	}
	return groupCompletedToolEntries(entries)
}

func compactionSummaryContent(msg api.Message) (string, bool) {
	return coreagent.CompactionSummaryContent(msg)
}

func groupCompletedToolEntries(entries []chatEntry) []chatEntry {
	grouped := make([]chatEntry, 0, len(entries))
	for i := 0; i < len(entries); {
		if !isCompletedToolHistoryEntry(entries[i]) {
			grouped = append(grouped, entries[i])
			i++
			continue
		}

		start := i
		for i < len(entries) && isCompletedToolHistoryEntry(entries[i]) {
			i++
		}

		tools := flattenToolHistory(entries[start:i])
		if len(tools) <= 1 {
			grouped = append(grouped, entries[start:i]...)
			continue
		}

		group := chatEntry{
			role:       "tool_group",
			label:      fmt.Sprintf("Tool calls (%d)", len(tools)),
			status:     aggregateToolStatus(tools),
			expanded:   anyToolExpanded(tools),
			startedAt:  firstToolStartedAt(tools),
			finishedAt: lastToolFinishedAt(tools),
			tools:      tools,
		}
		if group.status == "error" {
			group.err = "one or more tool calls failed"
		}
		grouped = append(grouped, newChatEntry(group))
	}
	return grouped
}

func anyToolExpanded(tools []chatEntry) bool {
	for _, tool := range tools {
		if tool.expanded {
			return true
		}
	}
	return false
}

func isCompletedToolHistoryEntry(entry chatEntry) bool {
	return (entry.role == "tool" && isToolResultStatus(entry.status)) ||
		(entry.role == "tool_group" && len(entry.tools) > 0)
}

func flattenToolHistory(entries []chatEntry) []chatEntry {
	var tools []chatEntry
	for _, entry := range entries {
		switch entry.role {
		case "tool":
			tools = append(tools, entry)
		case "tool_group":
			tools = append(tools, entry.tools...)
		}
	}
	return tools
}

func aggregateToolStatus(tools []chatEntry) string {
	for _, tool := range tools {
		if tool.err != "" || tool.status == "error" {
			return "error"
		}
	}
	return "done"
}

func firstToolStartedAt(tools []chatEntry) time.Time {
	for _, tool := range tools {
		if !tool.startedAt.IsZero() {
			return tool.startedAt
		}
	}
	return time.Time{}
}

func lastToolFinishedAt(tools []chatEntry) time.Time {
	for i := len(tools) - 1; i >= 0; i-- {
		if !tools[i].finishedAt.IsZero() {
			return tools[i].finishedAt
		}
	}
	return time.Time{}
}

func indentLines(lines []string, prefix string) []string {
	if len(lines) == 0 {
		return nil
	}
	out := make([]string, len(lines))
	for i, line := range lines {
		if line == "" {
			out[i] = prefix
		} else {
			out[i] = prefix + line
		}
	}
	return out
}

func wrapChatText(text string, width int) []string {
	if width < 20 {
		width = 20
	}
	var out []string
	for _, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		for runewidth.StringWidth(line) > width {
			cut := chatDisplayWidthCut(line, width)
			out = append(out, strings.TrimSpace(line[:cut]))
			line = strings.TrimSpace(line[cut:])
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func chatDisplayWidthCut(line string, width int) int {
	hardCut := 0
	currentWidth := 0
	spaceCut := 0
	spaceWidth := 0
	for i := 0; i < len(line); {
		r, size := utf8.DecodeRuneInString(line[i:])
		nextWidth := currentWidth + runewidth.RuneWidth(r)
		if nextWidth > width {
			break
		}
		currentWidth = nextWidth
		hardCut = i + size
		if (r == ' ' || r == '\t') && currentWidth > width/2 {
			spaceCut = i
			spaceWidth = currentWidth
		}
		i += size
	}
	if spaceCut > 0 && spaceWidth > 0 {
		return spaceCut
	}
	if hardCut > 0 {
		return hardCut
	}
	_, size := utf8.DecodeRuneInString(line)
	return size
}
