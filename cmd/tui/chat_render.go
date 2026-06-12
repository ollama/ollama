package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"

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

func (m *chatModel) toggleAllToolOutputs() {
	toolIndexes := m.toolOutputIndexes()
	if len(toolIndexes) == 0 {
		if !m.running {
			return
		}
		m.setToolOutputMode(!m.toolOutputOpen || !m.toolOutputMode)
		return
	}

	expand := false
	for _, index := range toolIndexes {
		if !m.entries[index].expanded {
			expand = true
			break
		}
	}

	m.setToolOutputMode(expand)
}

func (m chatModel) lastExpandableToolEntry() int {
	for i := len(m.entries) - 1; i >= 0; i-- {
		if m.isExpandableTool(i) {
			return i
		}
	}
	return -1
}

func (m chatModel) expandableToolIndexes() []int {
	var indexes []int
	for i := range m.entries {
		if m.isExpandableTool(i) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func (m chatModel) isExpandableTool(index int) bool {
	if index < 0 || index >= len(m.entries) {
		return false
	}
	return entryHasExpandableOutput(m.entries[index])
}

func (m chatModel) toolOutputIndexes() []int {
	var indexes []int
	for i := range m.entries {
		if entryHasToolOutputMode(m.entries[i]) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func entryHasExpandableOutput(entry chatEntry) bool {
	return (entry.role == "tool" && isToolResultStatus(entry.status) && entry.content != "") ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func entryHasToolOutputMode(entry chatEntry) bool {
	return (entry.role == "tool" && (isToolActiveStatus(entry.status) || isToolResultStatus(entry.status) || entry.content != "")) ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func (m *chatModel) setToolOutputMode(open bool) {
	m.toolOutputMode = true
	m.toolOutputOpen = open
	m.applyToolOutputMode()
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

func (m chatModel) visibleTranscriptLines(width, available int) []string {
	if available <= 0 {
		return nil
	}
	lines := m.transcriptLines(width)
	if len(lines) > available {
		maxScroll := len(lines) - available
		scroll := clamp(m.scroll, 0, maxScroll)
		start := maxScroll - scroll
		lines = lines[start : start+available]
	}
	return lines
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
	return max(0, height-2-len(m.bottomLines(width, height-2)))
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
	lines = append(lines, m.completionLines(width)...)
	lines = append(lines, m.queuedLines(width)...)
	if activity := m.activityLine(); activity != "" {
		lines = append(lines, chatMetaStyle.Render(activity))
	}
	fixedLines := len(lines) + 3
	inputBodyLines := maxInputBoxBodyLines
	if maxHeight > 0 {
		inputBodyLines = min(inputBodyLines, max(1, maxHeight-fixedLines))
	}
	lines = append(lines, renderInputBoxLines(string(m.input), width, inputBodyLines)...)
	lines = append(lines, m.renderFooterLine())
	return lines
}

func (m *chatModel) scrollBy(lines int) {
	if lines == 0 {
		return
	}
	m.scroll = clamp(m.scroll+lines, 0, m.maxScroll())
}

func (m chatModel) renderEntry(entry chatEntry) (string, string) {
	switch entry.role {
	case "user":
		return "", entry.content
	case "assistant":
		return chatAssistantStyle.Render("●") + " ", entry.content
	case "compaction_summary":
		prefix := toolStatusStyle(entry.status).Render("●") + " "
		return prefix, compactionSummaryStatusLine(entry)
	case "tool":
		prefix := toolStatusStyle(entry.status).Render("⏺") + " "
		return prefix, toolStatusLine(entry)
	case "tool_group":
		prefix := toolStatusStyle(entry.status).Render("●") + " "
		return prefix, toolGroupStatusLine(entry)
	case "error":
		return chatErrorStyle.Render("err ") + " ", entry.content
	case "system":
		return "", entry.content
	case "history":
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
	case "assistant", "system":
		return splitRenderedBody(renderMarkdownForView(body, width))
	case "history":
		return renderHistoryLines(body, width)
	case "user":
		return renderUserMessageLines(body, width)
	case "compaction_summary":
		return renderCompactionSummaryLines(entry, width)
	case "tool":
		if entry.status == "approval" {
			return m.renderApprovalEntryLines(entry, body, width)
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
	return renderPromptRow("> "+content, width)
}

func renderHistoryLines(history string, width int) []string {
	if width < 20 {
		width = 20
	}
	history = strings.TrimRight(history, "\n")
	if history == "" {
		return []string{""}
	}

	var lines []string
	inFence := false
	fence := ""
	fenceIndent := ""

	for _, rawLine := range strings.Split(history, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		if inFence {
			if fenceEnd(line, fence) {
				inFence = false
				fence = ""
				fenceIndent = ""
				continue
			}
			code := strings.TrimPrefix(line, fenceIndent)
			lines = append(lines, renderHistoryCodeLine(fenceIndent, code, width)...)
			continue
		}

		if nextFence, indent, ok := historyFenceStart(line); ok {
			inFence = true
			fence = nextFence
			fenceIndent = indent
			continue
		}

		lines = append(lines, renderHistoryLine(line, width)...)
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func renderHistoryLine(line string, width int) []string {
	if strings.TrimSpace(line) == "" {
		return []string{""}
	}

	if bold, ok := historyBoldLine(line); ok {
		if bold == "Message History" {
			return renderHistoryStyledLine("", bold, width, chatHistoryTitleStyle)
		}
		return renderHistoryStyledLine("", bold, width, historyRoleStyle(bold))
	}

	indent := leadingWhitespace(line)
	text := strings.TrimSpace(line)
	if strings.Contains(text, " · ") {
		return []string{indent + renderHistoryMetaLine(text)}
	}
	if label, value, ok := historyLabelValue(text); ok {
		return renderHistoryLabelValue(indent, label, value, width)
	}
	return renderHistoryStyledLine(indent, text, width, chatHistoryTextStyle)
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

func historyBoldLine(line string) (string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "**") || !strings.HasSuffix(trimmed, "**") || len(trimmed) <= 4 {
		return "", false
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(trimmed, "**"), "**")), true
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

func historyFenceStart(line string) (string, string, bool) {
	indent := leadingWhitespace(line)
	fence, _, ok := markdownFenceStart(line)
	return fence, indent, ok
}

func leadingWhitespace(line string) string {
	for i, r := range line {
		if r != ' ' && r != '\t' {
			return line[:i]
		}
	}
	return line
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

	if strings.TrimSpace(entry.content) == "" {
		return lines
	}
	lines = append(lines, "")
	lines = append(lines, renderToolOutputLines(entry, entry.content, width)...)
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
	status := toolStatusStyle(entry.status).Render(toolStatusLabel(entry))
	if entry.expanded {
		return fmt.Sprintf("▾ Compacted summary %s", status)
	}
	return fmt.Sprintf("▸ Compacted summary %s", status)
}

func toolGroupChildStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" && isToolResultStatus(entry.status) {
		status += suffix
	}

	return fmt.Sprintf("%s %s", boldToolInvocationName(label), toolStatusStyle(entry.status).Render(status))
}

func boldToolInvocationName(label string) string {
	name, rest, ok := strings.Cut(label, "(")
	if !ok || name == "" {
		return chatHeaderStyle.Render(label)
	}
	return chatHeaderStyle.Render(name) + "(" + rest
}

func renderToolOutputLines(entry chatEntry, output string, width int) []string {
	if looksLikeUnifiedDiff(output) {
		return splitRenderedBody(renderDiffForView(output, width))
	}
	if toolOutputUsesMarkdown(entry.detail) {
		return splitRenderedBody(renderMarkdownForView(output, width))
	}
	return wrapChatText(output, width)
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

func toolStatusLine(entry chatEntry) string {
	return toolStatusLineWithArrow(entry, true)
}

func toolStatusLineWithArrow(entry chatEntry, arrow bool) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" && isToolResultStatus(entry.status) {
		status += suffix
	}

	if arrow && isToolResultStatus(entry.status) && entry.content != "" {
		if entry.expanded {
			return fmt.Sprintf("▾ %s %s", label, toolStatusStyle(entry.status).Render(status))
		}
		return fmt.Sprintf("▸ %s %s", label, toolStatusStyle(entry.status).Render(status))
	}
	return fmt.Sprintf("%s %s", label, toolStatusStyle(entry.status).Render(status))
}

func toolGroupStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = fmt.Sprintf("Tool calls (%d)", len(entry.tools))
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" {
		status += suffix
	}

	if entry.expanded {
		return fmt.Sprintf("▾ %s %s", label, toolStatusStyle(entry.status).Render(status))
	}
	return fmt.Sprintf("▸ %s %s", label, toolStatusStyle(entry.status).Render(status))
}

func toolStatusLabel(entry chatEntry) string {
	if entry.err != "" || entry.status == "error" {
		return "failed"
	}
	if entry.status == "done" {
		return "done"
	}
	if entry.status == "approval" {
		return "needs approval"
	}
	return "in progress"
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
	switch name {
	case "web_search":
		if query, ok := stringArg(args, "query"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(query))
		}
	case "web_fetch":
		if targetURL, ok := stringArg(args, "url"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(targetURL))
		}
	case "bash":
		if command, ok := stringArg(args, "command"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(command))
		}
	case "read", "list":
		if path, ok := stringArg(args, "path"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(path))
		}
	case "edit":
		if path, ok := stringArg(args, "path"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(path))
		}
	}
	if len(args) == 0 {
		return displayName
	}
	return fmt.Sprintf("%s(%s)", displayName, formatToolArgs(args))
}

func toolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
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

func toolElapsedSuffix(startedAt, finishedAt time.Time) string {
	if startedAt.IsZero() || finishedAt.IsZero() || finishedAt.Before(startedAt) {
		return ""
	}
	elapsed := finishedAt.Sub(startedAt)
	if elapsed < time.Second {
		return " in " + elapsed.Round(time.Millisecond).String()
	}
	return " in " + elapsed.Round(time.Second).String()
}

func toolOutputUsesMarkdown(name string) bool {
	switch name {
	case "read", "skill", "web_search", "web_fetch":
		return true
	default:
		return false
	}
}

func formatToolArgs(args map[string]any) string {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", key, quoteToolArg(args[key])))
	}
	return strings.Join(parts, ", ")
}

func quoteToolArg(value any) string {
	switch v := value.(type) {
	case string:
		return strconv.Quote(truncateRunes(v, 100))
	default:
		return fmt.Sprint(v)
	}
}

func stringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return truncateRunes(value, 120), true
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func (m chatModel) statusLine() string {
	var parts []string
	if scroll := m.scrollStatus(); scroll != "" {
		parts = append(parts, scroll)
	}
	return strings.Join(parts, "  ")
}

func (m chatModel) footerLine() string {
	return strings.Join(m.footerParts(), " • ")
}

func (m chatModel) footerParts() []string {
	var parts []string
	if !m.running && !m.compacting && m.approvalPrompt == nil && m.status != "" && m.status != "ready" {
		parts = append(parts, m.status)
	}

	if len(m.queued) > 0 {
		parts = append(parts, fmt.Sprintf("queued %d", len(m.queued)))
	}

	action := "enter send"
	if m.approvalPrompt != nil {
		action = "enter approve"
	} else if m.running || m.compacting {
		action = "enter queue"
	}
	controls := action
	if m.approvalPrompt != nil {
		controls += " • ←/→ choose • o once • s session • d deny • esc deny"
	} else {
		controls += " • ctrl+c clear/cancel/quit"
	}
	controls += " • shift+tab"
	if m.lastExpandableToolEntry() >= 0 {
		controls += " • ctrl+o details"
	}
	parts = append(parts, controls)
	parts = append(parts, m.permissionModeStatus())
	if cwd := m.cwdStatus(); cwd != "" {
		parts = append(parts, cwd)
	}
	if contextStatus := m.contextStatus(); contextStatus != "" {
		parts = append(parts, contextStatus)
	}
	return parts
}

func (m chatModel) renderFooterLine() string {
	parts := m.footerParts()
	for i, part := range parts {
		if part == "full access" {
			parts[i] = chatFullAccessStyle.Render(part)
			continue
		}
		parts[i] = chatMetaStyle.Render(part)
	}
	return strings.Join(parts, chatMetaStyle.Render(" • "))
}

func (m chatModel) permissionModeStatus() string {
	if m.autoApproveTools() {
		return "full access"
	}
	return "review"
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

func (m chatModel) cwdStatus() string {
	workingDir := strings.TrimSpace(m.currentWorkingDir())
	rootDir := strings.TrimSpace(m.opts.RootDir)
	if workingDir == "" || rootDir == "" {
		return ""
	}
	rel, err := filepath.Rel(rootDir, workingDir)
	if err != nil || rel == "." {
		return ""
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "cwd " + workingDir
	}
	return "cwd ./" + filepath.ToSlash(rel)
}

func (m chatModel) activityLine() string {
	if !m.running && !m.compacting && m.approvalPrompt == nil {
		return ""
	}
	status := m.spinnerFrame()
	if label := m.activityLabel(); label != "" {
		status += " " + label
	}
	return status
}

func (m chatModel) activityLabel() string {
	if m.status == "canceling" {
		return "canceling"
	}
	if m.approvalPrompt != nil {
		return "waiting for approval"
	}
	if m.compacting {
		if m.compactingTokens > 0 {
			return "compacting " + formatTokenCount(m.compactingTokens)
		}
		return "compacting"
	}
	if m.thinking {
		if m.thinkingTokens > 0 {
			return "thinking " + formatTokenCount(m.thinkingTokens)
		}
		return "thinking"
	}
	for i := len(m.entries) - 1; i >= 0; i-- {
		entry := m.entries[i]
		switch entry.role {
		case "tool":
			if isToolActiveStatus(entry.status) {
				active := m.activeToolLabels()
				if len(active) == 1 {
					return "using " + active[0]
				}
				if len(active) > 1 {
					return fmt.Sprintf("using %d tools", len(active))
				}
				return "using tools"
			}
		case "assistant":
			if entry.content != "" {
				return ""
			}
		}
	}
	return "thinking"
}

func (m chatModel) activeToolLabels() []string {
	var labels []string
	for _, entry := range m.entries {
		if entry.role != "tool" || !isToolActiveStatus(entry.status) {
			continue
		}
		label := entry.label
		if label == "" {
			label = toolDisplayName(entry.detail)
		}
		labels = append(labels, label)
	}
	return labels
}

func (m *chatModel) applyResponseMetrics(response *api.ChatResponse) {
	if response == nil {
		return
	}
	if response.PromptEvalCount > 0 {
		m.contextTokens = response.PromptEvalCount
		m.contextEstimate = false
	}
}

func eventEvalCount(event coreagent.Event) int {
	if event.Response == nil {
		return 0
	}
	return event.Response.EvalCount
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
	requestMessages := slices.Clone(messages)
	if strings.TrimSpace(systemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: strings.TrimSpace(systemPrompt)})
		requestMessages = append(requestMessages, messages...)
	}
	if len(requestMessages) == 0 && len(tools) == 0 && strings.TrimSpace(format) == "" {
		return 0
	}

	payload := struct {
		Messages []api.Message   `json:"messages,omitempty"`
		Tools    api.Tools       `json:"tools,omitempty"`
		Format   json.RawMessage `json:"format,omitempty"`
	}{
		Messages: requestMessages,
		Tools:    tools,
	}
	if rawFormat, ok := promptFormatForEstimate(format); ok {
		payload.Format = rawFormat
	}

	if b, err := json.Marshal(payload); err == nil {
		return estimateTokenCount(string(b))
	}

	var runes int
	for _, msg := range requestMessages {
		runes += estimateMessageRunes(msg)
	}
	runes += len([]rune(tools.String()))
	runes += len([]rune(strings.TrimSpace(format)))
	if runes == 0 {
		return 0
	}
	return max(1, (runes+3)/4)
}

func promptFormatForEstimate(format string) (json.RawMessage, bool) {
	format = strings.TrimSpace(format)
	if format == "" {
		return nil, false
	}
	if format == "json" {
		format = `"` + format + `"`
	}
	if !json.Valid([]byte(format)) {
		return nil, false
	}
	return json.RawMessage(format), true
}

func estimateMessageRunes(msg api.Message) int {
	var runes int
	runes += len([]rune(msg.Role))
	runes += len([]rune(msg.Content))
	runes += len([]rune(msg.Thinking))
	runes += len([]rune(msg.ToolName))
	runes += len([]rune(msg.ToolCallID))
	for _, image := range msg.Images {
		runes += len(image)
	}
	for _, call := range msg.ToolCalls {
		runes += len([]rune(call.ID))
		runes += len([]rune(call.Function.Name))
		runes += len([]rune(fmt.Sprint(call.Function.Arguments.ToMap())))
	}
	return runes
}

func estimateTokenCount(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return max(1, (len([]rune(text))+3)/4)
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
	used := clamp(m.contextTokens, 0, window)
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
		return fmt.Sprintf("ctx %s%s/%s (%d%%) • compact due at %s", prefix, formatInteger(used), formatInteger(window), percent, formatInteger(compactAt))
	}

	noticeDistance := int(float64(window)*0.1 + 0.999999)
	if noticeDistance < 1 {
		noticeDistance = 1
	}
	if compactAt-used <= noticeDistance {
		return fmt.Sprintf("ctx %s%s/%s (%d%%) • compact at %s", prefix, formatInteger(used), formatInteger(window), percent, formatInteger(compactAt))
	}

	return fmt.Sprintf("ctx %s%s/%s (%d%%)", prefix, formatInteger(used), formatInteger(window), percent)
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

func (m chatModel) scrollStatus() string {
	maxScroll := m.maxScroll()
	if maxScroll <= 0 {
		return ""
	}
	scroll := clamp(m.scroll, 0, maxScroll)
	if scroll == 0 {
		return "↑ more"
	}
	if scroll == maxScroll {
		return "↓ more"
	}
	return "↑/↓ more"
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

func truncateRenderedLine(line string, width int) string {
	if width <= 0 || lipgloss.Width(line) <= width {
		return line
	}
	return lipgloss.NewStyle().MaxWidth(width).Render(line)
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
	if msg.Role != "user" && msg.Role != "system" {
		return "", false
	}
	if !strings.HasPrefix(msg.Content, chatCompactionSummaryPrefix) {
		return "", false
	}
	return strings.TrimSpace(strings.TrimPrefix(msg.Content, chatCompactionSummaryPrefix)), true
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
		for len([]rune(line)) > width {
			runes := []rune(line)
			cut := width
			for i := width; i > width/2; i-- {
				if runes[i-1] == ' ' || runes[i-1] == '\t' {
					cut = i
					break
				}
			}
			out = append(out, strings.TrimSpace(string(runes[:cut])))
			line = strings.TrimSpace(string(runes[cut:]))
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}
