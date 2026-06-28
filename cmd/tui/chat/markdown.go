package chat

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

func renderMarkdownForView(markdown string, width int) string {
	if width < 20 {
		width = 20
	}

	source := strings.Split(strings.TrimRight(markdown, "\n"), "\n")
	var rendered []string
	inCodeBlock := false
	for i := 0; i < len(source); i++ {
		line := strings.TrimRight(source[i], "\r")
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "```") {
			inCodeBlock = !inCodeBlock
			continue
		}
		if inCodeBlock {
			rendered = append(rendered, renderMarkdownCodeLine(line, width)...)
			continue
		}

		if table, consumed := renderMarkdownTable(source[i:], width); consumed > 0 {
			rendered = append(rendered, table...)
			i += consumed - 1
			continue
		}

		if heading, ok := markdownHeading(trimmed); ok {
			rendered = append(rendered, chatHeaderStyle.Render(heading))
			continue
		}

		if trimmed == "" {
			rendered = append(rendered, "")
			continue
		}
		for _, wrapped := range wrapChatText(line, width) {
			rendered = append(rendered, renderMarkdownInline(wrapped))
		}
	}
	return strings.Join(rendered, "\n")
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}

func markdownHeading(line string) (string, bool) {
	if !strings.HasPrefix(line, "#") {
		return "", false
	}
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(line) || line[level] != ' ' {
		return "", false
	}
	return strings.TrimSpace(line[level:]), true
}

func renderMarkdownInline(line string) string {
	var b strings.Builder
	for {
		before, rest, ok := strings.Cut(line, "`")
		b.WriteString(before)
		if !ok {
			break
		}
		code, after, ok := strings.Cut(rest, "`")
		if !ok {
			b.WriteString("`")
			b.WriteString(rest)
			break
		}
		b.WriteString(chatInlineCodeStyle.Render(code))
		line = after
	}
	return b.String()
}

func renderMarkdownCodeLine(line string, width int) []string {
	codeWidth := max(1, width-2)
	lines := wrapChatText(line, codeWidth)
	for i, wrapped := range lines {
		lines[i] = "  " + chatCodeBlockStyle.Render(wrapped)
	}
	return lines
}

func renderMarkdownTable(lines []string, width int) ([]string, int) {
	if len(lines) < 2 || !looksLikeMarkdownTableRow(lines[0]) || !isMarkdownTableSeparator(lines[1]) {
		return nil, 0
	}

	var rows [][]string
	consumed := 0
	for consumed < len(lines) && looksLikeMarkdownTableRow(lines[consumed]) {
		if consumed == 1 && isMarkdownTableSeparator(lines[consumed]) {
			consumed++
			continue
		}
		rows = append(rows, parseMarkdownTableRow(lines[consumed]))
		consumed++
	}
	if len(rows) == 0 {
		return nil, 0
	}

	columnCount := 0
	for _, row := range rows {
		columnCount = max(columnCount, len(row))
	}
	widths := make([]int, columnCount)
	for _, row := range rows {
		for i := 0; i < columnCount; i++ {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			widths[i] = max(widths[i], lipglossWidth(cell))
		}
	}

	var rendered []string
	for rowIndex, row := range rows {
		cells := make([]string, columnCount)
		for i := 0; i < columnCount; i++ {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			cells[i] = padPlainLine(cell, widths[i])
		}
		line := strings.Join(cells, chatTableBorderStyle.Render(" | "))
		if rowIndex == 0 {
			line = chatHeaderStyle.Render(stripANSIForWidth(line))
		}
		if lipglossWidth(line) > width {
			line = truncateRunes(stripANSIForWidth(line), width)
		}
		rendered = append(rendered, line)
	}
	return rendered, consumed
}

func looksLikeMarkdownTableRow(line string) bool {
	line = strings.TrimSpace(line)
	return strings.Contains(line, "|") && strings.Count(line, "|") >= 1
}

func isMarkdownTableSeparator(line string) bool {
	cells := parseMarkdownTableRow(line)
	if len(cells) == 0 {
		return false
	}
	for _, cell := range cells {
		cell = strings.Trim(cell, " :-")
		if cell != "" {
			return false
		}
	}
	return true
}

func parseMarkdownTableRow(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	raw := strings.Split(line, "|")
	cells := make([]string, 0, len(raw))
	for _, cell := range raw {
		cells = append(cells, strings.TrimSpace(cell))
	}
	return cells
}

func padPlainLine(line string, width int) string {
	if extra := width - lipglossWidth(line); extra > 0 {
		return line + strings.Repeat(" ", extra)
	}
	return line
}

func stripANSIForWidth(line string) string {
	return stripChatANSI(line)
}

func lipglossWidth(line string) int {
	return lipgloss.Width(line)
}
