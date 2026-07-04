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
	naturalWidths := make([]int, columnCount)
	for _, row := range rows {
		for i := 0; i < columnCount; i++ {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			naturalWidths[i] = max(naturalWidths[i], lipglossWidth(cell))
		}
	}
	widths := markdownTableColumnWidths(naturalWidths, width)

	var rendered []string
	for rowIndex, row := range rows {
		wrappedCells := make([][]string, columnCount)
		rowHeight := 1
		for i := 0; i < columnCount; i++ {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			wrappedCells[i] = wrapMarkdownTableCell(cell, widths[i])
			rowHeight = max(rowHeight, len(wrappedCells[i]))
		}
		for lineIndex := 0; lineIndex < rowHeight; lineIndex++ {
			cells := make([]string, columnCount)
			for i := 0; i < columnCount; i++ {
				cellLine := ""
				if lineIndex < len(wrappedCells[i]) {
					cellLine = wrappedCells[i][lineIndex]
				}
				cells[i] = padPlainLine(cellLine, widths[i])
			}
			line := strings.Join(cells, chatTableBorderStyle.Render(" | "))
			if rowIndex == 0 {
				line = chatHeaderStyle.Render(stripANSIForWidth(line))
			}
			rendered = append(rendered, line)
		}
	}
	return rendered, consumed
}

func markdownTableColumnWidths(naturalWidths []int, width int) []int {
	if len(naturalWidths) == 0 {
		return nil
	}
	separatorWidth := max(0, len(naturalWidths)-1) * lipglossWidth(" | ")
	available := max(1, width-separatorWidth)
	widths := make([]int, len(naturalWidths))
	minWidths := make([]int, len(naturalWidths))
	for i, natural := range naturalWidths {
		widths[i] = max(1, natural)
		minWidth := min(widths[i], 12)
		if i == 0 {
			minWidth = min(widths[i], 4)
		}
		minWidths[i] = max(1, minWidth)
	}

	for sumInts(widths) > available {
		index := widestShrinkableColumn(widths, minWidths)
		if index < 0 {
			break
		}
		widths[index]--
	}
	for sumInts(widths) > available {
		index := widestColumn(widths)
		if index < 0 || widths[index] <= 1 {
			break
		}
		widths[index]--
	}
	return widths
}

func widestShrinkableColumn(widths, minWidths []int) int {
	index := -1
	for i, width := range widths {
		if width <= minWidths[i] {
			continue
		}
		if index < 0 || width > widths[index] {
			index = i
		}
	}
	return index
}

func widestColumn(widths []int) int {
	index := -1
	for i, width := range widths {
		if index < 0 || width > widths[index] {
			index = i
		}
	}
	return index
}

func sumInts(values []int) int {
	sum := 0
	for _, value := range values {
		sum += value
	}
	return sum
}

func wrapMarkdownTableCell(cell string, width int) []string {
	width = max(1, width)
	var out []string
	line := strings.TrimSpace(cell)
	for lipglossWidth(line) > width {
		cut := chatDisplayWidthCut(line, width)
		out = append(out, strings.TrimSpace(line[:cut]))
		line = strings.TrimSpace(line[cut:])
	}
	out = append(out, line)
	if len(out) == 0 {
		return []string{""}
	}
	return out
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
