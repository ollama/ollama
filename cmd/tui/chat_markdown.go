package tui

import (
	"regexp"
	"slices"
	"strings"
	"sync"
	"unicode"

	"github.com/charmbracelet/glamour"
	glamouransi "github.com/charmbracelet/glamour/ansi"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
)

var markdownLinkPattern = regexp.MustCompile(`\[([^\]]+)\]\((https?://[^)\s]+)\)`)

var markdownTableSeparatorPattern = regexp.MustCompile(`^:?-{3,}:?$`)

const maxMarkdownRendererCacheEntries = 8

var chatMarkdownRenderers = newMarkdownRendererCache()

type markdownRendererCache struct {
	mu        sync.Mutex
	renderers map[int]*cachedMarkdownRenderer
	order     []int
}

func newMarkdownRendererCache() *markdownRendererCache {
	return &markdownRendererCache{renderers: make(map[int]*cachedMarkdownRenderer)}
}

type cachedMarkdownRenderer struct {
	renderer *glamour.TermRenderer
	mu       sync.Mutex
}

func renderMarkdownForView(markdown string, width int) string {
	if strings.TrimSpace(markdown) == "" {
		return markdown
	}
	if width < 20 {
		width = 20
	}
	return renderMarkdownBlocks(markdownBlocks(markdown), width)
}

func renderMarkdownPlain(markdown string, width int) string {
	return renderMarkdownBlocks(markdownBlocks(markdown), width)
}

func exposeMarkdownLinks(markdown string) string {
	return markdownLinkPattern.ReplaceAllString(markdown, "$1 ($2)")
}

func renderMarkdownChunk(markdown string, width int) string {
	renderer, err := markdownRendererForWidth(width)
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	renderer.mu.Lock()
	rendered, err := renderer.renderer.Render(markdown)
	renderer.mu.Unlock()
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	return trimRenderedLines(rendered)
}

type markdownBlockKind int

type markdownBlock struct {
	kind  markdownBlockKind
	text  string
	table markdownTable
}

type markdownTable struct {
	headers []string
	rows    [][]string
}

func markdownBlocks(markdown string) []markdownBlock {
	lines := strings.Split(markdown, "\n")
	blocks := make([]markdownBlock, 0, 1)
	var prose []string

	flushProse := func() {
		if len(prose) == 0 {
			return
		}
		blocks = append(blocks, markdownBlock{kind: markdownBlockProse, text: strings.Join(prose, "\n")})
		prose = nil
	}

	for i := 0; i < len(lines); {
		if fence, info, ok := markdownFenceStart(lines[i]); ok {
			flushProse()
			if markdownFenceIsDiff(info) {
				var diffLines []string
				i++
				for ; i < len(lines); i++ {
					if fenceEnd(lines[i], fence) {
						break
					}
					diffLines = append(diffLines, lines[i])
				}
				if i < len(lines) {
					i++
				}
				blocks = append(blocks, markdownBlock{kind: markdownBlockDiffFence, text: strings.Join(diffLines, "\n")})
				continue
			}

			codeLines := []string{lines[i]}
			i++
			for ; i < len(lines); i++ {
				codeLines = append(codeLines, lines[i])
				if fenceEnd(lines[i], fence) {
					i++
					break
				}
			}
			blocks = append(blocks, markdownBlock{kind: markdownBlockCodeFence, text: strings.Join(codeLines, "\n")})
			continue
		}

		if table, next, ok := markdownTableAt(lines, i); ok {
			flushProse()
			blocks = append(blocks, markdownBlock{kind: markdownBlockTable, table: table})
			i = next
			continue
		}

		prose = append(prose, lines[i])
		i++
	}
	flushProse()
	return blocks
}

func renderMarkdownBlocks(blocks []markdownBlock, width int) string {
	if width < 20 {
		width = 20
	}
	var rendered []string
	for _, block := range blocks {
		switch block.kind {
		case markdownBlockProse:
			chunk := renderMarkdownChunk(exposeMarkdownLinks(block.text), width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockTable:
			rendered = append(rendered, renderMarkdownTable(block.table, width))
		case markdownBlockCodeFence:
			chunk := renderMarkdownChunk(block.text, width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockDiffFence:
			rendered = append(rendered, renderDiffForView(block.text, width))
		}
	}
	return trimRenderedLines(strings.Join(rendered, "\n"))
}

func renderMarkdownTables(markdown string, width int) (string, bool) {
	blocks := markdownBlocks(markdown)
	found := false
	for _, block := range blocks {
		if block.kind == markdownBlockTable {
			found = true
			break
		}
	}
	if !found {
		return "", false
	}
	return renderMarkdownBlocks(blocks, width), true
}

func markdownTableAt(lines []string, start int) (markdownTable, int, bool) {
	if start+1 >= len(lines) {
		return markdownTable{}, start, false
	}
	if markdownLineIsIndentedCode(lines[start]) || markdownLineIsIndentedCode(lines[start+1]) {
		return markdownTable{}, start, false
	}
	headers, ok := parseMarkdownTableRow(lines[start])
	if !ok {
		return markdownTable{}, start, false
	}
	separator, ok := parseMarkdownTableRow(lines[start+1])
	if !ok || !isMarkdownTableSeparator(separator) {
		return markdownTable{}, start, false
	}

	table := markdownTable{headers: headers}
	columns := len(headers)
	i := start + 2
	for ; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "" {
			break
		}
		row, ok := parseMarkdownTableRow(lines[i])
		if !ok || isMarkdownTableSeparator(row) {
			break
		}
		if len(row) > columns {
			columns = len(row)
		}
		table.rows = append(table.rows, row)
	}
	table.headers = padMarkdownTableRow(table.headers, columns)
	for row := range table.rows {
		table.rows[row] = padMarkdownTableRow(table.rows[row], columns)
	}
	return table, i, true
}

func parseMarkdownTableRow(line string) ([]string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.Contains(trimmed, "|") {
		return nil, false
	}
	if strings.HasPrefix(trimmed, "|") {
		trimmed = strings.TrimPrefix(trimmed, "|")
	}
	if strings.HasSuffix(trimmed, "|") {
		trimmed = strings.TrimSuffix(trimmed, "|")
	}

	var cells []string
	var b strings.Builder
	escaped := false
	for _, r := range trimmed {
		switch {
		case escaped:
			if r != '|' {
				b.WriteRune('\\')
			}
			b.WriteRune(r)
			escaped = false
		case r == '\\':
			escaped = true
		case r == '|':
			cells = append(cells, strings.TrimSpace(b.String()))
			b.Reset()
		default:
			b.WriteRune(r)
		}
	}
	if escaped {
		b.WriteRune('\\')
	}
	cells = append(cells, strings.TrimSpace(b.String()))
	if len(cells) < 2 {
		return nil, false
	}
	return cells, true
}

func isMarkdownTableSeparator(cells []string) bool {
	if len(cells) < 2 {
		return false
	}
	for _, cell := range cells {
		normalized := strings.ReplaceAll(strings.TrimSpace(cell), " ", "")
		if !markdownTableSeparatorPattern.MatchString(normalized) {
			return false
		}
	}
	return true
}

func padMarkdownTableRow(row []string, columns int) []string {
	if len(row) >= columns {
		return row
	}
	out := slices.Clone(row)
	for len(out) < columns {
		out = append(out, "")
	}
	return out
}

func renderMarkdownTable(table markdownTable, width int) string {
	if width < 20 {
		width = 20
	}
	rows := make([][]string, 0, len(table.rows)+1)
	rows = append(rows, table.headers)
	rows = append(rows, table.rows...)

	widths := markdownTableColumnWidths(rows, width)
	var rendered []string
	rendered = append(rendered, renderMarkdownTableRow(table.headers, widths, true)...)
	rendered = append(rendered, strings.Repeat("─", min(width, markdownTableRenderedWidth(widths))))
	for _, row := range table.rows {
		rendered = append(rendered, renderMarkdownTableRow(row, widths, false)...)
	}
	return strings.Join(rendered, "\n")
}

func markdownTableColumnWidths(rows [][]string, width int) []int {
	columns := 0
	for _, row := range rows {
		columns = max(columns, len(row))
	}
	if columns == 0 {
		return nil
	}
	gapWidth := 2 * (columns - 1)
	available := max(columns, width-gapWidth)
	minWidth := 4
	if available < columns*minWidth {
		minWidth = max(1, available/columns)
	}

	maxWidths := make([]int, columns)
	for _, row := range rows {
		for col := 0; col < columns; col++ {
			cell := ""
			if col < len(row) {
				cell = cleanMarkdownTableCell(row[col])
			}
			for _, line := range strings.Split(cell, "\n") {
				maxWidths[col] = max(maxWidths[col], lipgloss.Width(line))
			}
		}
	}

	widths := make([]int, columns)
	even := max(minWidth, available/columns)
	for col := range widths {
		widths[col] = min(max(maxWidths[col], minWidth), even)
	}

	for leftover := available - sumInts(widths); leftover > 0; {
		grew := false
		for col := columns - 1; col >= 0 && leftover > 0; col-- {
			if widths[col] >= maxWidths[col] {
				continue
			}
			widths[col]++
			leftover--
			grew = true
		}
		if !grew {
			widths[columns-1] += leftover
			break
		}
	}

	for sumInts(widths) > available {
		col := widestColumn(widths)
		if widths[col] <= minWidth {
			break
		}
		widths[col]--
	}
	return widths
}

func renderMarkdownTableRow(row []string, widths []int, header bool) []string {
	wrapped := make([][]string, len(widths))
	height := 1
	for col := range widths {
		cell := ""
		if col < len(row) {
			cell = cleanMarkdownTableCell(row[col])
		}
		wrapped[col] = wrapTableCell(cell, widths[col])
		height = max(height, len(wrapped[col]))
	}

	lines := make([]string, 0, height)
	for lineIndex := 0; lineIndex < height; lineIndex++ {
		var b strings.Builder
		for col := range widths {
			if col > 0 {
				b.WriteString("  ")
			}
			part := ""
			if lineIndex < len(wrapped[col]) {
				part = wrapped[col][lineIndex]
			}
			if header || (col == 0 && strings.TrimSpace(part) != "") {
				part = chatHeaderStyle.Render(part)
			}
			if col < len(widths)-1 {
				part = padRenderedCell(part, widths[col])
			}
			b.WriteString(part)
		}
		lines = append(lines, strings.TrimRight(b.String(), " "))
	}
	return lines
}

func cleanMarkdownTableCell(cell string) string {
	cell = strings.TrimSpace(cell)
	cell = strings.ReplaceAll(cell, "<br>", "\n")
	cell = strings.ReplaceAll(cell, "<br/>", "\n")
	cell = strings.ReplaceAll(cell, "<br />", "\n")
	cell = exposeMarkdownLinks(cell)
	cell = strings.ReplaceAll(cell, "**", "")
	cell = strings.ReplaceAll(cell, "__", "")
	cell = strings.Trim(cell, "`")
	return strings.ReplaceAll(cell, `\|`, "|")
}

func wrapTableCell(text string, width int) []string {
	if width <= 0 {
		return []string{""}
	}
	var out []string
	for _, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		for lipgloss.Width(line) > width {
			runes := []rune(line)
			cut := min(width, len(runes))
			for i := cut; i > max(1, cut/2); i-- {
				if unicode.IsSpace(runes[i-1]) {
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

func markdownTableRenderedWidth(widths []int) int {
	if len(widths) == 0 {
		return 0
	}
	return sumInts(widths) + 2*(len(widths)-1)
}

func padRenderedCell(cell string, width int) string {
	padding := width - lipgloss.Width(cell)
	if padding <= 0 {
		return cell
	}
	return cell + strings.Repeat(" ", padding)
}

func sumInts(values []int) int {
	total := 0
	for _, value := range values {
		total += value
	}
	return total
}

func widestColumn(widths []int) int {
	widest := 0
	for i := 1; i < len(widths); i++ {
		if widths[i] > widths[widest] {
			widest = i
		}
	}
	return widest
}

func markdownRendererForWidth(width int) (*cachedMarkdownRenderer, error) {
	return chatMarkdownRenderers.renderer(width)
}

func (c *markdownRendererCache) renderer(width int) (*cachedMarkdownRenderer, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if cached, ok := c.renderers[width]; ok {
		return cached, nil
	}
	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(compactMarkdownStyle()),
		glamour.WithWordWrap(width),
		glamour.WithTableWrap(true),
		glamour.WithInlineTableLinks(true),
	)
	if err != nil {
		return nil, err
	}
	cached := &cachedMarkdownRenderer{renderer: renderer}
	if len(c.order) >= maxMarkdownRendererCacheEntries {
		evict := c.order[0]
		c.order = c.order[1:]
		delete(c.renderers, evict)
	}
	c.renderers[width] = cached
	c.order = append(c.order, width)
	return cached, nil
}

func (c *markdownRendererCache) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.renderers)
}

func renderMarkdownDiffFences(markdown string, width int) (string, bool) {
	blocks := markdownBlocks(markdown)
	found := false
	for _, block := range blocks {
		if block.kind == markdownBlockDiffFence {
			found = true
			break
		}
	}
	if !found {
		return "", false
	}
	return renderMarkdownBlocks(blocks, width), true
}

func diffFenceStart(line string) (string, bool) {
	fence, info, ok := markdownFenceStart(line)
	if !ok || !markdownFenceIsDiff(info) {
		return "", false
	}
	return fence, true
}

func markdownFenceStart(line string) (string, string, bool) {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "```") {
		return markdownFence(trimmed, '`')
	}
	if strings.HasPrefix(trimmed, "~~~") {
		return markdownFence(trimmed, '~')
	}
	return "", "", false
}

func markdownFence(line string, marker rune) (string, string, bool) {
	count := 0
	for _, r := range line {
		if r != marker {
			break
		}
		count++
	}
	if count < 3 {
		return "", "", false
	}
	info := strings.TrimSpace(line[count:])
	return strings.Repeat(string(marker), count), info, true
}

func markdownFenceIsDiff(info string) bool {
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(info)))
	if len(fields) == 0 {
		return false
	}
	first := strings.Trim(fields[0], "{}.")
	return first == "diff" || first == "patch"
}

func fenceEnd(line string, fence string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, fence)
}

func markdownLineIsIndentedCode(line string) bool {
	if strings.HasPrefix(line, "\t") {
		return true
	}
	spaces := 0
	for _, r := range line {
		if r != ' ' {
			break
		}
		spaces++
	}
	return spaces >= 4
}

func compactMarkdownStyle() glamouransi.StyleConfig {
	style := styles.ASCIIStyleConfig
	style.Document.Margin = uintPtr(0)
	style.Document.BlockPrefix = ""
	style.Document.BlockSuffix = ""
	style.CodeBlock.Margin = uintPtr(0)
	style.Table.Margin = uintPtr(0)
	style.Strong.BlockPrefix = ""
	style.Strong.BlockSuffix = ""
	style.Strong.Bold = boolPtr(true)
	style.Emph.BlockPrefix = ""
	style.Emph.BlockSuffix = ""
	style.Emph.Italic = boolPtr(true)
	return style
}

func trimRenderedLines(rendered string) string {
	rendered = strings.TrimRight(rendered, "\n")
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return strings.Join(lines, "\n")
}

func uintPtr(value uint) *uint {
	return &value
}

func boolPtr(value bool) *bool {
	return &value
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}
